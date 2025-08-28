#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import sys

import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from tqdm import tqdm

SHARD_POS = 100_000
FLUSH_EVERY = 25_000
LOGITS = 64 * 64 * 5
TOTAL_CHANNELS = 36
META_FILENAME = "_meta.json"

PIECE_PLANE = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
    'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11,
}

MISSING_PIECE_PLANE = {
    'p': 12, 'n': 13, 'b': 14, 'r': 15, 'q': 16,
    'P': 17, 'N': 18, 'B': 19, 'R': 20, 'Q': 21,
}

PIECE_VAL = {
    'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0,
    'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
}

PROMO_ADD = {
    None: 0,
    chess.QUEEN: 4_096,
    chess.ROOK: 8_192,
    chess.BISHOP: 12_288,
    chess.KNIGHT: 16_384,
}
assert max(PROMO_ADD.values()) + 64 * 64 - 1 == LOGITS - 1, \
    "PROMO_ADD table and LOGITS constant are out of sync."

def square_rc(sq: int) -> Tuple[int, int]:
    r, c = divmod(sq, 8)
    return 7 - r, c

def uci_index(uci: str) -> Optional[int]:
    try:
        mv = chess.Move.from_uci(uci)
        return mv.from_square * 64 + mv.to_square + PROMO_ADD[mv.promotion]
    except Exception:
        return None

def tensor_from_board(
    brd: chess.Board,
    prev: Optional[chess.Move],
    legal_moves: List[chess.Move],
) -> np.ndarray:
    t = np.zeros((8, 8, TOTAL_CHANNELS), np.float32)
    for sq, pc in brd.piece_map().items():
        r, c = square_rc(sq)
        t[r, c, PIECE_PLANE[pc.symbol()]] = 1.0
    present = {p.symbol() for p in brd.piece_map().values()}
    for sym, ch in MISSING_PIECE_PLANE.items():
        if sym not in present:
            t[:, :, ch] = 1.0
    t[:, :, 22] = 1.0 if brd.turn else -1.0
    t[:, :, 23] = 1.0 if brd.has_kingside_castling_rights(chess.WHITE) else 0.0
    t[:, :, 24] = 1.0 if brd.has_queenside_castling_rights(chess.WHITE) else 0.0
    t[:, :, 25] = 1.0 if brd.has_kingside_castling_rights(chess.BLACK) else 0.0
    t[:, :, 26] = 1.0 if brd.has_queenside_castling_rights(chess.BLACK) else 0.0
    t[:, :, 27] = 1.0 if brd.is_check() else 0.0
    t[:, :, 28] = np.tanh((brd.fullmove_number - 40) / 40.0)
    if prev is not None:
        fr_r, fr_c = square_rc(prev.from_square)
        to_r, to_c = square_rc(prev.to_square)
        t[fr_r, fr_c, 29] = 1.0
        t[to_r, to_c, 30] = 1.0
    if brd.ep_square is not None:
        ep_r, ep_c = square_rc(brd.ep_square)
        t[ep_r, ep_c, 31] = 1.0
    t[:, :, 32] = np.log1p(len(legal_moves)) / np.log1p(218.0)
    t[:, :, 33] = 1.0 if any(brd.gives_check(m) for m in legal_moves) else 0.0
    t[:, :, 34] = np.tanh(brd.halfmove_clock / 50.0)
    material = sum(
        PIECE_VAL[p.symbol()] * (1 if p.color else -1)
        for p in brd.piece_map().values()
    )
    t[:, :, 35] = np.tanh(material / 10.0)
    return t

def encode(args: Tuple[chess.Board, Optional[str], str]) -> Optional[bytes]:
    brd, last_uci, target_uci = args
    prev = chess.Move.from_uci(last_uci) if last_uci else None
    legal_moves = list(brd.legal_moves)
    label = uci_index(target_uci)
    if label is None:
        return None
    tensor = tensor_from_board(brd, prev, legal_moves)
    mask = np.zeros(LOGITS, np.uint8)
    idxs = [uci_index(mv.uci()) for mv in legal_moves]
    mask[[i for i in idxs if i is not None]] = 1
    packed = np.packbits(mask, bitorder="little")
    ex = tf.train.Example(
        features=tf.train.Features(
            feature={
                "tensor": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tensor.tobytes()])
                ),
                "legal": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[packed.tobytes()])
                ),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label])
                ),
            }
        )
    )
    return ex.SerializeToString()

def flush_buffer(
    buf: List[Tuple[chess.Board, Optional[str], str]],
    pool: mp.pool.Pool,
    writer_ref: List[Optional[tf.io.TFRecordWriter]],
    written_ref: List[int],
    rotate_writer,
    pbar: tqdm,
) -> None:
    for ex in pool.imap_unordered(encode, buf, chunksize=1024):
        if ex:
            if writer_ref[0] is None:
                writer_ref[0] = rotate_writer()
            writer_ref[0].write(ex)
            written_ref[0] += 1
            pbar.update(1)
            if written_ref[0] % SHARD_POS == 0:
                writer_ref[0].close()
                writer_ref[0] = rotate_writer()
    buf.clear()

def iter_positions(game: chess.pgn.Game) -> Iterable[Tuple[chess.Board, Optional[str], str]]:
    brd = game.board()
    prev_uci: Optional[str] = None
    for mv in game.mainline_moves():
        yield brd.copy(stack=False), prev_uci, mv.uci()
        brd.push(mv)
        prev_uci = mv.uci()

def stream_examples(paths: List[Path]) -> Iterable[Tuple[chess.Board, Optional[str], str]]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                yield from iter_positions(game)

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Convert Lichess PGNs to sharded TFRecord datasets.")
    ap.add_argument("pgn", nargs="+", type=Path, help="Input PGN file(s).")
    ap.add_argument("-o", "--out", required=True, type=Path,
                    help="Output base path (without extension).")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                    help="Worker processes (default: CPU count – 1).")
    ap.add_argument("--flush", type=int, default=FLUSH_EVERY,
                    help=f"Flush buffer every N positions (default {FLUSH_EVERY}).")
    args = ap.parse_args(argv)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    out_base: Path = args.out
    out_base.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_base.parent / META_FILENAME
    if not meta_path.exists():
        meta = {
            "version": 5.2,
            "tensor_planes": TOTAL_CHANNELS,
            "logits": LOGITS,
            "shard_size": SHARD_POS,
            "orientation": "white_bottom",
        }
        with meta_path.open("w") as fh:
            json.dump(meta, fh)
    shard_idx = 0
    def rotate_writer() -> tf.io.TFRecordWriter:
        nonlocal shard_idx
        shard_path = out_base.with_name(f"{out_base.name}-{shard_idx:05d}.tfrecord.gz")
        shard_idx += 1
        try:
            (shard_path.parent / f"{shard_path.stem}_meta.json").symlink_to(meta_path.name)
        except OSError:
            print(f"Warning: symlink to {META_FILENAME} failed for {shard_path.name}", file=sys.stderr)
        opts = tf.io.TFRecordOptions(compression_type="GZIP")
        return tf.io.TFRecordWriter(str(shard_path), opts)
    pool = mp.Pool(args.workers)
    pbar = tqdm(unit="pos", smoothing=0.05)
    writer_ref: List[Optional[tf.io.TFRecordWriter]] = [None]
    written_ref: List[int] = [0]
    buf: List[Tuple[chess.Board, Optional[str], str]] = []
    try:
        for triple in stream_examples(args.pgn):
            buf.append(triple)
            if len(buf) >= args.flush:
                flush_buffer(buf, pool, writer_ref, written_ref, rotate_writer, pbar)
        if buf:
            flush_buffer(buf, pool, writer_ref, written_ref, rotate_writer, pbar)
    finally:
        pool.close()
        pool.join()
        if writer_ref[0] is not None:
            writer_ref[0].close()
        pbar.close()
    print(f"Wrote {written_ref[0]:,} positions across {shard_idx} shard(s).", file=sys.stderr)

if __name__ == "__main__":
    main()
