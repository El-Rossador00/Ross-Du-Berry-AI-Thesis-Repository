#!/usr/bin/env python3
"""


Builds an evaluation set per Elo bin from the test set so that both Stockfish and my models can parse move d:
  • TFRecord with (tensor, label, legal) for model eval
  • JSON with (FEN, UCI) for Stockfish eval
  • Symlink sampled_data/{bin}_samples.json -> eval_sets/{bin}_eval_samples.json

Reads shards from GCS, the case for this project or local via tf.io.gfile.glob.

"""

import os, json, random
from pathlib import Path
import numpy as np
import tensorflow as tf
import chess

def _csv_ints(s, default):
    raw = os.getenv(s, default)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

BINS = _csv_ints("BINS", "1200,1500,1800,2000")
DATA_GLOB_TEMPLATE = os.getenv(
    "DATA_GLOB_TEMPLATE",
    "gs://rossd-chess-20250813/shards/{bin}_bin-*.tfrecord.gz",
)
OUT_DIR = os.getenv("OUT_DIR", "eval_sets")
TEST_FRAC = float(os.getenv("TEST_FRAC", "0.02"))
SAMPLES_PER_BIN = int(os.getenv("SAMPLES_PER_BIN", "10000"))
SEED = int(os.getenv("SEED", "42"))
VERBOSE = os.getenv("VERBOSE", "1").lower() not in ("0","false","no")

INPUT_SHAPE = (8, 8, 36)
LOGITS = 64 * 64 * 5
MASK_BYTES = (LOGITS + 7) // 8

_FEATURE_SPEC = {
    "tensor": tf.io.FixedLenFeature([], tf.string),
    "label":  tf.io.FixedLenFeature([], tf.int64),
    "legal":  tf.io.FixedLenFeature([], tf.string),
}

def _bytes_feature(b):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))

def _int64_feature(i):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(i)]))

PROMO_OFFSETS = [
    (chess.KNIGHT, 16384),
    (chess.BISHOP, 12288),
    (chess.ROOK,    8192),
    (chess.QUEEN,   4096),
    (None,             0),
]

def index_to_uci(idx: int) -> str:
    promo = None
    for p, off in PROMO_OFFSETS:
        if idx >= off:
            idx -= off
            promo = p
            break
    from_sq, to_sq = divmod(idx, 64)
    return chess.Move(from_sq, to_sq, promotion=promo).uci()

def tensor_bytes_to_board(t_bytes: bytes) -> chess.Board:
    t = np.frombuffer(t_bytes, dtype=np.float32).reshape(8, 8, 36)
    board = chess.Board.empty()

    idx_to_piece = {
        0:"p", 1:"n", 2:"b", 3:"r", 4:"q", 5:"k",
        6:"P", 7:"N", 8:"B", 9:"R", 10:"Q", 11:"K",
    }
    def rc_to_sq(r, c):
        return (7 - r) * 8 + c

    for r in range(8):
        for c in range(8):
            for plane, sym in idx_to_piece.items():
                if t[r, c, plane] > 0.5:
                    board.set_piece_at(rc_to_sq(r, c), chess.Piece.from_symbol(sym))
                    break

    board.turn = True if t[0, 0, 22] > 0 else False
    if t[0, 0, 23] > 0.5: board.castling_rights |= chess.BB_H1
    if t[0, 0, 24] > 0.5: board.castling_rights |= chess.BB_A1
    if t[0, 0, 25] > 0.5: board.castling_rights |= chess.BB_H8
    if t[0, 0, 26] > 0.5: board.castling_rights |= chess.BB_A8
    eps = np.argwhere(t[:, :, 31] > 0.5)
    if eps.size:
        board.ep_square = rc_to_sq(eps[0][0], eps[0][1])

    return board

def parse_example(raw):
    ex = tf.io.parse_single_example(raw, _FEATURE_SPEC)
    return ex["tensor"], ex["label"], ex["legal"]

def collect_samples(test_files, samples_needed, writer_tfr=None):
    out = []
    seen, illegal = 0, 0

    dataset = (
        tf.data.Dataset.from_tensor_slices(test_files)
        .interleave(
            lambda f: tf.data.TFRecordDataset(f, compression_type="GZIP"),
            cycle_length=min(16, len(test_files)),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        .map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    for tensor_bytes, label, legal_bytes in dataset.as_numpy_iterator():
        seen += 1
        try:
            board = tensor_bytes_to_board(tensor_bytes)
            uci = index_to_uci(int(label))
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                illegal += 1
                continue

            out.append({"fen": board.fen(), "uci": uci})

            if writer_tfr is not None:
                ex = tf.train.Example(features=tf.train.Features(feature={
                    "tensor": _bytes_feature(tensor_bytes),
                    "label":  _int64_feature(label),
                    "legal":  _bytes_feature(legal_bytes),
                }))
                writer_tfr.write(ex.SerializeToString())

            if len(out) >= samples_needed:
                break
        except Exception:
            illegal += 1
            continue

    stats = {"examined": int(seen), "illegal_or_decode_skipped": int(illegal)}
    return out, stats

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    Path("sampled_data").mkdir(parents=True, exist_ok=True)

    for elo_bin in BINS:
        print("=" * 70)
        print(f"[bin {elo_bin}] Building eval pack")

        rng = random.Random(SEED)
        pattern = DATA_GLOB_TEMPLATE.format(bin=elo_bin)
        all_files = sorted(tf.io.gfile.glob(pattern))
        if not all_files:
            print(f"[bin {elo_bin}] ERROR: no shards for pattern: {pattern}")
            continue

        rng.shuffle(all_files)
        n_test = max(1, int(len(all_files) * TEST_FRAC))
        test_files = all_files[:n_test]

        if VERBOSE:
            print(f"[bin {elo_bin}] total shards={len(all_files)}  test={n_test} ({TEST_FRAC:.2%})")
            preview = test_files[:5]
            for s in preview:
                print(f"  test shard: {s}")
            if n_test > 5:
                print(f"  ... (+{n_test - 5} more)")

        tfr_path = Path(OUT_DIR) / f"{elo_bin}_eval-00000-of-00001.tfrecord.gz"
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(str(tfr_path), options=options) as wr:
            samples, stats = collect_samples(test_files, SAMPLES_PER_BIN, writer_tfr=wr)

        json_path = Path(OUT_DIR) / f"{elo_bin}_eval_samples.json"
        with open(json_path, "w") as fh:
            json.dump(samples, fh)

        meta = {
            "bin": elo_bin,
            "seed": SEED,
            "test_frac": TEST_FRAC,
            "data_glob_template": DATA_GLOB_TEMPLATE,
            "selected_test_shards": test_files,
            "requested_samples": SAMPLES_PER_BIN,
            "written_samples": len(samples),
            "decode_stats": stats,
            "tfrecord_path": str(tfr_path),
        }
        meta_path = Path(OUT_DIR) / f"{elo_bin}_eval_meta.json"
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        link_path = Path("sampled_data") / f"{elo_bin}_samples.json"
        try:
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(json_path.resolve())
        except Exception as e:
            print(f"[bin {elo_bin}] WARN: could not create symlink: {e}")

        print(f"[bin {elo_bin}] wrote {len(samples):,} examples")
        print(f"           JSON -> {json_path}")
        print(f"           TFRecord -> {tfr_path}")
        print(f"           meta -> {meta_path}")
        print(f"           symlink -> {link_path} (for existing Stockfish scripts)")

    print("\n[done] All bins processed.")

if __name__ == "__main__":
    main()
