#!/usr/bin/env python3
# Stockfish evaluator - ELO-LIMITED
import os, sys, json, csv
from pathlib import Path
from typing import List, Dict, Tuple
import multiprocessing as mp
import chess, chess.engine
from tqdm import tqdm

BINS = [1200, 1500, 1800, 2000]
SAMPLE_DIR = "sampled_data"
OUT_DIR = "sf_eval_results_elo"
STOCKFISH_PATH = "/usr/local/bin/stockfish"
MULTIPV = 5
TIME_PER_MOVE = 0.5
SF_ELO_MIN = 1320
WORKERS = max(1, mp.cpu_count() - 1)

def _analyse_elo(task: Tuple[str, str, int]) -> Dict[str, int]:
    fen, played_uci, elo_bin = task
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({
            "Threads": 1, "Hash": 16,
            "UCI_LimitStrength": True,
            "UCI_Elo": max(SF_ELO_MIN, elo_bin),
        })
        info = engine.analyse(chess.Board(fen), chess.engine.Limit(time=TIME_PER_MOVE), multipv=MULTIPV)
        sf_moves = [pv["pv"][0].uci() for pv in info if "pv" in pv and pv["pv"]]
        if not sf_moves: return {}
        return {
            "in_top1": int(played_uci == sf_moves[0]),
            "in_top3": int(played_uci in set(sf_moves[:3])),
            "in_top5": int(played_uci in set(sf_moves[:5])),
        }
    except Exception as e:
        print(f"WORKER ERROR (bin {elo_bin}): {e}", file=sys.stderr)
        return {}
    finally:
        if engine: engine.quit()

def run_bin(elo_bin: int, samples: List[Dict]) -> Dict:
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_summary_path = Path(OUT_DIR) / f"sf_eval_bin{elo_bin}_elo.json"
    tasks = [(s['fen'], s['uci'], elo_bin) for s in samples]

    results: List[Dict[str, int]] = []
    with mp.Pool(processes=WORKERS) as pool:
        for r in tqdm(pool.imap_unordered(_analyse_elo, tasks, chunksize=16),
                      total=len(tasks), desc=f"Analyzing bin {elo_bin} ELO"):
            if r: results.append(r)

    n = len(results)
    summary = {
        "bin": elo_bin, "mode": "elo", "positions_analyzed": n,
        "top1_accuracy": sum(r["in_top1"] for r in results) / n if n else 0.0,
        "top3_accuracy": sum(r["in_top3"] for r in results) / n if n else 0.0,
        "top5_accuracy": sum(r["in_top5"] for r in results) / n if n else 0.0
    }
    with open(out_summary_path, "w") as fh: json.dump(summary, fh, indent=2)
    print(f"[bin {elo_bin}] ELO results -> {out_summary_path}")
    return summary

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for elo in BINS:
        sample_file = Path(SAMPLE_DIR) / f"{elo}_samples.json"
        if not sample_file.exists():
            print(f"[ERROR] Sample file not found: {sample_file}")
            continue
        with open(sample_file, "r") as f:
            samples = json.load(f)
        all_summaries.append(run_bin(elo, samples))

    if all_summaries:
        csv_path = Path(OUT_DIR) / "summary_elo.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader(); writer.writerows(all_summaries)
        print(f"[done] Combined ELO summary -> {csv_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
