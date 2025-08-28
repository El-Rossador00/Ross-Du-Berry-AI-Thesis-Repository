#!/usr/bin/env bash
# 4-engine NK round-robin (greedy), 18,432 games per pairing (total 110,592 games).
# and the static opening book at /home/rossduberry/starts60.epd

set -euo pipefail

BOOK=${BOOK:-/home/rossduberry/starts60.epd}  # EPD file created by make_starts60.py
OUT=${OUT:-/home/rossduberry/pgn}
CONC=${CONC:-16}      # cutechess concurrency
ST=${ST:-1.0}         # sec/move
GPP=${GPP:-18432}     # games per pairing (6 pairings with 4 engines)
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUT"

if [[ ! -s "$BOOK" ]]; then
  echo "ERROR: Opening book not found: $BOOK"
  echo "Generate it first (see note below), then re-run."
  exit 1
fi

TAG="nk4_rr_greedy_${GPP}gpp_${TS}"
PGN="$OUT/${TAG}.pgn"
LOG="$OUT/${TAG}.log"

echo "=== NK Greedy 4-engine RR ==="
echo "PGN: $PGN"
echo "LOG: $LOG"
echo "BOOK: $BOOK"
echo "GPP=$GPP  CONC=$CONC  ST=$ST"

cutechess-cli \
  -tournament round-robin \
  -engine cmd="/home/rossduberry/uci_greedy.sh" arg=1200 name="NK-1200" restart=off \
  -engine cmd="/home/rossduberry/uci_greedy.sh" arg=1500 name="NK-1500" restart=off \
  -engine cmd="/home/rossduberry/uci_greedy.sh" arg=1800 name="NK-1800" restart=off \
  -engine cmd="/home/rossduberry/uci_greedy.sh" arg=2000 name="NK-2000" restart=off \
  -each proto=uci st="$ST" timemargin=8000 \
  -openings file="$BOOK" format=epd order=random -repeat \
  -games "$GPP" \
  -concurrency "$CONC" \
  -event "NK Greedy 4-RR ($GPP per pairing) $TS" \
  -srand 4242 -recover \
  -pgnout "$PGN" |& tee "$LOG"

echo ">>> DONE: $PGN"
