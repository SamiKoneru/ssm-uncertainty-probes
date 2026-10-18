#!/bin/bash
# Train every arm x seed, TWO AT A TIME at 2 threads each. The machine is
# shared: while this was written another session in this repo was running
# fit_model.py jobs and an unrelated project had a 9-worker campaign up, which
# took the load average from 2 to 83 on 10 cores.
#
# Ordered BY SEED, not by arm, so an interruption leaves complete seed blocks
# (every arm at seed 0) rather than complete arms at missing seeds.
#
# Pairs put one bookkeeping arm next to one cheap arm where possible: `ours`,
# `ours+hist`, `ours+rate` and `ours+oracle` all run the per-address
# bookkeeping and cost ~2.5x, while `none` and `ours_pub` do not.
#
#   bash run_all.sh
cd "$(dirname "$0")"
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
mkdir -p logs checkpoints

run() {   # arm seed
  local tag="${1//+/_}_seed$2"
  if [ -f "checkpoints/$tag.pt" ]; then echo "skip $tag (done)"; return; fi
  $PY run_arm.py --arm "$1" --seed "$2" --threads 2 > "logs/$tag.log" 2>&1
  echo "done $tag: $(grep RESULT "logs/$tag.log")"
}

for s in 0 1 2; do
  for pair in "none ours" "ours+hist ours+rate" "ours+oracle ours_pub"; do
    set -- $pair
    run "$1" "$s" & P1=$!
    run "$2" "$s" & P2=$!
    wait $P1 $P2
  done
  echo "SEED $s COMPLETE"
done
echo ALL_RUNS_DONE
