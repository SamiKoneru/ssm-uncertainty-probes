#!/bin/bash
# Train and score `ours+cheap`, the cheap-features-only arm.
#
#   bash run_cheap.sh
#
# Bars, kill condition and stream plan are in PREREGISTRATION_cheap.md and
# were fixed before arms.py gained the arm. This script does not choose any
# of them; score.py's cheap_verdict() only compares against them.
#
# Protocol is the unmodified run_arm.py: 800 steps, batch 24, AdamW lr 2e-3,
# clip 1.0, gate_width 32, threads 2. Identical to every arm already in 05,
# which is what makes the comparison against `ours` legitimate.
#
# Three seeds run concurrently at 2 threads each (6 of 10 cores). Thread count
# changes speed, not arithmetic.
set -u
cd "$(dirname "$0")"
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
mkdir -p logs checkpoints

# ---- pre-flight. A failure here means the arm is not a valid control, so
# ---- there is no point spending 3 x 800 steps on it.
echo "=== PRE-FLIGHT ==="
if ! $PY check_cheap.py > logs/check_cheap.log 2>&1; then
  tail -20 logs/check_cheap.log
  echo "ABORT: check_cheap.py failed. Nothing trained."
  exit 1
fi
tail -3 logs/check_cheap.log

# ---- the comparison arms. `checkpoints/` is gitignored, so a fresh clone has
# ---- none of them and the 27 from the original campaign live wherever that
# ---- campaign ran. Without `ours` there is no baseline and no verdict.
SIB=../../.claude/worktrees/sharp-dubinsky-67ebc6/process/05_history_loss/checkpoints
if [ ! -f checkpoints/ours_seed0.pt ] && [ -d "$SIB" ]; then
  echo "=== importing $(ls "$SIB" | wc -l | tr -d ' ') existing checkpoints from the original campaign ==="
  cp -n "$SIB"/*.pt checkpoints/
fi
if [ ! -f checkpoints/ours_seed0.pt ]; then
  echo "ABORT: no checkpoints/ours_seed*.pt. The baseline arms must be trained"
  echo "       first (bash run_all.sh) or copied in; there is no verdict"
  echo "       without them."
  exit 1
fi

# ---- train
echo "=== TRAINING ours+cheap, seeds 0 1 2 ==="
PIDS=()
for s in 0 1 2; do
  tag="ours_cheap_seed$s"
  if [ -f "checkpoints/$tag.pt" ]; then echo "skip $tag (done)"; continue; fi
  $PY run_arm.py --arm "ours+cheap" --seed "$s" --threads 2 \
      > "logs/$tag.log" 2>&1 &
  PIDS+=($!)
done
[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
for s in 0 1 2; do
  echo "  seed $s: $(grep RESULT "logs/ours_cheap_seed$s.log" || echo MISSING)"
done

# ---- score. 5150 first for comparability with every other arm, then 7771,
# ---- which appears nowhere in the repo and is the only stream that can
# ---- CONFIRM a post-hoc arm.
echo "=== SCORING ==="
$PY score.py --stream 5150 > logs/score_5150_cheap.log 2>&1
$PY score.py --stream 7771 > logs/score_7771_cheap.log 2>&1
for st in 5150 7771; do
  echo "--- stream $st ---"
  sed -n '/CHEAP-ARM VERDICT/,/STATUS:/p' "logs/score_${st}_cheap.log"
done
echo ALL_CHEAP_DONE
