#!/bin/bash
# The two post-hoc CONTROL arms, queued after run_bayes.sh. Separate script
# again: editing a bash script while bash is executing it can resume at a
# stale byte offset.
#
# These were added after seeing seed 0 and they can only explain the headline
# gain AWAY, never inflate it. See PREREGISTRATION.md's fourth appendix.
cd "$(dirname "$0")"
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
while pgrep -f "run_all.sh|run_bayes.sh" > /dev/null 2>&1; do sleep 60; done
run() {   # arm seed
  local tag="${1//+/_}_seed$2"
  if [ -f "checkpoints/$tag.pt" ]; then echo "skip $tag (done)"; return; fi
  $PY run_arm.py --arm "$1" --seed "$2" --threads 2 > "logs/$tag.log" 2>&1
  echo "done $tag: $(grep RESULT "logs/$tag.log")"
}
for s in 0 1 2; do
  run "ours+nvis" "$s" & P1=$!
  run "ours+fresh" "$s" & P2=$!
  wait $P1 $P2
  echo "CONTROL SEED $s COMPLETE"
done
echo ALL_CONTROLS_DONE
