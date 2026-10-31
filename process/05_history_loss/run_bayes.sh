#!/bin/bash
# The ours+bayes ceiling arm, queued AFTER run_all.sh so the machine never
# carries more than two of my jobs. Separate script on purpose: editing
# run_all.sh while bash is executing it can make bash resume at a stale byte
# offset and run garbage.
cd "$(dirname "$0")"
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
while pgrep -f run_all.sh > /dev/null 2>&1; do sleep 60; done
run() {
  local tag="ours_bayes_seed$1"
  if [ -f "checkpoints/$tag.pt" ]; then echo "skip $tag (done)"; return; fi
  $PY run_arm.py --arm ours+bayes --seed "$1" --threads 2 > "logs/$tag.log" 2>&1
  echo "done $tag: $(grep RESULT "logs/$tag.log")"
}
run 0 & P1=$!
run 1 & P2=$!
wait $P1 $P2
run 2
echo ALL_BAYES_DONE
