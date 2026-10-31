#!/bin/sh
# Run every audit measurement for one model on one stream, unattended.
# Usage: sh run_all.sh <tag> <stream>
# Python is unbuffered (-u) so the logs are readable while a run is in flight.
set -e
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
TAG=${1:-gated_seed0}
STREAM=${2:-505}
mkdir -p logs
for s in audit_beta audit_counterfactual audit_controls audit_claim_d \
         audit_linear; do
  echo "=== $s  $TAG  $STREAM ==="
  $PY -u "$s.py" --tag "$TAG" --stream "$STREAM" \
      > "logs/${s#audit_}_${TAG}_${STREAM}.log" 2>&1 \
    && echo "  ok" || echo "  FAILED (exit $?)"
done
echo ALL_DONE_$TAG_$STREAM
