#!/bin/bash
# Fast parallel LIBERO eval
# Usage: ./run_fast_eval.sh [suite] [trials] [parallel]
set -e
cd /home/ubuntu/libero-finetune/openpi

SUITE="${1:-libero_spatial}"
TRIALS="${2:-50}"
NPAR="${3:-5}"
PYBIN="examples/libero/.venv/bin/python"

export PYTHONPATH="/home/ubuntu/libero-finetune/openpi/third_party/libero"
export MUJOCO_GL=egl

TMPD=$(mktemp -d)
trap "rm -rf $TMPD" EXIT

echo "Suite: $SUITE | $TRIALS trials/task | $NPAR parallel"

START_NS=$(date +%s%N)

# Launch tasks in batches
RUNNING=0
for TID in 0 1 2 3 4 5 6 7 8 9; do
    $PYBIN examples/libero/main_fast.py \
        --task-suite-name "$SUITE" \
        --num-trials-per-task "$TRIALS" \
        --task-id "$TID" \
        > "$TMPD/t${TID}.out" 2>&1 &
    RUNNING=$((RUNNING + 1))
    if [ $RUNNING -ge $NPAR ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$((RUNNING - 1))
    fi
done
wait

END_NS=$(date +%s%N)
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))

# Collect
TOK=0
TN=0
for TID in 0 1 2 3 4 5 6 7 8 9; do
    if grep -q "^RESULT" "$TMPD/t${TID}.out" 2>/dev/null; then
        LINE=$(grep "^RESULT" "$TMPD/t${TID}.out")
        OK=$(echo "$LINE" | cut -d'|' -f3)
        N=$(echo "$LINE" | cut -d'|' -f4)
        TOK=$((TOK + OK))
        TN=$((TN + N))
        grep "Task " "$TMPD/t${TID}.out"
    else
        echo "Task $TID: FAILED"
        tail -3 "$TMPD/t${TID}.out" 2>/dev/null
    fi
done

ELAPSED_S=$((ELAPSED_MS / 1000))
echo "================================"
echo "Suite: $SUITE"
echo "Success: $TOK / $TN"
echo "Time: ${ELAPSED_S}s total, ${TN} episodes"
echo "================================"
