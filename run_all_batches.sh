#!/bin/bash
# run_all_batches.sh — runs batches of 40 continuously until all done or rate limit hit
# Usage: bash run_all_batches.sh

BATCH_SIZE=40
DELAY_BETWEEN_BATCHES=5   # seconds between batches
LOG_FILE="batch_runner.log"

echo "======================================================"
echo "  Continuous Batch Runner — $(date)"
echo "  Batch size: $BATCH_SIZE | Log: $LOG_FILE"
echo "======================================================"

batch_num=0

while true; do
    batch_num=$((batch_num + 1))

    # Check how many remain
    remaining=$(python3 test_all_queries.py --status 2>&1 | grep "Remaining" | grep -oP '\d+(?= \()')
    if [ -z "$remaining" ] || [ "$remaining" -eq 0 ]; then
        echo ""
        echo "✅ ALL QUERIES COMPLETE!"
        python3 test_all_queries.py --summary
        break
    fi

    echo ""
    echo "--- Batch #$batch_num | Remaining: $remaining | $(date +%H:%M:%S) ---"

    # Run batch, capture output
    output=$(python3 test_all_queries.py --batch $BATCH_SIZE 2>&1)
    exit_code=$?
    echo "$output"
    echo "$output" >> "$LOG_FILE"

    # Check for rate limit signals
    if echo "$output" | grep -qi "rate.limit\|too.many.requests\|429\|quota.*exceeded\|rate_limit_exceeded"; then
        echo ""
        echo "🔴 GROQ RATE LIMIT HIT — stopping."
        echo "   Run again later to continue from where we stopped."
        python3 test_all_queries.py --status
        break
    fi

    # Check for repeated timeouts (API might be overloaded)
    timeout_count=$(echo "$output" | grep -c "timed out" || true)
    if [ "$timeout_count" -gt 10 ]; then
        echo ""
        echo "⚠️  Too many timeouts ($timeout_count) — pausing 60s then continuing..."
        sleep 60
    fi

    # Check if still remaining
    still_remaining=$(echo "$output" | grep "Still after run" | grep -oP '\d+')
    if [ -z "$still_remaining" ] || [ "$still_remaining" -eq 0 ]; then
        echo ""
        echo "✅ ALL DONE!"
        python3 test_all_queries.py --summary
        break
    fi

    sleep $DELAY_BETWEEN_BATCHES
done

echo ""
echo "======================================================"
echo "  Runner finished at $(date)"
echo "======================================================"
