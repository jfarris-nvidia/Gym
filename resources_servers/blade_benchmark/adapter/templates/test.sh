#!/usr/bin/env bash
# Harbor verifier entry point — calls BLADE judge.
set -euo pipefail

mkdir -p /logs/verifier

# Use blade_judge.py if available, fall back to test_runner.py
if [ -f /judge/blade_judge.py ]; then
    ARGS="--report /app/report.md --golden /tests/golden_report.md --checklist /judge/universal_checklist.json --output-dir /logs/verifier/"

    # Add trajectory if available
    if [ -f /agent/trajectory.json ]; then
        ARGS="$ARGS --trajectory /agent/trajectory.json"
    fi

    # Add anchor facts if available
    if [ -f /tests/anchor_facts.json ]; then
        ARGS="$ARGS --anchor-facts /tests/anchor_facts.json"
    fi

    python3 /judge/blade_judge.py $ARGS
else
    python3 /tests/test_runner.py
fi

# Always exit 0 — reward.txt carries the score.
exit 0
