#!/usr/bin/env bash
# Run BLADE discrimination study: 3 agent+model tiers on cobol_compiler and cvdp_agentic_heavy.
#
# Usage:
#   ./scripts/run_blade_discrimination.sh claude      # Claude Code + Sonnet 4.6
#   ./scripts/run_blade_discrimination.sh opencode    # OpenCode + Haiku 4.5 and OpenCode + Nemotron
#   ./scripts/run_blade_discrimination.sh all         # All 3 tiers
#
# Each tier runs on both benchmarks (cobol_compiler: 4 tasks, cvdp_agentic_heavy: 3 tasks).
# Harbor uses craft-bench's venv: ~/projects/craft-bench/.venv/bin/harbor

set -euo pipefail
cd "$(dirname "$0")/.."

HARBOR="${HARBOR:-$HOME/projects/craft-bench/.venv/bin/harbor}"

if [[ ! -x "$HARBOR" ]]; then
  echo "Error: harbor not found at $HARBOR"
  echo "Install: cd ~/projects/craft-bench && uv sync"
  exit 1
fi

if [[ -f ../../.env ]]; then
  set -a
  source ../../.env
  set +a
  echo "Loaded .env from nemo-gym root"
elif [[ -f "$HOME/projects/llm_evaluation_analysis/skills_hub/.env" ]]; then
  set -a
  source "$HOME/projects/llm_evaluation_analysis/skills_hub/.env"
  set +a
  echo "Loaded .env from skills_hub"
fi

export JUDGE_API_KEY="${JUDGE_API_KEY:-$OPENAI_API_KEY}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-$OPENAI_BASE_URL}"

N_CONCURRENT=1
COBOL_TASKS="harbor-tasks/blade-cobol-comp-gptoss120b"
CVDP_AH_TASKS="harbor-tasks/blade-cvdp-ah-claude-opus-46"
MODE="${1:-all}"
JOBS_DIR="${JOBS_DIR:-jobs}"

echo "============================================================"
echo "BLADE Discrimination Study"
echo "============================================================"
echo "Mode:    $MODE"
echo "Harbor:  $HARBOR"
echo "Tasks:   cobol_compiler (gptoss120b) + cvdp_agentic_heavy (claude_opus_46)"
echo ""

# Clean up stale containers
docker ps -a --filter "name=blade-" -q | xargs -r docker stop 2>/dev/null || true
docker ps -a --filter "name=blade-" -q | xargs -r docker rm 2>/dev/null || true

# =====================================================================
# Helper: run one tier on one benchmark
# =====================================================================
run_tier() {
  local job_name="$1" agent="$2" model="$3" task_path="$4"
  echo ""
  echo "--- $job_name ---"
  mkdir -p "$JOBS_DIR/$job_name"
  "$HARBOR" run \
    --agent "$agent" \
    --model "$model" \
    --path "$task_path" \
    --n-concurrent "$N_CONCURRENT" \
    --env docker \
    -o "$JOBS_DIR/$job_name/"
  echo "--- $job_name complete ---"
}

# =====================================================================
# TIER 1: Claude Code + Sonnet 4.6 (strong agent + strong model)
# =====================================================================
run_claude() {
  echo ""
  echo "=== Tier 1: Claude Code + Sonnet 4.6 ==="
  run_tier "blade-claudecode-sonnet-cobol" \
    claude-code "aws/anthropic/bedrock-claude-sonnet-4-6" \
    "$COBOL_TASKS"

  run_tier "blade-claudecode-sonnet-cvdpah" \
    claude-code "aws/anthropic/bedrock-claude-sonnet-4-6" \
    "$CVDP_AH_TASKS"
  echo "=== Tier 1 complete ==="
}

# =====================================================================
# TIER 2: OpenCode + Haiku 4.5 (open agent + weaker model)
# =====================================================================
run_opencode_haiku() {
  echo ""
  echo "=== Tier 2: OpenCode + Haiku 4.5 ==="
  run_tier "blade-opencode-haiku45-cobol" \
    opencode "nvidia/aws/anthropic/claude-haiku-4-5-v1" \
    "$COBOL_TASKS"

  run_tier "blade-opencode-haiku45-cvdpah" \
    opencode "nvidia/aws/anthropic/claude-haiku-4-5-v1" \
    "$CVDP_AH_TASKS"
  echo "=== Tier 2 complete ==="
}

# =====================================================================
# TIER 3: OpenCode + Nemotron (open agent + NVIDIA model)
# =====================================================================
run_opencode_nemotron() {
  echo ""
  echo "=== Tier 3: OpenCode + Nemotron Super ==="
  run_tier "blade-opencode-nemotron-cobol" \
    opencode "nvidia/nvidia/nvidia/nemotron-3-super-preview" \
    "$COBOL_TASKS"

  run_tier "blade-opencode-nemotron-cvdpah" \
    opencode "nvidia/nvidia/nvidia/nemotron-3-super-preview" \
    "$CVDP_AH_TASKS"
  echo "=== Tier 3 complete ==="
}

# =====================================================================
# Dispatch
# =====================================================================
case "$MODE" in
  claude)
    run_claude
    ;;
  opencode)
    run_opencode_haiku
    run_opencode_nemotron
    ;;
  haiku)
    run_opencode_haiku
    ;;
  nemotron)
    run_opencode_nemotron
    ;;
  all)
    run_claude
    run_opencode_haiku
    run_opencode_nemotron
    ;;
  *)
    echo "Usage: $0 {claude|opencode|haiku|nemotron|all}"
    exit 1
    ;;
esac

echo ""
echo "============================================================"
echo "BLADE Discrimination Study — Complete"
echo "============================================================"
echo "Results in jobs/blade-*/"
echo ""
echo "To score results:"
echo "  for d in jobs/blade-*/; do"
echo "    echo \"\$(basename \$d): \$(cat \$d/reward.txt 2>/dev/null || echo 'no score')\""
echo "  done"
