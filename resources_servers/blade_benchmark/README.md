# BLADE Benchmark

Evaluate LLM analysis quality on benchmark rollout data using Harbor for sandboxed execution.

BLADE (Benchmark for LLM Analysis & Diagnostic Evaluation) tests whether an agent can produce quality diagnostic analysis of benchmark results. The agent receives rollout JSONL data and a SKILL.md methodology guide, then must produce an analysis report. A judge scores the report against human-verified golden reports.

## Setup

### 1. Generate Harbor task directories

Convert BLADE benchmarks from skills_hub into Harbor task directories:

```bash
# All benchmarks with rollout data + golden reports
python resources_servers/blade_benchmark/adapter/convert_to_harbor.py \
  --skills-hub /path/to/skills_hub \
  --output-dir resources_servers/blade_benchmark/harbor-tasks

# Specific benchmark
python resources_servers/blade_benchmark/adapter/convert_to_harbor.py \
  --skills-hub /path/to/skills_hub \
  --output-dir resources_servers/blade_benchmark/harbor-tasks \
  --benchmark cobol_compiler
```

This generates one Harbor task per (benchmark, model) pair. Currently produces 10 tasks:
- cobol_compiler × 4 models (gptoss120b, haiku45, nemotron3super120b, qwen35_397b)
- cvdp_agentic_heavy × 3 models (claude_opus_46, gpt5, nemotron_super_49b)
- cvdp × 3 models (gptoss120b, nemotron_super, opus4.6)

Task IDs are sanitized for Docker: underscores become hyphens, common prefixes are shortened (`agentic-heavy` → `ah`, `compiler` → `comp`). Example: `blade-cobol-comp-gptoss120b`.

### 2. Run with Harbor

```bash
# Test with oracle (copies golden report — should score ~1.0)
harbor run --agent oracle --path resources_servers/blade_benchmark/harbor-tasks/ --env docker

# Run with Claude Code
harbor run --agent claude-code --path resources_servers/blade_benchmark/harbor-tasks/ --env docker
```

### 3. Run with NeMo-Gym

```bash
ng_run "+config_paths=[resources_servers/blade_benchmark/configs/blade_benchmark.yaml]"
```

## Architecture

```
Harbor Container:
  /data/rollouts/*.jsonl         ← benchmark rollout data to analyze
  /data/skill/SKILL.md           ← analysis methodology + failure taxonomy
  /data/skill/scripts/*.py       ← analysis scripts (optional tools)
  /judge/blade_judge.py          ← BLADE judge (baked into image)
  /judge/universal_checklist.json

Agent reads data, runs scripts, reasons about failures
  → writes /app/report.md

Verifier (/tests/):
  golden_report.md               ← human-verified ground truth
  golden_metrics.json            ← structured metrics
  anchor_facts.json              ← 25 pattern-level findings (Layer B criteria)
  test.sh                        ← runs blade_judge.py and preserves report

Outputs:
  /logs/verifier/reward.txt      ← final score
  /logs/verifier/reward.json     ← per-criterion breakdown
  /logs/verifier/agent_report.md ← preserved copy for re-scoring
```

## Scoring

The BLADE judge uses a two-track evaluation (see skills_hub/judge/blade_judge.py):

**Track 1: Evidence-anchored checklist (50% weight)**
- Layer A (5): structural completeness — universal preconditions
- Layer A+ (3): trajectory process checks — verifies the agent read the data
- Layer B (25): benchmark-specific anchor facts — pattern-level findings that discriminate
- Layer C (4): analytical depth with strictness=high — catches template-only reports

**Track 2: Holistic quality (50% weight)**
Single LLM call comparing candidate + trajectory against golden report on a 1-5 scale.

The judge requires `JUDGE_MODEL`, `JUDGE_API_KEY`, and `JUDGE_BASE_URL` environment variables. The NVIDIA gateway's `OPENAI_API_KEY` / `OPENAI_BASE_URL` are passed through via `task.toml.template`.

### Preserved Reports

`test.sh` copies `/app/report.md` to `/logs/verifier/agent_report.md` before scoring. This lets you re-score with updated criteria or anchor facts without re-running agents — just re-invoke `blade_judge.py` on the preserved report.
