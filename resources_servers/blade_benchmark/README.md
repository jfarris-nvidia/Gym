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

This generates one Harbor task per (benchmark, model) pair. Currently produces 7 tasks:
- cobol_compiler × 4 models (gptoss120b, haiku45, nemotron3super120b, qwen35_397b)
- cvdp_agentic_heavy × 3 models (claude_opus_46, gpt5, nemotron_super_49b)

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
Harbor Container (/data/):
  rollouts/*.jsonl          ← benchmark rollout data to analyze
  skill/SKILL.md            ← analysis methodology + failure taxonomy
  skill/scripts/*.py        ← analysis scripts (optional tools)

Agent reads data, runs scripts, reasons about failures
  → writes /app/report.md

Verifier (/tests/):
  golden_report.md          ← human-verified ground truth
  golden_metrics.json       ← structured metrics
  test_runner.py            ← scores report (30% structural + 70% LLM judge)
  → writes /logs/verifier/reward.txt
```

## Scoring

The verifier uses two axes:

1. **Structural (30%)** — Does the report have expected sections (summary, taxonomy, examples, root causes, metrics)?
2. **LLM Judge (70%)** — Analytical quality scored against golden report on 7 criteria:
   - Dominant failure mode identification
   - Root cause distinction
   - Causal narrative
   - Evidence citation
   - Cross-cutting insight
   - Within-task comparison
   - Non-obvious findings

Requires `JUDGE_API_KEY` and `JUDGE_BASE_URL` environment variables for LLM judge scoring. Without them, falls back to structural scoring only.
