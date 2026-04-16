"""Convert BLADE benchmarks from skills_hub to Harbor task directories.

For each benchmark with rollout data and golden reports, generates one Harbor
task per (benchmark, model) pair. The agent analyzes the model's rollout data
and produces a diagnostic report; the verifier scores it against the golden report.

Usage:
    python convert_to_harbor.py \
        --skills-hub /path/to/skills_hub \
        --output-dir harbor-tasks/blade/ \
        [--benchmark cobol_compiler]  # or omit for all benchmarks
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ADAPTER_DIR = Path(__file__).parent
TEMPLATES_DIR = ADAPTER_DIR / "templates"

TASK_DIFFICULTY = "hard"

# Cache templates at module level to avoid re-reading per task
_template_cache: dict[str, str] = {}


def read_template(name: str) -> str:
    if name not in _template_cache:
        _template_cache[name] = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
    return _template_cache[name]


def render(template: str, substitutions: dict[str, str]) -> str:
    """Replace {key} placeholders in a template string."""
    content = template
    for key, value in substitutions.items():
        content = content.replace(f"{{{key}}}", value)
    return content


def find_benchmarks(skills_hub: Path) -> list[Path]:
    """Find all benchmark directories that have a SKILL.md."""
    benchmarks_dir = skills_hub / "benchmarks"
    results = []

    for d in sorted(benchmarks_dir.iterdir()):
        if not d.is_dir():
            continue

        has_own_skill = (d / "skill" / "SKILL.md").exists() or (d / "SKILL.md").exists()

        if has_own_skill:
            results.append(d)
        else:
            # Parent dir with sub-benchmarks — only rglob if no own SKILL.md
            for skill_path in d.rglob("SKILL.md"):
                bench_dir = skill_path.parent
                if bench_dir.name == "skill":
                    bench_dir = bench_dir.parent
                if bench_dir != d:
                    results.append(bench_dir)

    # Deduplicate (possible if rglob finds multiple paths to same dir)
    seen = set()
    deduped = []
    for b in results:
        if b not in seen:
            seen.add(b)
            deduped.append(b)
    return deduped


def find_models_with_golden_reports(benchmark_dir: Path) -> list[dict]:
    """Find models that have both rollout data and golden reports."""
    golden_dir = benchmark_dir / "golden_reports"
    if not golden_dir.exists():
        golden_dir = benchmark_dir

    rollouts_dir = benchmark_dir / "rollouts"
    if not rollouts_dir.exists():
        return []

    # Collect rollout files once to avoid re-globbing per golden report
    all_rollout_files = list(rollouts_dir.glob("*.jsonl"))

    models = []
    for report in sorted(golden_dir.glob("*_golden_report.md")):
        stem = report.stem
        model_name = stem.replace("_golden_report", "")

        if "_vs_" in model_name:
            continue

        # Find matching rollout file from cached list
        rollout_files = [f for f in all_rollout_files if model_name.lower() in f.name.lower()]
        raw_files = [f for f in rollout_files if "profiled" not in f.name.lower()]
        rollout = raw_files[0] if raw_files else (rollout_files[0] if rollout_files else None)

        if not rollout:
            continue

        metrics_path = report.parent / f"{stem}_metrics.json"

        models.append({
            "model_name": model_name,
            "rollout_file": rollout,
            "golden_report": report,
            "golden_metrics": metrics_path if metrics_path.exists() else None,
        })

    return models


def generate_task(
    benchmark_dir: Path,
    benchmark_name: str,
    model_info: dict,
    output_dir: Path,
    judge_model: str,
    judge_dir: Path | None = None,
) -> str | None:
    """Generate a Harbor task directory for one (benchmark, model) pair."""
    model_name = model_info["model_name"]
    # Docker compose project names can't have underscores and must be short
    safe_bench = benchmark_name.replace("_", "-")
    safe_model = model_name.replace("_", "-").replace(".", "-")
    # Shorten common prefixes to keep Docker project names under ~40 chars
    safe_bench = safe_bench.replace("agentic-heavy", "ah").replace("compiler", "comp")
    task_id = f"blade-{safe_bench}-{safe_model}"
    task_dir = output_dir / task_id

    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    # task.toml
    task_toml = render(read_template("task.toml.template"), {
        "task_id": task_id,
        "difficulty": TASK_DIFFICULTY,
        "judge_model": judge_model,
    })
    (task_dir / "task.toml").write_text(task_toml)

    # instruction.md
    instruction = render(read_template("instruction.md.template"), {
        "benchmark_name": benchmark_name,
        "model_name": model_name,
    })
    (task_dir / "instruction.md").write_text(instruction)

    # environment/ (Dockerfile + data — build context is this directory)
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    shutil.copy(TEMPLATES_DIR / "Dockerfile.template", env_dir / "Dockerfile")

    # Data inside environment/ for Docker build context
    env_data_dir = env_dir / "data"
    env_data_dir.mkdir()

    rollouts_dest = env_data_dir / "rollouts"
    rollouts_dest.mkdir()
    shutil.copy(model_info["rollout_file"], rollouts_dest / model_info["rollout_file"].name)

    skill_src = benchmark_dir / "skill"
    if skill_src.exists():
        shutil.copytree(skill_src, env_data_dir / "skill")
    else:
        skill_dest = env_data_dir / "skill"
        skill_dest.mkdir()
        skill_md = benchmark_dir / "SKILL.md"
        if skill_md.exists():
            shutil.copy(skill_md, skill_dest / "SKILL.md")

    # tests/ (golden report + metrics + verifier — Harbor uploads at verify time)
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()

    shutil.copy(model_info["golden_report"], tests_dir / "golden_report.md")
    if model_info["golden_metrics"]:
        shutil.copy(model_info["golden_metrics"], tests_dir / "golden_metrics.json")

    shutil.copy(TEMPLATES_DIR / "test.sh", tests_dir / "test.sh")
    os.chmod(tests_dir / "test.sh", 0o755)

    # Copy anchor_facts.json if it exists alongside the golden report
    anchor_facts_path = model_info["golden_report"].parent / f"{model_info['model_name']}_anchor_facts.json"
    if anchor_facts_path.exists():
        shutil.copy(anchor_facts_path, tests_dir / "anchor_facts.json")

    # Copy judge into environment/ (Docker build context) if provided
    if judge_dir and judge_dir.exists():
        env_judge_dir = env_dir / "judge"
        env_judge_dir.mkdir(exist_ok=True)
        for f in ["blade_judge.py", "universal_checklist.json"]:
            src = judge_dir / f
            if src.exists():
                shutil.copy(src, env_judge_dir / f)
    else:
        # Fall back to old test_runner.py template
        shutil.copy(TEMPLATES_DIR / "test_runner.py.template", tests_dir / "test_runner.py")

    # solution/ (oracle — copies golden report)
    # Include golden report in solution/ so oracle can access it during agent phase
    # (tests/ is only uploaded during verifier phase, after the agent finishes)
    solution_dir = task_dir / "solution"
    solution_dir.mkdir()
    shutil.copy(model_info["golden_report"], solution_dir / "golden_report.md")
    solve_sh = "#!/usr/bin/env bash\ncp /solution/golden_report.md /app/report.md\n"
    (solution_dir / "solve.sh").write_text(solve_sh)
    os.chmod(solution_dir / "solve.sh", 0o755)

    return task_id


def main():
    parser = argparse.ArgumentParser(description="Convert BLADE benchmarks to Harbor tasks")
    parser.add_argument("--skills-hub", required=True, help="Path to skills_hub repo root")
    parser.add_argument("--output-dir", required=True, help="Output directory for Harbor tasks")
    parser.add_argument("--benchmark", default=None, help="Specific benchmark name (default: all with data)")
    parser.add_argument("--judge-model", default="anthropic/claude-sonnet-4-6",
                        help="Model for LLM-as-judge scoring")
    parser.add_argument("--judge-dir", default=None,
                        help="Path to judge/ directory in skills_hub (contains blade_judge.py + universal_checklist.json)")
    args = parser.parse_args()

    skills_hub = Path(args.skills_hub)
    output_dir = Path(args.output_dir)

    judge_dir = Path(args.judge_dir) if args.judge_dir else (skills_hub / "judge")

    if not (skills_hub / "benchmarks").exists():
        print(f"Error: {skills_hub}/benchmarks/ not found", file=sys.stderr)
        raise SystemExit(1)

    if judge_dir.exists() and (judge_dir / "blade_judge.py").exists():
        print(f"Using judge from {judge_dir}")
    else:
        print(f"Warning: judge not found at {judge_dir}, using fallback test_runner.py", file=sys.stderr)
        judge_dir = None

    benchmarks = find_benchmarks(skills_hub)
    print(f"Found {len(benchmarks)} benchmarks")

    if args.benchmark:
        benchmarks = [
            b for b in benchmarks
            if b.name == args.benchmark
            or str(b.relative_to(skills_hub / "benchmarks")) == args.benchmark
        ]
        if not benchmarks:
            print(f"Error: benchmark '{args.benchmark}' not found", file=sys.stderr)
            raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    registry = []

    for bench_dir in benchmarks:
        bench_name = str(bench_dir.relative_to(skills_hub / "benchmarks"))
        models = find_models_with_golden_reports(bench_dir)

        if not models:
            print(f"  {bench_name}: no models with rollouts + golden reports, skipping")
            continue

        print(f"  {bench_name}: {len(models)} model(s)")
        for model_info in models:
            task_id = generate_task(
                bench_dir, bench_name.replace("/", "-"), model_info,
                output_dir, args.judge_model, judge_dir,
            )
            if task_id:
                registry.append({
                    "task_id": task_id,
                    "benchmark": bench_name,
                    "model": model_info["model_name"],
                })
                print(f"    -> {task_id}")

    # Write registry
    registry_path = output_dir / "registry.json"
    with open(registry_path, "w") as f:
        json.dump({"tasks": registry, "count": len(registry)}, f, indent=2)

    # Write nemo-gym input JSONL (for ng_collect_rollouts)
    ng_data_dir = output_dir.parent / "data"
    ng_data_dir.mkdir(parents=True, exist_ok=True)
    tasks_jsonl = ng_data_dir / "blade_tasks.jsonl"
    with open(tasks_jsonl, "w") as f:
        for entry in registry:
            task = {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": f"Analyze the {entry['benchmark']} benchmark rollouts for model {entry['model']}.",
                        }
                    ],
                },
                "instance_id": f"blade::{entry['task_id']}",
                "benchmark": entry["benchmark"],
                "model": entry["model"],
            }
            f.write(json.dumps(task) + "\n")

    print(f"\nGenerated {len(registry)} Harbor tasks in {output_dir}")
    print(f"Registry: {registry_path}")
    print(f"NeMo-Gym input JSONL: {tasks_jsonl}")


if __name__ == "__main__":
    main()
