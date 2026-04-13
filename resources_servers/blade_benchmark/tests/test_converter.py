# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the BLADE skills_hub → Harbor task converter."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from resources_servers.blade_benchmark.adapter.convert_to_harbor import (
    find_benchmarks,
    find_models_with_golden_reports,
    generate_task,
)


@pytest.fixture
def mock_skills_hub(tmp_path):
    """Create a minimal skills_hub-like directory for testing."""
    bench = tmp_path / "benchmarks" / "test_bench"
    bench.mkdir(parents=True)

    # skill/SKILL.md
    skill_dir = bench / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-bench\ndescription: A test benchmark\n---\n"
        + "\n".join([f"## Section {i}\nContent line {j}" for i in range(5) for j in range(15)])
    )

    # skill/scripts/
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "analyze.py").write_text("#!/usr/bin/env python3\nprint('hello')\n")

    # rollouts/
    rollouts_dir = bench / "rollouts"
    rollouts_dir.mkdir()
    rollout_data = [
        {"reward": 1.0, "_ng_task_index": 0, "response": {"output": "pass"}},
        {"reward": 0.0, "_ng_task_index": 1, "response": {"output": "fail"}},
    ]
    rollout_file = rollouts_dir / "model_a_rollouts.jsonl"
    with open(rollout_file, "w") as f:
        for record in rollout_data:
            f.write(json.dumps(record) + "\n")

    # golden_reports/
    golden_dir = bench / "golden_reports"
    golden_dir.mkdir()
    (golden_dir / "model_a_golden_report.md").write_text("# Golden Report\n" + "Analysis content\n" * 50)
    (golden_dir / "model_a_golden_report_metrics.json").write_text(
        json.dumps({"pass_at_1": 50.0, "model": "model_a"})
    )

    return tmp_path


def test_find_benchmarks(mock_skills_hub):
    benchmarks = find_benchmarks(mock_skills_hub)
    assert len(benchmarks) == 1
    assert benchmarks[0].name == "test_bench"


def test_find_models_with_golden_reports(mock_skills_hub):
    bench_dir = mock_skills_hub / "benchmarks" / "test_bench"
    models = find_models_with_golden_reports(bench_dir)
    assert len(models) == 1
    assert models[0]["model_name"] == "model_a"
    assert models[0]["rollout_file"].name == "model_a_rollouts.jsonl"
    assert models[0]["golden_report"].name == "model_a_golden_report.md"
    assert models[0]["golden_metrics"].name == "model_a_golden_report_metrics.json"


def test_generate_task(mock_skills_hub, tmp_path):
    bench_dir = mock_skills_hub / "benchmarks" / "test_bench"
    models = find_models_with_golden_reports(bench_dir)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    task_id = generate_task(bench_dir, "test_bench", models[0], output_dir, "test-model")

    assert task_id == "blade-test_bench-model_a"

    task_dir = output_dir / task_id
    assert (task_dir / "task.toml").exists()
    assert (task_dir / "instruction.md").exists()
    assert (task_dir / "environment" / "Dockerfile").exists()
    # data/ lives inside environment/ (Docker build context)
    assert (task_dir / "environment" / "data" / "rollouts" / "model_a_rollouts.jsonl").exists()
    assert (task_dir / "environment" / "data" / "skill" / "SKILL.md").exists()
    assert (task_dir / "environment" / "data" / "skill" / "scripts" / "analyze.py").exists()
    assert (task_dir / "tests" / "test.sh").exists()
    assert (task_dir / "tests" / "test_runner.py").exists()
    assert (task_dir / "tests" / "golden_report.md").exists()
    assert (task_dir / "tests" / "golden_metrics.json").exists()
    assert (task_dir / "solution" / "solve.sh").exists()

    # Verify task.toml content
    toml_content = (task_dir / "task.toml").read_text()
    assert 'name = "blade-test_bench-model_a"' in toml_content
    assert 'difficulty = "hard"' in toml_content

    # Verify instruction references correct benchmark/model
    instruction = (task_dir / "instruction.md").read_text()
    assert "test_bench" in instruction
    assert "model_a" in instruction

    # Verify solve.sh copies golden report
    solve = (task_dir / "solution" / "solve.sh").read_text()
    assert "/tests/golden_report.md" in solve
    assert "/app/report.md" in solve


def test_find_benchmarks_skips_no_skill(mock_skills_hub):
    """Benchmarks without SKILL.md should be skipped."""
    empty_bench = mock_skills_hub / "benchmarks" / "no_skill"
    empty_bench.mkdir()
    (empty_bench / "rollouts").mkdir()

    benchmarks = find_benchmarks(mock_skills_hub)
    names = [b.name for b in benchmarks]
    assert "no_skill" not in names


def test_find_models_skips_comparison_reports(mock_skills_hub):
    """Comparison reports (model1_vs_model2) should not produce tasks."""
    golden_dir = mock_skills_hub / "benchmarks" / "test_bench" / "golden_reports"
    (golden_dir / "model_a_vs_model_b_golden_report.md").write_text("comparison")

    bench_dir = mock_skills_hub / "benchmarks" / "test_bench"
    models = find_models_with_golden_reports(bench_dir)
    model_names = [m["model_name"] for m in models]
    assert "model_a_vs_model_b" not in model_names
