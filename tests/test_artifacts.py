from __future__ import annotations

import json

import pytest

from skeletons import (
    RunArtifacts,
    make_run_name,
    resolve_checkpoint_reference,
    resolve_work_dir,
    sanitize_run_token,
    save_json,
    write_last_checkpoint,
)


def test_make_run_name_sanitizes_parts() -> None:
    assert sanitize_run_token("Human 3.6M++") == "human_3_6m"
    assert make_run_name("Human36M", "seed=7", None, "") == "human36m_seed_7"
    assert make_run_name(None, "") == "default"


def test_resolve_work_dir_uses_project_root_and_creates_directory(tmp_path) -> None:
    script_path = tmp_path / "examples" / "train_demo.py"
    script_path.parent.mkdir()
    script_path.write_text("", encoding="utf-8")

    work_dir = resolve_work_dir(
        script_path,
        run_name="Demo Run",
        project_root=tmp_path,
    )

    assert work_dir == tmp_path / "work_dirs" / "train_demo" / "demo_run"
    assert work_dir.is_dir()


def test_save_json_serializes_paths_with_stable_format(tmp_path) -> None:
    output = tmp_path / "metrics.json"

    save_json(output, {"path": tmp_path / "checkpoint.pt", "value": 3})

    assert output.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "path": str(tmp_path / "checkpoint.pt"),
        "value": 3,
    }


def test_checkpoint_reference_resolves_files_and_last_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    work_dir = tmp_path / "run"
    work_dir.mkdir()

    write_last_checkpoint(work_dir, checkpoint)

    assert resolve_checkpoint_reference(checkpoint) == checkpoint
    assert resolve_checkpoint_reference(work_dir) == checkpoint
    assert resolve_checkpoint_reference(work_dir / "last_checkpoint") == checkpoint


def test_checkpoint_reference_reports_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint file"):
        resolve_checkpoint_reference(tmp_path / "missing.pt")

    pointer = tmp_path / "last_checkpoint"
    pointer.write_text(str(tmp_path / "missing.pt"), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="referenced"):
        resolve_checkpoint_reference(pointer)


def test_run_artifacts_wraps_common_operations(tmp_path) -> None:
    script_path = tmp_path / "examples" / "demo.py"
    script_path.parent.mkdir()
    script_path.write_text("", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")

    artifacts = RunArtifacts.create(
        script_path,
        run_name="Sample Run",
        project_root=tmp_path,
    )
    artifacts.save_json("config.json", {"checkpoint": checkpoint})
    artifacts.write_last_checkpoint(checkpoint)

    assert artifacts.script_name == "demo"
    assert artifacts.run_name == "sample_run"
    assert artifacts.path("config.json").is_file()
    assert resolve_checkpoint_reference(artifacts.work_dir) == checkpoint
