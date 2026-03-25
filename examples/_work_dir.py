from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sanitize_token(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    return normalized or "run"


def make_run_name(*parts: object) -> str:
    tokens = [
        _sanitize_token(str(part))
        for part in parts
        if part is not None and str(part).strip()
    ]
    return "_".join(tokens) if tokens else "default"


def resolve_work_dir(
    script_path: Path,
    *,
    run_name: str,
    work_dir: Path | None = None,
) -> Path:
    path = (
        work_dir.expanduser()
        if work_dir is not None
        else PROJECT_ROOT / "work_dirs" / script_path.stem / run_name
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_status(script_name: str, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{script_name}] {message}", flush=True)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_last_checkpoint(work_dir: Path, checkpoint_path: Path) -> None:
    (work_dir / "last_checkpoint").write_text(
        str(checkpoint_path.resolve()) + "\n",
        encoding="utf-8",
    )


def resolve_checkpoint_reference(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_dir():
        candidate = candidate / "last_checkpoint"
    if candidate.name == "last_checkpoint":
        if not candidate.is_file():
            raise FileNotFoundError(f"No last_checkpoint file found at {candidate}.")
        resolved = Path(candidate.read_text(encoding="utf-8").strip()).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Checkpoint referenced by {candidate} does not exist: {resolved}."
            )
        return resolved
    if not candidate.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {candidate}.")
    return candidate
