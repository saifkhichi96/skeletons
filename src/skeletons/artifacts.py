from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def sanitize_run_token(value: str) -> str:
    """Normalize a string for use in run names.

    Parameters
    ----------
    value : str
        Raw run-name component.

    Returns
    -------
    str
        Lowercase alphanumeric token with non-alphanumeric spans replaced by
        underscores.
    """

    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    return normalized or "run"


def make_run_name(*parts: object) -> str:
    """Build a stable run name from arbitrary parts.

    Parameters
    ----------
    *parts : object
        Run-name components. Empty or ``None`` parts are ignored.

    Returns
    -------
    str
        Sanitized underscore-delimited run name.
    """

    tokens = [
        sanitize_run_token(str(part))
        for part in parts
        if part is not None and str(part).strip()
    ]
    return "_".join(tokens) if tokens else "default"


def resolve_work_dir(
    script_path: Path | str,
    *,
    run_name: str,
    work_dir: Path | str | None = None,
    project_root: Path | str | None = None,
) -> Path:
    """Resolve and create a run output directory.

    Parameters
    ----------
    script_path : pathlib.Path or str
        Script path used to name the default run directory.
    run_name : str
        Sanitized or raw run name.
    work_dir : pathlib.Path or str, optional
        Explicit output directory. When supplied, ``project_root`` is ignored.
    project_root : pathlib.Path or str, optional
        Root directory used for default ``work_dirs`` output. If omitted, this
        defaults to the parent of ``script_path``'s containing directory, which
        matches scripts stored under a repository-level ``examples`` directory.

    Returns
    -------
    pathlib.Path
        Created output directory.
    """

    if work_dir is not None:
        path = Path(work_dir).expanduser()
    else:
        script_path = Path(script_path).expanduser().resolve()
        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else script_path.parents[1]
        )
        path = root / "work_dirs" / script_path.stem / make_run_name(run_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_status(script_name: str, message: str) -> None:
    """Print a timestamped status line.

    Parameters
    ----------
    script_name : str
        Name shown in the log prefix.
    message : str
        Status message.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{script_name}] {message}", flush=True)


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def save_json(path: Path | str, payload: Any) -> None:
    """Write a JSON file with stable formatting.

    Parameters
    ----------
    path : pathlib.Path or str
        Destination path.
    payload : Any
        JSON-serializable object. ``Path`` values are written as strings.
    """

    Path(path).write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_last_checkpoint(work_dir: Path | str, checkpoint_path: Path | str) -> None:
    """Write a ``last_checkpoint`` pointer file.

    Parameters
    ----------
    work_dir : pathlib.Path or str
        Run output directory.
    checkpoint_path : pathlib.Path or str
        Checkpoint file path to record.
    """

    work_dir = Path(work_dir)
    checkpoint_path = Path(checkpoint_path).expanduser()
    (work_dir / "last_checkpoint").write_text(
        str(checkpoint_path.resolve()) + "\n",
        encoding="utf-8",
    )


def resolve_checkpoint_reference(path: Path | str) -> Path:
    """Resolve a checkpoint path or ``last_checkpoint`` reference.

    Parameters
    ----------
    path : pathlib.Path or str
        Checkpoint file, run directory containing ``last_checkpoint``, or a
        ``last_checkpoint`` file.

    Returns
    -------
    pathlib.Path
        Resolved checkpoint file path.

    Raises
    ------
    FileNotFoundError
        If the reference or resolved checkpoint does not exist.
    """

    candidate = Path(path).expanduser()
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


@dataclass(frozen=True)
class RunArtifacts:
    """Convenience wrapper around a run output directory.

    Parameters
    ----------
    script_name : str
        Name shown in status logs.
    run_name : str
        Run identifier used for output paths.
    work_dir : pathlib.Path
        Created output directory.
    """

    script_name: str
    run_name: str
    work_dir: Path

    @classmethod
    def create(
        cls,
        script_path: Path | str,
        *,
        run_name: str,
        work_dir: Path | str | None = None,
        project_root: Path | str | None = None,
    ) -> RunArtifacts:
        """Create run-artifact state from a script path.

        Parameters
        ----------
        script_path : pathlib.Path or str
            Script path used to derive ``script_name`` and default output path.
        run_name : str
            Sanitized or raw run name.
        work_dir : pathlib.Path or str, optional
            Explicit output directory.
        project_root : pathlib.Path or str, optional
            Root used for default output directory resolution.

        Returns
        -------
        RunArtifacts
            Wrapper for logging and artifact writing.
        """

        script_path = Path(script_path)
        resolved_run_name = make_run_name(run_name)
        return cls(
            script_name=script_path.stem,
            run_name=resolved_run_name,
            work_dir=resolve_work_dir(
                script_path,
                run_name=resolved_run_name,
                work_dir=work_dir,
                project_root=project_root,
            ),
        )

    def path(self, *parts: str) -> Path:
        """Build a path inside the run output directory.

        Parameters
        ----------
        *parts : str
            Path components under ``work_dir``.

        Returns
        -------
        pathlib.Path
            Joined artifact path.
        """

        return self.work_dir.joinpath(*parts)

    def log(self, message: str) -> None:
        """Print a timestamped status line for this run.

        Parameters
        ----------
        message : str
            Status message.
        """

        log_status(self.script_name, message)

    def save_json(self, filename: str, payload: Any) -> None:
        """Write a JSON artifact under ``work_dir``.

        Parameters
        ----------
        filename : str
            Artifact filename.
        payload : Any
            JSON-serializable object.
        """

        save_json(self.path(filename), payload)

    def write_last_checkpoint(self, checkpoint_path: Path | str) -> None:
        """Write this run's ``last_checkpoint`` pointer.

        Parameters
        ----------
        checkpoint_path : pathlib.Path or str
            Checkpoint file path to record.
        """

        write_last_checkpoint(self.work_dir, checkpoint_path)
