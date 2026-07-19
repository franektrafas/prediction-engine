"""Helpers for resuming the project inside Google Colab."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess


def resume_colab_workspace(
    repo_url: str = "https://github.com/franektrafas/betclic.v1.git",
    repo_dir: str = "/content/betclic.v1",
    drive_checkpoint_dir: str = "/content/drive/MyDrive/football_ai_checkpoints",
) -> dict[str, object]:
    """Restore repo state, install requirements, and inspect checkpoints."""
    repository_path = Path(repo_dir)
    if not repository_path.exists():
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

    subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
    subprocess.run(
        ["python3", "-m", "pip", "install", "-r", str(repository_path / "requirements-colab.txt")],
        check=True,
    )

    checkpoint_root = Path(drive_checkpoint_dir)
    processed_dir = checkpoint_root / "processed"
    artifact_dir = checkpoint_root / "artifacts"
    os.chdir(repo_dir)

    return {
        "repo_dir": str(repository_path),
        "processed_exists": processed_dir.exists(),
        "artifact_exists": artifact_dir.exists(),
        "processed_files": sorted(path.name for path in processed_dir.glob("*"))[:20]
        if processed_dir.exists()
        else [],
        "artifact_files": sorted(path.name for path in artifact_dir.glob("*"))[:20]
        if artifact_dir.exists()
        else [],
    }
