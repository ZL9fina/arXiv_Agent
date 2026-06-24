from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .models import CodeRepo

CLONEABLE_CODE_HOSTS = ("github.com", "gitlab.com", "bitbucket.org")


class SourceManager:
    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.repo_dir.mkdir(parents=True, exist_ok=True)

    def clone_repo(self, repo: CodeRepo) -> Path:
        if not is_cloneable_repo_url(repo.url):
            raise ValueError(f"URL is not a supported git repository host: {repo.url}")
        target = self.repo_dir / repo_folder_name(repo.url)
        if target.exists():
            write_deployment_notes(target, repo)
            return target
        subprocess.run(["git", "clone", repo.url, str(target)], check=True)
        write_deployment_notes(target, repo)
        return target


def repo_folder_name(url: str) -> str:
    clean = url.rstrip("/").removesuffix(".git")
    parts = clean.split("/")
    if len(parts) >= 2:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(parts[-2:]))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", clean)


def is_cloneable_repo_url(url: str) -> bool:
    lowered = url.lower()
    return any(host in lowered for host in CLONEABLE_CODE_HOSTS)


def detect_deployment_hints(path: Path) -> list[str]:
    hints: list[str] = []
    if (path / "Dockerfile").exists():
        hints.append("Dockerfile detected: consider `docker build -t paper-code .` then run in an isolated container.")
    if (path / "docker-compose.yml").exists() or (path / "compose.yml").exists():
        hints.append("Compose file detected: review services and volumes before `docker compose up`.")
    if (path / "requirements.txt").exists():
        hints.append("Python requirements detected: create a virtualenv, then `pip install -r requirements.txt`.")
    if (path / "pyproject.toml").exists():
        hints.append("Python project detected: inspect build backend, then install with `pip install -e .` if appropriate.")
    if (path / "environment.yml").exists():
        hints.append("Conda environment detected: review channels, then `conda env create -f environment.yml`.")
    if (path / "package.json").exists():
        hints.append("Node project detected: review scripts, then install dependencies with `npm install` or `pnpm install`.")
    if not hints:
        hints.append("No common deployment manifest was detected. Start by reading README files and examples.")
    return hints


def write_deployment_notes(path: Path, repo: CodeRepo) -> None:
    hints = detect_deployment_hints(path)
    notes = [
        "# Deployment Notes",
        "",
        f"- Repository: {repo.url}",
        f"- Detection source: {repo.source}",
        f"- Confidence: {repo.confidence:.2f}",
        "",
        "## Suggested next steps",
        "",
    ]
    notes.extend(f"- {hint}" for hint in hints)
    notes.extend(
        [
            "",
            "## Safety checklist",
            "",
            "- Review install scripts before running them.",
            "- Prefer an isolated virtualenv, Conda environment, or container.",
            "- Do not provide secrets to unknown code.",
            "- Run tests or examples before using the code with important data.",
            "",
        ]
    )
    (path / "DEPLOYMENT_NOTES.md").write_text("\n".join(notes), encoding="utf-8")
