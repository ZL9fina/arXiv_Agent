from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathsConfig:
    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/papers.sqlite3")
    chroma_dir: Path = Path("data/chroma")
    pdf_dir: Path = Path("data/pdfs")
    repo_dir: Path = Path("data/repos")


@dataclass(slots=True)
class ArxivConfig:
    default_queries: list[str] = field(default_factory=lambda: ["retrieval augmented generation"])
    categories: list[str] = field(default_factory=list)
    max_results_per_query: int = 25
    sort_by: str = "submittedDate"
    sort_order: str = "descending"


@dataclass(slots=True)
class IngestionConfig:
    download_pdfs: bool = True
    chunk_size: int = 1400
    chunk_overlap: int = 200
    min_chunk_chars: int = 300
    request_timeout_seconds: int = 45
    user_agent: str = "PaperScoutAgent/0.1"


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32


@dataclass(slots=True)
class GithubConfig:
    enabled: bool = True
    token_env: str = "GITHUB_TOKEN"
    max_search_results: int = 3


@dataclass(slots=True)
class LLMConfig:
    enabled: bool = True
    provider: str = "openai-compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    max_tokens: int = 1400
    timeout_seconds: int = 60


@dataclass(slots=True)
class SchedulerConfig:
    day: int = 1
    hour: int = 3
    minute: int = 30
    timezone: str = "Asia/Shanghai"


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    arxiv: ArxivConfig = field(default_factory=ArxivConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    github: GithubConfig = field(default_factory=GithubConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def ensure_dirs(self) -> None:
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.paths.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.paths.repo_dir.mkdir(parents=True, exist_ok=True)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_path_config(data: dict[str, Any]) -> PathsConfig:
    return PathsConfig(**{key: Path(value) for key, value in data.items()})


def load_config(path: str | Path | None = None) -> AppConfig:
    defaults = {
        "paths": {},
        "arxiv": {},
        "ingestion": {},
        "embedding": {},
        "github": {},
        "llm": {},
        "scheduler": {},
    }
    if path:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        data = _merge_dict(defaults, raw)
    else:
        data = defaults

    config = AppConfig(
        paths=_as_path_config(data["paths"]),
        arxiv=ArxivConfig(**data["arxiv"]),
        ingestion=IngestionConfig(**data["ingestion"]),
        embedding=EmbeddingConfig(**data["embedding"]),
        github=GithubConfig(**data["github"]),
        llm=LLMConfig(**data["llm"]),
        scheduler=SchedulerConfig(**data["scheduler"]),
    )
    config.ensure_dirs()
    return config
