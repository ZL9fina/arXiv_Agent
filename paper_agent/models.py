from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class Paper:
    arxiv_id: str
    title: str
    summary: str
    authors: list[str]
    published: datetime | None
    updated: datetime | None
    pdf_url: str | None
    entry_url: str | None
    primary_category: str | None = None
    categories: list[str] = field(default_factory=list)
    comment: str | None = None
    journal_ref: str | None = None
    doi: str | None = None
    links: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PaperChunk:
    paper_id: str
    chunk_index: int
    text: str
    section: str = "body"


@dataclass(slots=True)
class CodeRepo:
    paper_id: str
    url: str
    source: str
    confidence: float
    description: str | None = None


@dataclass(slots=True)
class SearchHit:
    paper_id: str
    title: str
    text: str
    score: float
    chunk_index: int


@dataclass(slots=True)
class Recommendation:
    paper: Paper
    score: float
    evidence: list[str]
    code_repos: list[CodeRepo] = field(default_factory=list)

