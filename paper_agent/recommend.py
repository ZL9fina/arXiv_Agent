from __future__ import annotations

from collections import defaultdict

from .config import AppConfig
from .db import MetadataStore
from .models import Recommendation, SearchHit
from .vector_store import VectorStore


class RecommendationEngine:
    def __init__(self, config: AppConfig):
        self.store = MetadataStore(config.paths.sqlite_path)
        self.config = config
        self._vectors: VectorStore | None = None

    def recommend(self, topic: str, top_k: int = 8) -> list[Recommendation]:
        hits = self.vectors.query(topic, top_k=max(top_k * 4, top_k))
        grouped: dict[str, list[SearchHit]] = defaultdict(list)
        for hit in hits:
            grouped[hit.paper_id].append(hit)
        recommendations: list[Recommendation] = []
        for paper_id, paper_hits in grouped.items():
            paper = self.store.get_paper(paper_id)
            if not paper:
                continue
            sorted_hits = sorted(paper_hits, key=lambda hit: hit.score, reverse=True)
            score = sum(hit.score for hit in sorted_hits[:3]) / min(3, len(sorted_hits))
            evidence = [compact_evidence(hit.text) for hit in sorted_hits[:2]]
            recommendations.append(
                Recommendation(
                    paper=paper,
                    score=score,
                    evidence=evidence,
                    code_repos=self.store.get_code_repos(paper_id),
                )
            )
        return sorted(recommendations, key=lambda rec: rec.score, reverse=True)[:top_k]

    def author_trace(self, author_query: str, limit: int = 25):
        papers = self.store.papers_by_author(author_query, limit=limit)
        coauthors = self.store.coauthors_for_author(author_query)
        return papers, coauthors

    @property
    def vectors(self) -> VectorStore:
        if self._vectors is None:
            self._vectors = VectorStore(self.config.paths.chroma_dir, self.config.embedding)
        return self._vectors


def compact_evidence(text: str, max_chars: int = 360) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."
