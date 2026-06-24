from __future__ import annotations

from collections.abc import Iterable

import arxiv

from .config import ArxivConfig
from .models import Paper


SORT_BY = {
    "relevance": arxiv.SortCriterion.Relevance,
    "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    "submittedDate": arxiv.SortCriterion.SubmittedDate,
}

SORT_ORDER = {
    "ascending": arxiv.SortOrder.Ascending,
    "descending": arxiv.SortOrder.Descending,
}


class ArxivPaperClient:
    def __init__(self, config: ArxivConfig):
        self.config = config
        self.client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)

    def search(self, query: str, max_results: int | None = None) -> Iterable[Paper]:
        search_query = self._build_query(query)
        search = arxiv.Search(
            query=search_query,
            max_results=max_results or self.config.max_results_per_query,
            sort_by=SORT_BY.get(self.config.sort_by, arxiv.SortCriterion.SubmittedDate),
            sort_order=SORT_ORDER.get(self.config.sort_order, arxiv.SortOrder.Descending),
        )
        for result in self.client.results(search):
            yield self._from_result(result)

    def _build_query(self, query: str) -> str:
        if not self.config.categories:
            return query
        category_query = " OR ".join(f"cat:{category}" for category in self.config.categories)
        return f"({query}) AND ({category_query})"

    @staticmethod
    def _from_result(result: arxiv.Result) -> Paper:
        links = [link.href for link in result.links if getattr(link, "href", None)]
        return Paper(
            arxiv_id=result.get_short_id(),
            title=_normalize_space(result.title),
            summary=_normalize_space(result.summary),
            authors=[str(author) for author in result.authors],
            published=result.published,
            updated=result.updated,
            pdf_url=result.pdf_url,
            entry_url=result.entry_id,
            primary_category=result.primary_category,
            categories=list(result.categories),
            comment=result.comment,
            journal_ref=result.journal_ref,
            doi=result.doi,
            links=links,
        )


def _normalize_space(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())

