from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

from .config import AppConfig
from .db import MetadataStore
from .models import Paper, PaperChunk
from .vector_store import VectorStore


@dataclass(slots=True)
class UpdateResult:
    query: str
    papers_seen: int
    papers_indexed: int
    chunks_indexed: int


class RAGBuilder:
    def __init__(self, config: AppConfig, console: Console | None = None):
        from .arxiv_client import ArxivPaperClient
        from .code_finder import CodeFinder
        from .pdf import PdfTextExtractor

        self.config = config
        self.console = console or Console()
        self.store = MetadataStore(config.paths.sqlite_path)
        self.arxiv = ArxivPaperClient(config.arxiv)
        self.pdf = PdfTextExtractor(config.paths.pdf_dir, config.ingestion)
        self.vectors = VectorStore(config.paths.chroma_dir, config.embedding)
        self.code_finder = CodeFinder(config.github, config.ingestion)

    def init(self) -> None:
        self.store.init_schema()

    def update_query(
        self,
        query: str,
        max_results: int | None = None,
        download_pdfs: bool | None = None,
    ) -> UpdateResult:
        self.init()
        should_download = self.config.ingestion.download_pdfs if download_pdfs is None else download_pdfs
        run_id = self.store.begin_run(query)
        papers_seen = 0
        papers_indexed = 0
        chunks_indexed = 0
        error: str | None = None
        try:
            for paper in self.arxiv.search(query, max_results=max_results):
                papers_seen += 1
                self.console.print(f"[dim]Indexing[/dim] {paper.arxiv_id} {paper.title}")
                chunks = self._chunks_for_paper(paper, should_download)
                self.store.upsert_paper(paper)
                repos = self.code_finder.discover(paper)
                self.store.upsert_code_repos(repos)
                chunks_indexed += self.vectors.upsert_chunks(
                    chunks,
                    title_by_paper={paper.arxiv_id: paper.title},
                )
                papers_indexed += 1
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            self.store.finish_run(run_id, papers_seen, papers_indexed, error)
        return UpdateResult(
            query=query,
            papers_seen=papers_seen,
            papers_indexed=papers_indexed,
            chunks_indexed=chunks_indexed,
        )

    def update_default_queries(self) -> list[UpdateResult]:
        results = []
        for query in self.config.arxiv.default_queries:
            results.append(
                self.update_query(
                    query=query,
                    max_results=self.config.arxiv.max_results_per_query,
                    download_pdfs=self.config.ingestion.download_pdfs,
                )
            )
        return results

    def _chunks_for_paper(self, paper: Paper, download_pdfs: bool) -> list[PaperChunk]:
        from .pdf import chunk_text

        source_text = f"{paper.title}\n\n{paper.summary}"
        section = "abstract"
        if download_pdfs and paper.pdf_url:
            try:
                pdf_path = self.pdf.download_pdf(paper.arxiv_id, paper.pdf_url)
                extracted = self.pdf.extract_text(pdf_path)
                if extracted:
                    source_text = extracted
                    section = "full_text"
            except Exception as exc:
                self.console.print(f"[yellow]PDF fallback for {paper.arxiv_id}: {exc}[/yellow]")
        chunks = chunk_text(
            source_text,
            chunk_size=self.config.ingestion.chunk_size,
            chunk_overlap=self.config.ingestion.chunk_overlap,
            min_chunk_chars=self.config.ingestion.min_chunk_chars,
        )
        return [
            PaperChunk(
                paper_id=paper.arxiv_id,
                chunk_index=index,
                text=chunk,
                section=section,
            )
            for index, chunk in enumerate(chunks)
        ]
