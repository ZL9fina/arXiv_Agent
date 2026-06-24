from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Iterable

from .models import CodeRepo, Paper


class MetadataStore:
    def __init__(self, sqlite_path: Path):
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    arxiv_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    published TEXT,
                    updated TEXT,
                    pdf_url TEXT,
                    entry_url TEXT,
                    primary_category TEXT,
                    categories_json TEXT NOT NULL,
                    comment TEXT,
                    journal_ref TEXT,
                    doi TEXT,
                    links_json TEXT NOT NULL,
                    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS authors (
                    name TEXT PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS paper_authors (
                    paper_id TEXT NOT NULL,
                    author_name TEXT NOT NULL,
                    author_order INTEGER NOT NULL,
                    PRIMARY KEY (paper_id, author_name),
                    FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                    FOREIGN KEY (author_name) REFERENCES authors(name) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS code_repos (
                    paper_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    description TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (paper_id, url),
                    FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ingest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT,
                    papers_seen INTEGER DEFAULT 0,
                    papers_indexed INTEGER DEFAULT 0,
                    error TEXT
                );
                """
            )

    def begin_run(self, query: str) -> int:
        with self.connect() as conn:
            cur = conn.execute("INSERT INTO ingest_runs (query) VALUES (?)", (query,))
            return int(cur.lastrowid)

    def finish_run(
        self,
        run_id: int,
        papers_seen: int,
        papers_indexed: int,
        error: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE ingest_runs
                SET finished_at = CURRENT_TIMESTAMP,
                    papers_seen = ?,
                    papers_indexed = ?,
                    error = ?
                WHERE id = ?
                """,
                (papers_seen, papers_indexed, error, run_id),
            )

    def upsert_paper(self, paper: Paper) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (
                    arxiv_id, title, summary, published, updated, pdf_url, entry_url,
                    primary_category, categories_json, comment, journal_ref, doi, links_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    title = excluded.title,
                    summary = excluded.summary,
                    published = excluded.published,
                    updated = excluded.updated,
                    pdf_url = excluded.pdf_url,
                    entry_url = excluded.entry_url,
                    primary_category = excluded.primary_category,
                    categories_json = excluded.categories_json,
                    comment = excluded.comment,
                    journal_ref = excluded.journal_ref,
                    doi = excluded.doi,
                    links_json = excluded.links_json,
                    indexed_at = CURRENT_TIMESTAMP
                """,
                (
                    paper.arxiv_id,
                    paper.title,
                    paper.summary,
                    paper.published.isoformat() if paper.published else None,
                    paper.updated.isoformat() if paper.updated else None,
                    paper.pdf_url,
                    paper.entry_url,
                    paper.primary_category,
                    json.dumps(paper.categories),
                    paper.comment,
                    paper.journal_ref,
                    paper.doi,
                    json.dumps(paper.links),
                ),
            )
            conn.execute("DELETE FROM paper_authors WHERE paper_id = ?", (paper.arxiv_id,))
            for index, author in enumerate(paper.authors):
                conn.execute("INSERT OR IGNORE INTO authors (name) VALUES (?)", (author,))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO paper_authors (paper_id, author_name, author_order)
                    VALUES (?, ?, ?)
                    """,
                    (paper.arxiv_id, author, index),
                )

    def get_paper(self, arxiv_id: str) -> Paper | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
            if not row:
                return None
            authors = [
                r["author_name"]
                for r in conn.execute(
                    """
                    SELECT author_name
                    FROM paper_authors
                    WHERE paper_id = ?
                    ORDER BY author_order
                    """,
                    (arxiv_id,),
                )
            ]
        return self._paper_from_row(row, authors)

    def upsert_code_repos(self, repos: Iterable[CodeRepo]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO code_repos (paper_id, url, source, confidence, description)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(paper_id, url) DO UPDATE SET
                    source = excluded.source,
                    confidence = excluded.confidence,
                    description = excluded.description
                """,
                [(repo.paper_id, repo.url, repo.source, repo.confidence, repo.description) for repo in repos],
            )

    def get_code_repos(self, paper_id: str) -> list[CodeRepo]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT paper_id, url, source, confidence, description
                FROM code_repos
                WHERE paper_id = ?
                ORDER BY confidence DESC, source
                """,
                (paper_id,),
            ).fetchall()
        return [
            CodeRepo(
                paper_id=row["paper_id"],
                url=row["url"],
                source=row["source"],
                confidence=float(row["confidence"]),
                description=row["description"],
            )
            for row in rows
        ]

    def papers_by_author(self, author_query: str, limit: int = 25) -> list[Paper]:
        like = f"%{author_query}%"
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT p.*
                FROM papers p
                JOIN paper_authors pa ON pa.paper_id = p.arxiv_id
                WHERE pa.author_name LIKE ?
                ORDER BY COALESCE(p.updated, p.published) DESC
                LIMIT ?
                """,
                (like, limit),
            ).fetchall()
            paper_ids = [row["arxiv_id"] for row in rows]
            authors_by_paper = self._authors_for_papers(conn, paper_ids)
        return [self._paper_from_row(row, authors_by_paper.get(row["arxiv_id"], [])) for row in rows]

    def coauthors_for_author(self, author_query: str, limit: int = 20) -> list[tuple[str, int]]:
        papers = self.papers_by_author(author_query, limit=200)
        counter: Counter[str] = Counter()
        matched_names = {
            author
            for paper in papers
            for author in paper.authors
            if author_query.lower() in author.lower()
        }
        for paper in papers:
            for author in paper.authors:
                if author not in matched_names:
                    counter[author] += 1
        return counter.most_common(limit)

    def stats(self) -> dict[str, int]:
        with self.connect() as conn:
            return {
                "papers": int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]),
                "authors": int(conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]),
                "code_repos": int(conn.execute("SELECT COUNT(*) FROM code_repos").fetchone()[0]),
                "ingest_runs": int(conn.execute("SELECT COUNT(*) FROM ingest_runs").fetchone()[0]),
            }

    def _authors_for_papers(self, conn: sqlite3.Connection, paper_ids: list[str]) -> dict[str, list[str]]:
        if not paper_ids:
            return {}
        placeholders = ",".join("?" for _ in paper_ids)
        rows = conn.execute(
            f"""
            SELECT paper_id, author_name
            FROM paper_authors
            WHERE paper_id IN ({placeholders})
            ORDER BY paper_id, author_order
            """,
            paper_ids,
        ).fetchall()
        authors_by_paper: dict[str, list[str]] = {paper_id: [] for paper_id in paper_ids}
        for row in rows:
            authors_by_paper[row["paper_id"]].append(row["author_name"])
        return authors_by_paper

    @staticmethod
    def _paper_from_row(row: sqlite3.Row, authors: list[str]) -> Paper:
        return Paper(
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            summary=row["summary"],
            authors=authors,
            published=_parse_datetime(row["published"]),
            updated=_parse_datetime(row["updated"]),
            pdf_url=row["pdf_url"],
            entry_url=row["entry_url"],
            primary_category=row["primary_category"],
            categories=json.loads(row["categories_json"] or "[]"),
            comment=row["comment"],
            journal_ref=row["journal_ref"],
            doi=row["doi"],
            links=json.loads(row["links_json"] or "[]"),
        )


def _parse_datetime(value: str | None):
    if not value:
        return None
    return datetime.fromisoformat(value)
