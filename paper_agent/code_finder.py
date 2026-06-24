from __future__ import annotations

import os
import re
from urllib.parse import quote_plus

import requests

from .config import GithubConfig, IngestionConfig
from .models import CodeRepo, Paper


URL_RE = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)
CODE_HOSTS = ("github.com", "gitlab.com", "bitbucket.org")


class CodeFinder:
    def __init__(self, github_config: GithubConfig, ingestion_config: IngestionConfig):
        self.github_config = github_config
        self.ingestion_config = ingestion_config

    def discover(self, paper: Paper) -> list[CodeRepo]:
        repos = self.from_paper_text(paper)
        if self.github_config.enabled:
            repos.extend(self.from_github_search(paper))
        return _dedupe_repos(repos)

    def from_paper_text(self, paper: Paper) -> list[CodeRepo]:
        haystack = "\n".join(
            [
                paper.title,
                paper.summary,
                paper.comment or "",
                paper.journal_ref or "",
                "\n".join(paper.links),
            ]
        )
        repos: list[CodeRepo] = []
        for url in extract_urls(haystack):
            normalized = normalize_url(url)
            if any(host in normalized.lower() for host in CODE_HOSTS):
                repos.append(
                    CodeRepo(
                        paper_id=paper.arxiv_id,
                        url=normalized,
                        source="paper_text",
                        confidence=0.95,
                        description="Code repository URL found in arXiv metadata.",
                    )
                )
            elif "paperswithcode.com" in normalized.lower():
                repos.append(
                    CodeRepo(
                        paper_id=paper.arxiv_id,
                        url=normalized,
                        source="paperswithcode_link",
                        confidence=0.75,
                        description="Papers with Code page found in arXiv metadata.",
                    )
                )
        return repos

    def from_github_search(self, paper: Paper) -> list[CodeRepo]:
        token = os.getenv(self.github_config.token_env)
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.ingestion_config.user_agent,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        query = quote_plus(f'"{paper.title}"')
        url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc"
        try:
            response = requests.get(url, headers=headers, timeout=self.ingestion_config.request_timeout_seconds)
            response.raise_for_status()
            items = response.json().get("items", [])[: self.github_config.max_search_results]
        except requests.RequestException:
            return []
        repos: list[CodeRepo] = []
        for item in items:
            full_name = item.get("full_name", "")
            description = item.get("description") or ""
            confidence = score_github_match(paper, full_name, description)
            if confidence >= 0.45:
                repos.append(
                    CodeRepo(
                        paper_id=paper.arxiv_id,
                        url=item.get("html_url", ""),
                        source="github_search",
                        confidence=confidence,
                        description=description,
                    )
                )
        return repos


def extract_urls(text: str) -> list[str]:
    return [match.group(0).rstrip(".,;:") for match in URL_RE.finditer(text or "")]


def normalize_url(url: str) -> str:
    clean = url.strip().rstrip("/")
    if clean.endswith(".git"):
        clean = clean[:-4]
    return clean


def score_github_match(paper: Paper, repo_name: str, description: str) -> float:
    paper_words = _meaningful_words(paper.title)
    repo_words = _meaningful_words(f"{repo_name} {description}")
    if not paper_words or not repo_words:
        return 0.0
    overlap = len(paper_words & repo_words) / max(1, len(paper_words))
    title_hint = 0.15 if any(word in repo_name.lower() for word in paper_words) else 0.0
    return min(0.9, overlap + title_hint)


def _meaningful_words(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "the",
        "to",
        "via",
        "with",
    }
    return {
        word
        for word in re.findall(r"[a-z0-9]{3,}", text.lower())
        if word not in stopwords
    }


def _dedupe_repos(repos: list[CodeRepo]) -> list[CodeRepo]:
    best: dict[str, CodeRepo] = {}
    for repo in repos:
        if not repo.url:
            continue
        key = normalize_url(repo.url).lower()
        existing = best.get(key)
        if not existing or repo.confidence > existing.confidence:
            best[key] = repo
    return sorted(best.values(), key=lambda repo: repo.confidence, reverse=True)

