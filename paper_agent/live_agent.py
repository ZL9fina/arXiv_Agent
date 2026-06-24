from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from .config import AppConfig
from .llm import ChatMessage, LLMClient
from .models import CodeRepo, Paper


@dataclass(slots=True)
class PaperWithCode:
    paper: Paper
    code_repos: list[CodeRepo] = field(default_factory=list)


@dataclass(slots=True)
class LivePaperReport:
    topic: str
    queries: list[str]
    papers: list[PaperWithCode]
    answer: str
    llm_used: bool


class LivePaperAgent:
    """On-demand paper discovery agent.

    This path avoids building a large public-paper RAG cache. It plans arXiv
    search queries from the user's goal, fetches current metadata directly,
    discovers code links, then asks the LLM to produce a learning-oriented
    recommendation report.
    """

    def __init__(self, config: AppConfig):
        from .arxiv_client import ArxivPaperClient
        from .code_finder import CodeFinder

        self.config = config
        self.arxiv = ArxivPaperClient(config.arxiv)
        self.code_finder = CodeFinder(config.github, config.ingestion)
        self.llm = LLMClient(config.llm)

    def search_papers(self, query: str, max_results: int = 8) -> list[PaperWithCode]:
        papers = list(self.arxiv.search(query, max_results=max_results))
        return [PaperWithCode(paper=paper, code_repos=self.code_finder.discover(paper)) for paper in papers]

    def discover(self, topic: str, max_results: int = 8, use_llm: bool = True) -> LivePaperReport:
        queries = self.plan_queries(topic, max_queries=3) if use_llm else [topic]
        results_by_id: dict[str, PaperWithCode] = {}
        per_query_limit = max(1, max_results)
        for query in queries:
            for item in self.search_papers(query, max_results=per_query_limit):
                results_by_id.setdefault(item.paper.arxiv_id, item)
                if len(results_by_id) >= max_results:
                    break
            if len(results_by_id) >= max_results:
                break
        papers = list(results_by_id.values())
        answer, llm_used = self.generate_report(topic, queries, papers, use_llm=use_llm)
        return LivePaperReport(
            topic=topic,
            queries=queries,
            papers=papers,
            answer=answer,
            llm_used=llm_used,
        )

    def plan_queries(self, topic: str, max_queries: int = 3) -> list[str]:
        if not self.llm.is_configured:
            return [topic]
        prompt = (
            "You are planning arXiv searches for a research-paper learning agent. "
            "Convert the user's Chinese or English learning goal into concise English arXiv search queries. "
            "Return only a JSON array of strings. Avoid broad words like AI alone.\n\n"
            f"User goal: {topic}\n"
            f"Max queries: {max_queries}"
        )
        try:
            content = self.llm.complete(
                [
                    ChatMessage("system", "You generate precise arXiv search queries."),
                    ChatMessage("user", prompt),
                ]
            )
            queries = parse_json_string_list(content)
        except Exception:
            queries = []
        queries = [query for query in queries if query.strip()]
        return queries[:max_queries] or [topic]

    def generate_report(
        self,
        topic: str,
        queries: list[str],
        papers: list[PaperWithCode],
        use_llm: bool = True,
    ) -> tuple[str, bool]:
        if use_llm and self.llm.is_configured and papers:
            prompt = build_recommendation_prompt(topic, queries, papers)
            try:
                answer = self.llm.complete(
                    [
                        ChatMessage(
                            "system",
                            "You are a research learning agent. Recommend papers with grounded evidence only.",
                        ),
                        ChatMessage("user", prompt),
                    ]
                )
                return answer, True
            except Exception as exc:
                return fallback_report(topic, queries, papers, llm_error=str(exc)), False
        return fallback_report(topic, queries, papers), False


def build_recommendation_prompt(topic: str, queries: list[str], papers: list[PaperWithCode]) -> str:
    contexts = []
    for index, item in enumerate(papers, start=1):
        paper = item.paper
        repos = ", ".join(repo.url for repo in item.code_repos) or "None detected"
        contexts.append(
            "\n".join(
                [
                    f"[{index}] {paper.title}",
                    f"arXiv: {paper.arxiv_id}",
                    f"Authors: {', '.join(paper.authors[:8])}",
                    f"Published: {_date_str(paper.published)}",
                    f"Categories: {', '.join(paper.categories[:5])}",
                    f"Entry: {paper.entry_url or ''}",
                    f"PDF: {paper.pdf_url or ''}",
                    f"Code: {repos}",
                    f"Abstract: {paper.summary[:900]}",
                ]
            )
        )
    return (
        "用户希望学习的主题：\n"
        f"{topic}\n\n"
        "实际使用的 arXiv 检索式：\n"
        f"{json.dumps(queries, ensure_ascii=False)}\n\n"
        "候选论文：\n"
        + "\n\n".join(contexts)
        + "\n\n请用中文输出："
        "1. 推荐阅读顺序；2. 每篇论文为什么相关；3. 可追溯的作者/团队线索；"
        "4. 如果有源码，给出复现优先级；5. 明确说明结论只基于上面的 arXiv 元数据。"
    )


def fallback_report(
    topic: str,
    queries: list[str],
    papers: list[PaperWithCode],
    llm_error: str | None = None,
) -> str:
    lines = [
        f"Topic: {topic}",
        f"arXiv queries: {', '.join(queries)}",
    ]
    if llm_error:
        lines.append(f"LLM fallback reason: {llm_error}")
    else:
        lines.append("LLM is not configured, so this is a deterministic paper list.")
    if not papers:
        lines.append("No papers found.")
        return "\n".join(lines)
    lines.append("")
    lines.append("Recommended papers:")
    for index, item in enumerate(papers, start=1):
        paper = item.paper
        code = ", ".join(repo.url for repo in item.code_repos) or "no code detected"
        lines.append(f"{index}. {paper.title}")
        lines.append(f"   arXiv: {paper.arxiv_id} | authors: {', '.join(paper.authors[:4])}")
        lines.append(f"   link: {paper.entry_url or paper.pdf_url or ''}")
        lines.append(f"   code: {code}")
    return "\n".join(lines)


def parse_json_string_list(text: str) -> list[str]:
    clean = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", clean, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        clean = fenced.group(1).strip()
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", clean)
        if not match:
            return []
        value = json.loads(match.group(0))
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def paper_to_dict(item: PaperWithCode) -> dict[str, object]:
    paper = item.paper
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "summary": paper.summary,
        "authors": paper.authors,
        "published": _date_str(paper.published),
        "updated": _date_str(paper.updated),
        "pdf_url": paper.pdf_url,
        "entry_url": paper.entry_url,
        "primary_category": paper.primary_category,
        "categories": paper.categories,
        "code_repos": [
            {
                "url": repo.url,
                "source": repo.source,
                "confidence": repo.confidence,
                "description": repo.description,
            }
            for repo in item.code_repos
        ],
    }


def report_to_dict(report: LivePaperReport) -> dict[str, object]:
    return {
        "topic": report.topic,
        "queries": report.queries,
        "llm_used": report.llm_used,
        "answer": report.answer,
        "papers": [paper_to_dict(item) for item in report.papers],
    }


def _date_str(value: datetime | None) -> str:
    return value.date().isoformat() if value else ""
