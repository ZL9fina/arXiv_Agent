from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .config import load_config
from .db import MetadataStore


console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-agent",
        description="Paper Scout Agent: arXiv RAG, recommendations, author tracing, and code discovery.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize local metadata and vector-store directories.")
    add_config_arg(init_parser)
    init_parser.set_defaults(handler=handle_init)

    update_parser = subparsers.add_parser("update", help="Update the RAG database from arXiv.")
    add_config_arg(update_parser)
    update_parser.add_argument("--query", "-q", help="arXiv search query")
    update_parser.add_argument("--max-results", "-n", type=int, help="Max papers to fetch")
    pdf_group = update_parser.add_mutually_exclusive_group()
    pdf_group.add_argument("--download-pdfs", dest="download_pdfs", action="store_true")
    pdf_group.add_argument("--no-download-pdfs", dest="download_pdfs", action="store_false")
    update_parser.set_defaults(download_pdfs=None, handler=handle_update)

    learn_parser = subparsers.add_parser(
        "learn",
        help="Recommend papers for a learning topic and optionally download source code.",
    )
    add_config_arg(learn_parser)
    learn_parser.add_argument("topic", help="What the user wants to learn")
    learn_parser.add_argument("--top-k", "-k", type=int, default=8, help="Number of papers to recommend")
    learn_parser.add_argument(
        "--mode",
        choices=["live", "local-rag"],
        default="live",
        help="live searches arXiv on demand; local-rag uses the prebuilt Chroma database.",
    )
    learn_parser.add_argument("--no-llm", action="store_true", help="Skip LLM generation in live mode.")
    learn_parser.set_defaults(handler=handle_learn)

    authors_parser = subparsers.add_parser("authors", help="Trace papers and coauthors for an author.")
    add_config_arg(authors_parser)
    authors_parser.add_argument("name", help="Author name or partial name")
    authors_parser.add_argument("--limit", "-n", type=int, default=25, help="Max papers to show")
    authors_parser.set_defaults(handler=handle_authors)

    stats_parser = subparsers.add_parser("stats", help="Show local database stats.")
    add_config_arg(stats_parser)
    stats_parser.set_defaults(handler=handle_stats)

    scheduler_parser = subparsers.add_parser("scheduler", help="Run the monthly update scheduler in the foreground.")
    add_config_arg(scheduler_parser)
    scheduler_parser.set_defaults(handler=handle_scheduler)

    mcp_parser = subparsers.add_parser("mcp-server", help="Run an MCP server exposing arXiv paper tools.")
    add_config_arg(mcp_parser)
    mcp_parser.set_defaults(handler=handle_mcp_server)

    return parser


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", type=Path, help="Path to config.yaml")


def handle_init(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    MetadataStore(cfg.paths.sqlite_path).init_schema()
    console.print(f"[green]Initialized[/green] metadata DB at {cfg.paths.sqlite_path}")
    console.print(f"[green]Ready[/green] vector DB directory at {cfg.paths.chroma_dir}")
    return 0


def handle_update(args: argparse.Namespace) -> int:
    from .rag import RAGBuilder

    cfg = load_config(args.config)
    builder = RAGBuilder(cfg, console=console)
    queries = [args.query] if args.query else cfg.arxiv.default_queries
    for query in queries:
        result = builder.update_query(
            query,
            max_results=args.max_results or cfg.arxiv.max_results_per_query,
            download_pdfs=args.download_pdfs,
        )
        console.print(
            f"[green]Updated[/green] {result.query}: "
            f"{result.papers_indexed}/{result.papers_seen} papers, "
            f"{result.chunks_indexed} chunks"
        )
    return 0


def handle_learn(args: argparse.Namespace) -> int:
    if args.mode == "live":
        return handle_live_learn(args)
    return handle_local_rag_learn(args)


def handle_live_learn(args: argparse.Namespace) -> int:
    from .live_agent import LivePaperAgent
    from .source_manager import SourceManager, is_cloneable_repo_url

    cfg = load_config(args.config)
    agent = LivePaperAgent(cfg)
    report = agent.discover(args.topic, max_results=args.top_k, use_llm=not args.no_llm)

    console.rule("[bold]Live arXiv Paper Agent[/bold]")
    console.print(f"[cyan]Queries[/cyan] {', '.join(report.queries)}")
    if not report.llm_used:
        console.print(
            f"[yellow]LLM was not used.[/yellow] Set {cfg.llm.api_key_env} "
            "or pass a provider-compatible config to enable generated recommendations."
        )
    console.print(Markdown(report.answer))
    print_paper_table(report.papers)

    repos_to_offer = [
        repo
        for item in report.papers
        for repo in item.code_repos
        if is_cloneable_repo_url(repo.url)
    ]
    if repos_to_offer and confirm("Download detected source repositories locally?", default=False):
        manager = SourceManager(cfg.paths.repo_dir)
        for repo in repos_to_offer:
            if confirm(f"Clone {repo.url}?", default=False):
                try:
                    path = manager.clone_repo(repo)
                    console.print(f"[green]Downloaded[/green] {repo.url} -> {path}")
                    console.print(f"[dim]Read {path / 'DEPLOYMENT_NOTES.md'} before running anything.[/dim]")
                except Exception as exc:
                    console.print(f"[red]Failed[/red] {repo.url}: {exc}")
    return 0


def handle_local_rag_learn(args: argparse.Namespace) -> int:
    from .recommend import RecommendationEngine
    from .source_manager import SourceManager, is_cloneable_repo_url

    cfg = load_config(args.config)
    engine = RecommendationEngine(cfg)
    recommendations = engine.recommend(args.topic, top_k=args.top_k)
    if not recommendations:
        console.print("[yellow]No recommendations found. Run `update` first.[/yellow]")
        return 1

    repos_to_offer = []
    for index, rec in enumerate(recommendations, start=1):
        paper = rec.paper
        console.rule(f"[bold]{index}. {paper.title}[/bold]")
        console.print(f"[cyan]arXiv[/cyan] {paper.arxiv_id}  [cyan]score[/cyan] {rec.score:.3f}")
        console.print(f"[cyan]authors[/cyan] {', '.join(paper.authors[:8])}")
        if paper.entry_url:
            console.print(f"[cyan]link[/cyan] {paper.entry_url}")
        console.print("[bold]Why this helps[/bold]")
        for evidence in rec.evidence:
            console.print(f"- {evidence}")
        if rec.code_repos:
            console.print("[bold]Detected code[/bold]")
            for repo in rec.code_repos:
                console.print(f"- {repo.url} ({repo.source}, confidence {repo.confidence:.2f})")
                if is_cloneable_repo_url(repo.url):
                    repos_to_offer.append(repo)

    if repos_to_offer and confirm("Download detected source repositories locally?", default=False):
        manager = SourceManager(cfg.paths.repo_dir)
        for repo in repos_to_offer:
            if confirm(f"Clone {repo.url}?", default=False):
                try:
                    path = manager.clone_repo(repo)
                    console.print(f"[green]Downloaded[/green] {repo.url} -> {path}")
                    console.print(f"[dim]Read {path / 'DEPLOYMENT_NOTES.md'} before running anything.[/dim]")
                except Exception as exc:
                    console.print(f"[red]Failed[/red] {repo.url}: {exc}")
    return 0


def print_paper_table(items) -> None:
    if not items:
        console.print("[yellow]No papers found.[/yellow]")
        return
    table = Table(title="Live arXiv results")
    table.add_column("arXiv")
    table.add_column("Title")
    table.add_column("Authors")
    table.add_column("Code")
    for item in items:
        paper = item.paper
        code = ", ".join(repo.url for repo in item.code_repos[:2]) or "-"
        table.add_row(paper.arxiv_id, paper.title, ", ".join(paper.authors[:3]), code)
    console.print(table)


def handle_authors(args: argparse.Namespace) -> int:
    from .recommend import RecommendationEngine

    cfg = load_config(args.config)
    engine = RecommendationEngine(cfg)
    papers, coauthors = engine.author_trace(args.name, limit=args.limit)

    if papers:
        table = Table(title=f"Papers matching author: {args.name}")
        table.add_column("Updated")
        table.add_column("arXiv")
        table.add_column("Title")
        table.add_column("Authors")
        for paper in papers:
            updated = paper.updated.date().isoformat() if paper.updated else ""
            table.add_row(updated, paper.arxiv_id, paper.title, ", ".join(paper.authors[:4]))
        console.print(table)
    else:
        console.print(f"[yellow]No papers found for author query: {args.name}[/yellow]")

    if coauthors:
        coauthor_table = Table(title="Frequent coauthors")
        coauthor_table.add_column("Author")
        coauthor_table.add_column("Papers")
        for author, count in coauthors:
            coauthor_table.add_row(author, str(count))
        console.print(coauthor_table)
    return 0


def handle_stats(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    store = MetadataStore(cfg.paths.sqlite_path)
    store.init_schema()
    table = Table(title="Paper Scout Agent Stats")
    table.add_column("Metric")
    table.add_column("Count")
    for key, value in store.stats().items():
        table.add_row(key, str(value))
    console.print(table)
    return 0


def handle_scheduler(args: argparse.Namespace) -> int:
    from .scheduler import run_scheduler

    cfg = load_config(args.config)
    run_scheduler(cfg)
    return 0


def handle_mcp_server(args: argparse.Namespace) -> int:
    from .mcp_server import run_server

    run_server(args.config)
    return 0


def confirm(prompt: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix} ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
