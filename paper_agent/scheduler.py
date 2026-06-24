from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from rich.console import Console

from .config import AppConfig
from .rag import RAGBuilder


def run_scheduler(config: AppConfig) -> None:
    console = Console()
    scheduler = BlockingScheduler(timezone=config.scheduler.timezone)
    builder = RAGBuilder(config, console=console)

    def update_job() -> None:
        console.print("[bold]Starting monthly arXiv update[/bold]")
        results = builder.update_default_queries()
        for result in results:
            console.print(
                f"[green]Done[/green] {result.query}: "
                f"{result.papers_indexed}/{result.papers_seen} papers, "
                f"{result.chunks_indexed} chunks"
            )

    trigger = CronTrigger(
        day=config.scheduler.day,
        hour=config.scheduler.hour,
        minute=config.scheduler.minute,
        timezone=config.scheduler.timezone,
    )
    scheduler.add_job(update_job, trigger=trigger, id="monthly_arxiv_update", replace_existing=True)
    console.print(
        "[bold]Scheduler started[/bold] "
        f"(monthly on day {config.scheduler.day} at "
        f"{config.scheduler.hour:02d}:{config.scheduler.minute:02d} "
        f"{config.scheduler.timezone})"
    )
    scheduler.start()

