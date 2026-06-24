from __future__ import annotations

import os
from pathlib import Path

from .config import load_config
from .live_agent import LivePaperAgent, paper_to_dict, report_to_dict


def run_server(config_path: str | Path | None = None) -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install the optional `mcp` package to run the MCP server.") from exc

    resolved_config = config_path or os.getenv("PAPER_AGENT_CONFIG")
    config = load_config(resolved_config)
    app = FastMCP("paper-scout-agent")

    @app.tool()
    def search_arxiv_papers(query: str, max_results: int = 8) -> dict[str, object]:
        """Search arXiv on demand and return paper metadata plus detected code links."""
        agent = LivePaperAgent(config)
        papers = agent.search_papers(query, max_results=max_results)
        return {"query": query, "papers": [paper_to_dict(item) for item in papers]}

    @app.tool()
    def recommend_papers(topic: str, max_results: int = 8, use_llm: bool = True) -> dict[str, object]:
        """Plan arXiv searches, fetch papers, and optionally ask the configured LLM for recommendations."""
        agent = LivePaperAgent(config)
        report = agent.discover(topic, max_results=max_results, use_llm=use_llm)
        return report_to_dict(report)

    app.run()


if __name__ == "__main__":
    run_server()
