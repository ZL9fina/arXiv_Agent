from __future__ import annotations

import re
from pathlib import Path

import requests

from .config import IngestionConfig


def safe_paper_filename(arxiv_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", arxiv_id) + ".pdf"


class PdfTextExtractor:
    def __init__(self, pdf_dir: Path, config: IngestionConfig):
        self.pdf_dir = pdf_dir
        self.config = config
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def download_pdf(self, arxiv_id: str, pdf_url: str) -> Path:
        target = self.pdf_dir / safe_paper_filename(arxiv_id)
        if target.exists() and target.stat().st_size > 0:
            return target
        headers = {"User-Agent": self.config.user_agent}
        response = requests.get(
            pdf_url,
            timeout=self.config.request_timeout_seconds,
            headers=headers,
        )
        response.raise_for_status()
        target.write_bytes(response.content)
        return target

    def extract_text(self, path: Path) -> str:
        import fitz

        pages: list[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                text = page.get_text("text")
                if text:
                    pages.append(text)
        return normalize_text("\n".join(pages))


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
) -> list[str]:
    clean_text = normalize_text(text)
    if not clean_text:
        return []
    chunks: list[str] = []
    start = 0
    text_length = len(clean_text)
    step = max(1, chunk_size - chunk_overlap)
    while start < text_length:
        end = min(text_length, start + chunk_size)
        chunk = clean_text[start:end].strip()
        if len(chunk) >= min_chunk_chars:
            chunks.append(chunk)
        start += step
    if not chunks and clean_text:
        chunks.append(clean_text[:chunk_size])
    return chunks
