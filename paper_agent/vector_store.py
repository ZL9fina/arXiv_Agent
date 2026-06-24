from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import EmbeddingConfig
from .models import PaperChunk, SearchHit


class VectorStore:
    def __init__(self, chroma_dir: Path, config: EmbeddingConfig):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name="paper_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(config.model_name)

    def upsert_chunks(self, chunks: Iterable[PaperChunk], title_by_paper: dict[str, str]) -> int:
        batch_ids: list[str] = []
        batch_docs: list[str] = []
        batch_meta: list[dict[str, str | int]] = []
        total = 0
        for chunk in chunks:
            batch_ids.append(self._chunk_id(chunk.paper_id, chunk.chunk_index))
            batch_docs.append(chunk.text)
            batch_meta.append(
                {
                    "paper_id": chunk.paper_id,
                    "title": title_by_paper.get(chunk.paper_id, ""),
                    "chunk_index": chunk.chunk_index,
                    "section": chunk.section,
                }
            )
            if len(batch_docs) >= self.config.batch_size:
                total += self._upsert_batch(batch_ids, batch_docs, batch_meta)
                batch_ids, batch_docs, batch_meta = [], [], []
        if batch_docs:
            total += self._upsert_batch(batch_ids, batch_docs, batch_meta)
        return total

    def query(self, query_text: str, top_k: int = 8) -> list[SearchHit]:
        query_embedding = self.embedder.encode([query_text], normalize_embeddings=True)[0].tolist()
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[SearchHit] = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            paper_id = str(meta.get("paper_id", ""))
            hits.append(
                SearchHit(
                    paper_id=paper_id,
                    title=str(meta.get("title", "")),
                    text=str(doc),
                    score=max(0.0, 1.0 - float(distance)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                )
            )
        return hits

    def _upsert_batch(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str | int]],
    ) -> int:
        embeddings = self.embedder.encode(documents, normalize_embeddings=True).tolist()
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(documents)

    @staticmethod
    def _chunk_id(paper_id: str, chunk_index: int) -> str:
        safe_id = paper_id.replace("/", "_")
        return f"{safe_id}:{chunk_index}"
