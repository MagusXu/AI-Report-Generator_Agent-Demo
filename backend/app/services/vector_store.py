from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from app.config import get_settings


COLLECTION_NAME = "report_references"


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    def upsert_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count and embedding count do not match")

        self.collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk["metadata"] for chunk in chunks],
        )

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self.collection.delete(ids=chunk_ids)

    def query(self, query_embedding: list[float], document_ids: list[str], top_k: int = 5) -> list[dict[str, Any]]:
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        rows: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            rows.append(
                {
                    "id": chunk_id,
                    "text": documents[index],
                    "metadata": metadatas[index],
                    "distance": distances[index],
                }
            )
        return rows
