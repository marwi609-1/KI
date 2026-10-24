"""Embedding-Service: Voyage AI mit In-Memory-Cache (spart API-Aufrufe und Rate-Limit)."""

import hashlib
import re

import numpy as np

from .config import settings


class Embedder:
    def __init__(self) -> None:
        self.backend = settings.embedding_backend
        self.model = settings.voyage_model if self.backend == "voyage" else "hash-demo-256"
        self._cache: dict[tuple[str | None, str], list[float]] = {}
        if self.backend == "voyage":
            import voyageai

            self._client = voyageai.AsyncClient(api_key=settings.voyage_api_key)

    async def embed(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        """input_type: "query" für Suchanfragen, "document" für Wissensbasis, None für symmetrische Vergleiche."""
        missing = [t for t in dict.fromkeys(texts) if (input_type, t) not in self._cache]
        if missing:
            if self.backend == "voyage":
                # Voyage erlaubt bis zu 128 Texte pro Request
                for i in range(0, len(missing), 128):
                    batch = missing[i : i + 128]
                    result = await self._client.embed(batch, model=self.model, input_type=input_type)
                    for text, vec in zip(batch, result.embeddings):
                        self._cache[(input_type, text)] = vec
            else:
                for text in missing:
                    self._cache[(input_type, text)] = _hash_embedding(text)
        return [self._cache[(input_type, t)] for t in texts]

    async def embed_one(self, text: str, input_type: str | None = None) -> list[float]:
        return (await self.embed([text], input_type))[0]


def _hash_embedding(text: str, dims: int = 256) -> list[float]:
    """Feature-Hashing über Wörter und Zeichen-Trigramme – erfasst nur Oberflächenähnlichkeit."""
    vec = np.zeros(dims)
    tokens = re.findall(r"\w+", text.lower())
    grams = tokens + [f"#{t[i:i + 3]}" for t in tokens for i in range(max(1, len(t) - 2))]
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        vec[h % dims] += 1 if (h >> 64) % 2 else -1
    norm = np.linalg.norm(vec)
    return (vec / norm if norm else vec).tolist()


embedder = Embedder()
