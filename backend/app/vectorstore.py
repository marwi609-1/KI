"""Qdrant-Anbindung: zwei Collections – 'saetze' für den 2D-Raum, 'wissen' für RAG."""

import uuid

from qdrant_client import AsyncQdrantClient, models

from .config import settings
from .corpus import KNOWLEDGE, SENTENCES
from .embeddings import embedder

SENTENCES_COLLECTION = "saetze"
KNOWLEDGE_COLLECTION = "wissen"

if settings.qdrant_url == ":memory:":
    client = AsyncQdrantClient(location=":memory:")
else:
    client = AsyncQdrantClient(url=settings.qdrant_url)


async def _ensure_collection(name: str, dims: int) -> bool:
    """Legt die Collection an. Gibt True zurück, wenn sie neu befüllt werden muss."""
    if await client.collection_exists(name):
        info = await client.get_collection(name)
        points, _ = await client.scroll(name, limit=1, with_payload=True)
        same_model = bool(points) and points[0].payload.get("model") == embedder.model
        if info.config.params.vectors.size == dims and same_model:
            return False
        await client.delete_collection(name)  # Modell gewechselt -> Vektoren sind inkompatibel
    await client.create_collection(
        name, vectors_config=models.VectorParams(size=dims, distance=models.Distance.COSINE)
    )
    return True


async def upsert(collection: str, items: list[dict], input_type: str | None) -> None:
    vectors = await embedder.embed([i["text"] for i in items], input_type=input_type)
    await client.upsert(
        collection,
        points=[
            models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload={**item, "model": embedder.model})
            for item, vec in zip(items, vectors)
        ],
    )


async def seed() -> None:
    dims = len(await embedder.embed_one("Dimensionstest"))
    if await _ensure_collection(SENTENCES_COLLECTION, dims):
        items = [{"text": t, "category": c, "custom": False} for c, ts in SENTENCES.items() for t in ts]
        await upsert(SENTENCES_COLLECTION, items, input_type="document")
    if await _ensure_collection(KNOWLEDGE_COLLECTION, dims):
        await upsert(KNOWLEDGE_COLLECTION, [{**k, "custom": False} for k in KNOWLEDGE], input_type="document")


async def all_points(collection: str) -> list[models.Record]:
    records, offset = [], None
    while True:
        batch, offset = await client.scroll(collection, limit=256, offset=offset, with_vectors=True, with_payload=True)
        records.extend(batch)
        if offset is None:
            return records


async def search(collection: str, vector: list[float], limit: int) -> list[models.ScoredPoint]:
    result = await client.query_points(collection, query=vector, limit=limit, with_payload=True)
    return result.points


async def delete_custom(collection: str) -> None:
    await client.delete(
        collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="custom", match=models.MatchValue(value=True))])
        ),
    )
