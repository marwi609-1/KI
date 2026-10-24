"""Trainingseinheiten in Qdrant: eine Collection mit zwei benannten Vektoren (Named Vectors)."""

import hashlib
import uuid
from dataclasses import asdict

from qdrant_client import models

from .demo_workouts import DEMO_ATHLETE, demo_activities
from .embeddings import embedder
from .fit_parser import Activity, AthleteSettings, compute_metrics, downsample_series
from .vectorizer import generate_metric_vector, generate_summary_text
from .vectorstore import client

COLLECTION = "workouts"
SEMANTIC = "dense_semantic"
METRICS = "dense_metrics"
NAMESPACE = uuid.UUID("6f1c2a52-3d0e-4b7c-9a51-0c6a6d1b2f10")


async def ensure_collection(dims: int) -> bool:
    """Legt die Collection an. Gibt True zurück, wenn sie neu befüllt werden muss."""
    if await client.collection_exists(COLLECTION):
        info = await client.get_collection(COLLECTION)
        vectors = info.config.params.vectors
        points, _ = await client.scroll(COLLECTION, limit=1, with_payload=["model"])
        same_model = bool(points) and points[0].payload.get("model") == embedder.model
        if isinstance(vectors, dict) and SEMANTIC in vectors and vectors[SEMANTIC].size == dims and same_model:
            return False
        await client.delete_collection(COLLECTION)
    await client.create_collection(
        COLLECTION,
        vectors_config={
            SEMANTIC: models.VectorParams(size=dims, distance=models.Distance.COSINE),
            METRICS: models.VectorParams(size=12, distance=models.Distance.EUCLID),
        },
    )
    return True


async def store_activity(
    act: Activity, settings: AthleteSettings, notes: str, title: str, point_id: str, source: str
) -> dict:
    metrics = compute_metrics(act, settings)
    summary = generate_summary_text(metrics, act.sport, act.sub_sport, act.start.strftime("%d.%m.%Y"), notes)
    metric_vector = generate_metric_vector(metrics, act.sport, settings)
    semantic_vector = await embedder.embed_one(summary, input_type="document")
    payload = {
        "title": title,
        "timestamp": act.start.isoformat(),
        "sport": act.sport,
        "sub_sport": act.sub_sport,
        "metrics": metrics,
        "summary_text": summary,
        "athlete_notes": notes,
        "settings": asdict(settings),
        "series": downsample_series(act),
        "source": source,
        "model": embedder.model,
    }
    await client.upsert(
        COLLECTION,
        points=[models.PointStruct(id=point_id, vector={SEMANTIC: semantic_vector, METRICS: metric_vector}, payload=payload)],
    )
    return {"id": point_id, **payload, "vectors": {SEMANTIC: semantic_vector, METRICS: metric_vector}}


def file_id(data: bytes) -> str:
    """Gleiche Datei -> gleiche ID, ein erneuter Upload überschreibt statt zu duplizieren."""
    return str(uuid.uuid5(NAMESPACE, hashlib.sha256(data).hexdigest()))


async def seed(dims: int) -> None:
    if await ensure_collection(dims):
        for name, act, notes in demo_activities():
            await store_activity(act, DEMO_ATHLETE, notes, name, str(uuid.uuid5(NAMESPACE, name)), "demo")


async def list_workouts() -> list[dict]:
    records, offset = [], None
    while True:
        batch, offset = await client.scroll(
            COLLECTION, limit=256, offset=offset, with_payload=["title", "timestamp", "sport", "sub_sport", "metrics", "source"]
        )
        records.extend(batch)
        if offset is None:
            break
    items = [{"id": str(r.id), **r.payload} for r in records]
    for item in items:
        item["metrics"] = {k: v for k, v in item["metrics"].items() if k not in ("intervals", "zones")}
    return sorted(items, key=lambda w: w["timestamp"], reverse=True)


async def get_workout(point_id: str) -> dict | None:
    records = await client.retrieve(COLLECTION, ids=[point_id], with_payload=True, with_vectors=True)
    if not records:
        return None
    r = records[0]
    return {"id": str(r.id), **r.payload, "vectors": r.vector}


async def search(vector: list[float], using: str, limit: int, exclude: str | None = None) -> list[dict]:
    query_filter = models.Filter(must_not=[models.HasIdCondition(has_id=[exclude])]) if exclude else None
    result = await client.query_points(
        COLLECTION,
        query=vector,
        using=using,
        limit=limit,
        query_filter=query_filter,
        with_payload=["title", "timestamp", "sport", "metrics", "summary_text"],
    )
    return [
        {
            "id": str(p.id),
            "score": p.score,
            "title": p.payload["title"],
            "timestamp": p.payload["timestamp"],
            "sport": p.payload["sport"],
            "tss": p.payload["metrics"].get("tss"),
            "summary_text": p.payload["summary_text"],
        }
        for p in result.points
    ]


async def delete(point_id: str) -> None:
    await client.delete(COLLECTION, points_selector=models.PointIdsList(points=[point_id]))
