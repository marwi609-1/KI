"""FastAPI-Backend des Lern-Dashboards für Vektorsuche & RAG."""

import json
import logging
import math
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import code_view, rag, vectorstore as vs, workouts
from .embeddings import embedder
from .fit_parser import AthleteSettings, parse_fit
from .vectorizer import METRIC_DIMENSIONS

log = logging.getLogger("lern-dashboard")
state: dict[str, str | None] = {"seed_error": None}


async def seed_all() -> None:
    await vs.seed()
    await workouts.seed(len(await embedder.embed_one("Dimensionstest")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await seed_all()
    except Exception as exc:  # App soll trotzdem starten und den Fehler im UI anzeigen
        log.exception("Seeding fehlgeschlagen")
        state["seed_error"] = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="Lern-Dashboard Vektorsuche & RAG", lifespan=lifespan)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MAX_FIT_BYTES = 25 * 1024 * 1024


@app.exception_handler(Exception)
async def api_error(request: Request, exc: Exception):
    status = 429 if "RateLimit" in type(exc).__name__ else 502
    return JSONResponse(status_code=status, content={"detail": f"{type(exc).__name__}: {exc}"})


# ---------- Hilfsfunktionen ----------

STOPWORDS = set(
    "der die das den dem des ein eine einer eines einem einen und oder aber im in ist sind war wird werden "
    "mit auf für von zu zum zur bei nach aus wie was wer wo wann sich es er sie wir ihr ich man nicht noch "
    "auch nur als am an so dass da hat haben kann können sollte the a of".split()
)


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS and len(t) > 1}


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def vector_stats(vec: list[float]) -> dict:
    v = np.asarray(vec)
    return {"dims": len(vec), "norm": float(np.linalg.norm(v)), "min": float(v.min()), "max": float(v.max())}


def fit_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gibt Mittelwert, die zwei Hauptachsen und deren erklärten Varianzanteil zurück."""
    mean = matrix.mean(axis=0)
    _, s, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    variance = s**2 / np.sum(s**2)
    return mean, vt[:2], variance[:2]


async def pca_space() -> dict:
    records = await vs.all_points(vs.SENTENCES_COLLECTION)
    matrix = np.asarray([r.vector for r in records])
    mean, axes, variance = fit_pca(matrix)
    coords = (matrix - mean) @ axes.T
    return {"records": records, "mean": mean, "axes": axes, "variance": variance, "coords": coords}


# ---------- API ----------


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class PairIn(BaseModel):
    a: str = Field(min_length=1, max_length=4000)
    b: str = Field(min_length=1, max_length=4000)


class SentenceIn(TextIn):
    category: str = Field(default="Eigene", max_length=40)


class DocIn(TextIn):
    title: str = Field(min_length=1, max_length=120)


class RagIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/api/status")
async def status():
    counts = {}
    for name in (vs.SENTENCES_COLLECTION, vs.KNOWLEDGE_COLLECTION, workouts.COLLECTION):
        counts[name] = (await vs.client.count(name)).count if await vs.client.collection_exists(name) else 0
    return {
        "embedding_backend": embedder.backend,
        "embedding_model": embedder.model,
        "llm_available": rag.llm_available(),
        "llm_model": rag.settings.claude_model,
        "collections": counts,
        "seed_error": state["seed_error"],
    }


@app.post("/api/seed")
async def reseed():
    await seed_all()
    state["seed_error"] = None
    return await status()


@app.post("/api/embed")
async def embed(body: TextIn):
    t0 = time.perf_counter()
    vec = await embedder.embed_one(body.text)
    return {"vector": vec, "ms": round((time.perf_counter() - t0) * 1000), **vector_stats(vec)}


@app.post("/api/compare")
async def compare(body: PairIn):
    va, vb = await embedder.embed([body.a, body.b])
    ta, tb = tokens(body.a), tokens(body.b)
    union = ta | tb
    return {
        "cosine": cosine(va, vb),
        "euclidean": float(np.linalg.norm(np.asarray(va) - np.asarray(vb))),
        "keyword_jaccard": len(ta & tb) / len(union) if union else 0.0,
        "shared_words": sorted(ta & tb),
        "vector_a": va,
        "vector_b": vb,
    }


@app.get("/api/space")
async def space():
    p = await pca_space()
    return {
        "variance": p["variance"].tolist(),
        "points": [
            {"text": r.payload["text"], "category": r.payload["category"], "x": float(x), "y": float(y)}
            for r, (x, y) in zip(p["records"], p["coords"])
        ],
    }


@app.post("/api/space/query")
async def space_query(body: TextIn):
    vec = await embedder.embed_one(body.text, input_type="query")
    p = await pca_space()
    x, y = (np.asarray(vec) - p["mean"]) @ p["axes"].T
    hits = await vs.search(vs.SENTENCES_COLLECTION, vec, limit=5)

    # Nächstes Cluster: höchste Kosinus-Ähnlichkeit zum Mittelpunkt (Zentroid) jeder Kategorie im Originalraum
    by_cat: dict[str, list] = {}
    for r in p["records"]:
        by_cat.setdefault(r.payload["category"], []).append(r.vector)
    clusters = sorted(
        ({"category": c, "similarity": cosine(vec, np.mean(vs_, axis=0).tolist())} for c, vs_ in by_cat.items()),
        key=lambda c: c["similarity"],
        reverse=True,
    )
    return {
        "x": float(x),
        "y": float(y),
        "neighbors": [{"text": h.payload["text"], "category": h.payload["category"], "score": h.score} for h in hits],
        "clusters": clusters,
    }


@app.post("/api/space/add")
async def space_add(body: SentenceIn):
    await vs.upsert(
        vs.SENTENCES_COLLECTION, [{"text": body.text, "category": body.category or "Eigene", "custom": True}], "document"
    )
    return {"ok": True}


@app.get("/api/knowledge")
async def knowledge():
    records = await vs.all_points(vs.KNOWLEDGE_COLLECTION)
    return [{"title": r.payload["title"], "text": r.payload["text"], "custom": r.payload.get("custom")} for r in records]


@app.post("/api/knowledge/add")
async def knowledge_add(body: DocIn):
    await vs.upsert(vs.KNOWLEDGE_COLLECTION, [{"title": body.title, "text": body.text, "custom": True}], "document")
    return {"ok": True}


@app.delete("/api/custom")
async def delete_custom():
    for name in (vs.SENTENCES_COLLECTION, vs.KNOWLEDGE_COLLECTION):
        await vs.delete_custom(name)
    return {"ok": True}


def keyword_search(question: str, docs: list[dict], limit: int) -> list[dict]:
    q = tokens(question)
    scored = []
    for d in docs:
        overlap = q & tokens(d["title"] + " " + d["text"])
        if overlap:
            scored.append({**d, "score": len(overlap) / math.sqrt(len(q) or 1), "matched": sorted(overlap)})
    return sorted(scored, key=lambda d: d["score"], reverse=True)[:limit]


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/rag")
async def rag_pipeline(body: RagIn):
    async def events():
        try:
            # Schritt 1: Frage vektorisieren
            t0 = time.perf_counter()
            qvec = await embedder.embed_one(body.question, input_type="query")
            t1 = time.perf_counter()
            yield sse("embedding", {"vector": qvec, "ms": round((t1 - t0) * 1000), **vector_stats(qvec)})

            # Schritt 2: Retrieval aus Qdrant (+ Schlüsselwortsuche zum Vergleich)
            hits = await vs.search(vs.KNOWLEDGE_COLLECTION, qvec, limit=body.top_k)
            t2 = time.perf_counter()
            sources = [{"title": h.payload["title"], "text": h.payload["text"], "score": h.score} for h in hits]
            all_docs = [{"title": r.payload["title"], "text": r.payload["text"]} for r in await vs.all_points(vs.KNOWLEDGE_COLLECTION)]
            yield sse(
                "retrieval",
                {
                    "sources": sources,
                    "keyword": keyword_search(body.question, all_docs, body.top_k),
                    "ms": round((t2 - t1) * 1000),
                    "prompt": {"system": rag.SYSTEM_PROMPT, "user": rag.build_user_prompt(body.question, sources)},
                },
            )

            # Schritt 3: Generierung
            if not rag.llm_available():
                yield sse("done", {"note": "Kein ANTHROPIC_API_KEY gesetzt – der Prompt oben würde an das LLM gehen."})
                return
            t3 = time.perf_counter()
            async for kind, payload in rag.stream_answer(body.question, sources):
                if kind == "delta":
                    yield sse("delta", {"text": payload})
                else:
                    yield sse("done", {"note": payload, "ms": round((time.perf_counter() - t3) * 1000)})
        except Exception as exc:
            log.exception("RAG-Pipeline fehlgeschlagen")
            yield sse("error", {"detail": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(events(), media_type="text/event-stream")


# ---------- Modul 5: Trainingsdaten (FIT) ----------


class WorkoutSearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)


class CoachIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)


@app.get("/api/workouts")
async def workout_list():
    return await workouts.list_workouts()


@app.get("/api/workouts/meta")
async def workout_meta():
    return {"dimensions": METRIC_DIMENSIONS, "defaults": AthleteSettings().__dict__}


@app.post("/api/workouts/upload-fit")
async def upload_fit(
    file: UploadFile = File(...),
    notes: str = Form(default="", max_length=2000),
    ftp: float = Form(default=250, gt=0),
    hr_max: float = Form(default=190, gt=0),
    lthr: float = Form(default=170, gt=0),
    threshold_pace: float = Form(default=300, gt=0),
):
    data = await file.read(MAX_FIT_BYTES + 1)
    if len(data) > MAX_FIT_BYTES:
        raise HTTPException(status_code=413, detail="Datei größer als 25 MB")
    try:
        act = parse_fit(data)
    except Exception as exc:  # fitdecode wirft je nach Defekt verschiedene Fehlertypen
        raise HTTPException(status_code=400, detail=f"FIT-Datei nicht lesbar: {exc}") from exc
    settings = AthleteSettings(ftp=ftp, hr_max=hr_max, lthr=lthr, threshold_pace=threshold_pace)
    title = file.filename or "Upload"
    return await workouts.store_activity(act, settings, notes, title, workouts.file_id(data), "upload")


@app.post("/api/workouts/search")
async def workout_search(body: WorkoutSearchIn):
    vec = await embedder.embed_one(body.query, input_type="query")
    return await workouts.search(vec, workouts.SEMANTIC, body.limit)


@app.get("/api/workouts/{workout_id}")
async def workout_detail(workout_id: str):
    workout = await workouts.get_workout(workout_id)
    if workout is None:
        raise HTTPException(status_code=404, detail="Training nicht gefunden")
    return workout


@app.get("/api/workouts/{workout_id}/similar")
async def workout_similar(workout_id: str, limit: int = 5):
    workout = await workouts.get_workout(workout_id)
    if workout is None:
        raise HTTPException(status_code=404, detail="Training nicht gefunden")
    v = workout["vectors"]
    return {
        "metrics": await workouts.search(v[workouts.METRICS], workouts.METRICS, limit, exclude=workout_id),
        "semantic": await workouts.search(v[workouts.SEMANTIC], workouts.SEMANTIC, limit, exclude=workout_id),
    }


@app.delete("/api/workouts/{workout_id}")
async def workout_delete(workout_id: str):
    await workouts.delete(workout_id)
    return {"ok": True}


@app.post("/api/workouts/coach")
async def workout_coach(body: CoachIn):
    async def events():
        try:
            t0 = time.perf_counter()
            qvec = await embedder.embed_one(body.question, input_type="query")
            hits = await workouts.search(qvec, workouts.SEMANTIC, body.top_k)
            sources = [{"title": h["title"], "text": h["summary_text"], "score": h["score"]} for h in hits]
            yield sse(
                "retrieval",
                {
                    "sources": sources,
                    "ms": round((time.perf_counter() - t0) * 1000),
                    "prompt": {"system": rag.COACH_PROMPT, "user": rag.build_user_prompt(body.question, sources)},
                },
            )
            if not rag.llm_available():
                yield sse("done", {"note": "Kein ANTHROPIC_API_KEY gesetzt – der Prompt oben würde an das LLM gehen."})
                return
            t1 = time.perf_counter()
            async for kind, payload in rag.stream_answer(body.question, sources, system=rag.COACH_PROMPT):
                if kind == "delta":
                    yield sse("delta", {"text": payload})
                else:
                    yield sse("done", {"note": payload, "ms": round((time.perf_counter() - t1) * 1000)})
        except Exception as exc:
            log.exception("Coach-Anfrage fehlgeschlagen")
            yield sse("error", {"detail": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/code/{module_id}")
async def code(module_id: str):
    result = code_view.module_code(module_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Unbekanntes Modul")
    return result


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
