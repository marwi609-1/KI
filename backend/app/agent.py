"""Modul 6: Agentic RAG mit LangGraph.

LangGraph steuert Zustand, Verzweigungen und die Schleife; die LLM-Aufrufe im Knoten `agent`
laufen direkt über das Anthropic-SDK (Tool Use). Ohne ANTHROPIC_API_KEY plant ein einfacher
regelbasierter Planer die Werkzeugaufrufe, damit der Graph trotzdem durchlaufen werden kann.
"""

import json
import operator
import re
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from qdrant_client import models

from . import rag, vectorstore as vs, workouts
from .config import settings
from .embeddings import embedder

MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = (
    "Du bist ein Recherche-Agent in einem Lern-Dashboard. Dir stehen zwei Datenquellen zur Verfügung: "
    "eine Wissensbasis (Fachwissen zu Embeddings/RAG und Infos zum Radsportverein RSV Grünwald) und die "
    "Trainingseinheiten des Athleten in Qdrant. Nutze die Werkzeuge, bis du die Frage sicher beantworten kannst; "
    "kombiniere sie bei Bedarf in mehreren Schritten (z. B. erst eine Einheit finden, dann ähnliche Einheiten "
    "über deren ID suchen). Jedes Suchergebnis trägt eine Quellennummer [n] – belege Aussagen damit. "
    "Erfinde nichts, was nicht in den Ergebnissen steht. Antworte knapp auf Deutsch."
)

TOOLS = [
    {
        "name": "suche_wissensbasis",
        "description": "Semantische Suche in der Wissensbasis: Fachwissen zu Embeddings, Vektordatenbanken, RAG "
        "sowie Fakten zum Radsportverein RSV Grünwald (Ausfahrten, Beiträge, Regeln, Rekorde).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Suchanfrage in natürlicher Sprache"},
                "limit": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "name": "suche_trainings",
        "description": "Semantische Suche über die Textzusammenfassungen der Trainingseinheiten (inkl. Notizen). "
        "Gut für qualitative Fragen wie 'müde', 'Hitze', 'Intervalle'. Liefert IDs für weitere Werkzeuge.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "name": "filtere_trainings",
        "description": "Strukturierte Abfrage über die Payload (kein Vektor): Einheiten nach Sportart und TSS-Bereich, "
        "sortiert nach TSS absteigend. Gut für 'härteste', 'längste', 'alle Läufe über TSS 100'.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "sportart": {"type": "string", "enum": ["alle", "cycling", "running", "swimming"]},
                "min_tss": {"type": "number"},
                "limit": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            },
            "required": ["sportart", "min_tss", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "name": "aehnliche_trainings",
        "description": "Findet Einheiten mit ähnlicher Belastungsstruktur über den 12D-Kennzahlenvektor "
        "(dense_metrics, euklidischer Abstand). Braucht die ID einer Einheit aus einem vorherigen Ergebnis.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "training_id": {"type": "string"},
                "limit": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            },
            "required": ["training_id", "limit"],
            "additionalProperties": False,
        },
    },
]


class AgentState(TypedDict):
    question: str
    messages: list  # Verlauf im Format der Messages API (inkl. tool_use/tool_result)
    rounds: int
    sources: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]
    answer: str


# ---------- Werkzeuge ----------


def _workout_line(n: int, w: dict, score_label: str = "") -> str:
    m = w.get("metrics") or {}
    tss = w.get("tss", m.get("tss"))
    return f"[{n}] {w['title']} (ID {w['id']}, {w['timestamp'][:10]}, {w['sport']}, TSS {tss}{score_label})\n{w.get('summary_text', '')}"


async def run_tool(name: str, args: dict, next_source: int) -> tuple[str, list[dict]]:
    """Führt ein Werkzeug aus. Gibt den Text für das LLM und die neuen Quellen zurück."""
    limit = int(args.get("limit", 3))
    sources: list[dict] = []
    if name == "suche_wissensbasis":
        vec = await embedder.embed_one(args["query"], input_type="query")
        for h in await vs.search(vs.KNOWLEDGE_COLLECTION, vec, limit):
            sources.append({"kind": "Wissen", "title": h.payload["title"], "text": h.payload["text"], "score": h.score})
    elif name == "suche_trainings":
        vec = await embedder.embed_one(args["query"], input_type="query")
        for w in await workouts.search(vec, workouts.SEMANTIC, limit):
            sources.append({"kind": "Training", **w, "text": w["summary_text"]})
    elif name == "filtere_trainings":
        must = [models.FieldCondition(key="metrics.tss", range=models.Range(gte=float(args.get("min_tss", 0))))]
        if args.get("sportart", "alle") != "alle":
            must.append(models.FieldCondition(key="sport", match=models.MatchValue(value=args["sportart"])))
        records, _ = await vs.client.scroll(
            workouts.COLLECTION, scroll_filter=models.Filter(must=must), limit=256,
            with_payload=["title", "timestamp", "sport", "metrics", "summary_text"],
        )
        ranked = sorted(records, key=lambda r: r.payload["metrics"].get("tss") or 0, reverse=True)[:limit]
        for r in ranked:
            sources.append({"kind": "Training", "id": str(r.id), "score": None, **r.payload, "text": r.payload["summary_text"]})
    elif name == "aehnliche_trainings":
        ref = await workouts.get_workout(args["training_id"])
        if ref is None:
            return f"Fehler: Keine Einheit mit ID {args['training_id']}.", []
        hits = await workouts.search(ref["vectors"][workouts.METRICS], workouts.METRICS, limit, exclude=args["training_id"])
        for w in hits:
            sources.append({"kind": "Training", **w, "text": w["summary_text"]})
    else:
        return f"Fehler: Unbekanntes Werkzeug {name}.", []

    if not sources:
        return "Keine Treffer.", []
    lines = []
    for i, s in enumerate(sources, start=next_source):
        s["n"] = i
        if s["kind"] == "Wissen":
            lines.append(f"[{i}] {s['title']} (Kosinus {s['score']:.3f})\n{s['text']}")
        else:
            label = "" if s.get("score") is None else (
                f", Abstand {s['score']:.3f}" if name == "aehnliche_trainings" else f", Kosinus {s['score']:.3f}"
            )
            lines.append(_workout_line(i, s, label))
    return "\n\n".join(lines), sources


# ---------- Planer ohne LLM (Fallback) ----------

TRAINING_WORDS = re.compile(
    r"\b(training\w*|einheit\w*|intervall\w*|puls|herzfrequenz|tss|lauf\w*|läuf\w*|radfahr\w*|ausfahrt\w*|"
    r"schwimm\w*|müde|härteste\w*|belastung|drift)\b",
    re.I,
)


def heuristic_plan(state: AgentState) -> list[dict]:
    """Einfache Regeln statt LLM – nur damit der Graph ohne API-Key durchlaufen werden kann."""
    q, done = state["question"], {t["tool"] for t in state["trace"] if t["node"] == "tools"}
    if state["rounds"] == 0:
        calls = []
        if re.search(r"härteste|stärkste|höchste", q, re.I):
            calls.append({"name": "filtere_trainings", "input": {"sportart": "alle", "min_tss": 0, "limit": 3}})
        elif TRAINING_WORDS.search(q):
            calls.append({"name": "suche_trainings", "input": {"query": q, "limit": 3}})
        if not TRAINING_WORDS.search(q) or re.search(r"verein|rsv|embedding|vektor|rag|qdrant", q, re.I):
            calls.append({"name": "suche_wissensbasis", "input": {"query": q, "limit": 3}})
        return calls
    if re.search(r"ähnlich", q, re.I) and "aehnliche_trainings" not in done:
        first = next((s for s in state["sources"] if s["kind"] == "Training"), None)
        if first:
            return [{"name": "aehnliche_trainings", "input": {"training_id": first["id"], "limit": 3}}]
    return []


# ---------- Knoten ----------


async def agent_node(state: AgentState) -> dict:
    """Entscheidet: weitere Werkzeuge aufrufen oder antworten."""
    write = get_stream_writer()
    if not rag.llm_available():
        calls = heuristic_plan(state)
        if calls:
            content = [{"type": "tool_use", "id": f"regel_{state['rounds']}_{i}", **c} for i, c in enumerate(calls)]
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": content}],
                "trace": [{"node": "agent", "decision": "tools", "calls": calls, "planner": "Regeln"}],
            }
        found = "\n".join(f"[{s['n']}] {s['title']}" for s in state["sources"]) or "keine"
        answer = f"(Ohne ANTHROPIC_API_KEY keine Antwortgenerierung – gefundene Quellen:)\n{found}"
        return {"answer": answer, "trace": [{"node": "agent", "decision": "antworten", "planner": "Regeln"}]}

    async with rag._client.beta.messages.stream(
        model=settings.claude_model,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        output_config={"effort": settings.claude_effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=state["messages"],
    ) as stream:
        async for text in stream.text_stream:
            write({"delta": text})
        final = await stream.get_final_message()

    # Antwort komplett (inkl. Thinking-Blöcken) unverändert in den Verlauf übernehmen
    messages = state["messages"] + [{"role": "assistant", "content": final.content}]
    text = "".join(b.text for b in final.content if b.type == "text")
    calls = [{"name": b.name, "input": b.input} for b in final.content if b.type == "tool_use"]
    usage = {"input_tokens": final.usage.input_tokens, "output_tokens": final.usage.output_tokens}
    if final.stop_reason == "tool_use" and calls:
        return {"messages": messages, "trace": [{"node": "agent", "decision": "tools", "calls": calls, "text": text, "planner": final.model, "usage": usage}]}
    if final.stop_reason == "refusal":
        text = "Das Modell hat die Anfrage abgelehnt."
    return {"messages": messages, "answer": text, "trace": [{"node": "agent", "decision": "antworten", "planner": final.model, "usage": usage}]}


async def tools_node(state: AgentState) -> dict:
    """Führt alle Werkzeugaufrufe der letzten Agent-Antwort aus."""
    last = state["messages"][-1]["content"]
    calls = [b for b in last if (b["type"] if isinstance(b, dict) else b.type) == "tool_use"]
    results, new_sources, trace = [], [], []
    next_n = len(state["sources"]) + 1
    for c in calls:
        cid, name, args = (c["id"], c["name"], c["input"]) if isinstance(c, dict) else (c.id, c.name, c.input)
        text, sources = await run_tool(name, args, next_n)
        next_n += len(sources)
        new_sources += sources
        results.append({"type": "tool_result", "tool_use_id": cid, "content": text, "is_error": text.startswith("Fehler")})
        trace.append({"node": "tools", "tool": name, "input": args, "hits": [{"n": s["n"], "title": s["title"], "kind": s["kind"]} for s in sources]})
    return {
        "messages": state["messages"] + [{"role": "user", "content": results}],
        "rounds": state["rounds"] + 1,
        "sources": new_sources,
        "trace": trace,
    }


async def final_node(state: AgentState) -> dict:
    """Notbremse: Nach MAX_TOOL_ROUNDS muss der Agent mit dem antworten, was er hat."""
    if not rag.llm_available():
        found = "\n".join(f"[{s['n']}] {s['title']}" for s in state["sources"])
        return {"answer": f"(Schleifenlimit erreicht, ohne LLM keine Antwort – Quellen:)\n{found}", "trace": [{"node": "final"}]}
    write = get_stream_writer()
    messages = state["messages"] + [
        {"role": "user", "content": "Das Werkzeuglimit ist erreicht. Antworte jetzt mit den vorhandenen Ergebnissen."}
    ]
    async with rag._client.beta.messages.stream(
        model=settings.claude_model,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        tool_choice={"type": "none"},
        output_config={"effort": settings.claude_effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            write({"delta": text})
        final = await stream.get_final_message()
    text = "".join(b.text for b in final.content if b.type == "text")
    return {"answer": text, "trace": [{"node": "final", "planner": final.model}]}


def route_after_agent(state: AgentState) -> str:
    return "werkzeuge" if state["trace"][-1].get("decision") == "tools" else "antworten"


def route_after_tools(state: AgentState) -> str:
    return "limit" if state["rounds"] >= MAX_TOOL_ROUNDS else "weiter"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("final", final_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"werkzeuge": "tools", "antworten": END})
    builder.add_conditional_edges("tools", route_after_tools, {"weiter": "agent", "limit": "final"})
    builder.add_edge("final", END)
    return builder.compile()


graph = build_graph()


def initial_state(question: str) -> AgentState:
    return {"question": question, "messages": [{"role": "user", "content": question}], "rounds": 0, "sources": [], "trace": [], "answer": ""}


def jsonable(value: Any) -> Any:
    """Trace-Einträge für SSE serialisierbar machen."""
    return json.loads(json.dumps(value, default=str))
