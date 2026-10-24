"""Code-Ansicht für Entwickler: Minimalbeispiele und der echte Projektcode je Lernmodul.

Der Projektcode wird zur Laufzeit per `inspect` aus den Quelldateien gelesen und ist damit immer aktuell.
"""

import importlib
import inspect
import textwrap
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

# (Modul, Attributpfad, Erklärung)
SOURCES: dict[str, list[tuple[str, str, str]]] = {
    "m1": [
        ("app.main", "embed", "Endpunkt: nimmt den Text entgegen und gibt Vektor und Kennzahlen zurück."),
        ("app.embeddings", "Embedder.embed", "Ruft Voyage AI im Batch auf und cacht jedes Ergebnis pro (input_type, Text)."),
        ("app.main", "vector_stats", "Kennzahlen: Dimensionen, L2-Norm, Minimum, Maximum."),
    ],
    "m2": [
        ("app.main", "compare", "Beide Sätze in einem Request einbetten, dann Ähnlichkeit und Wortüberlappung berechnen."),
        ("app.main", "cosine", "Kosinus-Ähnlichkeit = Skalarprodukt geteilt durch das Produkt der Längen."),
        ("app.main", "tokens", "Einfache Tokenisierung für den Vergleich mit der Schlüsselwort-Logik."),
    ],
    "m3": [
        ("app.main", "fit_pca", "PCA per Singulärwertzerlegung: Daten zentrieren, die zwei stärksten Achsen behalten."),
        ("app.main", "pca_space", "Alle Vektoren aus Qdrant lesen und auf die zwei Hauptachsen projizieren."),
        ("app.main", "space_query", "Suchanfrage projizieren, Nachbarn in Qdrant (volle Dimension) suchen, Cluster-Zentroiden vergleichen."),
        ("app.vectorstore", "all_points", "Seitenweises Auslesen einer Collection inklusive Vektoren (scroll)."),
        ("app.vectorstore", "upsert", "Texte einbetten und mit Payload als Punkte in Qdrant schreiben."),
    ],
    "m4": [
        ("app.main", "rag_pipeline", "Die komplette Pipeline als Server-Sent-Events-Stream: Embedding, Retrieval, Generierung."),
        ("app.vectorstore", "search", "Nächste-Nachbarn-Suche in Qdrant (HNSW-Index, Kosinus-Distanz)."),
        ("app.rag", "build_user_prompt", "Die gefundenen Dokumente als nummerierte Quellen in den Prompt einbauen."),
        ("app.rag", "stream_answer", "Claude streamt die Antwort; Stop-Grund wird am Ende ausgewertet."),
        ("app.main", "keyword_search", "Zum Vergleich: naive Schlüsselwortsuche über Wortüberlappung."),
    ],
    "m5": [
        ("app.main", "upload_fit", "Upload-Endpunkt: FIT-Datei lesen, Schwellenwerte übernehmen, Einheit speichern."),
        ("app.fit_parser", "parse_fit", "fitdecode liest record- und session-Nachrichten; alle Kanäle werden auf 1 Hz gebracht."),
        ("app.fit_parser", "normalized_power", "Normalized Power: 30-s-Mittel, 4. Potenz, Mittelwert, 4. Wurzel."),
        ("app.fit_parser", "aerobic_decoupling", "Aerobe Entkopplung (Pa:HR): Effizienzfaktor erste vs. zweite Hälfte."),
        ("app.fit_parser", "compute_metrics", "Alle Kennzahlen: NP, IF, TSS (Leistung, Pace oder HF), Zonen, Höhenmeter, Intervalle."),
        ("app.vectorizer", "generate_metric_vector", "Der 12D-Kennzahlenvektor – jede Dimension auf 0–1 skaliert."),
        ("app.vectorizer", "generate_summary_text", "Textzusammenfassung als Grundlage für das semantische Embedding."),
        ("app.workouts", "ensure_collection", "Qdrant-Collection mit zwei benannten Vektoren: Kosinus für Text, Euklid für Kennzahlen."),
        ("app.workouts", "store_activity", "Beide Vektoren und die Payload in einem Punkt speichern."),
        ("app.workouts", "search", "Suche über einen der beiden Vektoren (Parameter using)."),
    ],
    "m6": [
        ("app.agent", "build_graph", "Der Zustandsgraph: Knoten agent, tools, final und die bedingten Kanten dazwischen."),
        ("app.agent", "AgentState", "Gemeinsamer Zustand; sources und trace werden per Reducer (operator.add) über alle Schritte gesammelt."),
        ("app.agent", "agent_node", "Claude mit Werkzeugen aufrufen; die Antwort samt Thinking-Blöcken unverändert in den Verlauf übernehmen."),
        ("app.agent", "tools_node", "Alle Werkzeugaufrufe ausführen und als tool_result in einer Nachricht zurückgeben."),
        ("app.agent", "run_tool", "Die vier Werkzeuge: zwei semantische Suchen, ein Payload-Filter, eine Kennzahlen-Ähnlichkeitssuche."),
        ("app.agent", "route_after_agent", "Bedingte Kante: Werkzeuge oder Ende."),
        ("app.agent", "route_after_tools", "Bedingte Kante: zurück zum Agenten oder Notbremse nach MAX_TOOL_ROUNDS."),
        ("app.agent", "final_node", "Notbremse: Antwort mit tool_choice none erzwingen."),
        ("app.main", "agent_run", "Endpunkt: graph.astream mit stream_mode updates (Knoten) und custom (Text-Deltas) als SSE."),
    ],
}

MINIMAL: dict[str, str] = {
    "m1": '''# pip install voyageai
import voyageai

vo = voyageai.Client()  # liest VOYAGE_API_KEY aus der Umgebung

result = vo.embed(["Das Auto fährt schnell über die Autobahn."], model="voyage-3-lite")
vector = result.embeddings[0]

print(len(vector))   # 512 Dimensionen
print(vector[:10])   # die ersten 10 Werte
print(result.total_tokens)  # abgerechnete Token
''',
    "m2": '''# pip install voyageai numpy
import numpy as np
import voyageai

vo = voyageai.Client()
a, b = vo.embed(
    ["Das Auto fährt schnell", "Der Wagen ist zügig unterwegs"],
    model="voyage-3-lite",
).embeddings
a, b = np.array(a), np.array(b)

cosine = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Kosinus-Ähnlichkeit: {cosine:.3f}")

# Voyage-Vektoren sind auf Länge 1 normiert -> Skalarprodukt genügt
print(np.isclose(cosine, a @ b))
''',
    "m3": '''# pip install voyageai numpy
import numpy as np
import voyageai

sentences = {
    "Sport": ["Der Stürmer schoss das Siegtor.", "Sie gewann die Bergetappe."],
    "Kochen": ["Die Zwiebeln glasig anschwitzen.", "Der Teig braucht Hefe und Mehl."],
}
texts = [t for ts in sentences.values() for t in ts]

vo = voyageai.Client()
X = np.array(vo.embed(texts, model="voyage-3-lite", input_type="document").embeddings)

# PCA per SVD: zentrieren, zerlegen, auf die zwei stärksten Achsen projizieren
mean = X.mean(axis=0)
_, s, vt = np.linalg.svd(X - mean, full_matrices=False)
coords = (X - mean) @ vt[:2].T
print("erklärte Varianz:", (s[:2] ** 2 / np.sum(s ** 2)).round(3))

# Neue Anfrage mit denselben Achsen projizieren
q = np.array(vo.embed(["Wie koche ich Pasta?"], model="voyage-3-lite", input_type="query").embeddings[0])
print("Position der Anfrage:", (q - mean) @ vt[:2].T)
''',
    "m4": '''# pip install voyageai qdrant-client anthropic
import anthropic
import voyageai
from qdrant_client import QdrantClient, models

docs = [
    "Die Ausfahrten des RSV Grünwald starten dienstags und donnerstags um 18:00 Uhr am Marktplatz.",
    "Der Jahresbeitrag beträgt 84 € für Erwachsene und 150 € für Familien.",
]
vo = voyageai.Client()
qdrant = QdrantClient(":memory:")  # oder QdrantClient(url="http://localhost:6333")

# 1. Indexieren: Dokumente einbetten und speichern
vectors = vo.embed(docs, model="voyage-3-lite", input_type="document").embeddings
qdrant.create_collection("wissen", vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE))
qdrant.upsert("wissen", points=[models.PointStruct(id=i, vector=v, payload={"text": d}) for i, (v, d) in enumerate(zip(vectors, docs))])

# 2. Retrieval: Frage einbetten (input_type="query") und nächste Nachbarn holen
question = "Wann starten die Ausfahrten?"
qvec = vo.embed([question], model="voyage-3-lite", input_type="query").embeddings[0]
hits = qdrant.query_points("wissen", query=qvec, limit=2).points
context = "\\n".join(f"[{i}] {h.payload['text']}" for i, h in enumerate(hits, start=1))

# 3. Generierung: Claude antwortet nur auf Basis der Quellen
client = anthropic.Anthropic()  # liest ANTHROPIC_API_KEY
response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=1024,
    output_config={"effort": "low"},
    system="Antworte nur auf Basis der Quellen und zitiere sie mit [n].",
    messages=[{"role": "user", "content": f"Quellen:\\n{context}\\n\\nFrage: {question}"}],
)
if response.stop_reason == "refusal":
    print("Anfrage abgelehnt")
else:
    print(next(b.text for b in response.content if b.type == "text"))
''',
    "m5": '''# pip install fitdecode numpy voyageai qdrant-client
import fitdecode
import numpy as np
import voyageai
from qdrant_client import QdrantClient, models

FTP = 280

# 1. FIT-Datei lesen: Leistung und Herzfrequenz aus den record-Nachrichten
power, hr = [], []
with fitdecode.FitReader("training.fit") as fit:
    for frame in fit:
        if isinstance(frame, fitdecode.FitDataMessage) and frame.name == "record":
            if frame.has_field("power") and frame.get_value("power") is not None:
                power.append(frame.get_value("power"))
            if frame.has_field("heart_rate") and frame.get_value("heart_rate") is not None:
                hr.append(frame.get_value("heart_rate"))
power, hr = np.array(power, dtype=float), np.array(hr, dtype=float)

# 2. Kennzahlen (bei 1 Datenpunkt pro Sekunde)
rolling30 = np.convolve(power, np.ones(30) / 30, mode="valid")
np_ = np.mean(rolling30 ** 4) ** 0.25          # Normalized Power
if_ = np_ / FTP                                  # Intensity Factor
tss = len(power) / 3600 * if_ ** 2 * 100        # Training Stress Score
half = len(hr) // 2
ef1, ef2 = power[:half].mean() / hr[:half].mean(), power[half:].mean() / hr[half:].mean()
drift = (ef1 - ef2) / ef1 * 100                 # aerobe Entkopplung in %

# 3. Zwei Repräsentationen
metric_vector = [min(len(power) / 60 / 360, 1), min(tss / 400, 1), min(if_ / 1.5, 1)]  # gekürzt, im Projekt 12D
summary = f"Radfahren, {len(power) // 60} min, TSS {tss:.0f}, NP {np_:.0f} W, IF {if_:.2f}, HF-Drift {drift:+.1f} %"
semantic_vector = voyageai.Client().embed([summary], model="voyage-3-lite", input_type="document").embeddings[0]

# 4. Qdrant mit Named Vectors: ein Punkt, zwei Vektoren mit eigener Distanz
qdrant = QdrantClient(":memory:")
qdrant.create_collection("workouts", vectors_config={
    "dense_semantic": models.VectorParams(size=len(semantic_vector), distance=models.Distance.COSINE),
    "dense_metrics": models.VectorParams(size=len(metric_vector), distance=models.Distance.EUCLID),
})
qdrant.upsert("workouts", points=[models.PointStruct(
    id=1,
    vector={"dense_semantic": semantic_vector, "dense_metrics": metric_vector},
    payload={"summary": summary, "tss": round(tss, 1)},
)])

# 5. Suchen: mit using= wählt man den Vektor
hits = qdrant.query_points("workouts", query=metric_vector, using="dense_metrics", limit=5).points
print([(h.payload["summary"], h.score) for h in hits])
''',
    "m6": '''# pip install langgraph anthropic voyageai qdrant-client
import operator
from typing import Annotated, TypedDict

import anthropic
import voyageai
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient

client = anthropic.Anthropic()
vo = voyageai.Client()
qdrant = QdrantClient(url="http://localhost:6333")  # Collection "wissen" wie in Modul 4

TOOLS = [{
    "name": "suche_wissensbasis",
    "description": "Semantische Suche in der Wissensbasis.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}]


class State(TypedDict):
    messages: list
    rounds: int
    log: Annotated[list[str], operator.add]  # Reducer: Einträge werden angehängt statt ersetzt


def agent(state: State) -> dict:
    response = client.messages.create(
        model="claude-opus-5-5", max_tokens=8000, tools=TOOLS,
        output_config={"effort": "low"}, messages=state["messages"],
    )
    # Antwort komplett übernehmen - inklusive Thinking-Blöcken
    return {"messages": state["messages"] + [{"role": "assistant", "content": response.content}],
            "log": [f"agent: {response.stop_reason}"]}


def tools(state: State) -> dict:
    results = []
    for block in state["messages"][-1]["content"]:
        if block.type == "tool_use":
            vec = vo.embed([block.input["query"]], model="voyage-3-lite", input_type="query").embeddings[0]
            hits = qdrant.query_points("wissen", query=vec, limit=3).points
            text = "\\n".join(h.payload["text"] for h in hits)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": text})
    return {"messages": state["messages"] + [{"role": "user", "content": results}],
            "rounds": state["rounds"] + 1, "log": [f"tools: {len(results)} Aufruf(e)"]}


def route(state: State) -> str:
    last = state["messages"][-1]["content"]
    wants_tool = any(b.type == "tool_use" for b in last)
    return "tools" if wants_tool and state["rounds"] < 3 else END


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", tools)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route, ["tools", END])
builder.add_edge("tools", "agent")
graph = builder.compile()

print(graph.get_graph().draw_mermaid())  # Graph als Mermaid-Diagramm
result = graph.invoke({"messages": [{"role": "user", "content": "Was ist ein HNSW-Index?"}], "rounds": 0, "log": []})
print(result["log"])
print(next(b.text for b in result["messages"][-1]["content"] if b.type == "text"))
''',
}


def _resolve(module: str, path: str):
    obj = importlib.import_module(module)
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def module_code(module_id: str) -> dict | None:
    if module_id not in SOURCES:
        return None
    sources = []
    for module, path, explanation in SOURCES[module_id]:
        obj = _resolve(module, path)
        lines, start = inspect.getsourcelines(obj)
        file = Path(inspect.getsourcefile(obj)).resolve().relative_to(BACKEND_DIR)
        sources.append(
            {
                "name": path,
                "file": f"backend/{file.as_posix()}",
                "line": start,
                "explanation": explanation,
                "code": textwrap.dedent("".join(lines)).rstrip(),
            }
        )
    return {"minimal": MINIMAL[module_id], "sources": sources}
