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
