# Vektorsuche & RAG – Lernlabor

Interaktives Lern-Dashboard für den Homeserver: zeigt Schritt für Schritt, wie Embedding-Modelle (Voyage AI), eine Vektordatenbank (Qdrant) und Retrieval-Augmented Generation (Claude) zusammenspielen.

## Schnellstart

```bash
cp .env.example .env        # VOYAGE_API_KEY eintragen, optional ANTHROPIC_API_KEY
docker compose up -d --build
```

Die Ports sind nur an `127.0.0.1` des Servers gebunden (weder Dashboard noch Qdrant haben eine Authentifizierung). Der Zugriff läuft per SSH-Tunnel vom eigenen PC:

```bash
ssh -L 8000:localhost:8000 -L 6333:localhost:6333 <user>@<server>
```

- Dashboard: `http://localhost:8000`
- Qdrant-Web-UI: `http://localhost:6333/dashboard`

In VS Code (Remote-SSH) genügt es, die Ports 8000 und 6333 im Panel „Ports“ weiterzuleiten.

Beim ersten Start bettet das Backend die Beispielsätze und die Wissensbasis in **einem** Voyage-Request ein und speichert sie in Qdrant (Volume `qdrant_data`). Danach wird nur noch eingebettet, was du eingibst; identische Texte werden im Speicher gecacht. Wechselst du `VOYAGE_MODEL`, legt das Backend die Collections automatisch neu an.

**Ohne Voyage-Key testen:** `EMBEDDING_BACKEND=hash` nutzt Feature-Hashing über Wörter – die Oberfläche funktioniert, die Vektoren haben aber keine Semantik. **Ohne Anthropic-Key** läuft Modul 4 bis zum fertig gebauten Prompt; nur die Generierung fehlt.

## Architektur

| Komponente | Umsetzung |
|---|---|
| Backend | Python 3.13, FastAPI (`backend/app`) |
| Embeddings | Voyage AI, Standard `voyage-3-lite` (512 Dimensionen) |
| Vektordatenbank | Qdrant v1.20, Distanz: Kosinus, zwei Collections (`saetze`, `wissen`) |
| LLM | Claude (`claude-opus-5-5`, Effort `low`) mit Streaming |
| Frontend | HTML/JS, TailwindCSS, Plotly.js (`backend/static`) |

```
Browser ──► FastAPI ──► Voyage AI   (Text → Vektor)
                 ├────► Qdrant      (Vektoren speichern / Nächste-Nachbarn-Suche)
                 └────► Claude      (Antwort aus den Treffern, per SSE gestreamt)
```

## Module

1. **Vektorisierung** – Text → Vektor; Kennzahlen (Dimensionen, L2-Norm, Min/Max), die ersten 10 Werte und ein Heatmap-Streifen über alle Dimensionen (blau negativ, rot positiv).
2. **Distanz-Labor** – Zwei Sätze vergleichen: Tacho für die Kosinus-Ähnlichkeit, euklidischer Abstand, Wortüberlappung (Jaccard) als Gegenprobe, Heatmap mit dem Produkt pro Dimension (dessen Summe ist der Kosinus). Beispiel-Chips für Synonyme, Mehrdeutigkeit („Bank“), vertauschte Rollen und sprachübergreifende Sätze.
3. **2D-Vektorraum** – Alle Sätze werden aus Qdrant gelesen und per PCA (SVD in NumPy) auf 2D projiziert. Eine Suchanfrage wandert animiert an ihre Position; die Top-3-Nachbarn aus der vollen Dimension werden verbunden, das nächste Cluster wird über die Kategorie-Zentroiden bestimmt. Der erklärte Varianzanteil zeigt, wie viel Information die Projektion verliert. Eigene Sätze lassen sich dauerhaft speichern.
4. **RAG-Inspektor** – Frage → Query-Embedding → Top-k aus Qdrant mit Match-Prozent, daneben eine Schlüsselwortsuche zum Vergleich → vollständiger Prompt → gestreamte Antwort mit Quellenangaben `[n]`. Die Wissensbasis enthält einen erfundenen Radsportverein („RSV Grünwald 1923“): Fakten, die kein LLM aus dem Training kennen kann. Fragen außerhalb der Wissensbasis (z. B. „Tour de France 2024“) soll das Modell ablehnen statt zu raten.

### Code-Ansicht für Entwickler

Jedes Modul hat unten ein aufklappbares Panel **„Code hinter diesem Modul“** mit drei Reitern:

- **Minimalbeispiel** – eigenständiges Python-Skript, das den Kern des Moduls ohne dieses Projekt nachbaut (`voyageai`, `numpy`, `qdrant-client`, `anthropic`).
- **Backend (dieses Projekt)** – die tatsächlich laufenden Funktionen mit Datei und Zeilennummer. Sie werden per `inspect` live aus dem Quelltext gelesen (`backend/app/code_view.py`) und veralten daher nie.
- **API-Aufruf** – der letzte Request des Frontends als `curl`-Befehl mit deiner Eingabe, dazu die Antwort des Backends (Vektoren gekürzt, bei RAG die Server-Sent Events).

## API

| Methode | Pfad | Zweck |
|---|---|---|
| GET | `/api/status` | Modelle, Anzahl Punkte, Seeding-Fehler |
| POST | `/api/embed` | `{text}` → Vektor + Kennzahlen |
| POST | `/api/compare` | `{a, b}` → Kosinus, Abstand, Wortüberlappung, beide Vektoren |
| GET | `/api/space` | PCA-Koordinaten aller Sätze |
| POST | `/api/space/query` | `{text}` → 2D-Position, Nachbarn, Cluster-Ranking |
| POST | `/api/space/add` | `{text, category}` speichern |
| GET / POST | `/api/knowledge`, `/api/knowledge/add` | Wissensbasis lesen / erweitern |
| DELETE | `/api/custom` | Alle eigenen Einträge löschen |
| POST | `/api/rag` | `{question, top_k}` → Server-Sent Events: `embedding`, `retrieval`, `delta`, `done`, `error` |
| GET | `/api/code/{m1…m4}` | Minimalbeispiel und Projektcode für die Code-Ansicht |
| POST | `/api/seed` | Befüllung erneut versuchen (z. B. nach nachgetragenem Key) |

## Hinweise

- **Rate-Limits:** Voyage-Konten ohne hinterlegte Zahlungsmethode sind stark gedrosselt (wenige Requests pro Minute). Das Dashboard bündelt Requests und cacht, bei schnellem Klicken kann trotzdem ein 429 kommen – die Meldung erscheint im UI.
- **Absolute Ähnlichkeitswerte** sind modellabhängig; auch unverwandte Sätze liegen oft deutlich über 0. Aussagekräftig ist die Rangfolge.
- **Lokale Entwicklung ohne Docker:**
  ```bash
  cd backend && pip install -r requirements.txt
  QDRANT_URL=:memory: VOYAGE_API_KEY=... uvicorn app.main:app --reload
  ```
