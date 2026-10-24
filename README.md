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
| Vektordatenbank | Qdrant v1.20, Collections `saetze`, `wissen` (Kosinus) und `workouts` (Named Vectors: Kosinus + Euklid) |
| FIT-Parsing | `fitdecode`, Kennzahlen mit NumPy |
| Agenten-Orchestrierung | LangGraph (Zustandsgraph); LLM-Aufrufe darin direkt über das `anthropic`-SDK |
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
5. **Trainingsdaten (FIT)** – Upload einer `.FIT`-Datei (Garmin/ANT+) samt Notizen und Schwellenwerten (FTP, HFmax, Schwellen-HF, Schwellenpace). Pipeline:
   - **Parsing** mit `fitdecode`, alle Kanäle auf 1 Hz; kurze Lücken (Smart Recording) werden aufgefüllt, Pausen > 10 s entfernt.
   - **Kennzahlen:** NP, IF, TSS (aus Leistung; ohne Powermeter als rTSS aus der Pace bzw. hrTSS aus der HF – jeweils Näherungen), Arbeit, Ø/max HF, aerobe Entkopplung (Pa:HR bzw. Pace:HR, erste vs. zweite Hälfte), Zonen Z1–Z5 (Leistung nach % FTP, sonst HF nach % HFmax), Höhenmeter, automatisch erkannte Intervalle (≥ 2 min über 85 % FTP).
   - **Zwei Vektoren in einem Qdrant-Punkt:** `dense_metrics` (12D, jede Dimension auf 0–1 skaliert, euklidische Distanz) und `dense_semantic` (Voyage-Embedding einer Textzusammenfassung inkl. Notizen, Kosinus).
   - **Oberfläche:** Zeitreihen mit markierten Intervallen, 12D-Vektor als Balkendiagramm, Zusammenfassung, Zonenverteilung, „ähnliche Einheiten“ nach Kennzahlen vs. nach Bedeutung im direkten Vergleich, semantische Suche und „Frag den Coach“ (RAG über die eigenen Einheiten mit Claude).
   - 12 simulierte Beispiel-Einheiten (Rad, Lauf, Schwimmen; FTP 280 W) werden beim ersten Start angelegt. Dieselbe Datei erneut hochladen überschreibt die Einheit (ID aus dem Datei-Hash).

   Bewusste Abweichung vom Ursprungskonzept: Die Zusammenfassung wird per Template erzeugt, nicht zusätzlich per LLM angereichert – das spart Kosten und Latenz pro Upload und macht den Text reproduzierbar. Die Sportart ist wie im Konzept als eine Zahl kodiert; das Dashboard weist darauf hin, dass One-Hot-Kodierung sauberer wäre.
6. **Agentic RAG (LangGraph)** – Dieselbe Idee wie Modul 4, aber das LLM entscheidet selbst, welche Werkzeuge es wie oft aufruft, und kann Ergebnisse verketten. Der Graph hat drei Knoten:
   - `agent`: Claude mit Tool Use (Werkzeuge mit `strict: true`); die Antwort samt Thinking-Blöcken geht unverändert in den Verlauf.
   - `tools`: führt die Werkzeugaufrufe aus – `suche_wissensbasis` und `suche_trainings` (semantisch), `filtere_trainings` (Qdrant-Payload-Filter, kein Vektor) und `aehnliche_trainings` (12D-Kennzahlenvektor über die ID einer Einheit).
   - `final`: Notbremse nach 4 Werkzeugrunden, erzwingt mit `tool_choice: none` eine Antwort.

   Das Dashboard zeigt den Graphen als Diagramm (aus `draw_mermaid()`) mit dem durchlaufenen Pfad, jede Entscheidung und jedes Werkzeugergebnis als Zeitleiste und die gestreamte Antwort mit Quellenangaben. Ohne `ANTHROPIC_API_KEY` plant ein einfacher regelbasierter Planer die Werkzeugaufrufe – der Graph läuft dann trotzdem durch, nur ohne generierte Antwort.

### Code-Ansicht für Entwickler

Jedes Modul hat unten ein aufklappbares Panel **„Code hinter diesem Modul“** mit drei Reitern:

- **Minimalbeispiel** – eigenständiges Python-Skript, das den Kern des Moduls ohne dieses Projekt nachbaut (`voyageai`, `numpy`, `qdrant-client`, `anthropic`, `fitdecode`, `langgraph`).
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
| GET | `/api/workouts` | Alle Einheiten (ohne Vektoren) |
| POST | `/api/workouts/upload-fit` | Multipart: `file`, `notes`, `ftp`, `hr_max`, `lthr`, `threshold_pace` (s/km) → Kennzahlen, beide Vektoren |
| GET / DELETE | `/api/workouts/{id}` | Einheit inkl. Vektoren und Zeitreihen lesen / löschen |
| GET | `/api/workouts/{id}/similar` | Ähnliche Einheiten nach `dense_metrics` und nach `dense_semantic` |
| POST | `/api/workouts/search` | `{query, limit}` → semantische Suche |
| POST | `/api/workouts/coach` | `{question, top_k}` → Server-Sent Events: `retrieval`, `delta`, `done`, `error` |
| GET | `/api/agent/graph` | Graph als Mermaid-Text, Werkzeuge, aktiver Planer |
| POST | `/api/agent/run` | `{question}` → Server-Sent Events: `node` (pro durchlaufenem Knoten), `delta`, `done`, `error` |
| GET | `/api/code/{m1…m6}` | Minimalbeispiel und Projektcode für die Code-Ansicht |
| POST | `/api/seed` | Befüllung erneut versuchen (z. B. nach nachgetragenem Key) |

## Hinweise

- **Rate-Limits:** Voyage-Konten ohne hinterlegte Zahlungsmethode sind stark gedrosselt (wenige Requests pro Minute). Das Dashboard bündelt Requests und cacht, bei schnellem Klicken kann trotzdem ein 429 kommen – die Meldung erscheint im UI.
- **Absolute Ähnlichkeitswerte** sind modellabhängig; auch unverwandte Sätze liegen oft deutlich über 0. Aussagekräftig ist die Rangfolge.
- **Lokale Entwicklung ohne Docker:**
  ```bash
  cd backend && pip install -r requirements.txt
  QDRANT_URL=:memory: VOYAGE_API_KEY=... uvicorn app.main:app --reload
  ```
