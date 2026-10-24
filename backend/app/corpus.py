"""Beispieldaten: Sätze für den 2D-Vektorraum (Modul 3) und eine kleine Wissensbasis für RAG (Modul 4)."""

SENTENCES: dict[str, list[str]] = {
    "Sport": [
        "Der Stürmer schoss in der letzten Minute das Siegtor.",
        "Beim Intervalltraining wechseln sich harte und lockere Abschnitte ab.",
        "Die Radfahrerin gewann die Bergetappe mit großem Vorsprung.",
        "Nach dem Marathon brauchen die Muskeln mehrere Tage Erholung.",
        "Die Mannschaft trainiert dreimal pro Woche in der Halle.",
        "Er verbesserte seine Bestzeit über zehn Kilometer deutlich.",
        "Beim Tennis entscheidet oft der zweite Aufschlag das Spiel.",
        "Das Schwimmbad bietet morgens Bahnen für Kraulschwimmer an.",
    ],
    "Kochen": [
        "Die Zwiebeln werden glasig angeschwitzt, bevor der Reis dazukommt.",
        "Für den Teig braucht man Mehl, Hefe, Wasser und eine Prise Salz.",
        "Das Gemüse sollte bissfest gegart und nicht zerkocht werden.",
        "Frischer Basilikum gibt der Tomatensoße ein feines Aroma.",
        "Den Braten bei niedriger Temperatur langsam im Ofen garen.",
        "Zum Abschmecken fehlen noch etwas Pfeffer und Zitronensaft.",
        "Der Kuchen muss vor dem Anschneiden vollständig abkühlen.",
        "Eine gute Brühe köchelt mehrere Stunden auf kleiner Flamme.",
    ],
    "Informatik": [
        "Der Compiler übersetzt den Quellcode in Maschinensprache.",
        "Eine Hashtabelle ermöglicht Zugriffe in konstanter Zeit.",
        "Der Server antwortet nach dem Update deutlich schneller.",
        "Mit Docker lassen sich Anwendungen in Containern isolieren.",
        "Die Datenbankabfrage wurde durch einen Index beschleunigt.",
        "Ein neuronales Netz lernt Gewichte durch Backpropagation.",
        "Der Fehler trat nur auf, wenn zwei Threads gleichzeitig schrieben.",
        "Die API liefert ihre Antworten im JSON-Format zurück.",
    ],
    "Finanzen": [
        "Ein breit gestreuter Indexfonds senkt das Einzelwertrisiko.",
        "Die Zentralbank hat den Leitzins erneut angehoben.",
        "Durch den Zinseszins wächst das Vermögen über Jahrzehnte.",
        "Die Aktie fiel nach den schwachen Quartalszahlen deutlich.",
        "Ein Notgroschen sollte drei bis sechs Monatsausgaben abdecken.",
        "Die Inflation verringert die Kaufkraft des Ersparten.",
        "Die Bank verlangt für das Darlehen einen festen Sollzins.",
        "Anleihen gelten als schwankungsärmer als Aktien.",
    ],
    "Natur": [
        "Im Herbst färben sich die Blätter der Buchen goldgelb.",
        "Der Fuchs schleicht in der Dämmerung über die Wiese.",
        "Nach dem Gewitter roch der Wald nach feuchter Erde.",
        "Zugvögel fliegen im Winter in den warmen Süden.",
        "Der Bach schlängelt sich durch das enge Tal.",
        "Bienen bestäuben die Blüten der Obstbäume im Frühling.",
        "Auf dem Gipfel lag noch Schnee, obwohl es schon Mai war.",
        "Moos wächst bevorzugt auf der schattigen Seite der Bäume.",
    ],
}

# Wissensbasis: Teils Fachwissen zum Thema dieses Dashboards, teils frei erfundene Fakten
# (Radsportverein „RSV Grünwald 1923“), die kein LLM aus dem Training kennen kann.
# Damit lässt sich zeigen, dass die Antwort wirklich aus den abgerufenen Quellen stammt.
KNOWLEDGE: list[dict[str, str]] = [
    {
        "title": "Was ist ein Embedding?",
        "text": "Ein Embedding ist ein Vektor aus Gleitkommazahlen, der die Bedeutung eines Textes repräsentiert. "
        "Texte mit ähnlicher Bedeutung liegen im Vektorraum nahe beieinander, auch wenn sie keine gemeinsamen Wörter haben.",
    },
    {
        "title": "Kosinus-Ähnlichkeit",
        "text": "Die Kosinus-Ähnlichkeit misst den Winkel zwischen zwei Vektoren. Sie liegt zwischen -1 und 1. "
        "Bei auf Länge 1 normierten Vektoren entspricht sie dem Skalarprodukt, weshalb Vektordatenbanken sie sehr schnell berechnen können.",
    },
    {
        "title": "Vektordatenbank Qdrant",
        "text": "Qdrant ist eine Open-Source-Vektordatenbank, geschrieben in Rust. Sie speichert Vektoren zusammen mit einer Payload "
        "(Metadaten) und findet mit einem HNSW-Index die nächsten Nachbarn eines Suchvektors in Millisekunden.",
    },
    {
        "title": "HNSW-Index",
        "text": "HNSW (Hierarchical Navigable Small World) ist ein Graph-basierter Index für die approximative Nächste-Nachbarn-Suche. "
        "Statt jeden Vektor zu vergleichen, springt die Suche über mehrere Graph-Ebenen schnell in die Nähe des Ziels.",
    },
    {
        "title": "Retrieval-Augmented Generation",
        "text": "Bei RAG wird eine Frage zuerst vektorisiert, dann werden die ähnlichsten Dokumente aus einer Vektordatenbank geholt. "
        "Diese Dokumente werden dem Sprachmodell als Kontext mitgegeben, damit es seine Antwort auf überprüfbare Quellen stützt.",
    },
    {
        "title": "Chunking",
        "text": "Lange Dokumente werden vor dem Einbetten in kleinere Abschnitte (Chunks) zerlegt, typischerweise 200 bis 800 Token "
        "mit etwas Überlappung. Zu große Chunks verwässern die Bedeutung, zu kleine verlieren den Zusammenhang.",
    },
    {
        "title": "PCA",
        "text": "Die Hauptkomponentenanalyse (PCA) projiziert hochdimensionale Daten auf die Richtungen mit der größten Varianz. "
        "Eine 2D-Projektion von Embeddings ist anschaulich, verliert aber viel Information – Abstände im Plot sind nur eine Näherung.",
    },
    {
        "title": "Query- und Dokument-Embeddings",
        "text": "Voyage AI unterscheidet beim Einbetten zwischen input_type 'query' und 'document'. Dadurch werden kurze Suchanfragen "
        "und längere Dokumente so kodiert, dass sie im selben Raum besser zueinander passen.",
    },
    {
        "title": "Grenzen der Schlüsselwortsuche",
        "text": "Eine klassische Schlüsselwortsuche findet nur Dokumente, die dieselben Wörter enthalten. Synonyme wie 'Auto' und 'Wagen' "
        "oder mehrdeutige Wörter wie 'Bank' erkennt sie nicht. Semantische Suche über Embeddings löst genau dieses Problem.",
    },
    {
        "title": "RSV Grünwald 1923 – Vereinsgeschichte",
        "text": "Der Radsportverein RSV Grünwald 1923 wurde am 14. Mai 1923 von elf Bahnradfahrern gegründet. "
        "Erster Vorsitzender war der Schmied Anton Kreitmeier. Heute hat der Verein 412 Mitglieder.",
    },
    {
        "title": "RSV Grünwald – Ausfahrten",
        "text": "Die Vereinsausfahrten des RSV Grünwald starten jeden Dienstag und Donnerstag um 18:00 Uhr am Brunnen auf dem Marktplatz. "
        "Die Dienstagsrunde ist 65 km lang bei einem Schnitt von 28 km/h, die Donnerstagsrunde ist eine lockere Grundlagenrunde mit 45 km.",
    },
    {
        "title": "RSV Grünwald – Mitgliedsbeitrag",
        "text": "Der Jahresbeitrag im RSV Grünwald beträgt 84 € für Erwachsene, 36 € für Jugendliche unter 18 Jahren und 150 € für Familien. "
        "Neumitglieder erhalten im ersten Jahr ein Vereinstrikot kostenlos.",
    },
    {
        "title": "RSV Grünwald – Helmpflicht",
        "text": "Bei allen Vereinsausfahrten des RSV Grünwald gilt Helmpflicht. Ab Oktober ist zusätzlich ein Rücklicht vorgeschrieben. "
        "Wer ohne Helm erscheint, darf nicht mitfahren.",
    },
    {
        "title": "RSV Grünwald – Vereinsmeisterschaft",
        "text": "Die Vereinsmeisterschaft des RSV Grünwald ist ein Bergzeitfahren auf den Kreitmeier-Berg (4,2 km, 310 Höhenmeter). "
        "Der Streckenrekord liegt bei 11:48 Minuten und wurde 2019 von Lena Hofbauer aufgestellt.",
    },
]
