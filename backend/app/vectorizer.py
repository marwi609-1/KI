"""Aus Trainingskennzahlen zwei Repräsentationen bauen: 12D-Metrikvektor und Textzusammenfassung."""

from .fit_parser import AthleteSettings

# Kategorische Kodierung als eine Zahl – bewusst wie im Konzept. Didaktischer Haken: Der Abstand
# Laufen–Radfahren ist dadurch kleiner als Schwimmen–Radfahren, obwohl es keine Rangfolge gibt.
SPORT_IDS = {"cycling": 1.0, "running": 0.66, "swimming": 0.33}

METRIC_DIMENSIONS = [
    ("Dauer", "0–360 min"),
    ("TSS", "0–400"),
    ("IF", "0–1,5"),
    ("Ø HF / HFmax", "0–1"),
    ("HF-Drift", "−20 % … +20 %"),
    ("Zone 1", "Anteil"),
    ("Zone 2", "Anteil"),
    ("Zone 3", "Anteil"),
    ("Zone 4", "Anteil"),
    ("Zone 5", "Anteil"),
    ("Höhenmeter", "0–3000 m"),
    ("Sportart", "kategorisch"),
]

SPORT_DE = {"cycling": "Radfahren", "running": "Laufen", "swimming": "Schwimmen"}
SUB_SPORT_DE = {
    "indoor_cycling": "Indoor/Rolle",
    "road": "Straße",
    "mountain": "Mountainbike",
    "gravel_cycling": "Gravel",
    "treadmill": "Laufband",
    "trail": "Trail",
    "track": "Bahn",
    "lap_swimming": "Becken",
    "open_water": "Freiwasser",
}


def _scale(value: float | None, lo: float, hi: float) -> float:
    if value is None:
        return 0.0
    return round(min(max((value - lo) / (hi - lo), 0.0), 1.0), 4)


def generate_metric_vector(m: dict, sport: str, st: AthleteSettings) -> list[float]:
    zones = m.get("zones") or [0.0] * 5
    drift = m.get("hr_drift_percent")
    return [
        _scale(m["moving_s"] / 60, 0, 360),
        _scale(m.get("tss"), 0, 400),
        _scale(m.get("intensity_factor"), 0, 1.5),
        _scale(m.get("avg_hr", 0) / st.hr_max if m.get("avg_hr") else None, 0, 1),
        _scale(drift / 100 if drift is not None else 0.0, -0.2, 0.2),
        *[round(z, 4) for z in zones],
        _scale(m.get("elevation_gain_m"), 0, 3000),
        SPORT_IDS.get(sport, 0.0),
    ]


def _de(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _fmt_min(seconds: float) -> str:
    return f"{round(seconds / 60)} min"


def _fmt_pace(s_per_km: float) -> str:
    return f"{int(s_per_km // 60)}:{int(s_per_km % 60):02d} min/km"


def generate_summary_text(m: dict, sport: str, sub_sport: str, date: str, notes: str = "") -> str:
    """Strukturierte Textfassung des Trainings – Grundlage für das semantische Embedding."""
    lines = [
        f"Sportart: {SPORT_DE.get(sport, sport)} | Art: {SUB_SPORT_DE.get(sub_sport, sub_sport)}",
        f"Datum: {date} | Dauer: {_fmt_min(m['moving_s'])} | Distanz: {_de(m['distance_km'], 1)} km",
    ]
    key = []
    if "tss" in m:
        key.append(f"TSS {m['tss']:.0f}")
    if "normalized_power_w" in m:
        key.append(f"NP {m['normalized_power_w']} W")
    if "intensity_factor" in m:
        key.append(f"IF {_de(m['intensity_factor'], 2)}")
    if "avg_pace_s_per_km" in m:
        key.append(f"Pace {_fmt_pace(m['avg_pace_s_per_km'])}")
    if "avg_hr" in m:
        key.append(f"Ø HF {m['avg_hr']} bpm")
    if "hr_drift_percent" in m:
        key.append(f"HF-Drift {'+' if m['hr_drift_percent'] >= 0 else ''}{_de(m['hr_drift_percent'], 1)} %")
    if m.get("elevation_gain_m"):
        key.append(f"{m['elevation_gain_m']} Hm")
    lines.append("Kennzahlen: " + ", ".join(key))

    if m.get("zones"):
        dominant = max(range(5), key=lambda i: m["zones"][i])
        lines.append(
            "Zonen: "
            + ", ".join(f"Z{i + 1} {z * 100:.0f} %" for i, z in enumerate(m["zones"]))
            + f" – Schwerpunkt Zone {dominant + 1}"
        )

    intervals = m.get("intervals") or []
    if intervals:
        avg_dur = sum(i["duration_s"] for i in intervals) / len(intervals)
        avg_pow = sum(i["avg_power"] for i in intervals) / len(intervals)
        lines.append(f"Struktur: {len(intervals)} harte Intervalle à ca. {_fmt_min(avg_dur)} mit Ø {avg_pow:.0f} W")
    elif m.get("intensity_factor", 0) < 0.75:
        lines.append("Struktur: gleichmäßige, lockere Einheit ohne harte Intervalle")
    else:
        lines.append("Struktur: durchgehend zügige Belastung ohne abgesetzte Intervalle")

    drift = m.get("hr_drift_percent")
    if drift is not None:
        if drift >= 5:
            lines.append("Physiologische Reaktion: deutliche kardiale Drift, die Herzfrequenz stieg bei gleicher Leistung an")
        elif drift < -2:
            lines.append("Physiologische Reaktion: Herzfrequenz in der zweiten Hälfte relativ niedriger (negativer Drift)")
        else:
            lines.append("Physiologische Reaktion: stabile Herzfrequenz, gute aerobe Kopplung")
    if notes.strip():
        lines.append(f"Notizen des Athleten: {notes.strip()}")
    return "\n".join(lines)
