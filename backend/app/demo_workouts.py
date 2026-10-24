"""Simulierte Beispiel-Workouts, damit Modul 5 auch ohne eigene FIT-Dateien funktioniert."""

from datetime import datetime, timezone

import numpy as np

from .fit_parser import Activity, AthleteSettings

DEMO_ATHLETE = AthleteSettings(ftp=280, hr_max=188, lthr=170, threshold_pace=270)

# (Name, Sportart, Unterart, Datum, Segmente [(Dauer s, Watt bzw. Pace s/km)], HF-Drift bpm/h, Höhenmeter, Notiz)
DEMO: list[tuple] = [
    ("Intervalle 4×8 min", "cycling", "indoor_cycling", "2026-10-20T17:30",
     [(900, 170)] + [(480, 315), (240, 140)] * 4 + [(900, 150)], 4, 0,
     "Schwere Beine beim Einfahren, Wattvorgabe in allen Intervallen aber gehalten."),
    ("Grundlage 3 h", "cycling", "road", "2026-10-18T09:00", [(10800, 185)], 2, 450,
     "Lockere Grundlage mit der Dienstagsrunde, fühlte mich frisch."),
    ("Lange Bergtour 4,5 h", "cycling", "road", "2026-10-11T08:30", [(16200, 195)], 9, 2200,
     "Sehr heiß, zu wenig getrunken, die letzte Stunde war zäh."),
    ("Regeneration 45 min", "cycling", "indoor_cycling", "2026-10-13T18:00", [(2700, 140)], 0, 0,
     "Regeneration nach dem Rennwochenende, nur locker rollen."),
    ("Sweet Spot 3×15 min", "cycling", "indoor_cycling", "2026-10-15T18:00",
     [(900, 180)] + [(900, 250), (300, 150)] * 3 + [(600, 150)], 3, 0, "Gut gelaufen, Puls blieb stabil."),
    ("VO2max 5×4 min", "cycling", "indoor_cycling", "2026-10-08T17:30",
     [(900, 170)] + [(240, 335), (240, 130)] * 5 + [(900, 150)], 5, 0,
     "Letztes Intervall musste ich abbrechen, komplett leer."),
    ("Lockerer Lauf 50 min", "running", "generic", "2026-10-19T07:30", [(3000, 345)], 2, 60,
     "Lockerer Lauf am Bach entlang, Herbstlaub."),
    ("Bahnintervalle 6×1000 m", "running", "track", "2026-10-16T18:30",
     [(900, 360)] + [(240, 240), (150, 420)] * 6 + [(600, 380)], 3, 0, "Bahnintervalle, Beine fühlten sich gut an."),
    ("Langer Lauf 2 h", "running", "generic", "2026-10-12T08:00", [(7200, 330)], 8, 180,
     "Ab Kilometer 18 deutlicher Einbruch, Puls ging hoch."),
    ("Bergzeitfahren Kreitmeier-Berg", "cycling", "road", "2026-10-04T10:00",
     [(1200, 160), (708, 365), (900, 130)], 2, 310,
     "Vereinsmeisterschaft am Kreitmeier-Berg, neue persönliche Bestzeit."),
    ("Tempo 2×20 min", "cycling", "road", "2026-10-06T17:00",
     [(900, 180)] + [(1200, 255), (300, 150)] * 2 + [(900, 160)], 3, 250,
     "Fühlte mich stark, Gegenwind auf dem Rückweg."),
    ("Schwimmen Technik 45 min", "swimming", "lap_swimming", "2026-10-14T06:45", [(2700, 870)], 1, 0,
     "Technikeinheit im Hallenbad, Fokus auf Wasserlage."),
]


def simulate(sport: str, sub_sport: str, start: str, segments: list, drift_per_h: float, ascent: float, seed: int) -> Activity:
    rng = np.random.default_rng(seed)
    st = DEMO_ATHLETE
    target = np.concatenate([np.full(d, v, dtype=float) for d, v in segments])
    n = len(target)
    t_h = np.arange(n) / 3600

    if sport == "cycling":
        terrain = 1 + 0.12 * np.sin(np.arange(n) / 500) if ascent > 500 else 1
        power = np.clip(target * terrain + rng.normal(0, 10, n), 0, None)
        intensity = power / st.ftp
        speed = (5.3 * np.cbrt(power) - (6 if sub_sport != "indoor_cycling" and ascent > 500 else 0)) / 3.6
    else:
        power = None
        speed = np.clip(1000 / target + rng.normal(0, 0.08, n), 0.3, None)
        intensity = speed / (1000 / st.threshold_pace) if sport == "running" else np.full(n, 0.72)

    # HF folgt der Belastung mit Verzögerung (Zeitkonstante ~40 s) und driftet mit der Zeit nach oben
    hr_target = np.clip(55 + 110 * intensity**1.6 + drift_per_h * t_h, 60, st.hr_max - 2)
    hr = np.empty(n)
    hr[0] = 95
    for i in range(1, n):
        hr[i] = hr[i - 1] + (hr_target[i] - hr[i - 1]) / 40
    hr += rng.normal(0, 1.2, n)

    return Activity(
        start=datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
        sport=sport,
        sub_sport=sub_sport,
        elapsed_s=float(n),
        power=power,
        hr=hr,
        speed=speed,
        total_ascent=ascent,
        total_distance=float(speed.sum()),
    )


def demo_activities():
    for i, (name, sport, sub, start, segments, drift, ascent, notes) in enumerate(DEMO):
        yield name, simulate(sport, sub, start, segments, drift, ascent, seed=i), notes
