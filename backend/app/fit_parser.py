"""FIT-Dateien einlesen und Trainingskennzahlen berechnen (NP, IF, TSS, Entkopplung, Zonen)."""

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone

import fitdecode
import numpy as np

MAX_GAP_S = 10  # längere Lücken gelten als Pause und fließen nicht in die Berechnung ein
POWER_ZONES = [0, 0.55, 0.75, 0.90, 1.05]  # Anteil FTP, Untergrenzen Z1–Z5 (Coggan, Z5–Z7 zusammengefasst)
HR_ZONES = [0, 0.60, 0.70, 0.80, 0.90]  # Anteil HFmax, Untergrenzen Z1–Z5


@dataclass
class AthleteSettings:
    ftp: float = 250.0  # W
    hr_max: float = 190.0  # bpm
    lthr: float = 170.0  # Laktatschwellen-HF, bpm
    threshold_pace: float = 300.0  # Schwellenpace Laufen, s/km


@dataclass
class Activity:
    """Auf 1 Hz gebrachte Zeitreihen. Fehlende Kanäle sind None."""

    start: datetime
    sport: str
    sub_sport: str
    elapsed_s: float
    power: np.ndarray | None = None
    hr: np.ndarray | None = None
    speed: np.ndarray | None = None  # m/s
    altitude: np.ndarray | None = None
    total_ascent: float | None = None
    total_distance: float | None = None  # m
    extra: dict = field(default_factory=dict)


def parse_fit(data: bytes) -> Activity:
    records: list[dict] = []
    session: dict = {}
    with fitdecode.FitReader(io.BytesIO(data)) as reader:
        for frame in reader:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name == "record":
                records.append({f.name: f.value for f in frame.fields})
            elif frame.name == "session" and not session:
                session = {f.name: f.value for f in frame.fields if f.value is not None}

    records = [r for r in records if isinstance(r.get("timestamp"), datetime)]
    if len(records) < 60:
        raise ValueError("Die FIT-Datei enthält zu wenige Datenpunkte (mindestens 60 Sekunden nötig).")

    start = records[0]["timestamp"]
    t = np.array([(r["timestamp"] - start).total_seconds() for r in records])

    def channel(*names: str) -> np.ndarray | None:
        values = np.array(
            [next((r[n] for n in names if isinstance(r.get(n), (int, float))), np.nan) for r in records], dtype=float
        )
        return None if np.all(np.isnan(values)) else values

    return Activity(
        start=start.astimezone(timezone.utc),
        sport=str(session.get("sport", "generic")),
        sub_sport=str(session.get("sub_sport", "generic")),
        elapsed_s=float(session.get("total_elapsed_time", t[-1])),
        power=_resample(t, channel("power")),
        hr=_resample(t, channel("heart_rate")),
        speed=_resample(t, channel("enhanced_speed", "speed")),
        altitude=_resample(t, channel("enhanced_altitude", "altitude")),
        total_ascent=session.get("total_ascent"),
        total_distance=session.get("total_distance"),
    )


def _resample(t: np.ndarray, values: np.ndarray | None) -> np.ndarray | None:
    """Auf ein 1-Hz-Raster bringen; kurze Lücken (Smart Recording) auffüllen, Pausen entfernen."""
    if values is None:
        return None
    grid = np.arange(0, int(t[-1]) + 1)
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    out = values[idx]
    gap = grid - t[idx]
    keep = gap <= MAX_GAP_S
    # Zeitpunkte nach einer Pause entfernen, damit Pausen nicht als Leistung 0 zählen
    return _ffill(out[keep])


def _ffill(a: np.ndarray) -> np.ndarray:
    mask = np.isnan(a)
    if mask.all():
        return a
    idx = np.where(~mask, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    out[np.isnan(out)] = np.nanmean(a)  # führende NaNs
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return np.array([x.mean()])
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[window:] - c[:-window]) / window


def normalized_power(power: np.ndarray) -> float:
    """NP: 30-s-gleitender Mittelwert, 4. Potenz mitteln, 4. Wurzel ziehen."""
    return float(np.mean(rolling_mean(power, 30) ** 4) ** 0.25)


def aerobic_decoupling(output: np.ndarray, hr: np.ndarray) -> float | None:
    """Pa:HR in %: Effizienzfaktor (Leistung bzw. Tempo / HF) erste vs. zweite Hälfte. Positiv = HF driftet nach oben."""
    n = min(len(output), len(hr))
    if n < 600:
        return None
    half = n // 2
    ef1 = output[:half].mean() / max(hr[:half].mean(), 1)
    ef2 = output[half:n].mean() / max(hr[half:n].mean(), 1)
    return float((ef1 - ef2) / ef1 * 100) if ef1 > 0 else None


def zone_distribution(values: np.ndarray, reference: float, bounds: list[float]) -> list[float]:
    rel = values / reference
    zone = np.searchsorted(bounds, rel, side="right") - 1
    counts = np.bincount(np.clip(zone, 0, 4), minlength=5)
    return (counts / counts.sum()).round(4).tolist()


def elevation_gain(altitude: np.ndarray, threshold: float = 2.0) -> float:
    """Summe der Anstiege mit Hysterese, damit Messrauschen nicht als Höhenmeter zählt."""
    gain, ref = 0.0, altitude[0]
    for a in altitude[1:]:
        if a - ref >= threshold:
            gain += a - ref
            ref = a
        elif a < ref:
            ref = a
    return gain


def detect_intervals(power: np.ndarray, ftp: float) -> list[dict]:
    """Blöcke ≥ 2 min oberhalb 85 % FTP (30-s-Mittel), kurze Einbrüche < 20 s werden überbrückt."""
    smooth = rolling_mean(power, 30)
    hard = smooth >= 0.85 * ftp
    segments, start, last_hard = [], None, None
    for i, h in enumerate(hard):
        if h:
            start = i if start is None else start
            last_hard = i
        elif start is not None and i - last_hard > 20:
            segments.append((start, last_hard))
            start = None
    if start is not None:
        segments.append((start, last_hard))
    return [
        {"start_s": s, "duration_s": e - s + 30, "avg_power": round(float(power[s : e + 30].mean()))}
        for s, e in segments
        if e - s + 30 >= 120
    ]


def compute_metrics(act: Activity, st: AthleteSettings) -> dict:
    moving_s = max(len(a) for a in (act.power, act.hr, act.speed) if a is not None) if any(
        a is not None for a in (act.power, act.hr, act.speed)
    ) else act.elapsed_s
    hours = moving_s / 3600
    m: dict = {
        "duration_s": round(act.elapsed_s),
        "moving_s": int(moving_s),
        "distance_km": round((act.total_distance or (act.speed.sum() if act.speed is not None else 0)) / 1000, 2),
    }

    has_power = act.power is not None and np.nanmax(act.power) > 0
    if has_power:
        np_ = normalized_power(act.power)
        if_ = np_ / st.ftp
        m |= {
            "avg_power_w": round(float(act.power.mean())),
            "normalized_power_w": round(np_),
            "intensity_factor": round(if_, 3),
            "tss": round(hours * if_**2 * 100, 1),
            "work_kj": round(float(act.power.sum()) / 1000),
            "tss_method": "Leistung (NP/FTP)",
        }
    elif act.speed is not None and act.sport == "running":
        avg_speed = float(act.speed.mean())
        if_ = avg_speed / (1000 / st.threshold_pace)
        m |= {"intensity_factor": round(if_, 3), "tss": round(hours * if_**2 * 100, 1), "tss_method": "Pace (rTSS, Näherung)"}
    elif act.hr is not None:
        if_ = float(act.hr.mean()) / st.lthr
        m |= {"intensity_factor": round(if_, 3), "tss": round(hours * if_**2 * 100, 1), "tss_method": "Herzfrequenz (hrTSS, Näherung)"}

    if act.speed is not None and act.speed.mean() > 0:
        m["avg_speed_kmh"] = round(float(act.speed.mean()) * 3.6, 1)
        if act.sport == "running":
            m["avg_pace_s_per_km"] = round(1000 / float(act.speed.mean()))
            m["threshold_pace_ratio"] = round(m["avg_pace_s_per_km"] / st.threshold_pace, 3)

    if act.hr is not None:
        m["avg_hr"] = round(float(act.hr.mean()))
        m["max_hr"] = round(float(act.hr.max()))
        output = act.power if has_power else act.speed
        if output is not None:
            drift = aerobic_decoupling(output, act.hr)
            if drift is not None:
                m["hr_drift_percent"] = round(drift, 1)

    if has_power:
        m["zones"], m["zone_basis"] = zone_distribution(act.power, st.ftp, POWER_ZONES), "Leistung (% FTP)"
        m["intervals"] = detect_intervals(act.power, st.ftp)
    elif act.hr is not None:
        m["zones"], m["zone_basis"] = zone_distribution(act.hr, st.hr_max, HR_ZONES), "Herzfrequenz (% HFmax)"

    if act.total_ascent is not None:
        m["elevation_gain_m"] = round(float(act.total_ascent))
    elif act.altitude is not None:
        m["elevation_gain_m"] = round(elevation_gain(act.altitude))
    return m


def downsample_series(act: Activity, points: int = 600) -> dict:
    """Zeitreihen für Diagramme ausdünnen (Mittelwert je Block)."""
    series = {k: v for k, v in (("power", act.power), ("hr", act.hr), ("speed", act.speed)) if v is not None}
    n = max(len(v) for v in series.values())
    step = max(1, n // points)
    out = {"t_min": [round(i * step / 60, 2) for i in range(n // step)]}
    for k, v in series.items():
        v = v[: (len(v) // step) * step].reshape(-1, step).mean(axis=1)
        out[k] = [round(float(x), 2) for x in v]
    return out
