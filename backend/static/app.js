const $ = (id) => document.getElementById(id);
const fmt = (x, d = 3) => x.toLocaleString("de-DE", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (x) => `${fmt(x * 100, 1)} %`;
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const PLOT_CFG = { displayModeBar: false, responsive: true };
const DIVERGING = [[0, "#1d4ed8"], [0.5, "#f8fafc"], [1, "#dc2626"]];
const COLORS = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#0891b2", "#db2777", "#ca8a04", "#475569"];

async function api(path, body, method) {
  const res = await fetch(path, {
    method: method || (body ? "POST" : "GET"),
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ? JSON.stringify(data.detail) : res.statusText);
  if (CALL_MODULE[path]) recordCall(CALL_MODULE[path], { method: method || "POST", path, body, response: data });
  return data;
}

function busy(form, on) {
  const btn = form.querySelector("button:not([type=button])");
  btn.disabled = on;
  btn.classList.toggle("opacity-50", on);
}

function showError(err) {
  alert(`Fehler: ${err.message}`);
}

function stripe(el, rows, labels, height) {
  // Jede Zeile auf ihr eigenes Maximum skalieren, damit auch kleine Produktwerte sichtbar bleiben
  const scaled = rows.map((r) => { const m = Math.max(...r.map(Math.abs)) || 1; return r.map((v) => v / m); });
  Plotly.react(el, [{
    z: scaled, customdata: rows, y: labels, type: "heatmap", colorscale: DIVERGING,
    zmin: -1, zmax: 1, showscale: false,
    hovertemplate: "Dimension %{x}<br>Wert %{customdata:.4f}<extra>%{y}</extra>",
  }], {
    margin: { l: labels.some(Boolean) ? 80 : 10, r: 10, t: 5, b: 30 }, height, separators: ",.",
    xaxis: { title: { text: "Dimension", font: { size: 11 } } },
    yaxis: { automargin: true },
  }, PLOT_CFG);
}

// ---------- Tabs & Status ----------

function showTab(id) {
  document.querySelectorAll(".module").forEach((m) => m.classList.toggle("hidden", m.id !== id));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("tab-active", t.dataset.tab === id));
  if (id === "m3" && !spaceLoaded) loadSpace();
  if (id === "m4") loadKnowledge();
  if (id === "m5" && !workoutsLoaded) initWorkouts();
  if (id === "m6" && !graphInfo) initAgent();
  window.dispatchEvent(new Event("resize"));
}
document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => showTab(t.dataset.tab)));

async function loadStatus() {
  try {
    const s = await api("/api/status");
    const llm = s.llm_available ? `LLM: ${s.llm_model}` : "LLM: nicht konfiguriert";
    $("status").innerHTML =
      `Embeddings: <b>${esc(s.embedding_model)}</b> · ${llm}<br>` +
      `Qdrant: ${s.collections.saetze} Sätze, ${s.collections.wissen} Wissensdokumente, ${s.collections.workouts} Trainings`;
    const box = $("seed-error");
    if (s.seed_error) {
      box.querySelector("div").textContent = `Datenbank konnte nicht befüllt werden: ${s.seed_error}`;
    } else if (s.embedding_backend === "hash") {
      box.querySelector("div").textContent =
        "Testmodus (EMBEDDING_BACKEND=hash): Die Vektoren erfassen nur Wortähnlichkeit, keine Semantik. Für echte Ergebnisse VOYAGE_API_KEY setzen.";
    }
    box.classList.toggle("hidden", !s.seed_error && s.embedding_backend !== "hash");
  } catch (e) {
    $("status").textContent = "Backend nicht erreichbar";
  }
}

// ---------- Modul 1 ----------

$("f1").onsubmit = async (ev) => {
  ev.preventDefault();
  busy(ev.target, true);
  try {
    const r = await api("/api/embed", { text: $("t1").value });
    $("r1").classList.remove("hidden");
    $("stats1").innerHTML = [
      ["Dimensionen", r.dims.toLocaleString("de-DE")],
      ["L2-Norm", fmt(r.norm, 4)],
      ["Minimum", fmt(r.min, 4)],
      ["Maximum", fmt(r.max, 4)],
    ].map(([k, v]) => `<div class="bg-slate-50 rounded-lg p-3"><div class="text-xs text-slate-500">${k}</div><div class="text-lg font-semibold">${v}</div></div>`).join("");
    $("preview1").textContent = `[${r.vector.slice(0, 10).map((v) => v.toFixed(4)).join(", ")}, … (${r.dims - 10} weitere)]`;
    stripe("heat1", [r.vector], [""], 110);
  } catch (e) { showError(e); } finally { busy(ev.target, false); }
};

// ---------- Modul 2 ----------

const EXAMPLES2 = [
  ["Synonyme", "Das Auto fährt schnell", "Der Wagen ist zügig unterwegs"],
  ["Mehrdeutig", "Ich sitze auf der Bank im Park", "Ich habe Geld auf der Bank"],
  ["Gleiche Wörter, andere Aussage", "Der Hund beißt den Mann", "Der Mann beißt den Hund"],
  ["Sprachübergreifend", "Ich trinke gerne Kaffee am Morgen", "I enjoy drinking coffee in the morning"],
  ["Ohne Bezug", "Die Aktie ist gestiegen", "Der Kuchen ist im Ofen"],
];
EXAMPLES2.forEach(([label, a, b]) => {
  const chip = document.createElement("span");
  chip.className = "chip bg-slate-100 hover:bg-indigo-100 rounded-full px-3 py-1";
  chip.textContent = label;
  chip.onclick = () => { $("a2").value = a; $("b2").value = b; $("f2").requestSubmit(); };
  $("examples2").appendChild(chip);
});
const history2 = [];

$("f2").onsubmit = async (ev) => {
  ev.preventDefault();
  busy(ev.target, true);
  try {
    const a = $("a2").value, b = $("b2").value;
    const r = await api("/api/compare", { a, b });
    $("r2").classList.remove("hidden");
    Plotly.react("gauge2", [{
      type: "indicator", mode: "gauge+number", value: Math.max(0, r.cosine) * 100,
      number: { suffix: " %", valueformat: ".1f" },
      gauge: {
        axis: { range: [0, 100] }, bar: { color: "#4f46e5" },
        steps: [{ range: [0, 40], color: "#fee2e2" }, { range: [40, 70], color: "#fef9c3" }, { range: [70, 100], color: "#dcfce7" }],
      },
      title: { text: "Kosinus-Ähnlichkeit", font: { size: 14 } },
    }], { margin: { l: 30, r: 30, t: 50, b: 10 }, height: 250, separators: ",." }, PLOT_CFG);
    $("metrics2").innerHTML = [
      ["Kosinus", fmt(r.cosine, 4)],
      ["Euklid. Abstand", fmt(r.euclidean, 4)],
      ["Wortüberlappung", pct(r.keyword_jaccard)],
    ].map(([k, v]) => `<div class="bg-slate-50 rounded-lg p-2"><div class="text-xs text-slate-500">${k}</div><div class="font-semibold">${v}</div></div>`).join("");
    const product = r.vector_a.map((v, i) => v * r.vector_b[i]);
    stripe("heat2", [product, r.vector_b, r.vector_a], ["Produkt", "Satz B", "Satz A"], 220);

    history2.unshift({ a, b, cos: r.cosine, kw: r.keyword_jaccard });
    $("history2").innerHTML = `<h3 class="font-medium mb-2">Verlauf – vergleiche relativ, nicht absolut</h3>
      <table class="w-full text-left"><thead class="text-xs text-slate-500"><tr><th>Satz A</th><th>Satz B</th><th class="text-right">Kosinus</th><th class="text-right">Wortüberlappung</th></tr></thead><tbody>
      ${history2.slice(0, 10).map((h) => `<tr class="border-t"><td class="py-1 pr-2">${esc(h.a)}</td><td class="pr-2">${esc(h.b)}</td><td class="text-right font-medium">${fmt(h.cos, 3)}</td><td class="text-right text-slate-500">${pct(h.kw)}</td></tr>`).join("")}
      </tbody></table>
      <p class="text-xs text-slate-500 mt-2">Absolute Werte sind modellabhängig – auch unverwandte Sätze haben oft eine Kosinus-Ähnlichkeit deutlich über 0. Aussagekräftig ist die Rangfolge.</p>`;
  } catch (e) { showError(e); } finally { busy(ev.target, false); }
};

// ---------- Modul 3 ----------

let spaceLoaded = false;
let categories = [];

async function loadSpace() {
  try {
    const r = await api("/api/space");
    spaceLoaded = true;
    categories = [...new Set(r.points.map((p) => p.category))];
    const traces = categories.map((c, i) => {
      const pts = r.points.filter((p) => p.category === c);
      return {
        x: pts.map((p) => p.x), y: pts.map((p) => p.y), text: pts.map((p) => p.text), name: c,
        mode: "markers", type: "scatter", marker: { size: 10, color: COLORS[i % COLORS.length], opacity: 0.85 },
        hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
      };
    });
    Plotly.newPlot("plot3", traces, {
      margin: { l: 40, r: 10, t: 10, b: 40 }, hovermode: "closest", separators: ",.",
      xaxis: { title: "Hauptkomponente 1", zeroline: false }, yaxis: { title: "Hauptkomponente 2", zeroline: false },
      legend: { orientation: "h", y: -0.15 },
    }, PLOT_CFG);
    const [v1, v2] = r.variance;
    $("variance3").textContent = `Die zwei Achsen erklären zusammen ${pct(v1 + v2)} der Varianz (PC1 ${pct(v1)}, PC2 ${pct(v2)}). Der Rest der Information steckt in den übrigen Dimensionen – Nähe im Plot ist daher nur eine Näherung.`;
  } catch (e) { showError(e); }
}

$("f3").onsubmit = async (ev) => {
  ev.preventDefault();
  busy(ev.target, true);
  try {
    if (!spaceLoaded) await loadSpace();
    const text = $("t3").value;
    const r = await api("/api/space/query", { text });
    const plot = $("plot3");
    // alte Abfrage-Spuren entfernen
    const old = plot.data.map((t, i) => (t.meta === "query" ? i : -1)).filter((i) => i >= 0);
    if (old.length) Plotly.deleteTraces(plot, old);

    const top = r.neighbors.slice(0, 3);
    const neighborPts = top.map((n) => {
      const tr = plot.data.find((t) => t.name === n.category);
      const idx = tr ? tr.text.indexOf(n.text) : -1;
      return idx >= 0 ? [tr.x[idx], tr.y[idx]] : null;
    }).filter(Boolean);
    const lines = {
      x: neighborPts.flatMap(([x, y]) => [r.x, x, null]), y: neighborPts.flatMap(([x, y]) => [r.y, y, null]),
      mode: "lines", line: { color: "#111827", dash: "dot", width: 1 }, hoverinfo: "skip", showlegend: false, meta: "query",
    };
    const q = {
      x: [0], y: [0], text: [text], name: "Suchanfrage", meta: "query", mode: "markers", type: "scatter",
      marker: { size: 18, symbol: "star", color: "#111827", line: { color: "#fff", width: 1 } },
      hovertemplate: "%{text}<extra>Suchanfrage</extra>",
    };
    Plotly.addTraces(plot, [q]);
    const qi = plot.data.length - 1;
    await Plotly.animate(plot, { data: [{ x: [r.x], y: [r.y] }], traces: [qi] }, { transition: { duration: 900, easing: "cubic-in-out" }, frame: { duration: 900 } });
    Plotly.addTraces(plot, [lines]);

    const best = r.clusters[0];
    $("side3").innerHTML = `
      <h3 class="font-medium">Nächstes Cluster</h3>
      <p class="text-2xl font-semibold mt-1">${esc(best.category)}</p>
      <ul class="mt-2 space-y-1">${r.clusters.map((c) => `<li class="flex justify-between"><span>${esc(c.category)}</span><span class="text-slate-500">${fmt(c.similarity, 3)}</span></li>`).join("")}</ul>
      <h3 class="font-medium mt-4">Nächste Nachbarn (Qdrant, ${r.neighbors.length})</h3>
      <ol class="mt-2 space-y-2 list-decimal list-inside">${r.neighbors.map((n) => `<li>${esc(n.text)} <span class="text-xs text-slate-500">${esc(n.category)} · ${fmt(n.score, 3)}</span></li>`).join("")}</ol>
      <p class="text-xs text-slate-500 mt-3">Die Nachbarn werden im vollen Vektorraum gesucht, nicht im 2D-Plot. Die gepunkteten Linien zeigen die Top-3 – liegen sie im Plot nicht am nächsten, ist das die Projektion, die täuscht.</p>`;
  } catch (e) { showError(e); } finally { busy(ev.target, false); }
};

$("f3add").onsubmit = async (ev) => {
  ev.preventDefault();
  if (!$("t3add").value.trim()) return;
  busy(ev.target, true);
  try {
    await api("/api/space/add", { text: $("t3add").value, category: $("c3add").value || "Eigene" });
    $("t3add").value = "";
    await loadSpace();
    loadStatus();
  } catch (e) { showError(e); } finally { busy(ev.target, false); }
};

// ---------- Modul 4 ----------

const EXAMPLES4 = [
  "Wann und wo starten die Ausfahrten des Radvereins?",
  "Was kostet die Mitgliedschaft für eine Familie?",
  "Wer hält den Rekord am Kreitmeier-Berg?",
  "Warum findet eine normale Suche keine Synonyme?",
  "Wie schnell ist ein HNSW-Index?",
  "Wer hat die Tour de France 2024 gewonnen?",
];
EXAMPLES4.forEach((q) => {
  const chip = document.createElement("span");
  chip.className = "chip bg-slate-100 hover:bg-indigo-100 rounded-full px-3 py-1";
  chip.textContent = q;
  chip.onclick = () => { $("t4").value = q; $("f4").requestSubmit(); };
  $("examples4").appendChild(chip);
});
$("k4").oninput = () => ($("k4v").textContent = $("k4").value);

async function loadKnowledge() {
  try {
    const docs = await api("/api/knowledge");
    $("kb4").innerHTML = docs.map((d) => `<li><b>${esc(d.title)}</b>${d.custom ? ' <span class="text-indigo-600">(eigenes)</span>' : ""}: ${esc(d.text)}</li>`).join("");
  } catch (e) { /* Status-Banner zeigt Probleme bereits an */ }
}

$("f4add").onsubmit = async (ev) => {
  ev.preventDefault();
  if (!$("title4add").value.trim() || !$("text4add").value.trim()) return;
  busy(ev.target, true);
  try {
    await api("/api/knowledge/add", { title: $("title4add").value, text: $("text4add").value });
    $("title4add").value = $("text4add").value = "";
    loadKnowledge(); loadStatus();
  } catch (e) { showError(e); } finally { busy(ev.target, false); }
};

$("reset").onclick = async () => {
  if (!confirm("Alle selbst hinzugefügten Sätze und Dokumente löschen?")) return;
  try {
    await api("/api/custom", null, "DELETE");
    spaceLoaded = false;
    loadKnowledge(); loadStatus();
  } catch (e) { showError(e); }
};

function scoreBar(score) {
  const w = Math.max(0, Math.min(1, score)) * 100;
  return `<div class="h-1.5 bg-slate-100 rounded mt-1"><div class="h-1.5 bg-indigo-500 rounded" style="width:${w}%"></div></div>`;
}

function renderAnswer(text) {
  $("answer").innerHTML = esc(text).replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>');
}

const handlers4 = {
  embedding(d) {
    $("ms4a").textContent = `· ${d.dims} Dimensionen · ${d.ms} ms`;
    stripe("heat4", [d.vector], [""], 90);
  },
  retrieval(d) {
    $("ms4b").textContent = `· ${d.ms} ms`;
    $("sources4").innerHTML = d.sources.map((s, i) => `
      <li class="border border-slate-200 rounded-lg p-3 text-sm">
        <div class="flex justify-between gap-2"><b>[${i + 1}] ${esc(s.title)}</b><span class="text-indigo-700 font-semibold">${pct(s.score)}</span></div>
        ${scoreBar(s.score)}<p class="text-slate-600 mt-1">${esc(s.text)}</p></li>`).join("");
    $("keyword4").innerHTML = d.keyword.length
      ? d.keyword.map((s) => `
        <li class="border border-dashed border-slate-300 rounded-lg p-3 text-sm text-slate-600">
          <b>${esc(s.title)}</b> <span class="text-xs">Treffer: ${s.matched.map(esc).join(", ")}</span></li>`).join("")
      : '<li class="text-sm text-slate-500">Kein Dokument enthält eines der Suchwörter.</li>';
    $("prompt4").textContent = `SYSTEM:\n${d.prompt.system}\n\nUSER:\n${d.prompt.user}`;
  },
  delta(d) { answerText += d.text; renderAnswer(answerText); },
  done(d) { $("note4").textContent = d.note + (d.ms ? ` · ${d.ms} ms` : ""); },
  error(d) { $("note4").textContent = `Fehler: ${d.detail}`; },
};
let answerText = "";

$("f4").onsubmit = async (ev) => {
  ev.preventDefault();
  busy(ev.target, true);
  answerText = "";
  renderAnswer("");
  $("note4").textContent = "";
  ["sources4", "keyword4", "prompt4", "ms4a", "ms4b"].forEach((id) => ($(id).textContent = ""));
  $("r4").classList.remove("hidden");
  const ragBody = { question: $("t4").value, top_k: Number($("k4").value) };
  const ragEvents = [];
  try {
    const res = await fetch("/api/rag", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ragBody),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        const event = raw.match(/^event: (.*)$/m)?.[1];
        const data = raw.match(/^data: (.*)$/m)?.[1];
        if (event && data && handlers4[event]) {
          const parsed = JSON.parse(data);
          ragEvents.push({ event, data: parsed });
          handlers4[event](parsed);
        }
      }
    }
    recordCall("m4", { method: "POST", path: "/api/rag", body: ragBody, events: mergeDeltas(ragEvents) });
  } catch (e) { showError(e); } finally { busy(ev.target, false); }
};

// ---------- Modul 5: Trainingsdaten (FIT) ----------

let workoutsLoaded = false;
let metricDims = [];
let currentWorkout = null;
const SPORT_DE = { cycling: "Rad", running: "Lauf", swimming: "Schwimmen" };
const fmtDate = (iso) => new Date(iso).toLocaleDateString("de-DE", { day: "2-digit", month: "2-digit", year: "numeric" });
const fmtDur = (s) => `${Math.floor(s / 3600)}:${String(Math.round((s % 3600) / 60)).padStart(2, "0")} h`;
const fmtPace = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

async function initWorkouts() {
  try {
    const meta = await api("/api/workouts/meta");
    metricDims = meta.dimensions;
    const d = meta.defaults;
    $("ftp5").value = d.ftp; $("hrmax5").value = d.hr_max; $("lthr5").value = d.lthr; $("pace5").value = fmtPace(d.threshold_pace);
    workoutsLoaded = true;
    await loadWorkouts();
  } catch (e) { showError(e); }
}

async function loadWorkouts() {
  const items = await api("/api/workouts");
  $("count5").textContent = `· ${items.length} in Qdrant`;
  $("list5").innerHTML = items.map((w) => `
    <tr class="border-t hover:bg-indigo-50 cursor-pointer ${currentWorkout === w.id ? "bg-indigo-50" : ""}" data-id="${w.id}">
      <td class="py-1.5 pr-2 whitespace-nowrap">${fmtDate(w.timestamp)}</td>
      <td class="pr-2">${esc(w.title)} <span class="text-xs text-slate-400">${SPORT_DE[w.sport] || esc(w.sport)}${w.source === "demo" ? " · Demo" : ""}</span></td>
      <td class="text-right whitespace-nowrap">${fmtDur(w.metrics.moving_s)}</td>
      <td class="text-right">${w.metrics.tss != null ? fmt(w.metrics.tss, 0) : "–"}</td>
    </tr>`).join("");
  $("list5").querySelectorAll("tr").forEach((tr) => (tr.onclick = () => showWorkout(tr.dataset.id)));
}

function resultItem(r, metric) {
  const score = metric ? `Abstand ${fmt(r.score, 3)}` : `Kosinus ${fmt(r.score, 3)}`;
  return `<li class="border border-slate-200 rounded-lg p-2 cursor-pointer hover:bg-indigo-50" data-id="${r.id}">
    <div class="flex justify-between gap-2"><b>${esc(r.title)}</b><span class="text-indigo-700 whitespace-nowrap">${score}</span></div>
    <div class="text-xs text-slate-500">${fmtDate(r.timestamp)} · ${SPORT_DE[r.sport] || esc(r.sport)} · TSS ${r.tss != null ? fmt(r.tss, 0) : "–"}</div></li>`;
}

function bindResults(el) {
  el.querySelectorAll("li[data-id]").forEach((li) => (li.onclick = () => showWorkout(li.dataset.id)));
}

async function showWorkout(id) {
  try {
    const [w, sim] = await Promise.all([api(`/api/workouts/${id}`), api(`/api/workouts/${id}/similar`)]);
    renderWorkout(w);
    $("simm5").innerHTML = sim.metrics.map((r) => resultItem(r, true)).join("");
    $("sims5").innerHTML = sim.semantic.map((r) => resultItem(r, false)).join("");
    bindResults($("simm5")); bindResults($("sims5"));
    currentWorkout = id;
    loadWorkouts();
    $("detail5").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) { showError(e); }
}

function renderWorkout(w) {
  const m = w.metrics;
  $("detail5").classList.remove("hidden");
  $("title5").textContent = `${w.title} – ${fmtDate(w.timestamp)}`;
  const tiles = [
    ["Dauer", fmtDur(m.moving_s)],
    ["Distanz", `${fmt(m.distance_km, 1)} km`],
    ["TSS", m.tss != null ? fmt(m.tss, 0) : "–"],
    ["IF", m.intensity_factor != null ? fmt(m.intensity_factor, 2) : "–"],
    m.normalized_power_w ? ["NP", `${m.normalized_power_w} W`] : m.avg_pace_s_per_km ? ["Pace", `${fmtPace(m.avg_pace_s_per_km)} /km`] : ["Ø Tempo", m.avg_speed_kmh ? `${fmt(m.avg_speed_kmh, 1)} km/h` : "–"],
    ["Ø / max HF", m.avg_hr ? `${m.avg_hr} / ${m.max_hr}` : "–"],
    ["HF-Drift", m.hr_drift_percent != null ? `${m.hr_drift_percent > 0 ? "+" : ""}${fmt(m.hr_drift_percent, 1)} %` : "–"],
    ["Höhenmeter", m.elevation_gain_m != null ? `${m.elevation_gain_m} m` : "–"],
  ];
  $("tiles5").innerHTML = tiles.map(([k, v]) => `<div class="bg-slate-50 rounded-lg p-2"><div class="text-xs text-slate-500">${k}</div><div class="font-semibold">${v}</div></div>`).join("");

  const sr = w.series, traces = [];
  if (sr.power) traces.push({ x: sr.t_min, y: sr.power, name: "Leistung (W)", line: { color: "#4f46e5", width: 1 } });
  else if (sr.speed) traces.push({ x: sr.t_min, y: sr.speed.map((v) => v * 3.6), name: "Tempo (km/h)", line: { color: "#4f46e5", width: 1 } });
  if (sr.hr) traces.push({ x: sr.t_min, y: sr.hr, name: "Herzfrequenz (bpm)", yaxis: "y2", line: { color: "#dc2626", width: 1 } });
  Plotly.react("chart5", traces.map((t) => ({ ...t, type: "scatter", mode: "lines" })), {
    margin: { l: 45, r: 45, t: 10, b: 35 }, height: 250, separators: ",.", legend: { orientation: "h", y: 1.12 },
    xaxis: { title: { text: "Minuten", font: { size: 11 } } },
    yaxis: { title: { text: traces[0]?.name || "", font: { size: 11 } } },
    yaxis2: { overlaying: "y", side: "right", showgrid: false, title: { text: "HF", font: { size: 11 } } },
    shapes: (m.intervals || []).map((iv) => ({ type: "rect", xref: "x", yref: "paper", x0: iv.start_s / 60, x1: (iv.start_s + iv.duration_s) / 60, y0: 0, y1: 1, fillcolor: "#fde68a", opacity: 0.35, line: { width: 0 } })),
  }, PLOT_CFG);

  const mv = w.vectors.dense_metrics;
  Plotly.react("mvec5", [{
    type: "bar", orientation: "h", x: mv.slice().reverse(), y: metricDims.map(([n]) => n).reverse(),
    customdata: metricDims.map(([, r]) => r).reverse(), marker: { color: "#4f46e5" },
    hovertemplate: "%{y}: %{x:.3f}<br>Skala: %{customdata}<extra></extra>",
  }], { margin: { l: 95, r: 10, t: 5, b: 30 }, height: 310, separators: ",.", xaxis: { range: [0, 1] } }, PLOT_CFG);

  $("summary5").textContent = w.summary_text;
  stripe("svec5", [w.vectors.dense_semantic], [""], 80);
  if (m.zones) {
    Plotly.react("zones5", [{
      type: "bar", x: ["Z1", "Z2", "Z3", "Z4", "Z5"], y: m.zones.map((z) => z * 100),
      marker: { color: ["#94a3b8", "#60a5fa", "#34d399", "#fbbf24", "#f87171"] }, hovertemplate: "%{x}: %{y:.1f} %<extra></extra>",
    }], { margin: { l: 35, r: 10, t: 25, b: 25 }, height: 150, separators: ",.", title: { text: `Zonenverteilung nach ${m.zone_basis}`, font: { size: 12 } }, yaxis: { ticksuffix: " %" } }, PLOT_CFG);
  } else Plotly.purge("zones5");
}

function paceToSeconds(text) {
  const [mm, ss] = text.split(":").map(Number);
  return mm * 60 + (ss || 0);
}

$("f5").onsubmit = async (ev) => {
  ev.preventDefault();
  const file = $("file5").files[0];
  if (!file) return showError(new Error("Bitte eine .FIT-Datei auswählen."));
  busy(ev.target, true);
  try {
    const fields = { notes: $("notes5").value, ftp: $("ftp5").value, hr_max: $("hrmax5").value, lthr: $("lthr5").value, threshold_pace: paceToSeconds($("pace5").value) };
    const form = new FormData();
    form.append("file", file);
    Object.entries(fields).forEach(([k, v]) => form.append(k, v));
    const res = await fetch("/api/workouts/upload-fit", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.statusText);
    recordCall("m5", { method: "POST", path: "/api/workouts/upload-fit", form: { file: file.name, ...fields }, response: data });
    await showWorkout(data.id);
    loadStatus();
  } catch (e) { showError(e); } finally { busy(ev.target, false); }
};

$("del5").onclick = async () => {
  if (!currentWorkout || !confirm("Diese Einheit aus Qdrant löschen?")) return;
  try {
    await api(`/api/workouts/${currentWorkout}`, null, "DELETE");
    currentWorkout = null;
    $("detail5").classList.add("hidden");
    loadWorkouts(); loadStatus();
  } catch (e) { showError(e); }
};

const EXAMPLES5 = ["Harte Intervalle, bei denen ich müde war", "Lange lockere Ausfahrt", "Hitze und zu wenig getrunken", "Lauftraining auf der Bahn", "Persönliche Bestzeit"];
EXAMPLES5.forEach((q) => {
  const chip = document.createElement("span");
  chip.className = "chip bg-slate-100 hover:bg-indigo-100 rounded-full px-3 py-1";
  chip.textContent = q;
  chip.onclick = () => { $("q5").value = q; $("f5s").requestSubmit(); };
  $("examples5").appendChild(chip);
});

$("f5s").onsubmit = async (ev) => {
  ev.preventDefault();
  busy(ev.target, true);
  try {
    const hits = await api("/api/workouts/search", { query: $("q5").value, limit: 5 });
    $("search5").innerHTML = hits.map((r) => resultItem(r, false)).join("");
    bindResults($("search5"));
  } catch (e) { showError(e); } finally { busy(ev.target, false); }
};

async function streamSSE(path, body, onEvent) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      const event = raw.match(/^event: (.*)$/m)?.[1];
      const data = raw.match(/^data: (.*)$/m)?.[1];
      if (event && data) onEvent(event, JSON.parse(data));
    }
  }
}

const EXAMPLESC5 = ["Bei welchen Einheiten ist meine Herzfrequenz stark gedriftet und woran lag es?", "Wie verteilt sich meine Intensität zwischen locker und hart?", "Welche Einheit war die härteste und warum?"];
EXAMPLESC5.forEach((q) => {
  const chip = document.createElement("span");
  chip.className = "chip bg-slate-100 hover:bg-indigo-100 rounded-full px-3 py-1";
  chip.textContent = q;
  chip.onclick = () => { $("qc5").value = q; $("f5c").requestSubmit(); };
  $("examplesc5").appendChild(chip);
});

$("f5c").onsubmit = async (ev) => {
  ev.preventDefault();
  busy(ev.target, true);
  $("coach5").classList.remove("hidden");
  ["coachsrc5", "coachans5", "coachnote5", "coachprompt5"].forEach((id) => ($(id).textContent = ""));
  const body = { question: $("qc5").value, top_k: 4 };
  const events = [];
  let text = "";
  try {
    await streamSSE("/api/workouts/coach", body, (event, d) => {
      events.push({ event, data: d });
      if (event === "retrieval") {
        $("coachsrc5").innerHTML = `<li class="font-medium text-slate-700">Gefundene Einheiten (${d.ms} ms):</li>` +
          d.sources.map((s, i) => `<li>[${i + 1}] ${esc(s.title)} · Kosinus ${fmt(s.score, 3)}</li>`).join("");
        $("coachprompt5").textContent = `SYSTEM:\n${d.prompt.system}\n\nUSER:\n${d.prompt.user}`;
      } else if (event === "delta") {
        text += d.text;
        $("coachans5").innerHTML = esc(text).replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>');
      } else if (event === "done") {
        $("coachnote5").textContent = d.note + (d.ms ? ` · ${d.ms} ms` : "");
      } else if (event === "error") {
        $("coachnote5").textContent = `Fehler: ${d.detail}`;
      }
    });
    recordCall("m5", { method: "POST", path: "/api/workouts/coach", body, events: mergeDeltas(events) });
  } catch (e) { showError(e); } finally { busy(ev.target, false); }
};

// ---------- Modul 6: Agentic RAG mit LangGraph ----------

let graphInfo = null;
const TOOL_DE = {
  suche_wissensbasis: "Wissensbasis durchsuchen",
  suche_trainings: "Trainings semantisch suchen",
  filtere_trainings: "Trainings filtern (Payload)",
  aehnliche_trainings: "Ähnliche Trainings (Kennzahlen)",
};

// mermaid.render darf nicht parallel laufen – Aufrufe werden hintereinander ausgeführt
let graphRender = Promise.resolve();
function renderGraph(visited) {
  const snapshot = new Set(visited);
  graphRender = graphRender.then(() => drawGraph(snapshot)).catch((e) => console.warn("Graph", e));
  return graphRender;
}

async function drawGraph(visited) {
  if (!graphInfo || !window.mermaid) return;
  let text = graphInfo.mermaid;
  if (visited.size) {
    text += "\n\tclassDef visited fill:#c7d2fe,stroke:#4f46e5,stroke-width:2px";
    text += `\n\tclass ${[...visited].join(",")} visited`;
  }
  const { svg } = await mermaid.render(`g6_${Date.now()}`, text);
  $("graph6").innerHTML = svg;
}

async function initAgent() {
  try {
    graphInfo = await api("/api/agent/graph");
    if (window.mermaid) mermaid.initialize({ startOnLoad: false, theme: "neutral" });
    await renderGraph(new Set());
    $("planner6").textContent = `Planer im Knoten „agent“: ${graphInfo.planner} · höchstens ${graphInfo.max_tool_rounds} Werkzeugrunden, danach erzwingt „final“ eine Antwort.`;
    $("tools6").innerHTML = graphInfo.tools.map((t) => `<li><code class="text-indigo-700 font-semibold">${esc(t.name)}</code><br>${esc(t.description)}</li>`).join("");
  } catch (e) { showError(e); }
}

const EXAMPLES6 = [
  "Was war meine härteste Einheit und welche Einheiten hatten eine ähnliche Belastung?",
  "Wann starten die Vereinsausfahrten und welche meiner Einheiten war eine lockere Grundlage?",
  "Bei welchen Einheiten ist mein Puls gedriftet?",
  "Was ist ein HNSW-Index?",
];
EXAMPLES6.forEach((q) => {
  const chip = document.createElement("span");
  chip.className = "chip bg-slate-100 hover:bg-indigo-100 rounded-full px-3 py-1";
  chip.textContent = q;
  chip.onclick = () => { $("q6").value = q; $("f6").requestSubmit(); };
  $("examples6").appendChild(chip);
});

function traceCard(t, ms) {
  const time = `<span class="text-xs text-slate-400">${ms} ms</span>`;
  if (t.node === "agent") {
    const head = `<b>agent</b> · Planer: ${esc(t.planner || "")}`;
    const body = t.decision === "tools"
      ? `entscheidet sich für ${t.calls.length} Werkzeugaufruf${t.calls.length > 1 ? "e" : ""}:` +
        t.calls.map((c) => `<div class="mt-1"><code class="text-indigo-700">${esc(c.name)}</code> <code class="text-xs text-slate-500">${esc(JSON.stringify(c.input))}</code></div>`).join("") +
        (t.text ? `<div class="mt-1 text-slate-500 italic">${esc(t.text)}</div>` : "")
      : "entscheidet: genug Informationen → antworten";
    return `<li class="border-l-4 border-indigo-500 bg-indigo-50 rounded-r-lg p-3 text-sm"><div class="flex justify-between">${head}${time}</div><div class="mt-1">${body}</div></li>`;
  }
  if (t.node === "tools") {
    const hits = t.hits.length
      ? t.hits.map((h) => `<div>[${h.n}] ${esc(h.title)} <span class="text-xs text-slate-400">${esc(h.kind)}</span></div>`).join("")
      : '<div class="text-slate-500">keine Treffer</div>';
    return `<li class="border-l-4 border-emerald-500 bg-emerald-50 rounded-r-lg p-3 text-sm ml-4"><div class="flex justify-between"><span><b>tools</b> · ${esc(TOOL_DE[t.tool] || t.tool)}</span>${time}</div><div class="mt-1">${hits}</div></li>`;
  }
  return `<li class="border-l-4 border-amber-500 bg-amber-50 rounded-r-lg p-3 text-sm"><div class="flex justify-between"><b>final</b>${time}</div><div class="mt-1">Werkzeuglimit erreicht – Antwort wird erzwungen.</div></li>`;
}

$("f6").onsubmit = async (ev) => {
  ev.preventDefault();
  if (!graphInfo) await initAgent();
  busy(ev.target, true);
  $("run6").classList.remove("hidden");
  ["trace6", "answer6", "sources6", "ms6"].forEach((id) => ($(id).textContent = ""));
  const visited = new Set(["__start__"]);
  await renderGraph(visited);
  const body = { question: $("q6").value };
  const events = [];
  const sources = [];
  let text = "";
  const showAnswer = (t) => ($("answer6").innerHTML = esc(t).replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>'));
  try {
    await streamSSE("/api/agent/run", body, (event, d) => {
      events.push({ event, data: d });
      if (event === "delta") {
        text += d.text;
        showAnswer(text);
      } else if (event === "node") {
        visited.add(d.node);
        renderGraph(visited);
        d.trace.forEach((t) => {
          $("trace6").insertAdjacentHTML("beforeend", traceCard(t, d.ms));
          // Text vor einem Werkzeugaufruf gehört zum Schritt, nicht zur Antwort
          if (t.node === "agent" && t.decision === "tools") { text = ""; showAnswer(""); }
        });
        sources.push(...d.sources);
        if (d.answer) showAnswer(d.answer);
      } else if (event === "done") {
        visited.add("__end__");
        renderGraph(visited);
        $("ms6").textContent = `· ${d.ms} ms gesamt`;
      } else if (event === "error") {
        $("answer6").textContent = `Fehler: ${d.detail}`;
      }
    });
    $("sources6").innerHTML = sources.map((s) => `<li class="border border-slate-200 rounded p-2"><b>[${s.n}] ${esc(s.title)}</b> <span class="text-slate-400">${esc(s.kind)}</span><div class="text-slate-600 whitespace-pre-wrap">${esc(s.text || "")}</div></li>`).join("");
    recordCall("m6", { method: "POST", path: "/api/agent/run", body, events: mergeDeltas(events) });
  } catch (e) { showError(e); } finally { busy(ev.target, false); }
};

// ---------- Code-Ansicht für Entwickler ----------

const CALL_MODULE = { "/api/embed": "m1", "/api/compare": "m2", "/api/space/query": "m3", "/api/workouts/search": "m5" };
const lastCall = {};
const codeCache = {};
const panelTab = {};

function recordCall(mod, call) {
  lastCall[mod] = call;
  if (panelTab[mod] === "api") renderPanel(mod);
}

// Viele Text-Deltas zu einem Ereignis zusammenfassen, damit die Anzeige lesbar bleibt
function mergeDeltas(events) {
  const out = [];
  for (const e of events) {
    const prev = out[out.length - 1];
    if (e.event === "delta" && prev?.event === "delta") prev.data = { text: prev.data.text + e.data.text, chunks: (prev.data.chunks || 1) + 1 };
    else out.push({ event: e.event, data: { ...e.data } });
  }
  return out;
}

// Lange Zahlenarrays (Vektoren) kürzen
function compact(value) {
  return JSON.stringify(value, (key, v) =>
    Array.isArray(v) && v.length > 6 && v.every((x) => typeof x === "number")
      ? `[${v.slice(0, 5).map((x) => Number(x.toFixed(5))).join(", ")}, … ${v.length - 5} weitere Werte]`
      : v, 2);
}

function codeBlock(code, lang) {
  const id = `c${Math.random().toString(36).slice(2)}`;
  return `<div class="relative mt-2">
    <button data-copy="${id}" class="copy absolute right-2 top-2 text-xs bg-slate-700 hover:bg-slate-600 text-white rounded px-2 py-1">Kopieren</button>
    <pre><code id="${id}" class="language-${lang}">${esc(code)}</code></pre></div>`;
}

function curlFor(call) {
  if (call.form) {
    const fields = Object.entries(call.form).map(([k, v]) => ` \\\n  -F ${k === "file" ? `'file=@${v}'` : `'${k}=${String(v).replace(/'/g, "'\\''")}'`}`).join("");
    return `curl -s -X POST ${location.origin}${call.path}${fields}`;
  }
  const body = call.body ? ` \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(call.body).replace(/'/g, "'\\''")}'` : "";
  return `curl -s${call.events ? "N" : ""} -X ${call.method} ${location.origin}${call.path}${body}`;
}

function renderPanel(mod) {
  const panel = document.querySelector(`.code-panel[data-module=${mod}] .code-content`);
  const tab = panelTab[mod];
  const data = codeCache[mod];
  let html = "";
  if (tab === "minimal") {
    html = `<p class="text-sm text-slate-600">Eigenständiges Skript, das den Kern dieses Moduls ohne das restliche Projekt nachbaut.</p>${codeBlock(data.minimal, "python")}`;
  } else if (tab === "backend") {
    html = `<p class="text-sm text-slate-600">Der Code, der in diesem Dashboard gerade läuft – live aus den Quelldateien gelesen.</p>` +
      data.sources.map((src) => `
        <div class="mt-4">
          <div class="flex flex-wrap items-baseline gap-x-2 text-sm"><code class="font-semibold text-indigo-700">${esc(src.name)}</code><span class="text-xs text-slate-500">${esc(src.file)}:${src.line}</span></div>
          <p class="text-sm text-slate-600">${esc(src.explanation)}</p>
          ${codeBlock(src.code, "python")}
        </div>`).join("");
  } else {
    const call = lastCall[mod];
    if (!call) {
      html = `<p class="text-sm text-slate-500">Führe das Modul einmal aus – dann erscheint hier der exakte HTTP-Aufruf mit deiner Eingabe und die Antwort des Backends.</p>`;
    } else {
      const resp = call.events
        ? call.events.map((e) => `event: ${e.event}\ndata: ${compact(e.data)}`).join("\n\n")
        : compact(call.response);
      html = `<p class="text-sm text-slate-600">Diesen Aufruf hat das Frontend gerade an das Backend geschickt. Auf dem Server ausführbar (Vektoren in der Antwort gekürzt).</p>
        ${codeBlock(curlFor(call), "bash")}
        <h4 class="text-sm font-medium mt-4">Antwort${call.events ? " (Server-Sent Events – zur Lesbarkeit formatiert, Text-Deltas zusammengefasst)" : ""}</h4>
        ${codeBlock(resp, call.events ? "plaintext" : "json")}`;
    }
  }
  panel.innerHTML = html;
  panel.querySelectorAll("pre code").forEach((el) => window.hljs && hljs.highlightElement(el));
  panel.querySelectorAll(".copy").forEach((btn) => (btn.onclick = async () => {
    await navigator.clipboard.writeText($(btn.dataset.copy).textContent);
    btn.textContent = "Kopiert ✓";
    setTimeout(() => (btn.textContent = "Kopieren"), 1500);
  }));
  document.querySelectorAll(`.code-panel[data-module=${mod}] .code-tab`).forEach((b) => b.classList.toggle("active", b.dataset.ctab === tab));
}

document.querySelectorAll(".code-panel").forEach((panel) => {
  const mod = panel.dataset.module;
  panel.innerHTML = `
    <details class="bg-white rounded-xl shadow-sm">
      <summary class="p-5 cursor-pointer font-semibold select-none"><span class="font-mono text-indigo-600">&lt;/&gt;</span> Code hinter diesem Modul</summary>
      <div class="px-5 pb-5">
        <div class="flex flex-wrap gap-2 text-sm">
          <button data-ctab="minimal" class="code-tab px-3 py-1.5 rounded-lg bg-slate-100">Minimalbeispiel</button>
          <button data-ctab="backend" class="code-tab px-3 py-1.5 rounded-lg bg-slate-100">Backend (dieses Projekt)</button>
          <button data-ctab="api" class="code-tab px-3 py-1.5 rounded-lg bg-slate-100">API-Aufruf</button>
        </div>
        <div class="code-content mt-3"></div>
      </div>
    </details>`;
  const open = async (tab) => {
    panelTab[mod] = tab;
    try {
      codeCache[mod] ||= await api(`/api/code/${mod}`);
      renderPanel(mod);
    } catch (e) { showError(e); }
  };
  panel.querySelector("details").addEventListener("toggle", (ev) => { if (ev.target.open && !panelTab[mod]) open("minimal"); });
  panel.querySelectorAll(".code-tab").forEach((b) => (b.onclick = () => open(b.dataset.ctab)));
});

showTab("m1");
loadStatus();
