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
  window.dispatchEvent(new Event("resize"));
}
document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => showTab(t.dataset.tab)));

async function loadStatus() {
  try {
    const s = await api("/api/status");
    const llm = s.llm_available ? `LLM: ${s.llm_model}` : "LLM: nicht konfiguriert";
    $("status").innerHTML =
      `Embeddings: <b>${esc(s.embedding_model)}</b> · ${llm}<br>` +
      `Qdrant: ${s.collections.saetze} Sätze, ${s.collections.wissen} Wissensdokumente`;
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
  try {
    const res = await fetch("/api/rag", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: $("t4").value, top_k: Number($("k4").value) }),
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
        if (event && data && handlers4[event]) handlers4[event](JSON.parse(data));
      }
    }
  } catch (e) { showError(e); } finally { busy(ev.target, false); }
};

showTab("m1");
loadStatus();
