import { createModel, predict, spanishReason } from "./inference.js";

const els = {
  input: document.querySelector("#case-text"),
  button: document.querySelector("#classify-btn"),
  clear: document.querySelector("#clear-btn"),
  status: document.querySelector("#model-status"),
  result: document.querySelector("#result"),
  empty: document.querySelector("#empty-state"),
  badge: document.querySelector("#result-badge"),
  title: document.querySelector("#result-title"),
  probability: document.querySelector("#probability"),
  probabilityBar: document.querySelector("#probability-bar"),
  threshold: document.querySelector("#threshold"),
  thresholdMarker: document.querySelector("#threshold-marker"),
  explanation: document.querySelector("#explanation"),
  meta: document.querySelector("#result-meta"),
  examples: document.querySelectorAll("[data-example]"),
  oodNote: document.querySelector("#ood-note"),
};

let model = null;

// Contadores de sesión (se Reinician al cargar un nuevo modelo)
let sessionClassified = 0;
let sessionFinance = 0;
let sessionMarket = 0;
let sessionReject = 0;

const EXAMPLES = {
  finance:
    "La inversión inicial del proyecto es de USD 1,250,000 e incluye la compra de maquinaria, el capital de trabajo y los costos de instalación. El flujo de efectivo proyectado a cinco años tiene un margen bruto del 32%, un VAN positivo de USD 410,000 y una TIR del 18.4%. El punto de equilibrio se alcanza en el mes 14. Los principales riesgos son el aumento del costo de los materiales y la demora en la cadena de suministro; se propone una contingencia del 8% del presupuesto.",
  market:
    "El cliente objetivo son resorts de aventura en la península de Yucatán que necesitan una flota de vehículos todo terreno para tours guiados. El mercado de turismo de aventura crece 7% anual y los competidores principales son Polaris, Can-Am y proveedores locales de renta. La propuesta se dirige a gerentes de operaciones y dueños de hospedajes boutique. El canal principal será la venta directa con demostraciones en sitio, apoyada por marketing digital y alianzas con operadores turísticos.",
  oodEnglish:
    "Hello, I would like to book a flight to New York next week. Please confirm the price and the baggage allowance for two passengers.",
  oodRecipe:
    "Preheat the oven to 180 degrees. Mix flour, sugar and butter, then bake the cake for forty minutes until golden. Let it cool before serving.",
};

function setStatus(text, kind = "loading") {
  els.status.textContent = text;
  els.status.dataset.kind = kind;
}

async function loadModel() {
  try {
    setStatus("Cargando modelo…", "loading");
    // Reiniciar contadores de sesión al cargar un nuevo modelo
    sessionClassified = 0;
    sessionFinance = 0;
    sessionMarket = 0;
    sessionReject = 0;
    const setSession = (id, value) => {
      const el = document.querySelector(id);
      if (el) el.textContent = value;
    };
    setSession("#session-count", 0);
    setSession("#session-finance", 0);
    setSession("#session-market", 0);
    setSession("#session-reject", 0);
    const response = await fetch("./model.json?v=1", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const raw = await response.json();
    model = createModel(raw);
    const summary = model.metrics_summary || {};
    updateLabelCounter(summary);
    const suffix = model.provisional ? " · evaluación provisional" : "";
    const trained = summary.train_size ?? "?";
    setStatus(`Entrenado con ${trained} fragmentos${suffix}`, "ready");
    const trainNote = document.querySelector("#train-size");
    if (trainNote) {
      const total = summary.fragment_count ?? "?";
      const pos = summary.train_positive ?? "?";
      const neg = summary.train_negative ?? "?";
      const augmented = summary.train_augmented ?? 0;
      trainNote.textContent =
        `El modelo se entrenó con ${trained} fragmentos (${pos} finanzas/riesgo y ${neg} mercado/propuesta). ` +
        `Incluye ${augmented} fragmentos extra de los mismos equipos de entrenamiento. ` +
        `Validación: ${summary.val_size ?? "?"}. Test no usado para entrenar: ${summary.test_size ?? "?"}. ` +
        `Dataset base: ${total} fragmentos.`;
    }
    els.button.disabled = false;
    if (els.input.value.trim()) classify();
  } catch (error) {
    console.error(error);
    setStatus("No se pudo cargar el modelo. Recarga la página.", "error");
    els.button.disabled = true;
  }
}

function formatPct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function updateLabelCounter(summary) {
  const total = summary.total_with_augmentation ?? summary.fragment_count ?? 0;
  const pos =
    (summary.train_positive ?? 0) + (summary.val_positive ?? 0) + (summary.test_positive ?? 0);
  const neg = total - pos;
  const set = (id, value) => {
    const el = document.querySelector(id);
    if (el) el.textContent = value;
  };
  set("#label-count", total);
  set("#count-train", summary.train_size ?? "—");
  set("#count-val", summary.val_size ?? "—");
  set("#count-test", summary.test_size ?? "—");
  set("#balance-text", `${pos} finanzas/riesgo · ${neg} mercado/propuesta`);
  const bar = document.querySelector("#balance-pos");
  if (bar && total > 0) bar.style.width = `${((pos / total) * 100).toFixed(1)}%`;
}

function render(result) {
  els.empty.hidden = true;
  els.result.hidden = false;
  els.oodNote.hidden = true;

  const p = result.probability;
  const t = result.threshold;
  els.probability.textContent = formatPct(p);
  els.threshold.textContent = formatPct(t);
  els.probabilityBar.style.width = `${Math.max(0, Math.min(100, p * 100))}%`;
  els.thresholdMarker.style.left = `${Math.max(0, Math.min(100, t * 100))}%`;
  els.meta.textContent = `${result.tokenCount} tokens · ${result.featureCount} features activas`;

  if (result.rejectReason) {
    els.badge.textContent = "No clasificable";
    els.badge.dataset.kind = "reject";
    els.title.textContent = "Entrada fuera de dominio";
    els.explanation.textContent = spanishReason(result.rejectReason);
    els.oodNote.hidden = false;
    els.oodNote.textContent =
      "La app prefiere rechazar antes que forzar una clase. Prueba con un fragmento de un business case.";
    return;
  }

  const isFinance = result.label === 1;
  els.badge.textContent = isFinance ? "Finanzas / riesgo" : "Mercado / propuesta";
  els.badge.dataset.kind = isFinance ? "finance" : "market";
  els.title.textContent = isFinance
    ? "El fragmento parece financiero o de riesgo"
    : "El fragmento parece de mercado o propuesta";
  els.explanation.textContent = isFinance
    ? "El modelo encontró vocabulario asociado con inversión, ingresos, costos, rentabilidad o riesgos."
    : "El modelo encontró vocabulario asociado con cliente, mercado, competencia, propuesta, operación o marketing.";

  if (result.reviewReason) {
    els.oodNote.hidden = false;
    els.oodNote.textContent =
      "La probabilidad está cerca del umbral. La clase es tentativa; revisa el fragmento antes de decidir.";
  }
}

function classify() {
  if (!model) return;
  const text = els.input.value;
  const result = predict(model, text);
  render(result);

  // Actualizar contador de sesión
  sessionClassified++;
  const kind = els.badge.dataset.kind;
  if (kind === 'finance') sessionFinance++;
  else if (kind === 'market') sessionMarket++;
  else if (kind === 'reject') sessionReject++;
  const setSession = (id, value) => {
    const el = document.querySelector(id);
    if (el) el.textContent = value;
  };
  setSession("#session-count", sessionClassified);
  setSession("#session-finance", sessionFinance);
  setSession("#session-market", sessionMarket);
  setSession("#session-reject", sessionReject);
}

function clearAll() {
  els.input.value = "";
  els.result.hidden = true;
  els.empty.hidden = false;
  els.input.focus();
}

els.button.addEventListener("click", classify);
els.clear.addEventListener("click", clearAll);
els.input.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") classify();
});

for (const button of els.examples) {
  button.addEventListener("click", () => {
    const key = button.dataset.example;
    els.input.value = EXAMPLES[key] || "";
    els.input.focus();
    classify();
  });
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js").catch((error) => console.warn("SW", error));
  });
}

document.querySelector("#open-eval")?.addEventListener("click", () => {
  const popup = window.open(
    "./evaluacion.html",
    "practica1-evaluacion",
    "popup=yes,width=1180,height=820,resizable=yes,scrollbars=yes"
  );
  if (!popup) window.location.href = "./evaluacion.html";
});

window.addEventListener("message", (event) => {
  if (event.origin !== window.location.origin) return;
  if (event.data?.type !== "try-fragment") return;
  els.input.value = event.data.text || "";
  classify();
  els.input.focus();
});

const pending = sessionStorage.getItem("try-fragment");
if (pending) {
  sessionStorage.removeItem("try-fragment");
  els.input.value = pending;
}

loadModel();
