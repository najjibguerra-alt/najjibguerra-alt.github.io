const CATEGORIES = {
  seccion_mixta: "Sección mixta",
  negacion_o_contraste: "Negación o contraste",
  confusion_de_clase: "Clase confundida",
  texto_corto: "Texto corto",
  tabla_numerica: "Tabla numérica",
  cerca_del_umbral: "Cerca del umbral",
};

function money(value) {
  return Number(value).toFixed(1);
}

function labelName(value) {
  return value === 1 ? "Finanzas" : "Mercado";
}

function tryFragment(text) {
  if (window.opener && !window.opener.closed) {
    window.opener.postMessage({ type: "try-fragment", text }, window.location.origin);
    window.opener.focus();
    return;
  }
  sessionStorage.setItem("try-fragment", text);
  window.location.href = "./index.html";
}

function render(data) {
  const note = document.querySelector("#eval-note");
  const dataset = data.dataset || {};
  note.textContent = data.provisional
    ? `Evaluación provisional. Entrenado con ${dataset.train_size ?? "?"} fragmentos; el test aún usa etiquetas débiles.`
    : `Corrida final. Entrenado con ${dataset.train_size ?? "?"} fragmentos.`;

  const costs = data.costs || {};
  document.querySelector("#eval-kpis").innerHTML = [
    ["Entrenamiento", dataset.train_size],
    ["Modelo / 1000", costs.model_per_1000],
    ["Keyword / 1000", costs.keyword_per_1000],
    ["Mayoría / 1000", costs.majority_per_1000],
  ]
    .map(([name, value]) => `<div class="kpi"><span>${name}</span><strong>${name === "Entrenamiento" ? value ?? "—" : money(value)}</strong></div>`)
    .join("");

  const c = data.confusion || {};
  document.querySelector("#confusion").innerHTML = `
    <div></div><div class="mh">Pred. mercado</div><div class="mh">Pred. finanzas</div>
    <div class="mh">Real mercado</div><div>TN ${c.tn ?? "—"}</div><div>FP ${c.fp ?? "—"}</div>
    <div class="mh">Real finanzas</div><div>FN ${c.fn ?? "—"}</div><div>TP ${c.tp ?? "—"}</div>`;

  const cal = data.calibration || {};
  document.querySelector("#calibration").innerHTML = `
    <p>ECE antes: <strong>${Number(cal.ece_before).toFixed(3)}</strong>. ECE después: <strong>${Number(cal.ece_after).toFixed(3)}</strong>. Temperatura: ${Number(cal.temperature).toFixed(2)}.</p>
    <p>Costo/1000 sin calibrar: ${money(costs.raw_per_1000)}. Con calibración y umbral ${money((data.threshold || 0) * 100)}%: ${money(costs.model_per_1000)}.</p>
    <p>La calibración baja el ECE, pero en esta corrida no baja el costo. La probabilidad ordena mejor de lo que estima una frecuencia exacta.</p>`;

  const filters = document.querySelector("#error-filters");
  const list = document.querySelector("#error-list");
  const groups = Object.entries(data.errors_grouped || {});
  filters.innerHTML =
    `<button type="button" data-cat="all" aria-pressed="true">Todos (${(data.errors || []).length})</button>` +
    groups
      .map(([key, count]) => `<button type="button" data-cat="${key}" aria-pressed="false">${CATEGORIES[key] || key} (${count})</button>`)
      .join("");

  function paint(category) {
    const rows = (data.errors || []).filter((row) => category === "all" || row.category === category);
    list.innerHTML = rows
      .map((row, index) => {
        const kind = row.true_label === 1 ? "FN" : "FP";
        return `<article class="error-item">
          <header>
            <span class="tag ${kind === "FN" ? "bad" : "warn"}">${kind} · ${CATEGORIES[row.category] || row.category}</span>
            <button class="load" type="button" data-error="${index}">Probar</button>
          </header>
          <p>${row.hypothesis || ""} Real: ${labelName(row.true_label)}. Predicho: ${labelName(row.predicted_label)} a ${(row.probability * 100).toFixed(1)}% (umbral ${(row.threshold * 100).toFixed(1)}%).</p>
          <p class="quote">${row.text || ""}</p>
        </article>`;
      })
      .join("");
    list.querySelectorAll("[data-error]").forEach((button) => {
      button.addEventListener("click", () => tryFragment(rows[Number(button.dataset.error)]?.text || ""));
    });
  }

  paint("all");
  filters.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      filters.querySelectorAll("button").forEach((item) => item.setAttribute("aria-pressed", "false"));
      button.setAttribute("aria-pressed", "true");
      paint(button.dataset.cat);
    });
  });

  document.querySelector("#ood-list").innerHTML = (data.ood || [])
    .map((row) => {
      const ok = row.rejected;
      return `<article class="ood-item">
        <header>
          <span class="tag ${ok ? "" : "bad"}">${row.id} · ${ok ? "rechazada" : "no rechazada"}</span>
          <span class="meta">${row.actual_reject_reason || "clasificó"} · ${(row.probability * 100).toFixed(1)}%</span>
        </header>
        <p>Esperado: ${row.expected} (${row.reason_expected || "fuera de dominio"}). ${ok ? "La app rechazó la entrada." : "La app no la rechazó; la clasificó igual."}</p>
        <p class="quote">${row.text || "(vacío)"}</p>
      </article>`;
    })
    .join("");
}

fetch("./evaluation.json?v=2", { cache: "no-cache" })
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then(render)
  .catch((error) => {
    console.error(error);
    document.querySelector("#eval-note").textContent = "No se pudo cargar evaluation.json.";
  });
