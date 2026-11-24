/**
 * Inferencia pura de la PWA: TF-IDF + MLP de una capa oculta.
 * No usa DOM, para poder probarla con Node.
 */

const TOKEN_RE = /[a-z0-9_]{2,}/g;

export function stripAccents(text) {
  return String(text || "")
    .normalize("NFKD")
    .replace(/\p{M}/gu, "");
}

export function tokenize(text) {
  const normalized = stripAccents(String(text || "").toLowerCase());
  return normalized.match(TOKEN_RE) || [];
}

export function createModel(raw) {
  if (!raw || !raw.vocab || !raw.idf || !raw.w1 || !raw.w2) {
    throw new Error("Modelo inválido");
  }
  const hidden = raw.w1.length ? raw.w1[0].length : 0;
  return {
    ...raw,
    hiddenSize: hidden,
    stopwords: new Set(raw.ood?.spanish_stopwords || []),
  };
}

function featureCounts(tokens) {
  const counts = new Map();
  for (let i = 0; i < tokens.length; i += 1) {
    const uni = tokens[i];
    counts.set(uni, (counts.get(uni) || 0) + 1);
    if (i + 1 < tokens.length) {
      const bi = `${uni} ${tokens[i + 1]}`;
      counts.set(bi, (counts.get(bi) || 0) + 1);
    }
  }
  return counts;
}

function vectorize(model, text) {
  const tokens = tokenize(text);
  const counts = featureCounts(tokens);
  const sparse = [];
  let norm = 0;
  for (const [token, count] of counts.entries()) {
    if (!Object.prototype.hasOwnProperty.call(model.vocab, token)) continue;
    const idx = model.vocab[token];
    const tf = model.preprocess?.sublinear_tf ? 1 + Math.log(count) : count;
    const value = tf * model.idf[idx];
    if (!Number.isFinite(value) || value === 0) continue;
    sparse.push([idx, value]);
    norm += value * value;
  }
  const denom = Math.sqrt(norm) || 1;
  return { tokens, sparse: sparse.map(([idx, value]) => [idx, value / denom]) };
}

function relu(value) {
  return value > 0 ? value : 0;
}

function sigmoid(value) {
  const z = Math.max(-60, Math.min(60, value));
  return 1 / (1 + Math.exp(-z));
}

export function oodDecision(model, text, probability) {
  const trimmed = String(text || "").trim();
  const ood = model.ood || {};
  if (trimmed.length < (ood.min_chars ?? 80)) return "texto_vacio_o_muy_corto";
  const tokens = tokenize(text);
  if (tokens.length < (ood.min_tokens ?? 8)) return "muy_pocas_palabras";
  if (tokens.length >= (ood.spanish_min_tokens ?? 20)) {
    let hits = 0;
    for (const token of tokens) if (model.stopwords.has(token)) hits += 1;
    if (hits < (ood.spanish_min_hits ?? 2)) return "idioma_no_espanol";
  }
  let known = 0;
  for (const token of tokens) {
    if (Object.prototype.hasOwnProperty.call(model.vocab, token)) known += 1;
  }
  const oovRate = 1 - known / Math.max(1, tokens.length);
  if (oovRate > (ood.max_oov_rate ?? 0.6)) return "fuera_de_vocabulario";
  const margin = ood.review_margin ?? 0.08;
  if (Math.abs(probability - Number(model.threshold || 0.5)) <= margin) {
    return "cerca_del_umbral";
  }
  return null;
}

export function predict(model, text) {
  const { tokens, sparse } = vectorize(model, text);
  const hidden = new Array(model.hiddenSize).fill(0);
  for (const [idx, value] of sparse) {
    const row = model.w1[idx];
    if (!row) continue;
    for (let h = 0; h < model.hiddenSize; h += 1) {
      hidden[h] += value * row[h];
    }
  }
  for (let h = 0; h < model.hiddenSize; h += 1) {
    hidden[h] = relu(hidden[h] + (model.b1?.[h] || 0));
  }
  let logit = Number(model.b2 || 0);
  for (let h = 0; h < model.hiddenSize; h += 1) {
    logit += hidden[h] * (model.w2?.[h] || 0);
  }
  const temperature = Number(model.temperature || 1);
  const probability = sigmoid(logit / temperature);
  const threshold = Number(model.threshold || 0.5);
  const reason = oodDecision(model, text, probability);
  const rejected = Boolean(reason && reason !== "cerca_del_umbral");
  return {
    probability,
    threshold,
    label: probability >= threshold ? 1 : 0,
    labelName: probability >= threshold ? "Finanzas / riesgo" : "Mercado / propuesta",
    rejectReason: rejected ? reason : null,
    reviewReason: reason === "cerca_del_umbral" ? reason : null,
    tokenCount: tokens.length,
    featureCount: sparse.length,
  };
}

export function spanishReason(reason) {
  const map = {
    texto_vacio_o_muy_corto: "El texto está vacío o tiene muy poco contenido.",
    muy_pocas_palabras: "Hay muy pocas palabras para clasificar.",
    idioma_no_espanol: "El texto no parece estar en español.",
    fuera_de_vocabulario: "El texto está fuera del dominio del modelo.",
    cerca_del_umbral: "La probabilidad está cerca del umbral; conviene revisarlo.",
  };
  return map[reason] || "Entrada no clasificable.";
}
