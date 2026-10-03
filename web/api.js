/**
 * Client for the Convergence FastAPI backend (backend/app.py).
 *
 * All the scoring happens in Python (operators.py); these helpers only fetch
 * results and reshape them into camelCase objects for the pages.
 */

export const OPERATOR_NAMES = ["centroid", "maximin", "product", "textual"];

async function getJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {
      // Not a JSON error body; fall through to the status code.
    }
    throw new Error(typeof detail === "string" ? detail : `Request to ${url} failed (${response.status})`);
  }
  return response.json();
}

const toCandidate = (c) => ({ rank: c.rank, candidate: c.candidate, score: c.score, simA: c.sim_a, simB: c.sim_b });

/** Models that have a vocabulary in the database, as `{name, label, provider}`. */
export async function listModels() {
  return (await getJSON("/api/models")).models;
}

const vocabCache = new Map();

/** The candidate vocabulary for one model (fetched once per model). */
export async function getVocab(model) {
  if (!vocabCache.has(model)) {
    vocabCache.set(model, getJSON(`/api/vocab?${new URLSearchParams({ model })}`).then((data) => data.words));
  }
  return vocabCache.get(model);
}

/** Hand-picked, familiar words for random starting pairs. */
export async function getStartingWords() {
  return (await getJSON("/api/starting-words")).words;
}

/**
 * Rank vocabulary words with one operator. Inputs don't need to be in the
 * vocabulary; the backend embeds them on demand.
 *
 * @returns {Promise<{rank: number, candidate: string, score: number, simA: number, simB: number}[]>}
 */
export async function combine(operator, model, wordA, wordB, { topK = 5, includeInputs = false } = {}) {
  const params = new URLSearchParams({ model, a: wordA, b: wordB, operator, top_k: topK, include_inputs: includeInputs });
  return (await getJSON(`/api/combine?${params}`)).candidates.map(toCandidate);
}

/** Every vocabulary word's similarity to each input, plus the inputs' similarity to each other. */
export async function similarities(model, wordA, wordB) {
  const data = await getJSON(`/api/similarities?${new URLSearchParams({ model, a: wordA, b: wordB })}`);
  return { words: data.words, simA: data.sim_a, simB: data.sim_b, cosAB: data.cos_ab };
}

/**
 * Play the Convergence game on the backend, either between two chosen models
 * or (with `randomModels`) between two random models that change every round.
 *
 * @returns {Promise<{path: object[], outcome: string, termination: {round: number, reason?: string, repeatedRound?: number}}>}
 */
export async function simulateConvergence({ modelA, modelB, randomModels = false, wordA, wordB, operator = "centroid", maxRounds = 10, topK = 5 }) {
  const data = await getJSON("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_a: modelA, model_b: modelB, random_models: randomModels, word_a: wordA, word_b: wordB, operator, max_rounds: maxRounds, top_k: topK }),
  });
  return {
    outcome: data.outcome,
    termination: { round: data.termination.round, reason: data.termination.reason, repeatedRound: data.termination.repeated_round },
    path: data.path.map((step) => ({
      round: step.round,
      pairIn: step.pair_in,
      pairOut: step.pair_out,
      outputs: step.outputs && { A: step.outputs[0], B: step.outputs[1] },
      models: step.models && { A: step.models[0], B: step.models[1] },
      candidates: step.candidates && { A: step.candidates[0].map(toCandidate), B: step.candidates[1].map(toCandidate) },
    })),
  };
}
