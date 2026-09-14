/**
 * Browser-side port of the Python "Convergence" experiment.
 *
 * This mirrors operators.py's scoring math and simulate_convergence.py's
 * round loop, but only against a fixed, precomputed vocabulary -- there is
 * no model inference in the browser. Vectors for each model are precomputed
 * offline by export_web_embeddings.py and simply fetched as JSON here.
 *
 * Only the operators that need nothing but vocabulary vectors are supported
 * (centroid, balanced, mean, geometric_mean). The "textual" operator, which
 * needs to embed an arbitrary phrase at request time, is intentionally not
 * ported since that would require running a model in the browser.
 */

export const OPERATOR_NAMES = ["centroid", "balanced", "mean", "geometric_mean"];

/** Cache of loaded {model -> {words, vectors}} so repeated fetches are avoided. */
const modelCache = new Map();

/**
 * Fetch and parse one model's precomputed vocabulary embeddings.
 *
 * @param {string} modelName - Short model name, matching the JSON file's basename
 *   (e.g. "minilm-l6" loads "data/minilm-l6.json").
 * @param {string} dataDir - Directory (relative to the page) containing the JSON files.
 * @returns {Promise<{model: string, words: string[], vectors: number[][], index: Map<string, number>}>}
 */
export async function loadModelEmbeddings(modelName, dataDir = "data") {
  if (modelCache.has(modelName)) {
    return modelCache.get(modelName);
  }
  const response = await fetch(`${dataDir}/${modelName}.json`);
  if (!response.ok) {
    throw new Error(`Failed to load embeddings for model "${modelName}": ${response.status}`);
  }
  const payload = await response.json();
  payload.index = new Map(payload.words.map((word, i) => [word, i]));
  modelCache.set(modelName, payload);
  return payload;
}

/** Dot product of two equal-length arrays (vectors are pre-normalized, so this is cosine similarity). */
function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function normalize(vector) {
  let normSq = 0;
  for (const x of vector) normSq += x * x;
  const norm = Math.sqrt(normSq);
  if (norm === 0) return vector.slice();
  return vector.map((x) => x / norm);
}

/** Cosine similarity of every vocabulary word to a query vector. */
function similaritiesTo(vectors, query) {
  return vectors.map((row) => dot(row, query));
}

function vectorFor(modelData, word) {
  const idx = modelData.index ? modelData.index.get(word) : modelData.words.indexOf(word);
  return idx === undefined || idx === -1 ? null : modelData.vectors[idx];
}

const PAIRWISE_OPERATORS = {
  balanced: (simA, simB) => simA.map((a, i) => Math.min(a, simB[i])),
  mean: (simA, simB) => simA.map((a, i) => (a + simB[i]) / 2),
  geometric_mean: (simA, simB) => simA.map((a, i) => Math.sqrt(Math.max(a, 0) * Math.max(simB[i], 0))),
};

/**
 * Rank vocabulary words by score, descending, with alphabetical tie-breaking
 * (matching operators.rank_candidates: scores are rounded before comparing
 * so floating-point noise never overrides the alphabetical tie-break).
 */
function rankCandidates(words, scores, simA, simB, exclude, topK) {
  const excludeSet = exclude || new Set();
  const indices = words.map((_, i) => i);
  indices.sort((i, j) => {
    const scoreI = round(scores[i], 9);
    const scoreJ = round(scores[j], 9);
    if (scoreI !== scoreJ) return scoreJ - scoreI;
    return words[i] < words[j] ? -1 : words[i] > words[j] ? 1 : 0;
  });
  const kept = indices.filter((i) => !excludeSet.has(words[i])).slice(0, topK);
  return kept.map((i, rank) => ({
    rank: rank + 1,
    candidate: words[i],
    score: scores[i],
    simA: simA[i],
    simB: simB[i],
  }));
}

function round(value, digits) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

/**
 * Apply one operator for a single model over its precomputed vocabulary.
 *
 * @param {string} operatorName - One of OPERATOR_NAMES.
 * @param {{words: string[], vectors: number[][]}} modelData - Loaded model embeddings.
 * @param {string} wordA - First input word (must be present in modelData.words).
 * @param {string} wordB - Second input word (must be present in modelData.words).
 * @param {{topK?: number, exclude?: Set<string>}} [options]
 */
export function applyOperator(operatorName, modelData, wordA, wordB, options = {}) {
  const { topK = 5, exclude = null } = options;
  if (!OPERATOR_NAMES.includes(operatorName)) {
    throw new Error(`Unknown operator "${operatorName}"; choose from ${OPERATOR_NAMES}`);
  }
  const { words, vectors } = modelData;
  const vecA = vectorFor(modelData, wordA);
  const vecB = vectorFor(modelData, wordB);
  if (!vecA || !vecB) {
    throw new Error(`Both words must be in the vocabulary (got "${wordA}", "${wordB}")`);
  }

  const simA = similaritiesTo(vectors, vecA);
  const simB = similaritiesTo(vectors, vecB);

  let scores;
  if (operatorName === "centroid") {
    const query = normalize(vecA.map((x, i) => x + vecB[i]));
    scores = similaritiesTo(vectors, query);
  } else {
    scores = PAIRWISE_OPERATORS[operatorName](simA, simB);
  }

  return rankCandidates(words, scores, simA, simB, exclude, topK);
}

/**
 * Simulate the Convergence game: two models repeatedly apply the same
 * operator to a pair of words, and their outputs become the next round's
 * pair. Mirrors simulate_convergence.simulate_convergence()'s outcomes:
 * "converged", "loop", "stalled", or "round_limit".
 *
 * @param {{words: string[], vectors: number[][]}} modelDataA
 * @param {{words: string[], vectors: number[][]}} modelDataB
 * @param {string} wordA
 * @param {string} wordB
 * @param {{operator?: string, maxRounds?: number, topK?: number, includeInputs?: boolean}} [options]
 * @returns {{path: object[], outcome: string, termination: object}}
 */
export function simulateConvergence(modelDataA, modelDataB, wordA, wordB, options = {}) {
  const { operator = "centroid", maxRounds = 10, topK = 5, includeInputs = false } = options;

  let currentPair = [wordA, wordB];
  const visited = new Map([[currentPair.join("\u0000"), 0]]);
  const path = [{ round: 0, pairIn: currentPair, outputs: null, pairOut: currentPair }];
  let outcome = "round_limit";
  let termination = { round: maxRounds };

  for (let roundNumber = 1; roundNumber <= maxRounds; roundNumber++) {
    const exclude = includeInputs ? null : new Set(currentPair);
    let outA;
    let outB;
    let candidatesA;
    let candidatesB;
    try {
      candidatesA = applyOperator(operator, modelDataA, currentPair[0], currentPair[1], { topK, exclude });
      candidatesB = applyOperator(operator, modelDataB, currentPair[0], currentPair[1], { topK, exclude });
      if (candidatesA.length === 0 || candidatesB.length === 0) {
        outcome = "stalled";
        termination = { round: roundNumber, reason: "No candidates remained after excluding the current input words." };
        break;
      }
      outA = candidatesA[0].candidate;
      outB = candidatesB[0].candidate;
    } catch (err) {
      outcome = "stalled";
      termination = { round: roundNumber, reason: err.message };
      break;
    }

    const newPair = [outA, outB];
    path.push({ round: roundNumber, pairIn: currentPair, outputs: { A: outA, B: outB }, pairOut: newPair, candidates: { A: candidatesA, B: candidatesB } });

    if (outA === outB) {
      outcome = "converged";
      termination = { round: roundNumber };
      currentPair = newPair;
      break;
    }
    const key = newPair.join("\u0000");
    if (visited.has(key)) {
      outcome = "loop";
      termination = { round: roundNumber, repeatedRound: visited.get(key) };
      currentPair = newPair;
      break;
    }
    visited.set(key, roundNumber);
    currentPair = newPair;
  }

  return { path, outcome, termination };
}
