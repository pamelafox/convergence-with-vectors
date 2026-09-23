import { loadModelEmbeddings, simulateConvergence } from "./convergence.js";

const partyView = document.getElementById("party-view");
const labView = document.getElementById("lab-view");
const playButton = document.getElementById("party-play");
const skipButton = document.getElementById("party-skip");
const loadingStatus = document.getElementById("party-loading");
const countdown = document.getElementById("party-countdown");
const roundLabel = document.getElementById("party-round");
const prompt = document.getElementById("party-prompt");
const announcer = document.getElementById("party-announcer");
const bubbleA = document.getElementById("party-bubble-a");
const bubbleB = document.getElementById("party-bubble-b");
const playerA = document.getElementById("party-player-a");
const playerB = document.getElementById("party-player-b");
const modelLabelA = document.getElementById("party-model-a");
const modelLabelB = document.getElementById("party-model-b");
const history = document.getElementById("party-history");
const outcome = document.getElementById("party-outcome");
const confettiLayer = document.getElementById("party-confetti");

const COUNTDOWN_DELAY = 560;
const SHOUT_DELAY = 850;
const MAX_ROUNDS = 10;

let models = [];
let vocabulary = [];
let currentRun = 0;
let skipAnimation = false;

function friendlyModelName(name) {
  return name
    .replace("minilm", "MiniLM")
    .replace("-l", " L")
    .replaceAll("-", " ");
}

function setMode(mode) {
  const partyMode = mode === "party";
  partyView.hidden = !partyMode;
  labView.hidden = partyMode;
  for (const button of document.querySelectorAll("[data-mode]")) {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  }
  window.history.replaceState(null, "", partyMode ? "#game-night" : "#vector-lab");
}

for (const button of document.querySelectorAll("[data-mode]")) {
  button.addEventListener("click", () => setMode(button.dataset.mode));
}

function randomPair() {
  const first = Math.floor(Math.random() * vocabulary.length);
  const offset = 1 + Math.floor(Math.random() * (vocabulary.length - 1));
  return [vocabulary[first], vocabulary[(first + offset) % vocabulary.length]];
}

function setBubbles(wordA, wordB, visible = true) {
  bubbleA.textContent = wordA;
  bubbleB.textContent = wordB;
  bubbleA.classList.toggle("show", visible);
  bubbleB.classList.toggle("show", visible);
}

function setPlayersState(state) {
  for (const player of [playerA, playerB]) {
    player.classList.toggle("is-thinking", state === "thinking");
    player.classList.toggle("is-shouting", state === "shouting");
  }
}

function delay(milliseconds, runId) {
  if (skipAnimation || runId !== currentRun) return Promise.resolve();
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function showCountdown(label, pair, runId) {
  roundLabel.textContent = label;
  prompt.textContent = pair ? `Finding the middle of ${pair[0]} + ${pair[1]}` : "Two random sparks to start the game";
  setBubbles("", "", false);
  setPlayersState("thinking");

  for (const value of ["3", "2", "1"]) {
    if (runId !== currentRun) return false;
    countdown.textContent = value;
    countdown.classList.remove("pop");
    void countdown.offsetWidth;
    countdown.classList.add("pop");
    announcer.textContent = `${label}: ${value}…`;
    await delay(COUNTDOWN_DELAY, runId);
  }

  if (runId !== currentRun) return false;
  countdown.textContent = "SHOUT!";
  countdown.classList.remove("pop");
  void countdown.offsetWidth;
  countdown.classList.add("pop");
  return true;
}

function addHistoryRow(label, words, converged = false) {
  if (history.querySelector(".empty-history")) history.replaceChildren();
  const item = document.createElement("li");
  if (label === "Opening") item.classList.add("opening-round");
  if (converged) item.classList.add("converged-round");
  item.setAttribute("aria-label", `${label}: ${words[0]} and ${words[1]}${converged ? ", converged" : ""}`);

  const first = document.createElement("span");
  first.className = "history-word";
  first.textContent = words[0];
  const plus = document.createElement("span");
  plus.className = "history-plus";
  plus.textContent = converged ? "✓" : "+";
  plus.setAttribute("aria-hidden", "true");
  const second = document.createElement("span");
  second.className = "history-word second";
  second.textContent = words[1];

  item.append(first, plus, second);
  history.appendChild(item);
}

function celebrate() {
  confettiLayer.replaceChildren();
  const colors = ["#ffd85c", "#f54f9a", "#55d6be", "#ffffff", "#9f7aea"];
  for (let index = 0; index < 80; index++) {
    const piece = document.createElement("span");
    piece.className = "confetti-piece";
    piece.style.left = `${Math.random() * 100}%`;
    piece.style.background = colors[index % colors.length];
    piece.style.setProperty("--drift", `${-90 + Math.random() * 180}px`);
    piece.style.setProperty("--spin", `${360 + Math.random() * 900}deg`);
    piece.style.setProperty("--fall-duration", `${1.8 + Math.random() * 1.6}s`);
    piece.style.setProperty("--fall-delay", `${Math.random() * 0.7}s`);
    confettiLayer.appendChild(piece);
  }
}

function describeEnding(result) {
  const last = result.path[result.path.length - 1];
  if (result.outcome === "loop") return `The models circled back to ${last.pairOut.join(" + ")}. Even models get stuck in a thought loop!`;
  if (result.outcome === "stalled") return `The models ran out of eligible words in round ${result.termination.round}.`;
  return `Ten rounds, no match yet. These two need another game night.`;
}

async function animateGame(result, startWords, runId) {
  if (!(await showCountdown("Opening shout", null, runId))) return;
  setPlayersState("shouting");
  setBubbles(...startWords);
  prompt.textContent = "The game begins!";
  announcer.textContent = `${friendlyModelName(models[0].model)} says “${startWords[0]}!” ${friendlyModelName(models[1].model)} says “${startWords[1]}!”`;
  addHistoryRow("Opening", startWords);
  await delay(SHOUT_DELAY, runId);

  for (const step of result.path.slice(1)) {
    if (!(await showCountdown(`Round ${step.round}`, step.pairIn, runId))) return;
    setPlayersState("shouting");
    setBubbles(step.outputs.A, step.outputs.B);
    const converged = step.outputs.A === step.outputs.B;
    prompt.textContent = converged ? "They found the same word!" : "New pair unlocked";
    announcer.textContent = converged
      ? `Both models shouted “${step.outputs.A}!”`
      : `${friendlyModelName(models[0].model)} says “${step.outputs.A}!” ${friendlyModelName(models[1].model)} says “${step.outputs.B}!”`;
    addHistoryRow(`Round ${step.round}`, step.pairOut, converged);
    await delay(SHOUT_DELAY, runId);
  }

  if (runId !== currentRun) return;
  setPlayersState("idle");
  if (result.outcome === "converged") {
    const last = result.path[result.path.length - 1];
    countdown.textContent = "🎉";
    roundLabel.textContent = "Converged!";
    prompt.textContent = `One shared thought: ${last.pairOut[0]}`;
    announcer.textContent = `Converged on “${last.pairOut[0]}” in ${last.round} round${last.round === 1 ? "" : "s"}!`;
    outcome.textContent = `${last.round} round${last.round === 1 ? "" : "s"} to converge`;
    outcome.className = "outcome-pill converged";
    celebrate();
  } else {
    countdown.textContent = "↻";
    roundLabel.textContent = result.outcome === "loop" ? "Thought loop" : "Game over";
    prompt.textContent = "Not every pair finds its way together.";
    announcer.textContent = describeEnding(result);
    outcome.textContent = result.outcome === "loop" ? "Loop detected" : "No convergence";
    outcome.className = "outcome-pill not-converged";
  }

  playButton.disabled = false;
  playButton.innerHTML = '<span aria-hidden="true">↻</span> Play again with new words';
  skipButton.hidden = true;
  loadingStatus.textContent = "Ready for another game whenever you are.";
}

async function startGame() {
  const runId = ++currentRun;
  skipAnimation = false;
  playButton.disabled = true;
  skipButton.hidden = false;
  skipButton.disabled = false;
  skipButton.textContent = "Skip to the result";
  loadingStatus.textContent = "The models are thinking in vectors. The shouts are revealed live.";
  outcome.textContent = "Game in progress";
  outcome.className = "outcome-pill";
  history.innerHTML = '<li class="empty-history">The opening words are almost ready…</li>';
  confettiLayer.replaceChildren();
  const startWords = randomPair();
  const result = simulateConvergence(models[0], models[1], ...startWords, {
    operator: "centroid",
    maxRounds: MAX_ROUNDS,
  });
  await animateGame(result, startWords, runId);
}

skipButton.addEventListener("click", () => {
  skipAnimation = true;
  skipButton.disabled = true;
  skipButton.textContent = "Skipping…";
});

playButton.addEventListener("click", startGame);

async function init() {
  const response = await fetch("data/manifest.json");
  if (!response.ok) throw new Error(`Manifest request failed: ${response.status}`);
  const manifest = await response.json();
  if (manifest.models.length < 2) throw new Error("Game night needs at least two models.");
  models = await Promise.all(manifest.models.slice(0, 2).map((model) => loadModelEmbeddings(model)));
  vocabulary = models[0].words.filter((word) => models[1].index.has(word));
  if (vocabulary.length < 2) throw new Error("The models do not share enough vocabulary words.");

  modelLabelA.textContent = friendlyModelName(models[0].model);
  modelLabelB.textContent = friendlyModelName(models[1].model);
  const friendlyNames = models.map((model) => friendlyModelName(model.model));
  for (const [index, label] of [...document.querySelectorAll("[data-seat-label]")].entries()) {
    label.textContent = friendlyNames[index % friendlyNames.length];
  }

  loadingStatus.textContent = `${friendlyNames.join(" and ")} are ready.`;
  playButton.disabled = false;
}

const requestedMode = window.location.hash === "#vector-lab" ? "lab" : "party";
setMode(requestedMode);
init().catch((error) => {
  loadingStatus.textContent = `Could not start game night: ${error.message}`;
  announcer.textContent = "The vector lab is still available while game night is offline.";
});
