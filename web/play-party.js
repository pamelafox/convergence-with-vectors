import { getVocab, listModels, simulateConvergence } from "./api.js";

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
const explainer = document.getElementById("party-explainer");
const operatorSelect = document.getElementById("party-operator");
const operatorHelp = document.getElementById("party-operator-help");
const modelPickers = document.getElementById("party-model-pickers");
const selectA = document.getElementById("party-select-a");
const selectB = document.getElementById("party-select-b");
const playerChoices = document.querySelectorAll('input[name="party-players"]');
const wordInputA = document.getElementById("party-word-a");
const wordInputB = document.getElementById("party-word-b");
const vocabList = document.getElementById("party-vocab");
const settingsControls = [operatorSelect, selectA, selectB, ...playerChoices];

const COUNTDOWN_DELAY = 560;
const SHOUT_DELAY = 850;
const MAX_ROUNDS = 10;

const OPERATORS = {
  centroid: { anchor: "centroid", help: "Each player picks the word closest to the midpoint of the two words' vectors." },
  balanced: { anchor: "balanced", help: "Each player picks the word whose weaker similarity to the two words is highest." },
  geometric_mean: { anchor: "geometric-mean", help: "Each player picks the word with the biggest product of its similarities to the two words." },
  textual: { anchor: "textual", help: "Each player embeds the phrase “A and B” and picks the word closest to it." },
};

let models = [];
let vocabulary = [];
let currentRun = 0;
let skipAnimation = false;

function labelFor(name) {
  return models.find((model) => model.name === name)?.label ?? name;
}

function randomPlayers() {
  return document.querySelector('input[name="party-players"]:checked').value === "random";
}

function setSeatLabels(names) {
  modelLabelA.textContent = names ? labelFor(names.A) : "Random model";
  modelLabelB.textContent = names ? labelFor(names.B) : "Random model";
}

function updateSettings() {
  const operator = operatorSelect.value;
  const operatorName = operatorSelect.selectedOptions[0].textContent;
  const link = document.createElement("a");
  link.href = `operators.html#${OPERATORS[operator].anchor}`;
  link.textContent = `How ${operatorName.toLowerCase()} works`;
  operatorHelp.replaceChildren(`${OPERATORS[operator].help} `, link);

  const random = randomPlayers();
  modelPickers.hidden = random;
  setSeatLabels(random ? null : { A: selectA.value, B: selectB.value });
  explainer.textContent = random
    ? `Every round, two different models are drawn at random from ${models.length} and both use the ${operatorName.toLowerCase()} operator.`
    : `${labelFor(selectA.value)} and ${labelFor(selectB.value)} both use the ${operatorName.toLowerCase()} operator for every round.`;
}

function setSettingsDisabled(disabled) {
  for (const control of [...settingsControls, wordInputA, wordInputB]) control.disabled = disabled;
}

function startingWords() {
  const random = randomPair();
  return [wordInputA, wordInputB].map((input, index) => input.value.trim().toLowerCase() || random[index]);
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

function historyWord(word, model, second) {
  const span = document.createElement("span");
  span.className = second ? "history-word second" : "history-word";
  span.textContent = word;
  if (model) {
    const modelLabel = document.createElement("span");
    modelLabel.className = "history-model";
    modelLabel.textContent = labelFor(model);
    span.appendChild(modelLabel);
  }
  return span;
}

function addHistoryRow(label, words, converged = false, players = null) {
  if (history.querySelector(".empty-history")) history.replaceChildren();
  const item = document.createElement("li");
  if (label === "Opening") item.classList.add("opening-round");
  if (converged) item.classList.add("converged-round");
  const said = players ? `${labelFor(players.A)} said ${words[0]}, ${labelFor(players.B)} said ${words[1]}` : `${words[0]} and ${words[1]}`;
  item.setAttribute("aria-label", `${label}: ${said}${converged ? ", converged" : ""}`);

  const plus = document.createElement("span");
  plus.className = "history-plus";
  plus.textContent = converged ? "✓" : "+";
  plus.setAttribute("aria-hidden", "true");

  item.append(historyWord(words[0], players?.A), plus, historyWord(words[1], players?.B, true));
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
  return `${MAX_ROUNDS} rounds, no match yet. Time for another game!`;
}

async function animateGame(result, startWords, runId) {
  if (!(await showCountdown("Opening shout", null, runId))) return;
  setPlayersState("shouting");
  setBubbles(...startWords);
  prompt.textContent = "The game begins!";
  announcer.textContent = `The opening words are “${startWords[0]}” and “${startWords[1]}”!`;
  addHistoryRow("Opening", startWords);
  await delay(SHOUT_DELAY, runId);

  for (const step of result.path.slice(1)) {
    setSeatLabels(step.models);
    if (!(await showCountdown(`Round ${step.round}`, step.pairIn, runId))) return;
    setPlayersState("shouting");
    setBubbles(step.outputs.A, step.outputs.B);
    const converged = step.outputs.A === step.outputs.B;
    prompt.textContent = converged ? "They found the same word!" : "New pair unlocked";
    announcer.textContent = converged
      ? `Both models shouted “${step.outputs.A}!”`
      : `${labelFor(step.models.A)} says “${step.outputs.A}!” ${labelFor(step.models.B)} says “${step.outputs.B}!”`;
    addHistoryRow(`Round ${step.round}`, step.pairOut, converged, step.models);
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
  setSettingsDisabled(false);
  playButton.innerHTML = '<span aria-hidden="true">↻</span> Play again';
  skipButton.hidden = true;
  loadingStatus.textContent = "Ready for another game whenever you are.";
}

async function startGame() {
  const startWords = startingWords();
  if (startWords[0] === startWords[1]) {
    loadingStatus.textContent = "Pick two different starting words.";
    return;
  }
  const runId = ++currentRun;
  skipAnimation = false;
  playButton.disabled = true;
  setSettingsDisabled(true);
  skipButton.hidden = false;
  skipButton.disabled = false;
  skipButton.textContent = "Skip to the result";
  loadingStatus.textContent = "The models are thinking in vectors. The shouts are revealed live.";
  outcome.textContent = "Game in progress";
  outcome.className = "outcome-pill";
  history.innerHTML = '<li class="empty-history">The opening words are almost ready…</li>';
  confettiLayer.replaceChildren();
  const random = randomPlayers();
  let result;
  try {
    result = await simulateConvergence({
      modelA: random ? undefined : selectA.value,
      modelB: random ? undefined : selectB.value,
      randomModels: random,
      wordA: startWords[0],
      wordB: startWords[1],
      operator: operatorSelect.value,
      maxRounds: MAX_ROUNDS,
    });
  } catch (error) {
    loadingStatus.textContent = `The models couldn't play: ${error.message}`;
    playButton.disabled = false;
    setSettingsDisabled(false);
    skipButton.hidden = true;
    return;
  }
  await animateGame(result, startWords, runId);
}

skipButton.addEventListener("click", () => {
  skipAnimation = true;
  skipButton.disabled = true;
  skipButton.textContent = "Skipping…";
});

playButton.addEventListener("click", startGame);
for (const control of settingsControls) control.addEventListener("change", updateSettings);

async function init() {
  models = await listModels();
  if (models.length < 2) throw new Error("The game needs at least two models.");
  for (const select of [selectA, selectB]) {
    select.replaceChildren(...models.map((model) => new Option(model.label, model.name)));
  }
  selectB.selectedIndex = 1;
  // Every model's vocabulary comes from the same word list, so the first one works for picking start words.
  vocabulary = await getVocab(models[0].name);
  if (vocabulary.length < 2) throw new Error("The vocabulary is too small to play.");
  vocabList.replaceChildren(...vocabulary.map((word) => new Option(word)));

  updateSettings();
  loadingStatus.textContent = `${models.length} models are ready to play.`;
  playButton.disabled = false;
}

init().catch((error) => {
  loadingStatus.textContent = `Could not start the game: ${error.message}`;
});
