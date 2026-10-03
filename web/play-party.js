import { combine, getStartingWords, getVocab, listModels, simulateConvergence } from "./api.js";

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
const pickerA = document.getElementById("party-picker-a");
const pickerBLabel = document.getElementById("party-picker-b-label");
const humanForm = document.getElementById("party-human-form");
const humanWordInput = document.getElementById("party-human-word");
const humanError = document.getElementById("party-human-error");
const settingsControls = [operatorSelect, selectA, selectB, ...playerChoices];

const HUMAN = "you";
const RANDOM_OPPONENT = "random";
const randomOpponentOption = new Option("Random model each round", RANDOM_OPPONENT);

const COUNTDOWN_DELAY = 560;
const SHOUT_DELAY = 1600;
const MAX_ROUNDS = 10;

const OPERATORS = {
  centroid: { anchor: "centroid", help: "Each player picks the word closest to the midpoint of the two words' vectors." },
  maximin: { anchor: "maximin", help: "Each player picks the word whose weaker similarity to the two words is highest." },
  product: { anchor: "product", help: "Each player picks the word with the biggest product of its similarities to the two words." },
  textual: { anchor: "textual", help: "Each player embeds the phrase “A and B” and picks the word closest to it." },
};

let models = [];
let vocabulary = [];
let startingVocabulary = [];
let currentRun = 0;
let skipAnimation = false;

function labelFor(name) {
  if (name === HUMAN) return "You";
  if (name === RANDOM_OPPONENT) return "Random model";
  return models.find((model) => model.name === name)?.label ?? name;
}

function playersMode() {
  return document.querySelector('input[name="party-players"]:checked').value;
}

function randomPlayers() {
  return playersMode() === "random";
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
  const human = playersMode() === "human";
  modelPickers.hidden = random;
  pickerA.hidden = human;
  pickerBLabel.textContent = human ? "Your opponent" : "Player B";
  if (human && !randomOpponentOption.parentElement) selectB.add(randomOpponentOption);
  if (!human && randomOpponentOption.parentElement) {
    if (selectB.value === RANDOM_OPPONENT) selectB.selectedIndex = 1;
    randomOpponentOption.remove();
  }
  if (human) {
    setSeatLabels({ A: HUMAN, B: selectB.value });
    explainer.textContent =
      selectB.value === RANDOM_OPPONENT
        ? `Each round, you play against a model drawn at random from ${models.length}. Every model uses the ${operatorName.toLowerCase()} operator.`
        : `You and ${labelFor(selectB.value)} each pick a word between the current pair. ${labelFor(selectB.value)} uses the ${operatorName.toLowerCase()} operator.`;
    return;
  }
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
  const first = Math.floor(Math.random() * startingVocabulary.length);
  const offset = 1 + Math.floor(Math.random() * (startingVocabulary.length - 1));
  return [startingVocabulary[first], startingVocabulary[(first + offset) % startingVocabulary.length]];
}

function setBubbles(wordA, wordB, visible = true) {
  for (const [bubble, word] of [[bubbleA, wordA], [bubbleB, wordB]]) {
    bubble.textContent = word;
    bubble.classList.toggle("long-word", word.length > 10);
    bubble.classList.toggle("show", visible);
  }
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
  const colors = ["#ffd85c", "#f54f9a", "#55d6be", "#ffffff", "#9f7aea", "#ff9f68"];
  const shapes = ["heart", "star", "dot", "ribbon"];
  for (let index = 0; index < 70; index++) {
    const piece = document.createElement("span");
    piece.className = `confetti-piece confetti-${shapes[index % shapes.length]}`;
    const angle = Math.random() * Math.PI * 2;
    const distance = 70 + Math.random() * 160;
    piece.style.background = colors[index % colors.length];
    piece.style.setProperty("--x", `${Math.cos(angle) * distance}px`);
    piece.style.setProperty("--y", `${Math.sin(angle) * distance - 40}px`);
    piece.style.setProperty("--drift", `${-40 + Math.random() * 80}px`);
    piece.style.setProperty("--fall", `${140 + Math.random() * 160}px`);
    piece.style.setProperty("--spin", `${(Math.random() < 0.5 ? -1 : 1) * (240 + Math.random() * 480)}deg`);
    piece.style.setProperty("--size", `${0.8 + Math.random() * 0.7}`);
    piece.style.setProperty("--fall-duration", `${2.2 + Math.random() * 1.4}s`);
    piece.style.setProperty("--fall-delay", `${0.35 + Math.random() * 0.25}s`);
    confettiLayer.appendChild(piece);
  }
}

const TROPHY_SVG = `<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <path d="M17 14H9v5c0 8 5 13 12 13M47 14h8v5c0 8-5 13-12 13" fill="none" stroke="#f5b82e" stroke-width="4" stroke-linecap="round"/>
  <path d="M16 8h32v14c0 12-7 20-16 20S16 34 16 22z" fill="#ffd85c" stroke="#c98a12" stroke-width="2"/>
  <path d="M21 12v9c0 6 2 10 5 13" fill="none" stroke="#fff6cf" stroke-width="3" stroke-linecap="round"/>
  <path d="M32 15l2.6 5.3 5.8.8-4.2 4.1 1 5.8-5.2-2.7-5.2 2.7 1-5.8-4.2-4.1 5.8-.8z" fill="#fff"/>
  <rect x="28" y="41" width="8" height="8" fill="#e0a21f"/>
  <rect x="19" y="48" width="26" height="8" rx="2" fill="#c98a12"/>
</svg>`;

function convergeMark() {
  const mark = document.createElement("span");
  mark.className = "converge-mark";
  mark.setAttribute("aria-hidden", "true");
  for (const part of ["spark spark-a", "spark spark-b", "ring"]) {
    const piece = document.createElement("span");
    piece.className = part;
    mark.appendChild(piece);
  }
  const trophy = document.createElement("span");
  trophy.className = "trophy";
  trophy.innerHTML = TROPHY_SVG;
  mark.appendChild(trophy);
  return mark;
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
  const last = result.path[result.path.length - 1];
  finishGame({
    converged: result.outcome === "converged",
    word: last.pairOut[0],
    rounds: last.round,
    title: result.outcome === "loop" ? "Thought loop" : "Game over",
    pill: result.outcome === "loop" ? "Loop detected" : "No convergence",
    message: describeEnding(result),
  });
}

function finishGame({ converged, word, rounds, title = "Game over", pill = "No convergence", message }) {
  setPlayersState("idle");
  humanForm.hidden = true;
  if (converged) {
    countdown.replaceChildren(convergeMark());
    roundLabel.textContent = "Converged!";
    prompt.textContent = `One shared thought: ${word}`;
    announcer.textContent = `Converged on “${word}” in ${rounds} round${rounds === 1 ? "" : "s"}!`;
    outcome.textContent = `${rounds} round${rounds === 1 ? "" : "s"} to converge`;
    outcome.className = "outcome-pill converged";
    celebrate();
  } else {
    countdown.textContent = "↻";
    roundLabel.textContent = title;
    prompt.textContent = "Not every pair finds its way together.";
    announcer.textContent = message;
    outcome.textContent = pill;
    outcome.className = "outcome-pill not-converged";
  }

  playButton.disabled = false;
  setSettingsDisabled(false);
  playButton.innerHTML = '<span aria-hidden="true">↻</span> Play again';
  skipButton.hidden = true;
  loadingStatus.textContent = "Ready for another game whenever you are.";
}

function waitForHumanWord(pair) {
  return new Promise((resolve) => {
    humanForm.hidden = false;
    humanWordInput.value = "";
    humanError.textContent = "";
    humanWordInput.focus();
    humanForm.onsubmit = (event) => {
      event.preventDefault();
      const word = humanWordInput.value.trim().toLowerCase();
      if (!word) {
        humanError.textContent = "Type a word first.";
        return;
      }
      if (pair.includes(word)) {
        humanError.textContent = "Pick a new word, not one of the current pair.";
        return;
      }
      humanForm.hidden = true;
      humanForm.onsubmit = null;
      resolve(word);
    };
  });
}

async function playHumanGame(startWords, runId) {
  const choice = selectB.value;
  const operator = operatorSelect.value;
  setSeatLabels({ A: HUMAN, B: choice });

  if (!(await showCountdown("Opening shout", null, runId))) return;
  setPlayersState("shouting");
  setBubbles(...startWords);
  prompt.textContent = "The game begins!";
  announcer.textContent = `The opening words are “${startWords[0]}” and “${startWords[1]}”!`;
  addHistoryRow("Opening", startWords);
  await delay(SHOUT_DELAY, runId);

  let pair = startWords;
  for (let round = 1; round <= MAX_ROUNDS; round++) {
    const opponent = choice === RANDOM_OPPONENT ? models[Math.floor(Math.random() * models.length)].name : choice;
    const players = { A: HUMAN, B: opponent };
    // The model only sees the current pair, so fetching now can't peek at the human's word
    const modelPick = combine(operator, opponent, ...pair, { topK: 1 }).then((candidates) => candidates[0]?.candidate);
    modelPick.catch(() => {});

    roundLabel.textContent = `Round ${round}`;
    setSeatLabels({ A: HUMAN, B: choice });
    countdown.textContent = "?";
    prompt.textContent = `What's between ${pair[0]} + ${pair[1]}?`;
    announcer.textContent = `Your turn: what word is between “${pair[0]}” and “${pair[1]}”?`;
    setBubbles("", "", false);
    setPlayersState("thinking");
    const humanWord = await waitForHumanWord(pair);
    if (runId !== currentRun) return;

    if (!(await showCountdown(`Round ${round}`, pair, runId))) return;
    let modelWord;
    try {
      modelWord = await modelPick;
    } catch (error) {
      finishGame({ converged: false, message: `${labelFor(opponent)} couldn't answer: ${error.message}` });
      return;
    }
    if (!modelWord) {
      finishGame({ converged: false, message: `${labelFor(opponent)} ran out of eligible words.` });
      return;
    }

    setPlayersState("shouting");
    setSeatLabels(players);
    setBubbles(humanWord, modelWord);
    const converged = humanWord === modelWord;
    prompt.textContent = converged ? "You found the same word!" : "New pair unlocked";
    announcer.textContent = converged ? `You and ${labelFor(opponent)} both shouted “${humanWord}!”` : `You say “${humanWord}!” ${labelFor(opponent)} says “${modelWord}!”`;
    addHistoryRow(`Round ${round}`, [humanWord, modelWord], converged, players);
    if (converged) {
      await delay(SHOUT_DELAY, runId);
      finishGame({ converged: true, word: humanWord, rounds: round });
      return;
    }
    pair = [humanWord, modelWord];
    await delay(SHOUT_DELAY, runId);
  }
  finishGame({ converged: false, message: `${MAX_ROUNDS} rounds, no match yet. Time for another game!` });
}

async function startGame() {
  const startWords = startingWords();
  if (startWords[0] === startWords[1]) {
    loadingStatus.textContent = "Pick two different starting words.";
    return;
  }
  const runId = ++currentRun;
  const human = playersMode() === "human";
  skipAnimation = false;
  playButton.disabled = true;
  setSettingsDisabled(true);
  skipButton.hidden = human;
  skipButton.disabled = false;
  skipButton.textContent = "Skip to the result";
  loadingStatus.textContent = human ? "Type your word each round, then the model's word is revealed with yours." : "The models are thinking in vectors. The shouts are revealed live.";
  outcome.textContent = "Game in progress";
  outcome.className = "outcome-pill";
  history.innerHTML = '<li class="empty-history">The opening words are almost ready…</li>';
  confettiLayer.replaceChildren();
  if (human) {
    await playHumanGame(startWords, runId);
    return;
  }
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
  // Every model's vocabulary comes from the same word list, so the first one works for suggestions.
  [vocabulary, startingVocabulary] = await Promise.all([getVocab(models[0].name), getStartingWords()]);
  if (startingVocabulary.length < 2) throw new Error("There aren't enough starting words to play.");
  vocabList.replaceChildren(...vocabulary.map((word) => new Option(word)));

  updateSettings();
  loadingStatus.textContent = `${models.length} models are ready to play.`;
  playButton.disabled = false;
}

init().catch((error) => {
  loadingStatus.textContent = `Could not start the game: ${error.message}`;
});
