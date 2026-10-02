"""FastAPI backend for Convergence: embeddings in SQLite, operators computed on demand.

Serves a JSON API under ``/api`` plus the existing pages: the slides at
``/``, the browser game and explainer under ``/web``. Only those folders
are exposed, never the whole repository.

Run locally with:

    uv run fastapi dev backend/app.py
"""

from __future__ import annotations

import os
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, get_args

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints, model_validator

from backend import db
from embeddings import MODELS, ModelUnavailableError, VocabEmbeddings, embed_texts
from operators import OPERATOR_NAMES, EmbedFn, apply_operator
from simulate_convergence import run_simulation

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT / "data" / "convergence.db"

OperatorName = Literal["centroid", "balanced", "geometric_mean", "textual"]
assert set(get_args(OperatorName)) == set(OPERATOR_NAMES), "OperatorName must list every operator in operators.OPERATOR_NAMES"

# Inputs are normalized the same way as the vocabulary file (lowercase, trimmed).
Word = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=1, max_length=60)]
Template = Annotated[str, StringConstraints(min_length=1, max_length=200)]


class Candidate(BaseModel):
    rank: int
    candidate: str
    score: float
    sim_a: float
    sim_b: float


class ModelInfo(BaseModel):
    name: str
    label: str
    provider: str


class SimulateRequest(BaseModel):
    model_a: str | None = None
    model_b: str | None = None
    random_models: bool = Field(False, description="Pick two different random models every round instead of model_a and model_b")
    seed: int | None = Field(None, description="Seed for random_models, to replay the same game")
    word_a: Word
    word_b: Word
    operator: OperatorName = "centroid"
    template: Template = "{a} and {b}"
    max_rounds: int = Field(10, ge=1, le=50)
    top_k: int = Field(5, ge=1, le=50)
    include_inputs: bool = False

    @model_validator(mode="after")
    def require_models_unless_random(self):
        if not self.random_models and not (self.model_a and self.model_b):
            raise ValueError("Choose model_a and model_b, or set random_models")
        return self


class SimulationStep(BaseModel):
    round: int
    pair_in: tuple[str, str]
    pair_out: tuple[str, str]
    outputs: tuple[str, str] | None
    models: tuple[str, str] | None
    candidates: tuple[list[Candidate], list[Candidate]] | None


class SimulationResult(BaseModel):
    path: list[SimulationStep]
    outcome: Literal["converged", "loop", "stalled", "round_limit"]
    termination: dict


def create_app(db_path: Path | None = None, embed_fn: EmbedFn | None = None) -> FastAPI:
    """Build the app. Tests pass a temporary database and a fake ``embed_fn``."""
    db_path = Path(db_path or os.environ.get("CONVERGENCE_DB", DEFAULT_DB_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not db_path.exists():
            raise RuntimeError(f"Embeddings database not found at {db_path}. Build it first with: uv run build_db.py")
        conn = db.connect(db_path)
        stored = set(db.vocab_models(conn))
        app.state.vocabs = {model: db.load_vocab(conn, model) for model in MODELS if model in stored}
        app.state.store = db.EmbeddingStore(conn, embed_fn or embed_texts)
        yield
        conn.close()

    app = FastAPI(title="Convergence in vector space", lifespan=lifespan)

    @app.exception_handler(ModelUnavailableError)
    def model_unavailable(request: Request, err: ModelUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(err)})

    def get_vocab(request: Request, model: str) -> VocabEmbeddings:
        vocab = request.app.state.vocabs.get(model)
        if vocab is None:
            raise HTTPException(status_code=404, detail=f"Unknown model {model!r}")
        return vocab

    def rounded(candidate) -> Candidate:
        return Candidate(rank=candidate.rank, candidate=candidate.candidate, score=round(candidate.score, 6), sim_a=round(candidate.sim_a, 6), sim_b=round(candidate.sim_b, 6))

    @app.get("/api/models")
    def list_models(request: Request) -> dict[str, list[ModelInfo]]:
        return {"models": [ModelInfo(name=name, label=MODELS[name].label, provider=MODELS[name].provider) for name in request.app.state.vocabs]}

    @app.get("/api/vocab")
    def vocabulary(request: Request, model: str) -> dict[str, list[str]]:
        return {"words": get_vocab(request, model).words}

    @app.get("/api/similarities")
    def similarities(request: Request, model: str, a: Annotated[Word, Query()], b: Annotated[Word, Query()]) -> dict:
        """Every vocabulary word's cosine similarity to ``a`` and to ``b``: all the explainer views need."""
        vocab = get_vocab(request, model)
        vec_a, vec_b = request.app.state.store.embed(model, [a, b])
        return {
            "words": vocab.words,
            "sim_a": [round(float(x), 5) for x in vocab.similarities_to(vec_a)],
            "sim_b": [round(float(x), 5) for x in vocab.similarities_to(vec_b)],
            "cos_ab": float(vec_a @ vec_b),
        }

    @app.get("/api/combine")
    def combine(
        request: Request,
        model: str,
        a: Annotated[Word, Query()],
        b: Annotated[Word, Query()],
        operator: OperatorName = "centroid",
        top_k: Annotated[int, Query(ge=1, le=10_000)] = 10,
        template: Annotated[Template, Query()] = "{a} and {b}",
        include_inputs: bool = False,
    ) -> dict[str, list[Candidate]]:
        """Rank vocabulary words for one operator, including "textual", which embeds a phrase on demand."""
        vocab = get_vocab(request, model)
        try:
            candidates = apply_operator(operator, model, vocab, a, b, top_k=top_k, exclude=None if include_inputs else {a, b}, template=template, embed=request.app.state.store.embed)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        return {"candidates": [rounded(c) for c in candidates]}

    @app.post("/api/simulate")
    def simulate(request: Request, body: SimulateRequest) -> SimulationResult:
        """Play a full game of Convergence between two models, or between random models that change every round."""
        pick_models = None
        if body.random_models:
            pool = list(request.app.state.vocabs)
            if len(pool) < 2:
                raise HTTPException(status_code=422, detail="Random players need at least two models")
            rng = random.Random(body.seed)

            def pick_models(_round: int) -> list[str]:
                return rng.sample(pool, 2)

            models = pick_models(0)
        else:
            models = [body.model_a, body.model_b]
            for model in models:
                get_vocab(request, model)
        try:
            result = run_simulation(
                models,
                request.app.state.vocabs,
                body.word_a,
                body.word_b,
                operator=body.operator,
                template=body.template,
                max_rounds=body.max_rounds,
                top_k=body.top_k,
                include_inputs=body.include_inputs,
                embed=request.app.state.store.embed,
                pick_models=pick_models,
            )
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        path = [{**step, "candidates": step["candidates"] and [[rounded(c) for c in player] for player in step["candidates"]]} for step in result["path"]]
        return SimulationResult(path=path, outcome=result["outcome"], termination=result["termination"])

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse("/web/play.html")

    app.mount("/web", StaticFiles(directory=ROOT / "web", html=True), name="web")
    return app


app = create_app()
