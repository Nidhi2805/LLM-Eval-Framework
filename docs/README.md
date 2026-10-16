# 🔬 RAG Eval Platform

**Automated quality evaluation and regression testing for Retrieval-Augmented Generation (RAG) systems.**

Bring your own documents, auto-generate a test set, evaluate your RAG pipeline on four research-grounded metrics, and track quality across versions — catching regressions before they ship. Think *"unit tests for LLM apps."*

---

## The problem

Every team shipping an LLM feature hits the same wall: **how do you know a change didn't make it worse?** Tweak a prompt, swap a model, adjust chunk size — did quality improve or quietly degrade? Most teams can't answer this, because building trustworthy evaluation ground truth is tedious, so they ship on vibes.

This platform closes that gap. It turns subjective "the answers seem fine" into **measurable, repeatable, versioned quality scores** — and fails loudly when quality drops.

---

## What it does

- **Bring your own data** — upload documents and the system builds a knowledge base and a RAG pipeline over them.
- **Auto-generate test sets** — point it at your documents and it generates a *candidate golden dataset*: answerable questions with known sources, plus deliberately out-of-scope questions to test refusal. It self-validates each generated question by re-running retrieval. *(Removes the biggest barrier to LLM evaluation: hand-writing ground truth.)*
- **Four evaluation metrics** — grounded in LLM-evaluation research:
  | Metric | What it measures | How |
  |---|---|---|
  | **Faithfulness** | Is the answer grounded in retrieved context, or hallucinated? | LLM-as-a-judge |
  | **Answer Correctness** | Does the answer match known-correct ground truth? | LLM-as-a-judge |
  | **Retrieval Quality** | Did we fetch the right documents? (Hit@k, MRR, Precision@k) | Deterministic math |
  | **Refusal Accuracy** | Does it correctly decline out-of-scope questions? | Logic |
- **Regression detection** — compare any two runs; the system flags metric drops beyond a threshold and returns a CI-ready pass/fail exit code.
- **Trend tracking** — a dashboard shows how a project's quality moves across runs over time.

---

## Why the design is the way it is

A few deliberate engineering decisions (and the reasoning behind them):

- **LLM-as-a-judge, hardened against known failure modes.** Judges run at `temperature=0` (repeatable), emit structured JSON with a score *and* a reason (auditable, not a black box), and follow a strict 1–5 rubric (fights leniency drift). Each judge is scoped narrowly to one question. The judges were validated by injecting known hallucinations and confirming they are caught — *an evaluator you haven't tested is just another untested model.*
- **Faithfulness and correctness are separate metrics, on purpose.** They can disagree, and the disagreement localizes the fault: *faithful-but-incorrect* implicates retrieval; *correct-but-unfaithful* implicates the generator ignoring its context. A single "is it good?" score can't do this.
- **Provider-agnostic LLM layer.** All model calls route through one module, so switching between Groq / Gemini / local models is a one-file change. Built-in exponential-backoff retry means a single transient API failure can't kill a multi-call run.
- **Decoupled architecture.** A FastAPI backend and a Streamlit dashboard communicate over REST — two independent services, not one monolith.
- **Pure, data-driven eval engine.** The engine takes documents and test cases as arguments and returns a result; persistence is the caller's job. This makes it reusable across the CLI, the API, and CI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard                     │
│   Projects · Upload docs · Generate tests · Run eval ·   │
│   Score history / regression trend                       │
└───────────────────────────┬─────────────────────────────┘
                            │ REST (HTTP)
┌───────────────────────────▼─────────────────────────────┐
│                     FastAPI Backend                      │
│   /projects  /documents  /testsets/generate              │
│   /evaluate  /runs                                       │
└───────────────────────────┬─────────────────────────────┘
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
   ┌────────────┐    ┌──────────────┐   ┌──────────────┐
   │  Database  │    │ Eval Engine  │   │ Test-set Gen │
   │ (SQLite,   │    │ 4 metrics +  │   │ auto golden  │
   │ Postgres-  │    │ regression   │   │ set + self-  │
   │ ready)     │    │ detection    │   │ validation   │
   └────────────┘    └──────┬───────┘   └──────────────┘
                            ▼
                   ┌──────────────────┐
                   │ Provider-agnostic│
                   │ LLM client       │
                   │ (Groq, retries)  │
                   └──────────────────┘
```

---

## Tech stack

**Python · FastAPI · Streamlit · SQLAlchemy (SQLite→Postgres) · sentence-transformers · Groq (Llama 3.3 70B) · Pydantic · Docker · GitHub Actions**

---

## Screenshots

> _Add your screenshots to a `docs/` folder and link them here._

- **Regression trend** — quality across runs, with the degraded version visibly dropping: `docs/history.png`
- **Evaluation scorecard** — the five headline metrics: `docs/scorecard.png`
- **Auto-generated test set** — questions, sources, and confidence flags: `docs/testset.png`
- **API docs** — FastAPI's interactive Swagger UI: `docs/api.png`

---

## Quickstart

```bash
# 1. Clone and set up
git clone <your-repo-url>
cd llm-eval-framework
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your API key
echo "GROQ_API_KEY=your_key_here" > .env   # get a free key at console.groq.com

# 3. Start the backend (terminal 1)
uvicorn api:app --reload

# 4. Start the dashboard (terminal 2)
streamlit run dashboard/app.py
```

Open the dashboard at `localhost:8501`. Create a project → add documents → generate a test set → run an evaluation → view the trend.

### Try the regression demo
Run an evaluation with the **grounded** config, then again with **ungrounded** (which removes the grounding instruction — simulating someone accidentally weakening a system prompt). Watch faithfulness and refusal accuracy drop in the History tab. That's the regression detector catching a realistic failure.

---

## Project structure

```
rag/            RAG pipeline: retriever, generator, LLM client, test-set generator
metrics/        The four metrics: retrieval, faithfulness, correctness, refusal
db/             SQLAlchemy models + data-access layer
dashboard/      Streamlit UI + API client
data/           Example corpus and golden dataset
runner.py       The evaluation engine
regression.py   Run-to-run regression detector (CI-ready exit codes)
ci_check.py     CI gate: fresh eval vs. committed baseline
api.py          FastAPI backend
```

---

## Roadmap / future work

- **Async evaluation jobs** — evaluation currently runs synchronously inside the HTTP request; the production design is a background job queue returning a job ID to poll, so the API never blocks on a multi-minute task.
- **Evaluate external endpoints** — the schema already includes a `target` abstraction so a future `http_endpoint` target type can evaluate a user's *own deployed* RAG API, not just the built-in pipeline.
- **More retrieval strategies** — hybrid search and reranking, compared head-to-head by the framework itself.
- **Human-in-the-loop review** of auto-generated test sets, with accept/edit/reject.

---

## Note on auto-generated test sets

The generator produces a strong *candidate* test set intended for human curation — not a replacement for human judgment. LLM-generated ground truth should always be reviewed before being trusted as a benchmark.