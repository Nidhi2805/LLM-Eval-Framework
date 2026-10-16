"""
FastAPI backend for the RAG Eval Platform.

Thin HTTP layer over three things: the data-access layer (db.repository), the
evaluation engine (runner), and the test-set generator. Endpoints map to the user
workflow: create project -> upload docs -> generate/define tests -> evaluate ->
review runs over time.

Request validation is handled by Pydantic models, so malformed input is rejected
with a clear 422 before it reaches any logic.
"""

from fastapi import FastAPI, HTTPException
from fastapi import BackgroundTasks
from pydantic import BaseModel, Field

from db.models import init_db
from db import repository as repo
from runner import run_evaluation
from rag.testset_generator import generate_test_set

app = FastAPI(title="RAG Eval Platform", version="1.0")


@app.on_event("startup")
def _startup():
    init_db()  # ensure tables exist


# ------------------------------------------------------------- request models
class ProjectIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""


class DocumentIn(BaseModel):
    external_id: str | None = None
    title: str = "Untitled"
    text: str = Field(..., min_length=1)


class DocumentsIn(BaseModel):
    documents: list[DocumentIn]


class GenerateIn(BaseModel):
    name: str = "generated-set"
    per_doc: int = Field(2, ge=1, le=5)
    unanswerable_count: int = Field(2, ge=0, le=10)


class EvaluateIn(BaseModel):
    test_set_id: int | None = None
    target_id: int | None = None       # if set + external, evaluate that endpoint
    config: str = "grounded"           # for built-in: grounded | ungrounded
    k: int = Field(2, ge=1, le=10)
    label: str = ""

class TargetIn(BaseModel):
    name: str = Field(..., min_length=1)
    type: str = Field("http_endpoint")   # builtin_rag | http_endpoint
    # For http_endpoint:
    url: str | None = None
    query_field: str = "query"
    answer_path: str = "answer"
    contexts_path: str | None = None
    headers: dict = {}


# --------------------------------------------------------------------- health
@app.get("/")
def health():
    return {"status": "ok", "service": "RAG Eval Platform"}


# ------------------------------------------------------------------- projects
@app.post("/projects")
def create_project(body: ProjectIn):
    return repo.create_project(body.name, body.description)


@app.get("/projects")
def list_projects():
    return repo.list_projects()


def _require_project(project_id: int):
    proj = repo.get_project(project_id)
    if not proj:
        raise HTTPException(404, f"Project {project_id} not found")
    return proj

def _run_eval_job(job_id: int, project_id: int, params: dict):
    """Background worker: runs an evaluation and records the outcome on the job.

    Runs detached from the HTTP request, so the request that started it returns
    immediately and can never time out. Status transitions:
    pending -> running -> done (with run_id) | failed (with error).
    """
    repo.update_job(job_id, status="running")
    try:
        test_set_id = params.get("test_set_id") or repo.latest_test_set_id(project_id)
        if not test_set_id:
            raise ValueError("No test set found.")
        test_cases = repo.get_test_cases(test_set_id)
        if not test_cases:
            raise ValueError("Test set has no cases.")

        target_id = params.get("target_id")
        if target_id:
            target = repo.get_target(target_id)
            if not target:
                raise ValueError(f"Target {target_id} not found.")
            if target["type"] == "http_endpoint":
                result = run_evaluation(test_cases, target_type="http_endpoint",
                                        target_config=target["config"])
            else:
                documents = repo.get_documents(project_id)
                result = run_evaluation(test_cases, target_type="builtin_rag",
                                        documents=documents, k=params.get("k", 2),
                                        generator_config=params.get("config", "grounded"))
        else:
            documents = repo.get_documents(project_id)
            if not documents:
                raise ValueError("Project has no documents.")
            result = run_evaluation(test_cases, target_type="builtin_rag",
                                    documents=documents, k=params.get("k", 2),
                                    generator_config=params.get("config", "grounded"))

        run_id = repo.save_run(project_id, result, label=params.get("label", ""),
                               test_set_id=test_set_id)
        repo.update_job(job_id, status="done", run_id=run_id)
    except Exception as e:
        # Any failure is recorded on the job, not lost -> client can see it.
        repo.update_job(job_id, status="failed", error=str(e))


# ------------------------------------------------------------------ documents
@app.post("/projects/{project_id}/documents")
def add_documents(project_id: int, body: DocumentsIn):
    _require_project(project_id)
    docs = [d.model_dump() for d in body.documents]
    count = repo.add_documents(project_id, docs)
    return {"added": count}


@app.get("/projects/{project_id}/documents")
def get_documents(project_id: int):
    _require_project(project_id)
    return repo.get_documents(project_id)


# ------------------------------------------------------------------ test sets
@app.post("/projects/{project_id}/testsets/generate")
def generate_testset(project_id: int, body: GenerateIn):
    _require_project(project_id)
    documents = repo.get_documents(project_id)
    if not documents:
        raise HTTPException(400, "Project has no documents to generate from.")
    cases = generate_test_set(
        documents, per_doc=body.per_doc,
        unanswerable_count=body.unanswerable_count,
    )
    ts_id = repo.save_test_set(project_id, body.name, cases, source="generated")
    return {"test_set_id": ts_id, "num_cases": len(cases), "cases": cases}


@app.get("/projects/{project_id}/testsets")
def list_testsets(project_id: int):
    _require_project(project_id)
    return repo.list_test_sets(project_id)


# ------------------------------------------------------------------- evaluate
@app.post("/projects/{project_id}/evaluate")
def evaluate(project_id: int, body: EvaluateIn):
    _require_project(project_id)

    test_set_id = body.test_set_id or repo.latest_test_set_id(project_id)
    if not test_set_id:
        raise HTTPException(400, "No test set found. Generate or add one first.")
    test_cases = repo.get_test_cases(test_set_id)
    if not test_cases:
        raise HTTPException(400, "Selected test set has no cases.")

    # Decide what we're evaluating: a saved external target, or the built-in RAG.
    if body.target_id:
        target = repo.get_target(body.target_id)
        if not target:
            raise HTTPException(404, f"Target {body.target_id} not found")
        if target["type"] == "http_endpoint":
            result = run_evaluation(
                test_cases,
                target_type="http_endpoint",
                target_config=target["config"],
            ) if False else run_evaluation(      # keep call simple/explicit:
                test_cases,
                target_type="http_endpoint",
                target_config=target["config"],
            )
        else:
            documents = repo.get_documents(project_id)
            result = run_evaluation(test_cases, target_type="builtin_rag",
                                    documents=documents, k=body.k,
                                    generator_config=body.config)
    else:
        documents = repo.get_documents(project_id)
        if not documents:
            raise HTTPException(400, "Project has no documents.")
        result = run_evaluation(test_cases, target_type="builtin_rag",
                                documents=documents, k=body.k,
                                generator_config=body.config)

    run_id = repo.save_run(project_id, result, label=body.label,
                           test_set_id=test_set_id)
    return {"run_id": run_id, "summary": result["summary"]}

@app.post("/projects/{project_id}/evaluate_async")
def evaluate_async(project_id: int, body: EvaluateIn, background: BackgroundTasks):
    """Start an evaluation in the background. Returns a job_id immediately;
    poll /jobs/{job_id} for status, then GET /runs/{run_id} when done."""
    _require_project(project_id)
    params = body.model_dump()
    job_id = repo.create_job(project_id, params)
    background.add_task(_run_eval_job, job_id, project_id, params)
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}")
def job_status(job_id: int):
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return job

# -------------------------------------------------------------------- targets
@app.post("/projects/{project_id}/targets")
def add_target(project_id: int, body: TargetIn):
    _require_project(project_id)
    if body.type == "http_endpoint" and not body.url:
        raise HTTPException(400, "http_endpoint target requires a 'url'.")
    config = {
        "url": body.url,
        "query_field": body.query_field,
        "answer_path": body.answer_path,
        "contexts_path": body.contexts_path,
        "headers": body.headers,
    }
    target_id = repo.save_target(project_id, body.name, body.type, config)
    return {"target_id": target_id}


@app.get("/projects/{project_id}/targets")
def get_targets(project_id: int):
    _require_project(project_id)
    return repo.list_targets(project_id)

# ----------------------------------------------------------------------- runs
@app.get("/projects/{project_id}/runs")
def list_runs(project_id: int):
    _require_project(project_id)
    return repo.list_runs(project_id)


@app.get("/runs/{run_id}")
def get_run(run_id: int):
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    return run