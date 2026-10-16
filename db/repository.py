"""
Data-access layer -- all database reads/writes live here, so the API stays thin
and focused on HTTP. Converts between DB models and plain dicts that the rest of
the app (engine, generator) already speaks.

Keeping this separate from the API means DB logic is testable without a server,
and swapping SQLite -> Postgres touches only db.models, not this file.
"""

import json
from db.models import (
    get_session, Project, Document, Target, TestSet, TestCase, Run, Job,
)


# ----------------------------------------------------------------- projects
def create_project(name: str, description: str = "") -> dict:
    s = get_session()
    try:
        proj = Project(name=name, description=description)
        s.add(proj)
        s.commit()
        return {"id": proj.id, "name": proj.name, "description": proj.description}
    finally:
        s.close()


def list_projects() -> list[dict]:
    s = get_session()
    try:
        return [
            {"id": p.id, "name": p.name, "description": p.description,
             "num_documents": len(p.documents), "num_runs": len(p.runs)}
            for p in s.query(Project).order_by(Project.created_at.desc()).all()
        ]
    finally:
        s.close()


def get_project(project_id: int) -> dict | None:
    s = get_session()
    try:
        p = s.get(Project, project_id)
        if not p:
            return None
        return {"id": p.id, "name": p.name, "description": p.description}
    finally:
        s.close()


# ---------------------------------------------------------------- documents
def add_documents(project_id: int, docs: list[dict]) -> int:
    """docs: [{"external_id","title","text"}]. Returns count added."""
    s = get_session()
    try:
        for d in docs:
            s.add(Document(
                project_id=project_id,
                external_id=d.get("external_id") or f"doc_{d['title'][:20]}",
                title=d.get("title", "Untitled"),
                text=d["text"],
            ))
        s.commit()
        return len(docs)
    finally:
        s.close()


def get_documents(project_id: int) -> list[dict]:
    """Returns documents in the engine's expected shape: {id,title,text}."""
    s = get_session()
    try:
        return [
            {"id": d.external_id, "title": d.title, "text": d.text}
            for d in s.query(Document).filter_by(project_id=project_id).all()
        ]
    finally:
        s.close()


# ----------------------------------------------------------------- test sets
def save_test_set(project_id: int, name: str, cases: list[dict],
                  source: str = "generated") -> int:
    """Persist a test set and its cases. Returns the test_set id."""
    s = get_session()
    try:
        ts = TestSet(project_id=project_id, name=name, source=source)
        s.add(ts)
        s.flush()  # get ts.id before adding cases
        for c in cases:
            s.add(TestCase(
                test_set_id=ts.id,
                external_id=c["id"],
                question=c["question"],
                expected_answer=c["expected_answer"],
                expected_doc_ids_json=json.dumps(c["expected_doc_ids"]),
                category=c["category"],
            ))
        s.commit()
        return ts.id
    finally:
        s.close()


def list_test_sets(project_id: int) -> list[dict]:
    s = get_session()
    try:
        out = []
        for ts in s.query(TestSet).filter_by(project_id=project_id).all():
            out.append({
                "id": ts.id, "name": ts.name, "source": ts.source,
                "num_cases": len(ts.cases),
            })
        return out
    finally:
        s.close()


def get_test_cases(test_set_id: int) -> list[dict]:
    """Returns cases in the engine's expected shape."""
    s = get_session()
    try:
        return [
            {
                "id": c.external_id,
                "question": c.question,
                "expected_answer": c.expected_answer,
                "expected_doc_ids": json.loads(c.expected_doc_ids_json),
                "category": c.category,
            }
            for c in s.query(TestCase).filter_by(test_set_id=test_set_id).all()
        ]
    finally:
        s.close()


def latest_test_set_id(project_id: int) -> int | None:
    s = get_session()
    try:
        ts = (s.query(TestSet).filter_by(project_id=project_id)
              .order_by(TestSet.created_at.desc()).first())
        return ts.id if ts else None
    finally:
        s.close()


# ---------------------------------------------------------------------- runs
def save_run(project_id: int, result: dict, label: str = "",
             test_set_id: int | None = None) -> int:
    s = get_session()
    try:
        overall = result["summary"]["overall"]
        run = Run(
            project_id=project_id,
            test_set_id=test_set_id,
            label=label or result["config"].get("generator_config", "run"),
            faithfulness_avg=overall["faithfulness_avg"],
            correctness_avg=overall["correctness_avg"],
            refusal_accuracy=overall["refusal_accuracy"],
            retrieval_hit=overall["retrieval_hit"],
            retrieval_mrr=overall["retrieval_mrr"],
            result_json=json.dumps(result),
        )
        s.add(run)
        s.commit()
        return run.id
    finally:
        s.close()


def list_runs(project_id: int) -> list[dict]:
    """Run history for trend charts -- summary columns only, no heavy JSON."""
    s = get_session()
    try:
        return [
            {
                "id": r.id, "label": r.label,
                "created_at": r.created_at.isoformat(),
                "faithfulness_avg": r.faithfulness_avg,
                "correctness_avg": r.correctness_avg,
                "refusal_accuracy": r.refusal_accuracy,
                "retrieval_hit": r.retrieval_hit,
                "retrieval_mrr": r.retrieval_mrr,
            }
            for r in (s.query(Run).filter_by(project_id=project_id)
                      .order_by(Run.created_at.asc()).all())
        ]
    finally:
        s.close()


def get_run(run_id: int) -> dict | None:
    s = get_session()
    try:
        r = s.get(Run, run_id)
        if not r:
            return None
        return {
            "id": r.id, "label": r.label, "project_id": r.project_id,
            "created_at": r.created_at.isoformat(),
            "result": json.loads(r.result_json),
        }
    finally:
        s.close()

# -------------------------------------------------------------------- targets
def save_target(project_id: int, name: str, target_type: str,
                config: dict) -> int:
    """Persist a target (builtin_rag or http_endpoint). Returns target id."""
    s = get_session()
    try:
        t = Target(
            project_id=project_id,
            name=name,
            type=target_type,
            config_json=json.dumps(config),
        )
        s.add(t)
        s.commit()
        return t.id
    finally:
        s.close()


def list_targets(project_id: int) -> list[dict]:
    s = get_session()
    try:
        return [
            {"id": t.id, "name": t.name, "type": t.type,
             "config": json.loads(t.config_json)}
            for t in s.query(Target).filter_by(project_id=project_id).all()
        ]
    finally:
        s.close()


def get_target(target_id: int) -> dict | None:
    s = get_session()
    try:
        t = s.get(Target, target_id)
        if not t:
            return None
        return {"id": t.id, "name": t.name, "type": t.type,
                "config": json.loads(t.config_json)}
    finally:
        s.close()

# ----------------------------------------------------------------------- jobs
def create_job(project_id: int, params: dict, kind: str = "evaluation") -> int:
    s = get_session()
    try:
        job = Job(project_id=project_id, kind=kind, status="pending",
                  params_json=json.dumps(params))
        s.add(job)
        s.commit()
        return job.id
    finally:
        s.close()


def update_job(job_id: int, *, status: str = None, run_id: int = None,
               error: str = None):
    s = get_session()
    try:
        job = s.get(Job, job_id)
        if not job:
            return
        if status is not None:
            job.status = status
        if run_id is not None:
            job.run_id = run_id
        if error is not None:
            job.error = error
        s.commit()
    finally:
        s.close()


def get_job(job_id: int) -> dict | None:
    s = get_session()
    try:
        j = s.get(Job, job_id)
        if not j:
            return None
        return {"id": j.id, "project_id": j.project_id, "status": j.status,
                "kind": j.kind, "run_id": j.run_id, "error": j.error,
                "created_at": j.created_at.isoformat(),
                "updated_at": j.updated_at.isoformat()}
    finally:
        s.close()

