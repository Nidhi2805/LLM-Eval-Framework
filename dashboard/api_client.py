"""
Thin HTTP client for the dashboard -> FastAPI backend.

Isolating all network calls here keeps the Streamlit UI focused on presentation,
and means the backend URL is configured in exactly one place (so pointing the
dashboard at a deployed API instead of localhost is a one-line change / env var).
"""

import os
import requests

# Base URL of the backend. Overridable by env var so the same dashboard works
# locally and in deployment without code changes.
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

TIMEOUT = 600  # eval calls are slow (many LLM calls); generous timeout


def _url(path: str) -> str:
    return f"{API_BASE}{path}"


def health() -> bool:
    try:
        r = requests.get(_url("/"), timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_projects() -> list[dict]:
    return requests.get(_url("/projects"), timeout=TIMEOUT).json()


def create_project(name: str, description: str = "") -> dict:
    return requests.post(_url("/projects"),
                         json={"name": name, "description": description},
                         timeout=TIMEOUT).json()


def add_documents(project_id: int, documents: list[dict]) -> dict:
    return requests.post(_url(f"/projects/{project_id}/documents"),
                         json={"documents": documents}, timeout=TIMEOUT).json()


def get_documents(project_id: int) -> list[dict]:
    return requests.get(_url(f"/projects/{project_id}/documents"),
                        timeout=TIMEOUT).json()


def generate_testset(project_id: int, name: str, per_doc: int,
                     unanswerable_count: int) -> dict:
    return requests.post(
        _url(f"/projects/{project_id}/testsets/generate"),
        json={"name": name, "per_doc": per_doc,
              "unanswerable_count": unanswerable_count},
        timeout=TIMEOUT).json()


def list_testsets(project_id: int) -> list[dict]:
    return requests.get(_url(f"/projects/{project_id}/testsets"),
                        timeout=TIMEOUT).json()


def evaluate(project_id: int, config: str = "grounded", k: int = 2,
             label: str = "") -> dict:
    return requests.post(
        _url(f"/projects/{project_id}/evaluate"),
        json={"config": config, "k": k, "label": label},
        timeout=TIMEOUT).json()


def list_runs(project_id: int) -> list[dict]:
    return requests.get(_url(f"/projects/{project_id}/runs"),
                        timeout=TIMEOUT).json()