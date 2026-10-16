"""
Target abstraction -- 'the thing being evaluated'.

A Target answers one question: given a query, return an answer and the context
that informed it. The eval engine depends only on this interface, not on any
concrete implementation, so new target types (built-in RAG, external HTTP endpoint,
and future ones) plug in without changing the engine or the metrics.

This is the extensibility seam the database was designed around (targets.type +
config_json). Adding a target type here + a branch in build_target() is the entire
cost of supporting a new kind of system-under-test.

Two implementations:
- BuiltinRagTarget : our own retriever + generator over uploaded documents.
- HttpEndpointTarget: calls the USER'S deployed RAG API. Makes the platform usable
  by anyone with a deployed system, not just our built-in pipeline.
"""

import json
import requests
from rag.retriever import Retriever
from rag.generator import generate_answer


class BuiltinRagTarget:
    """Evaluates our own RAG pipeline over a set of documents."""

    def __init__(self, documents: list[dict], k: int = 2, config: str = "grounded"):
        self.retriever = Retriever(documents)
        self.k = k
        self.config = config

    def answer(self, question: str) -> dict:
        chunks = self.retriever.retrieve(question, k=self.k)
        text = generate_answer(question, chunks, config=self.config)
        return {
            "answer": text,
            "contexts": chunks,                       # full chunks (for faithfulness)
            "retrieved_ids": [c["id"] for c in chunks],
        }


class HttpEndpointTarget:
    """
    Evaluates a user's OWN deployed RAG API.

    The user supplies an adapter config describing their API's shape, because every
    RAG API differs:
      - url               : the endpoint to POST to
      - query_field       : request JSON key holding the question (default "query")
      - answer_path       : dot-path into the response JSON for the answer text
                            (e.g. "answer" or "data.response.text")
      - contexts_path     : optional dot-path to a list of retrieved context strings
      - headers           : optional auth headers
    Faithfulness/retrieval can only be scored if the endpoint returns its contexts;
    if it doesn't, those metrics are skipped and we note why -- an honest degradation,
    not a silent wrong number.
    """

    def __init__(self, config: dict):
        self.url = config["url"]
        self.query_field = config.get("query_field", "query")
        self.answer_path = config.get("answer_path", "answer")
        self.contexts_path = config.get("contexts_path")  # may be None
        self.headers = config.get("headers", {})
        self.timeout = config.get("timeout", 30)

    @staticmethod
    def _dig(data: dict, path: str):
        """Follow a dot-path like 'data.response.text' into nested JSON."""
        cur = data
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
        return cur

    def answer(self, question: str) -> dict:
        payload = {self.query_field: question}
        try:
            resp = requests.post(self.url, json=payload, headers=self.headers,
                                 timeout=self.timeout)
            resp.raise_for_status()
            body = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            # A failing endpoint is a real result, not a crash: record the failure.
            return {"answer": f"[ENDPOINT ERROR: {e}]", "contexts": [],
                    "retrieved_ids": [], "endpoint_error": True}

        answer_text = self._dig(body, self.answer_path)
        if answer_text is None:
            answer_text = f"[COULD NOT FIND answer at path '{self.answer_path}']"

        # Contexts are optional; many endpoints don't expose them.
        contexts = []
        retrieved_ids = []
        if self.contexts_path:
            raw_ctx = self._dig(body, self.contexts_path)
            if isinstance(raw_ctx, list):
                for i, c in enumerate(raw_ctx):
                    text = c if isinstance(c, str) else json.dumps(c)
                    contexts.append({"id": f"ctx_{i}", "title": f"context_{i}",
                                     "text": text, "score": None})
                    retrieved_ids.append(f"ctx_{i}")

        return {"answer": str(answer_text), "contexts": contexts,
                "retrieved_ids": retrieved_ids}


def build_target(target_type: str, *, documents: list[dict] | None = None,
                 config: dict | None = None, k: int = 2,
                 generator_config: str = "grounded"):
    """Factory: construct the right Target from its type. The engine calls this
    and then only uses the common .answer() method."""
    if target_type == "builtin_rag":
        if not documents:
            raise ValueError("builtin_rag target requires documents.")
        return BuiltinRagTarget(documents, k=k, config=generator_config)
    if target_type == "http_endpoint":
        if not config or "url" not in config:
            raise ValueError("http_endpoint target requires config with a 'url'.")
        return HttpEndpointTarget(config)
    raise ValueError(f"Unknown target type: {target_type}")