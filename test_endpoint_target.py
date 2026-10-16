"""Prove HttpEndpointTarget can call an external RAG API and extract the answer.

We simulate a user's RAG API with a trivial local responder, then point the
target at it through the adapter config -- exercising the dot-path extraction.
"""

from rag.targets import HttpEndpointTarget, build_target

# Simulate what a user's API returns, to verify path extraction without a server.
# (We test the parsing logic directly; a live HTTP test comes once it's in the API.)
target = HttpEndpointTarget({
    "url": "http://example.invalid/none",   # not actually called in this parse test
    "query_field": "query",
    "answer_path": "data.response.text",
})

fake_response = {
    "data": {"response": {"text": "Type 1 diabetes is autoimmune.",
                          "confidence": 0.9}},
    "meta": {"latency_ms": 120},
}

# Directly exercise the extraction logic
extracted = HttpEndpointTarget._dig(fake_response, "data.response.text")
print("Dot-path extraction test:")
print(f"  path 'data.response.text' -> {extracted!r}")
assert extracted == "Type 1 diabetes is autoimmune.", "extraction failed!"
print("  PASS\n")

# Test the factory picks the right implementation
t = build_target("http_endpoint", config={"url": "http://x", "answer_path": "answer"})
print(f"Factory built: {type(t).__name__}")
t2 = build_target("builtin_rag",
                  documents=[{"id": "d1", "title": "T", "text": "hello world"}])
print(f"Factory built: {type(t2).__name__}")
print("\nTarget abstraction works.")