"""
Retrieval metrics -- measure whether the retriever fetched the correct document(s).

No LLM needed: this is pure set/rank math against the golden dataset's
expected_doc_ids. These metrics are deterministic and cheap, which makes them a
reliable first line of defense in regression testing.

Metrics:
- hit@k:        1.0 if any expected doc is in the top-k retrieved, else 0.0
- reciprocal rank: 1/rank of the first correct doc (0 if not found) -> averaged = MRR
- precision@k:  fraction of retrieved docs that are relevant

Items with no expected docs (unanswerable questions) are not scored here.
"""


def hit_at_k(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """1.0 if at least one expected doc is among the retrieved ids."""
    return 1.0 if any(doc_id in expected_ids for doc_id in retrieved_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """1 / (rank of first correct doc). Rank is 1-indexed. 0.0 if none found."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Fraction of retrieved docs that are relevant."""
    if not retrieved_ids:
        return 0.0
    relevant = sum(1 for doc_id in retrieved_ids if doc_id in expected_ids)
    return relevant / len(retrieved_ids)


def score_retrieval(retrieved_ids: list[str], expected_ids: list[str]) -> dict:
    """Compute all retrieval metrics for a single query. Returns None-friendly dict."""
    return {
        "hit": hit_at_k(retrieved_ids, expected_ids),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, expected_ids),
        "precision": precision_at_k(retrieved_ids, expected_ids),
    }