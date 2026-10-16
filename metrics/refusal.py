"""
Refusal-accuracy metric -- checks whether the system declines exactly when it should.

Pure logic, no LLM. Guards against two opposite failures:
- answering an unanswerable question  -> unsafe (hallucination risk)
- refusing an answerable question      -> uselessly over-cautious

This metric is the counterweight to faithfulness: because a refusal is trivially
faithful, a system could score perfectly on faithfulness by refusing everything.
Refusal accuracy catches that degenerate behavior.
"""

# Multiple phrasings, because different systems decline differently.
# (A production version might use a tiny classifier or an LLM judge for this;
# a pattern set covers the common cases cheaply and deterministically.)
REFUSAL_PATTERNS = [
    "don't have enough information",
    "don't have information",
    "do not have enough information",
    "do not have information",
    "cannot answer",
    "can't answer",
    "not enough information",
    "no information",
    "unable to answer",
]


def is_refusal(answer: str) -> bool:
    """True if the answer is a refusal, tolerant of varied phrasings."""
    low = answer.lower()
    return any(pattern in low for pattern in REFUSAL_PATTERNS)


def score_refusal(answer: str, category: str) -> dict:
    """
    Judge refusal correctness for one item.
    - unanswerable: correct iff the system refused
    - answerable/trap: correct iff the system did NOT refuse
    Returns correctness (1.0/0.0) and a human-readable verdict.
    """
    refused = is_refusal(answer)
    should_refuse = category == "unanswerable"

    correct = (refused == should_refuse)

    if should_refuse and refused:
        verdict = "correctly refused unanswerable question"
    elif should_refuse and not refused:
        verdict = "FAILED: answered an unanswerable question (hallucination risk)"
    elif not should_refuse and refused:
        verdict = "FAILED: refused an answerable question (over-cautious)"
    else:
        verdict = "correctly answered answerable question"

    return {"correct": 1.0 if correct else 0.0, "verdict": verdict}