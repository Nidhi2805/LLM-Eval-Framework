"""
Auto test-set generator -- builds a candidate golden evaluation set from a set of
documents, removing the biggest barrier to LLM evaluation: hand-writing ground truth.

Approach:
- ANSWERABLE cases: for each document, generate question/answer pairs grounded in
  THAT document. Because the question derives from a known doc, the expected source
  is known automatically, and the answer is extracted from the doc (verifiable).
- UNANSWERABLE cases: generate domain-plausible questions NOT covered by any
  document, to test that the system correctly refuses. Expected source is empty.
- VALIDATION pass: each generated answerable question is run back through retrieval;
  if its intended source doc isn't retrieved, the case is flagged low-confidence.
  The generator checks its own output rather than trusting it blindly.

Output is a CANDIDATE set intended for human review, not a replacement for it.
"""

import json
from rag.llm_client import generate
from rag.retriever import Retriever

_GEN_RUBRIC = """You generate evaluation questions for a document-grounded QA system.

Given a single DOCUMENT, produce question-answer pairs that are FULLY answerable
from that document alone. Each answer must be directly supported by the document
text -- do not add outside knowledge. Questions should be natural and specific.

Respond ONLY with valid JSON in this exact format:
{"pairs": [{"question": "...", "answer": "..."}, ...]}"""

_UNANSWERABLE_RUBRIC = """You generate questions that sound relevant to a topic area
but are NOT answerable from a given set of document titles -- to test whether a QA
system correctly declines to answer out-of-scope questions.

Given a list of document TITLES, produce questions about the same broad domain
(e.g. medicine) whose specific subject is clearly NOT among the titles. The
questions must be plausible but genuinely out of scope.

Respond ONLY with valid JSON in this exact format:
{"questions": ["...", "..."]}"""


def _generate_answerable(documents: list[dict], per_doc: int) -> list[dict]:
    """Generate answerable Q/A pairs grounded in each document."""
    cases = []
    counter = 1
    for doc in documents:
        prompt = (
            f"DOCUMENT TITLE: {doc['title']}\n"
            f"DOCUMENT TEXT:\n{doc['text']}\n\n"
            f"Generate {per_doc} question-answer pairs answerable from this document."
        )
        try:
            raw = generate(_GEN_RUBRIC, prompt, temperature=0.4, json_mode=True)
            pairs = json.loads(raw)["pairs"]
        except (json.JSONDecodeError, KeyError):
            continue  # skip a doc whose generation failed rather than crash
        for p in pairs:
            cases.append({
                "id": f"gen_{counter:03d}",
                "question": p["question"],
                "expected_answer": p["answer"],
                "expected_doc_ids": [doc["id"]],   # known source: this document
                "category": "answerable",
            })
            counter += 1
    return cases


def _generate_unanswerable(documents: list[dict], count: int,
                           start_idx: int) -> list[dict]:
    """Generate out-of-scope questions that should be refused."""
    titles = [d["title"] for d in documents]
    prompt = (
        f"DOCUMENT TITLES:\n- " + "\n- ".join(titles) +
        f"\n\nGenerate {count} out-of-scope questions."
    )
    try:
        raw = generate(_UNANSWERABLE_RUBRIC, prompt, temperature=0.5, json_mode=True)
        questions = json.loads(raw)["questions"]
    except (json.JSONDecodeError, KeyError):
        return []
    cases = []
    for i, q in enumerate(questions):
        cases.append({
            "id": f"gen_{start_idx + i:03d}",
            "question": q,
            "expected_answer": "I don't have enough information to answer that.",
            "expected_doc_ids": [],
            "category": "unanswerable",
        })
    return cases


def _validate(cases: list[dict], documents: list[dict], k: int = 2) -> list[dict]:
    """Flag answerable cases whose intended source doc isn't retrieved.

    Adds a 'confidence' field: 'high' if the expected doc is retrieved, else 'low'.
    This is the generator checking its own output -- low-confidence cases are kept
    but marked, so a human reviewer knows where to look.
    """
    retriever = Retriever(documents)
    for case in cases:
        if case["category"] != "answerable":
            case["confidence"] = "n/a"
            continue
        retrieved = [c["id"] for c in retriever.retrieve(case["question"], k=k)]
        expected = case["expected_doc_ids"][0]
        case["confidence"] = "high" if expected in retrieved else "low"
    return cases


def generate_test_set(
    documents: list[dict],
    per_doc: int = 2,
    unanswerable_count: int = 2,
) -> list[dict]:
    """Generate a validated candidate golden test set from documents."""
    answerable = _generate_answerable(documents, per_doc)
    unanswerable = _generate_unanswerable(
        documents, unanswerable_count, start_idx=len(answerable) + 1
    )
    all_cases = answerable + unanswerable
    return _validate(all_cases, documents)