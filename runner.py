"""
Evaluation runner -- the target-agnostic eval engine.

It asks a Target for an answer to each test question and scores the result on the
four metrics. The Target may be our built-in RAG or a user's external RAG endpoint;
the engine depends only on Target.answer(), so the same scoring works for both.

Graceful degradation: if a target returns no contexts (common for external
endpoints), faithfulness and retrieval are skipped with a recorded reason rather
than fabricated. Correctness and refusal are always scored.
"""

from datetime import datetime, timezone
from collections import defaultdict

from rag.targets import build_target
from rag.llm_client import DEFAULT_MODEL
from metrics.retrieval import score_retrieval
from metrics.faithfulness import score_faithfulness
from metrics.correctness import score_correctness
from metrics.refusal import score_refusal

TOP_K = 2


def run_evaluation(
    test_cases: list[dict],
    target_type: str = "builtin_rag",
    documents: list[dict] | None = None,
    target_config: dict | None = None,
    k: int = TOP_K,
    generator_config: str = "grounded",
) -> dict:
    """
    Evaluate a target over the given test cases.

    target_type: "builtin_rag" (needs documents) | "http_endpoint" (needs target_config)
    """
    target = build_target(
        target_type, documents=documents, config=target_config,
        k=k, generator_config=generator_config,
    )
    started = datetime.now(timezone.utc)
    per_question = []

    for item in test_cases:
        result = target.answer(item["question"])
        answer = result["answer"]
        chunks = result["contexts"]
        retrieved_ids = result["retrieved_ids"]
        has_context = len(chunks) > 0

        # Retrieval: only scorable with known expected docs AND returned contexts
        if item["expected_doc_ids"] and has_context:
            retrieval = score_retrieval(retrieved_ids, item["expected_doc_ids"])
        else:
            retrieval = None

        # Faithfulness: needs context to check grounding against
        if has_context:
            faith = score_faithfulness(answer, chunks)
        else:
            faith = {"score": None,
                     "reason": "Skipped: target returned no contexts to check grounding."}

        # Correctness and refusal: always scorable
        correctness = score_correctness(item["question"], answer,
                                        item["expected_answer"])
        refusal = score_refusal(answer, item["category"])

        per_question.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "expected_doc_ids": item["expected_doc_ids"],
            "retrieval": retrieval,
            "faithfulness": faith,
            "correctness": correctness,
            "refusal": refusal,
            "endpoint_error": result.get("endpoint_error", False),
        })

    finished = datetime.now(timezone.utc)
    summary = _summarize(per_question)

    return {
        "config": {
            "timestamp": started.isoformat(),
            "duration_seconds": round((finished - started).total_seconds(), 1),
            "target_type": target_type,
            "generator_model": DEFAULT_MODEL,
            "judge_model": DEFAULT_MODEL,
            "generator_config": generator_config,
            "top_k": k,
            "num_questions": len(test_cases),
        },
        "summary": summary,
        "per_question": per_question,
    }


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _summarize(per_question: list[dict]) -> dict:
    faith_scores = [q["faithfulness"]["score"] for q in per_question]
    corr_scores = [q["correctness"]["score"] for q in per_question]
    refusal_scores = [q["refusal"]["correct"] for q in per_question]
    retr_items = [q for q in per_question if q["retrieval"] is not None]

    overall = {
        "faithfulness_avg": _mean(faith_scores),
        "correctness_avg": _mean(corr_scores),
        "refusal_accuracy": _mean(refusal_scores),
        "retrieval_hit": _mean([q["retrieval"]["hit"] for q in retr_items]),
        "retrieval_mrr": _mean([q["retrieval"]["reciprocal_rank"] for q in retr_items]),
    }

    by_cat = defaultdict(list)
    for q in per_question:
        by_cat[q["category"]].append(q)

    per_category = {}
    for cat, items in by_cat.items():
        per_category[cat] = {
            "count": len(items),
            "faithfulness_avg": _mean([q["faithfulness"]["score"] for q in items]),
            "correctness_avg": _mean([q["correctness"]["score"] for q in items]),
            "refusal_accuracy": _mean([q["refusal"]["correct"] for q in items]),
        }

    return {"overall": overall, "per_category": per_category}