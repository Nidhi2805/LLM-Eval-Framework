"""
Faithfulness metric -- measures whether the generated answer is grounded in the
retrieved context, using an LLM as a judge.

WHY an LLM judge: grounding is a semantic judgment with no closed-form formula.
A second LLM call, scoped narrowly and given a strict rubric, approximates a human
rater at scale.

Reliability safeguards against known LLM-judge failure modes:
- temperature = 0            -> repeatable, deterministic scores
- structured JSON output     -> auditable (score + reason), not a black box
- explicit 1-5 rubric        -> reduces leniency drift and arbitrary scoring
- narrow definition          -> judge ONLY checks grounding, not overall quality

A refusal ("I don't have enough information...") is treated as trivially faithful,
because declining to answer cannot introduce unsupported claims.
"""

import json
from rag.llm_client import generate
from metrics.refusal import is_refusal

FAITHFULNESS_RUBRIC = """You are a strict evaluator of factual grounding.

You will be given a CONTEXT and an ANSWER. Your ONLY job is to judge whether every
claim in the ANSWER is directly supported by the CONTEXT. Do NOT judge whether the
answer is helpful, complete, or well-written. Do NOT use any outside knowledge --
even if a claim is true in reality, if it is not in the CONTEXT it is unsupported.

Score on this exact scale:
5 = Every claim in the answer is fully supported by the context.
4 = All claims supported, but includes minor phrasing that slightly overstates.
3 = Mostly supported, but contains at least one claim not found in the context.
2 = Several claims are unsupported by the context.
1 = The answer is largely or entirely unsupported by the context (hallucinated).

Respond ONLY with valid JSON in this exact format:
{"score": <1-5>, "reason": "<one sentence explaining the score>"}"""


def score_faithfulness(answer: str, context_chunks: list[dict]) -> dict:
    """Judge how well the answer is grounded in the retrieved context."""
    # A refusal cannot contain unsupported claims -> trivially faithful.
    if is_refusal(answer):
        return {"score": 5, "reason": "Refusal; makes no unsupported claims."}

    context_text = "\n\n".join(
        f"[{c['title']}]\n{c['text']}" for c in context_chunks
    )
    prompt = (
        f"CONTEXT:\n{context_text}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"Evaluate the faithfulness of the ANSWER to the CONTEXT."
    )

    response_text = generate(
        system_instruction=FAITHFULNESS_RUBRIC,
        prompt=prompt,
        temperature=0.0,
        json_mode=True,
    )
    try:
        result = json.loads(response_text)
        return {"score": int(result["score"]), "reason": result["reason"]}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Defensive: never let a malformed judge response crash an eval run.
        return {"score": 0, "reason": f"Judge parse error: {e}"}