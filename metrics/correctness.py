"""
Answer-correctness metric -- measures whether the generated answer matches the
known-correct reference answer from the golden dataset, using an LLM judge.

Distinct from faithfulness:
- faithfulness compares answer <-> retrieved context (grounding)
- correctness   compares answer <-> golden expected_answer (actual rightness)

The two together localize failures: faithful-but-incorrect implicates retrieval;
correct-but-unfaithful implicates the generator ignoring its context.

Judge compares for SEMANTIC equivalence, not exact wording, so valid paraphrases
are not penalized. Same reliability safeguards as the faithfulness judge.
"""

import json
from rag.llm_client import generate

CORRECTNESS_RUBRIC = """You are a strict evaluator of answer correctness.

You will be given a QUESTION, a REFERENCE ANSWER (the known-correct answer), and a
CANDIDATE ANSWER. Judge whether the CANDIDATE ANSWER is correct, meaning it conveys
the same essential factual information as the REFERENCE ANSWER. Judge MEANING, not
exact wording -- valid paraphrases are fully correct. The candidate may include extra
correct detail without penalty, but must not contradict or omit the core fact.

Score on this exact scale:
5 = Fully correct; conveys the same core facts as the reference.
4 = Correct core fact, but missing a minor secondary detail.
3 = Partially correct; captures some but not all of the core fact.
2 = Mostly incorrect; misses or distorts the core fact.
1 = Incorrect or contradicts the reference answer.

Respond ONLY with valid JSON in this exact format:
{"score": <1-5>, "reason": "<one sentence explaining the score>"}"""


def score_correctness(question: str, candidate: str, reference: str) -> dict:
    """Judge whether the candidate answer matches the reference answer."""
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"CANDIDATE ANSWER:\n{candidate}\n\n"
        f"Evaluate the correctness of the CANDIDATE ANSWER."
    )
    response_text = generate(
        system_instruction=CORRECTNESS_RUBRIC,
        prompt=prompt,
        temperature=0.0,
        json_mode=True,
    )
    try:
        result = json.loads(response_text)
        return {"score": int(result["score"]), "reason": result["reason"]}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return {"score": 0, "reason": f"Judge parse error: {e}"}