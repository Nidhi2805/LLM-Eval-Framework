"""
Generator: takes a question plus retrieved context and produces an answer.

Now config-driven: the system prompt is selected by config name, so we can run
the SAME pipeline in a healthy mode and in deliberately-degraded modes. This is
what lets the framework prove it can DETECT a regression -- we compare a known-good
config against a known-bad one and confirm the metrics drop.

Configs:
- "grounded"   : the correct system. Answers only from context, refuses otherwise.
- "ungrounded" : grounding instruction removed. The model is free to use outside
                 knowledge and is not told to refuse. Expected to hallucinate more
                 and stop refusing -> faithfulness and refusal accuracy should fall.
                 This mirrors a real-world mistake: someone "simplifies" the system
                 prompt and silently breaks grounding.
"""

from rag.llm_client import generate

GENERATOR_CONFIGS = {
    "grounded": (
        "You are a medical information assistant. Answer the user's question using "
        "ONLY the information in the provided context. If the context does not contain "
        "enough information to answer, respond exactly with: "
        "'I don't have enough information to answer that.' "
        "Do not use outside knowledge. Be concise."
    ),
    "ungrounded": (
        "You are a helpful medical information assistant. Answer the user's question "
        "as helpfully as you can. Be concise."
    ),
}

DEFAULT_CONFIG = "grounded"


def generate_answer(
    question: str,
    context_chunks: list[dict],
    config: str = DEFAULT_CONFIG,
) -> str:
    """Generate an answer. `config` selects the system prompt / behavior."""
    system_instruction = GENERATOR_CONFIGS[config]
    context_text = "\n\n".join(
        f"[{c['title']}]\n{c['text']}" for c in context_chunks
    )
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"

    return generate(
        system_instruction=system_instruction,
        prompt=prompt,
        temperature=0.1,
    )