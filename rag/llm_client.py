"""
Provider-agnostic LLM client -- the single choke point for every LLM call in the
framework. All generation and judging goes through `generate()`, so the rest of
the codebase is decoupled from any specific vendor.

Currently wired to Groq (fast, generous free tier). Because every caller depends
only on this module's `generate()` signature -- not on a vendor SDK -- switching
providers (Gemini, OpenRouter, local Ollama) is a change to this file alone.

Resilience: transient API failures (429 rate-limit, 5xx) are retried with
exponential backoff, so a single hiccup cannot kill a multi-call eval run.
"""

import os
from dotenv import load_dotenv
from groq import Groq, APIError, APIStatusError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

load_dotenv()
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DEFAULT_MODEL = "openai/gpt-oss-120b"


@retry(
    retry=retry_if_exception_type((APIError, APIStatusError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
def generate(
    system_instruction: str,
    prompt: str,
    temperature: float = 0.1,
    json_mode: bool = False,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Single entry point for all LLM calls. Returns the response text directly.

    Args:
        system_instruction: the system prompt (role/rules for the model)
        prompt: the user message
        temperature: 0.0 for judges (repeatable), low for generation
        json_mode: force valid-JSON output (used by the LLM judges)
        model: override the default model if needed
    """
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = _client.chat.completions.create(**kwargs)
    except (APIError, APIStatusError) as e:
        print(f"  [retrying after API error: {e}]")
        raise
    return response.choices[0].message.content.strip()