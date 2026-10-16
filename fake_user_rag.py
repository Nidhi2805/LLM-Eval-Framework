"""
A tiny standalone 'user's RAG API' -- stands in for a real user's deployed system
so we can test external-endpoint evaluation end to end. Run it on its own port.

It deliberately uses a DIFFERENT response shape than our platform, to prove the
adapter config handles arbitrary endpoints:  {"result": {"text": "..."}}
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# A trivial "knowledge base" baked in
KB = {
    "diabetes": "Type 1 diabetes is an autoimmune condition requiring insulin therapy.",
    "asthma": "Asthma is treated with inhaled corticosteroids and albuterol for relief.",
    "hypertension": "Hypertension is treated with lifestyle changes and medications like ACE inhibitors.",
}


class Q(BaseModel):
    question: str   # note: field is 'question', not 'query' -> tests adapter config


@app.post("/ask")
def ask(q: Q):
    text = "I don't have information on that."
    for key, ans in KB.items():
        if key in q.question.lower():
            text = ans
            break
    # Deliberately nested, different shape from our platform:
    return {"result": {"text": text}, "meta": {"source": "fake-rag"}}