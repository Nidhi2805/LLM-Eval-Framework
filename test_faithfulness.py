from rag.retriever import Retriever
from rag.generator import generate_answer
from metrics.faithfulness import score_faithfulness

retriever = Retriever()

# Test on a few golden questions + one deliberately hallucinated answer
tests = [
    "What causes Type 1 diabetes?",
    "What are the symptoms of COVID-19?",   # should refuse -> faithful 5
]

for q in tests:
    chunks = retriever.retrieve(q, k=2)
    answer = generate_answer(q, chunks)
    result = score_faithfulness(answer, chunks)
    print(f"\nQ: {q}")
    print(f"A: {answer}")
    print(f"  Faithfulness: {result['score']}/5 -- {result['reason']}")

# Now the key demonstration: inject a FAKE unfaithful answer and prove the judge catches it
print("\n" + "="*60)
print("INJECTED HALLUCINATION TEST (the judge must catch this):")
chunks = retriever.retrieve("What causes Type 1 diabetes?", k=2)
fake_answer = (
    "Type 1 diabetes is caused by eating too much sugar and can be fully "
    "cured by following a keto diet for three weeks."
)
result = score_faithfulness(fake_answer, chunks)
print(f"Fake answer: {fake_answer}")
print(f"  Faithfulness: {result['score']}/5 -- {result['reason']}")