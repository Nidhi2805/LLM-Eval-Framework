from data.corpus import CORPUS
from rag.testset_generator import generate_test_set

documents = [
    {"id": d["id"], "title": d["title"], "text": d["text"]} for d in CORPUS
]

print("Generating a candidate test set from the corpus...\n")
cases = generate_test_set(documents, per_doc=2, unanswerable_count=2)

print(f"Generated {len(cases)} candidate test cases:\n")
for c in cases:
    conf = c.get("confidence", "?")
    flag = "" if conf in ("high", "n/a") else "  <-- REVIEW (low confidence)"
    print(f"[{c['category']:<12} | conf:{conf:<4}] {c['id']}")
    print(f"   Q: {c['question']}")
    print(f"   A: {c['expected_answer'][:80]}")
    print(f"   source: {c['expected_doc_ids']}{flag}\n")