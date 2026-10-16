"""Confirm the target-agnostic runner still scores the built-in RAG correctly."""

from data.corpus import CORPUS
from data.golden_dataset import GOLDEN_DATASET
from runner import run_evaluation

documents = [{"id": d["id"], "title": d["title"], "text": d["text"]} for d in CORPUS]
test_cases = [
    {"id": t["id"], "question": t["question"],
     "expected_answer": t["expected_answer"],
     "expected_doc_ids": t["expected_doc_ids"], "category": t["category"]}
    for t in GOLDEN_DATASET
]

print("Testing built-in target through the new engine...\n")
result = run_evaluation(test_cases, target_type="builtin_rag", documents=documents)

print(f"target_type: {result['config']['target_type']}")
print("OVERALL")
for metric, value in result["summary"]["overall"].items():
    print(f"  {metric:<22} {value}")