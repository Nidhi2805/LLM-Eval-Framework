from rag.retriever import Retriever
from data.golden_dataset import GOLDEN_DATASET
from metrics.retrieval import score_retrieval

retriever = Retriever()

# Only score items that have expected docs (skip unanswerable)
scorable = [item for item in GOLDEN_DATASET if item["expected_doc_ids"]]

total_hit = total_rr = total_prec = 0.0

print(f"{'ID':<7} {'Cat':<12} {'Hit':<5} {'RR':<6} {'Prec':<5}  Retrieved")
print("-" * 70)

for item in scorable:
    chunks = retriever.retrieve(item["question"], k=2)
    retrieved_ids = [c["id"] for c in chunks]
    scores = score_retrieval(retrieved_ids, item["expected_doc_ids"])

    total_hit += scores["hit"]
    total_rr += scores["reciprocal_rank"]
    total_prec += scores["precision"]

    print(f"{item['id']:<7} {item['category']:<12} "
          f"{scores['hit']:<5.0f} {scores['reciprocal_rank']:<6.2f} "
          f"{scores['precision']:<5.2f}  {retrieved_ids}")

n = len(scorable)
print("-" * 70)
print(f"Hit@2:        {total_hit/n:.3f}")
print(f"MRR:          {total_rr/n:.3f}")
print(f"Precision@2:  {total_prec/n:.3f}")