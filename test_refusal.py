from rag.retriever import Retriever
from rag.generator import generate_answer
from metrics.refusal import score_refusal
from data.golden_dataset import GOLDEN_DATASET

retriever = Retriever()

total = 0.0
print(f"{'ID':<7} {'Category':<13} {'Correct':<8} Verdict")
print("-" * 75)

for item in GOLDEN_DATASET:
    chunks = retriever.retrieve(item["question"], k=2)
    answer = generate_answer(item["question"], chunks)
    result = score_refusal(answer, item["category"])
    total += result["correct"]
    print(f"{item['id']:<7} {item['category']:<13} "
          f"{result['correct']:<8.0f} {result['verdict']}")

print("-" * 75)
print(f"Refusal accuracy: {total/len(GOLDEN_DATASET):.3f}  "
      f"({total:.0f}/{len(GOLDEN_DATASET)})")