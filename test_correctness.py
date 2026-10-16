from rag.retriever import Retriever
from rag.generator import generate_answer
from metrics.correctness import score_correctness
from data.golden_dataset import GOLDEN_DATASET

retriever = Retriever()

# Score correctness on the first few answerable golden items
for item in GOLDEN_DATASET[:4]:
    if item["category"] == "unanswerable":
        continue
    chunks = retriever.retrieve(item["question"], k=2)
    answer = generate_answer(item["question"], chunks)
    result = score_correctness(item["question"], answer, item["expected_answer"])
    print(f"\nQ: {item['question']}")
    print(f"A: {answer[:100]}...")
    print(f"  Correctness: {result['score']}/5 -- {result['reason']}")

# Demonstration: a WRONG answer vs the reference, to prove the judge catches it
print("\n" + "="*60)
print("WRONG-ANSWER TEST (judge must score low):")
q = "What is the standard treatment for hypothyroidism?"
ref = "The standard treatment is daily oral levothyroxine."
wrong = "The standard treatment for hypothyroidism is insulin injections."
result = score_correctness(q, wrong, ref)
print(f"Wrong answer: {wrong}")
print(f"  Correctness: {result['score']}/5 -- {result['reason']}")