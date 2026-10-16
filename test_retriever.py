from rag.retriever import Retriever

retriever = Retriever()

queries = [
    "What causes Type 1 diabetes?",
    "How is high blood pressure treated?",
    "What medication is used for an asthma attack?",
]

for q in queries:
    print(f"\nQuery: {q}")
    results = retriever.retrieve(q, k=2)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['id']} - {r['title']}")