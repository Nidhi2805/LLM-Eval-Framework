from rag.retriever import Retriever
from rag.generator import generate_answer

retriever = Retriever()

questions = [
    "What causes Type 1 diabetes?",                    # in corpus - should answer well
    "How is high blood pressure treated?",             # in corpus - should answer well
    "What are the symptoms of COVID-19?",              # NOT in corpus - should decline
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    chunks = retriever.retrieve(q, k=2)
    print(f"Retrieved: {', '.join(c['title'] for c in chunks)}")
    answer = generate_answer(q, chunks)
    print(f"A: {answer}")