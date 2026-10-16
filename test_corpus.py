from data.corpus import CORPUS

print(f"Loaded {len(CORPUS)} documents")
for doc in CORPUS:
    print(f"  {doc['id']}: {doc['title']} ({len(doc['text'].split())} words)")