from data.golden_dataset import GOLDEN_DATASET
from data.corpus import CORPUS

corpus_ids = {doc["id"] for doc in CORPUS}

print(f"Loaded {len(GOLDEN_DATASET)} golden questions\n")

# Count categories
from collections import Counter
cats = Counter(item["category"] for item in GOLDEN_DATASET)
print("By category:", dict(cats))

# Integrity check: every expected_doc_id must actually exist in the corpus
print("\nIntegrity check:")
all_good = True
for item in GOLDEN_DATASET:
    for doc_id in item["expected_doc_ids"]:
        if doc_id not in corpus_ids:
            print(f"  ERROR: {item['id']} references missing doc {doc_id}")
            all_good = False
if all_good:
    print("  All expected_doc_ids exist in corpus. ✓")