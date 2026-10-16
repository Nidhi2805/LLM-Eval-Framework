"""
Golden evaluation dataset -- the ground truth the whole framework measures against.

Each item records:
- question:        what we ask the RAG system
- expected_answer: a reference answer (used by answer-correctness metric)
- expected_doc_ids: which corpus doc(s) SHOULD be retrieved (grades retrieval)
- category:        answerable | unanswerable | trap  (enables per-category scoring)
- ground_truth_facts: specific checkable facts the answer should contain / respect
                      (used by the faithfulness metric)

Design intent:
- 'answerable' items verify normal correct behavior.
- 'unanswerable' items verify the system safely declines (expected_doc_ids is empty;
  the correct answer is a refusal).
- 'trap' items are answerable BUT the question is phrased so a weak retriever might
  fetch a topically-similar wrong doc. These stress-test retrieval precision.
"""

GOLDEN_DATASET = [
    {
        "id": "q_001",
        "question": "What causes Type 1 diabetes?",
        "expected_answer": "Type 1 diabetes is caused by an autoimmune reaction in which the immune system destroys the insulin-producing beta cells of the pancreas.",
        "expected_doc_ids": ["doc_001"],
        "category": "answerable",
        "ground_truth_facts": [
            "It is an autoimmune condition.",
            "The immune system destroys insulin-producing beta cells.",
        ],
    },
    {
        "id": "q_002",
        "question": "How is high blood pressure treated?",
        "expected_answer": "Treatment begins with lifestyle changes such as reducing sodium, exercise, weight loss, and limiting alcohol. If needed, first-line medications include thiazide diuretics, ACE inhibitors, and calcium channel blockers.",
        "expected_doc_ids": ["doc_004"],
        "category": "answerable",
        "ground_truth_facts": [
            "Lifestyle measures are the first step.",
            "First-line medications include thiazide diuretics, ACE inhibitors, and calcium channel blockers.",
        ],
    },
    {
        "id": "q_003",
        "question": "What is the standard treatment for hypothyroidism?",
        "expected_answer": "The standard treatment for hypothyroidism is daily oral levothyroxine, a synthetic thyroid hormone.",
        "expected_doc_ids": ["doc_007"],
        "category": "answerable",
        "ground_truth_facts": [
            "The treatment is levothyroxine.",
            "It is a synthetic thyroid hormone taken daily.",
        ],
    },
    {
        "id": "q_004",
        "question": "What medication provides rapid relief during an asthma attack?",
        "expected_answer": "Short-acting beta-agonists such as albuterol provide rapid symptom relief during an asthma attack.",
        "expected_doc_ids": ["doc_005"],
        "category": "answerable",
        "ground_truth_facts": [
            "Short-acting beta-agonists provide rapid relief.",
            "Albuterol is an example.",
        ],
    },
    {
        "id": "q_005",
        "question": "What percentage of diabetes cases are Type 2?",
        "expected_answer": "Type 2 diabetes accounts for about 90 to 95 percent of all diabetes cases.",
        "expected_doc_ids": ["doc_002"],
        "category": "answerable",
        "ground_truth_facts": [
            "Type 2 accounts for about 90 to 95 percent of diabetes cases.",
        ],
    },
    {
        "id": "q_006",
        "question": "What are the symptoms of COVID-19?",
        "expected_answer": "I don't have enough information to answer that.",
        "expected_doc_ids": [],
        "category": "unanswerable",
        "ground_truth_facts": [
            "The system should decline because the corpus has no COVID-19 information.",
        ],
    },
    {
        "id": "q_007",
        "question": "How is cancer treated with chemotherapy?",
        "expected_answer": "I don't have enough information to answer that.",
        "expected_doc_ids": [],
        "category": "unanswerable",
        "ground_truth_facts": [
            "The system should decline because the corpus has no cancer information.",
        ],
    },
    {
        "id": "q_008",
        "question": "What defines high blood pressure?",
        "expected_answer": "Hypertension is defined as a sustained blood pressure of 130/80 mmHg or higher.",
        "expected_doc_ids": ["doc_003"],
        "category": "trap",
        "ground_truth_facts": [
            "High blood pressure is defined as 130/80 mmHg or higher.",
        ],
    },
    {
        "id": "q_009",
        "question": "What is used to treat iron-deficiency anemia?",
        "expected_answer": "Iron-deficiency anemia is treated with oral iron supplements and dietary changes.",
        "expected_doc_ids": ["doc_006"],
        "category": "answerable",
        "ground_truth_facts": [
            "It is treated with oral iron supplements.",
            "Dietary changes are also used.",
        ],
    },
    {
        "id": "q_010",
        "question": "Which type of diabetes requires lifelong insulin therapy?",
        "expected_answer": "Type 1 diabetes requires lifelong insulin therapy because the body produces little or no insulin.",
        "expected_doc_ids": ["doc_001"],
        "category": "trap",
        "ground_truth_facts": [
            "Type 1 diabetes requires lifelong insulin therapy.",
        ],
    },
]