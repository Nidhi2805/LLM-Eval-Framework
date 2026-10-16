"""
Healthcare knowledge base for the RAG system under test.

Design notes (important for evaluation integrity):
- Each doc states clear, checkable facts so faithfulness can be scored objectively.
- Some docs overlap in topic (e.g. diabetes types, hypertension stages) on purpose,
  so the retriever can plausibly fetch a near-but-wrong chunk. This gives our
  retrieval and faithfulness metrics something real to detect.
- Content is general medical-education material, not patient-specific advice.
"""

CORPUS = [
    {
        "id": "doc_001",
        "title": "Type 1 Diabetes",
        "text": (
            "Type 1 diabetes is an autoimmune condition in which the immune system "
            "destroys the insulin-producing beta cells of the pancreas. It usually "
            "appears in childhood or adolescence. People with Type 1 diabetes produce "
            "little or no insulin and require lifelong insulin therapy. It accounts for "
            "roughly 5 to 10 percent of all diabetes cases."
        ),
    },
    {
        "id": "doc_002",
        "title": "Type 2 Diabetes",
        "text": (
            "Type 2 diabetes is a metabolic disorder characterized by insulin resistance "
            "and relative insulin deficiency. It is strongly associated with obesity, "
            "physical inactivity, and age. Unlike Type 1, it often develops in adults and "
            "can initially be managed with lifestyle changes and oral medications such as "
            "metformin. It accounts for about 90 to 95 percent of diabetes cases."
        ),
    },
    {
        "id": "doc_003",
        "title": "Hypertension Overview",
        "text": (
            "Hypertension, or high blood pressure, is defined as a sustained blood "
            "pressure of 130/80 mmHg or higher under current guidelines. It is often "
            "called a silent condition because it typically produces no symptoms. Chronic "
            "hypertension increases the risk of stroke, heart attack, and kidney disease."
        ),
    },
    {
        "id": "doc_004",
        "title": "Hypertension Management",
        "text": (
            "Management of hypertension begins with lifestyle measures: reducing sodium "
            "intake, regular aerobic exercise, weight loss, and limiting alcohol. When "
            "these are insufficient, first-line medications include thiazide diuretics, "
            "ACE inhibitors, and calcium channel blockers. Blood pressure targets are "
            "individualized based on age and comorbidities."
        ),
    },
    {
        "id": "doc_005",
        "title": "Asthma",
        "text": (
            "Asthma is a chronic inflammatory disease of the airways causing wheezing, "
            "shortness of breath, chest tightness, and coughing. Symptoms are triggered "
            "by allergens, exercise, cold air, or respiratory infections. Treatment "
            "includes inhaled corticosteroids for long-term control and short-acting "
            "beta-agonists such as albuterol for rapid symptom relief."
        ),
    },
    {
        "id": "doc_006",
        "title": "Iron-Deficiency Anemia",
        "text": (
            "Iron-deficiency anemia occurs when the body lacks enough iron to produce "
            "adequate hemoglobin, reducing the blood's oxygen-carrying capacity. Common "
            "causes include blood loss, poor dietary intake, and malabsorption. Symptoms "
            "include fatigue, pallor, and shortness of breath. It is typically treated "
            "with oral iron supplements and dietary changes."
        ),
    },
    {
        "id": "doc_007",
        "title": "Hypothyroidism",
        "text": (
            "Hypothyroidism is a condition in which the thyroid gland produces "
            "insufficient thyroid hormone. Symptoms include fatigue, weight gain, cold "
            "intolerance, and dry skin. It is commonly caused by Hashimoto's thyroiditis, "
            "an autoimmune disorder. Standard treatment is daily oral levothyroxine, a "
            "synthetic thyroid hormone."
        ),
    },
    {
        "id": "doc_008",
        "title": "Migraine",
        "text": (
            "Migraine is a neurological disorder marked by recurrent moderate to severe "
            "headaches, often on one side of the head, and frequently accompanied by "
            "nausea and sensitivity to light and sound. Some people experience an aura "
            "beforehand. Treatment includes acute medications such as triptans and "
            "preventive medications for frequent attacks."
        ),
    },
]