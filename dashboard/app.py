"""
Product dashboard for the RAG Eval Platform.

A multi-project control panel that drives the FastAPI backend over HTTP:
- pick or create a project
- upload documents
- auto-generate a test set
- run an evaluation
- view score history / regression trend across runs

Talks only to the API (dashboard.api_client) -- it never touches the database,
so UI and backend are cleanly decoupled.

Run with:  streamlit run dashboard/app.py   (backend must be running)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from dashboard import api_client as api

st.set_page_config(page_title="RAG Eval Platform", layout="wide")
st.title("🔬 RAG Eval Platform")
st.caption("Bring your own documents, auto-generate a test set, evaluate your RAG "
           "system, and track quality over time.")

# ---- Backend health gate -------------------------------------------------
if not api.health():
    st.error(f"Cannot reach the backend at {api.API_BASE}. "
             f"Start it with:  uvicorn api:app --reload")
    st.stop()

# ---- Sidebar: project selection / creation -------------------------------
st.sidebar.header("Project")
projects = api.list_projects()
proj_labels = {p["name"]: p for p in projects}

choice = st.sidebar.selectbox(
    "Select a project",
    ["➕ New project..."] + list(proj_labels.keys()),
)

if choice == "➕ New project...":
    with st.sidebar.form("new_project"):
        new_name = st.text_input("Project name")
        new_desc = st.text_input("Description", "")
        if st.form_submit_button("Create") and new_name:
            api.create_project(new_name, new_desc)
            st.rerun()
    st.info("Create a project in the sidebar to begin.")
    st.stop()

project = proj_labels[choice]
project_id = project["id"]
st.sidebar.caption(f"ID: {project_id} · {project.get('num_documents', 0)} docs "
                   f"· {project.get('num_runs', 0)} runs")

# ---- Tabs: the workflow --------------------------------------------------
tab_docs, tab_tests, tab_eval, tab_history = st.tabs(
    ["📄 Documents", "🧪 Test Set", "▶️ Evaluate", "📈 History"]
)

# --- Documents ---
with tab_docs:
    st.subheader("Documents")
    docs = api.get_documents(project_id)
    if docs:
        st.dataframe(
            pd.DataFrame([{"id": d["id"], "title": d["title"],
                           "words": len(d["text"].split())} for d in docs]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No documents yet. Add some below.")

    st.markdown("**Add a document**")
    with st.form("add_doc"):
        d_id = st.text_input("Document ID", "doc_001")
        d_title = st.text_input("Title", "")
        d_text = st.text_area("Text", "", height=150)
        if st.form_submit_button("Add document") and d_text:
            api.add_documents(project_id, [
                {"external_id": d_id, "title": d_title or "Untitled", "text": d_text}
            ])
            st.success("Added.")
            st.rerun()

# --- Test set ---
with tab_tests:
    st.subheader("Auto-generate a test set")
    st.caption("Generates a candidate golden set from this project's documents, "
               "including out-of-scope questions to test refusal. Review before trusting.")
    existing = api.list_testsets(project_id)
    if existing:
        st.dataframe(pd.DataFrame(existing), use_container_width=True, hide_index=True)

    with st.form("gen_testset"):
        name = st.text_input("Test set name", "generated-set")
        per_doc = st.slider("Questions per document", 1, 5, 2)
        unans = st.slider("Unanswerable questions", 0, 10, 2)
        if st.form_submit_button("Generate test set"):
            with st.spinner("Generating (this calls the LLM)..."):
                res = api.generate_testset(project_id, name, per_doc, unans)
            st.success(f"Generated {res['num_cases']} cases.")
            st.dataframe(pd.DataFrame(res["cases"]), use_container_width=True,
                         hide_index=True)

# --- Evaluate ---
with tab_eval:
    st.subheader("Run an evaluation")
    config = st.radio(
        "Generator config",
        ["grounded", "ungrounded"],
        help="'ungrounded' removes the grounding instruction — use it to demo a "
             "regression (faithfulness and refusal should drop).",
    )
    label = st.text_input("Run label (optional)", "")
    if st.button("▶️ Run evaluation", type="primary"):
        with st.spinner("Running evaluation (many LLM calls, ~30–60s)..."):
            res = api.evaluate(project_id, config=config, label=label)
        if "summary" in res:
            overall = res["summary"]["overall"]
            st.success(f"Run #{res['run_id']} complete.")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Faithfulness", f"{overall['faithfulness_avg']}/5")
            c2.metric("Correctness", f"{overall['correctness_avg']}/5")
            c3.metric("Refusal", f"{overall['refusal_accuracy']*100:.0f}%")
            c4.metric("Retrieval Hit", f"{overall['retrieval_hit']*100:.0f}%")
            c5.metric("Retrieval MRR", f"{overall['retrieval_mrr']}")
        else:
            st.error(res)

# --- History / trend ---
with tab_history:
    st.subheader("Score history")
    st.caption("How this project's quality has moved across runs — your regression "
               "trend over time.")
    runs = api.list_runs(project_id)
    if not runs:
        st.info("No runs yet. Run an evaluation first.")
    else:
        hist = pd.DataFrame(runs)
        st.dataframe(
            hist[["id", "label", "created_at", "faithfulness_avg",
                  "correctness_avg", "refusal_accuracy"]],
            use_container_width=True, hide_index=True,
        )
        chart_df = hist.set_index("id")[
            ["faithfulness_avg", "correctness_avg", "refusal_accuracy"]
        ]
        st.line_chart(chart_df)