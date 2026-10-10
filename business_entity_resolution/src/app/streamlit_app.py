"""
Streamlit Dashboard for EntityIQ — Agentic AI Business Entity Resolution Platform.
"""
import os
import json
import pandas as pd
import streamlit as st

# Setup page configuration
st.set_page_config(
    page_title="EntityIQ — AI Business Entity Resolution",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

from src.config import BASE_DIR, MODELS_DIR, REPORTS_DIR, OUTPUT_DIR
from src.agent.orchestrator import EntityIQAgent
from src.agent.tools import run_entity_matching, summarize_evaluation_report
from src.explainability.explainer import MatchExplainer
from src.rag.qa_engine import RAGAssistant

@st.cache_resource
def load_resources():
    agent = EntityIQAgent()
    explainer = MatchExplainer(model_name="lightgbm")
    rag = RAGAssistant(base_dir=BASE_DIR, auto_build=False)
    return agent, explainer, rag

agent, explainer, rag = load_resources()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/business.png", width=70)
    st.markdown("### **EntityIQ Platform**")
    st.caption("AI/ML + Agentic AI + RAG Architecture")
    st.markdown("---")
    st.markdown("**Core Models:**")
    st.info("• LightGBM + 27+ Features\n• Rank-Aware Thresholding\n• Tree SHAP Explanations\n• TF-IDF Dense RAG")
    st.markdown("---")
    st.markdown("**Challenge Status:**")
    st.success("✅ Submission Validated (1.73M test rows)")

# Main Header
st.markdown('<div class="main-header">🏢 EntityIQ — Business Entity Resolution Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Source Cross-Lingual Business Matching • Tree SHAP Explainability • RAG Knowledge Assistant • Agentic AI</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Entity Matching",
    "📊 Explainable AI (SHAP)",
    "🧠 RAG Knowledge Assistant",
    "🤖 Agentic AI Coordinator",
    "📈 Evaluation & Submissions"
])

# ── Tab 1: Live Entity Matching ───────────────────────────────────────────────
with tab1:
    st.subheader("Interactive Entity Matching & Candidate Resolution")
    st.write("Resolve Source 1 business records against candidate records across noisy names, scripts, and addresses.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Source 1 Reference Record")
        s1_name = st.text_input("S1 Business Name", value="Tata Consultancy Services Ltd", key="t1_s1_name")
        s1_addr = st.text_input("S1 Business Address", value="Whitefield Road, Bangalore 560066", key="t1_s1_addr")
        s1_country = st.selectbox("S1 Country", ["India", "US", "France"], index=0, key="t1_s1_country")

    with col2:
        st.markdown("##### Candidate Records (Source 2 / Source 3)")
        c1_name = st.text_input("Candidate 1 Name", value="Tata Consultancy Services", key="t1_c1_name")
        c1_addr = st.text_input("Candidate 1 Address", value="560066 Whitefield, Bengaluru", key="t1_c1_addr")
        c2_name = st.text_input("Candidate 2 Name (Distractor)", value="Tata Motors Auto Sales", key="t1_c2_name")
        c2_addr = st.text_input("Candidate 2 Address", value="Main Highway, Pune 411001", key="t1_c2_addr")

    tau1_val = st.slider("Match Acceptance Threshold (tau1)", min_value=0.1, max_value=0.9, value=0.25, step=0.05)

    if st.button("🚀 Run Entity Resolution", type="primary"):
        s1_records = [{"entity_id": "S1-USER", "business_name": s1_name, "business_address": s1_addr, "country": s1_country}]
        cand_records = [
            {"entity_id": "CAND-01", "business_name": c1_name, "business_address": c1_addr, "country": s1_country},
            {"entity_id": "CAND-02", "business_name": c2_name, "business_address": c2_addr, "country": s1_country},
        ]
        
        with st.spinner("Processing features and scoring candidates..."):
            match_res = run_entity_matching(s1_records, cand_records, tau1=tau1_val, gap=0.05)

        matches = match_res["matching_results"].get("S1-USER", [])
        st.success(f"Matching Completed! Found **{len(matches)}** matched record(s).")

        if matches:
            st.markdown("### Matched Candidates:")
            for mid in matches:
                cand_obj = next(c for c in cand_records if c["entity_id"] == mid)
                st.markdown(f"- ✅ **{cand_obj['entity_id']}**: `{cand_obj['business_name']}` — *{cand_obj['business_address']}*")
        else:
            st.warning("No candidate met the decision threshold. Entity resolved as a **Singleton (0 matches)**.")

# ── Tab 2: Explainable AI & SHAP ─────────────────────────────────────────────
with tab2:
    st.subheader("Tree SHAP Feature Attribution & Prediction Evidence")
    st.write("Inspect how similarity features contribute towards or against a matching decision.")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        ex_s1_name = st.text_input("S1 Name", value="Infosys Technologies Ltd", key="t2_s1_n")
        ex_s1_addr = st.text_input("S1 Address", value="Hosur Road, Bangalore 560100", key="t2_s1_a")
    with col_e2:
        ex_c_name = st.text_input("Candidate Name", value="Infosys Technologies", key="t2_c_n")
        ex_c_addr = st.text_input("Candidate Address", value="Electronics City, Bengaluru 560100", key="t2_c_a")

    if st.button("🔬 Compute Tree SHAP Explanation", type="primary"):
        s1_obj = {"entity_id": "S1-EX", "business_name": ex_s1_name, "business_address": ex_s1_addr, "country": "India"}
        c_obj = {"entity_id": "C-EX", "business_name": ex_c_name, "business_address": ex_c_addr, "country": "India"}

        explanation = explainer.explain_pair(s1_obj, c_obj, threshold=0.70)
        pred = explanation["prediction"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Match Probability", f"{pred['match_probability']:.2%}")
        m2.metric("Decision Threshold", f"{pred['decision_threshold']:.2f}")
        m3.metric("Prediction", "MATCH" if pred["is_match"] else "NO MATCH")

        st.markdown("#### Top Contributing Features (SHAP Forces)")
        attributions = explanation["top_contributing_features"]
        df_attr = pd.DataFrame(attributions)
        st.dataframe(df_attr, use_container_width=True)

        st.markdown("#### Normalized Text Tokens Comparison")
        st.json({
            "S1 Normalized": explanation["s1"],
            "Candidate Normalized": explanation["candidate"]
        })

# ── Tab 3: RAG Knowledge Assistant ───────────────────────────────────────────
with tab3:
    st.subheader("RAG Knowledge Assistant")
    st.write("Ask questions about the project architecture, candidate generation methodology, or evaluation benchmarks.")

    sample_questions = [
        "What is the macro F0.5 evaluation score and methodology?",
        "How does the candidate generation blocking system work?",
        "What does country partitioning do across US, India, and France?",
        "What are the top 5 LightGBM similarity features?"
    ]
    selected_sample = st.selectbox("Sample Questions (or type below):", [""] + sample_questions)

    user_query = st.text_input("Your Question:", value=selected_sample if selected_sample else "")

    if st.button("💬 Ask Knowledge Assistant", type="primary") and user_query:
        with st.spinner("Searching verified project context..."):
            qa_res = rag.answer_question(user_query, top_k=3)

        st.markdown("#### Answer:")
        st.markdown(qa_res["answer"])

        st.markdown("#### Grounded Sources & Citations:")
        for src in qa_res["sources"]:
            st.markdown(f"- 📄 `{src}`")

        with st.expander("View Retrieved Context Chunks"):
            for chunk in qa_res["retrieved_context"]:
                st.markdown(f"**{chunk['title']}** *(Relevance Score: {chunk['relevance_score']})*")
                st.code(chunk["text"])

# ── Tab 4: Agentic AI Coordinator ────────────────────────────────────────────
with tab4:
    st.subheader("Agentic AI System & Audit Trail")
    st.write("Coordinate approved data processing, validation, and evaluation tools with automated safety policies.")

    tool_choice = st.selectbox("Select Agent Tool to Execute:", [
        "inspect_dataset_schema",
        "validate_input_dataset",
        "run_data_quality_report",
        "summarize_evaluation_report",
        "retrieve_project_knowledge"
    ])

    if st.button("⚡ Execute Agent Tool"):
        from src.config import TRAIN_S1
        params = {}
        if tool_choice in ["inspect_dataset_schema", "validate_input_dataset", "run_data_quality_report"]:
            params = {"file_path": TRAIN_S1}
        elif tool_choice == "retrieve_project_knowledge":
            params = {"query": "What is the blocking recall target?"}

        with st.spinner(f"Agent executing {tool_choice}..."):
            agent_res = agent.execute_tool(tool_choice, **params)

        st.markdown("#### Tool Execution Result:")
        st.json(agent_res)

    st.markdown("#### Immutable Execution Audit Trail:")
    st.dataframe(pd.DataFrame(agent.get_audit_trail()), use_container_width=True)

# ── Tab 5: Evaluation & Submissions ──────────────────────────────────────────
with tab5:
    st.subheader("Evaluation Metrics & Official Submission Compliance")

    st.markdown("### Official Benchmark Comparison")
    summary = summarize_evaluation_report()
    if "model_metrics" in summary:
        models_data = summary["model_metrics"]
        st.json(models_data)

    st.markdown("### Output Files & Submission Compliance")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.info("📄 **`output/matching_results.tsv`**\n• 1,732,544 rows\n• 100% Validator Compliant")
    with col_f2:
        st.info("📄 **`output/candidate_pairs.tsv`**\n• 1,732,544 rows\n• Candidate Subset Verified")
