import re
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Job Fit Proof", page_icon="✓", layout="wide")
BASE = Path(__file__).parent

LABEL_MAP = {
    "supported": "Verified",
    "partially_supported": "Partially Supported",
    "contradicted": "Not Verified",
    "insufficient_evidence": "Insufficient Evidence",
}

VERDICT_HELP = {
    "Verified": "The trained model predicts that the candidate evidence supports the requirement.",
    "Partially Supported": "The trained model predicts partial but incomplete support.",
    "Not Verified": "The trained model predicts evidence that conflicts with or clearly fails the requirement.",
    "Insufficient Evidence": "The trained model predicts that there is not enough explicit evidence to decide.",
}

@st.cache_resource
def load_model():
    return joblib.load(BASE / "careerfit_tfidf_model.joblib")

def extract_requirements(jd):
    rows = []
    for raw in jd.splitlines():
        raw = raw.strip()
        raw = re.sub(r"^[•\-\*\d\.\)\s]+", "", raw).strip()
        if len(raw) >= 8:
            rows.append(raw)

    if len(rows) <= 1 and jd.strip():
        candidates = re.split(r"(?<=[.;])\s+|\n+", jd)
        rows = [
            re.sub(r"^[•\-\*\d\.\)\s]+", "", x).strip()
            for x in candidates if len(x.strip()) >= 8
        ]
    return rows[:30]

def make_live_model_text(requirement, cv_text):
    return f"{requirement} | candidate_evidence | {cv_text}"

def confidence_tier(x):
    if x >= 0.85:
        return "High"
    if x >= 0.65:
        return "Medium"
    return "Low"

st.title("Job Fit Proof")
st.caption("Evidence-Based Job Requirement Verification · CareerFit portfolio prototype")

with st.sidebar:
    st.header("Research status")
    st.success("Live inference uses the saved trained TF-IDF baseline.")
    st.info(
        "The hybrid TF-IDF + SBERT + NLI experiment is preserved as a benchmark result. "
        "It is not used for live web inference because the hybrid gain was only +0.002 Macro-F1 "
        "and the NLI model is too heavy for a lightweight portfolio deployment."
    )
    st.warning(
        "Current metrics use analytical seed labels, not a completed human-gold benchmark."
    )
    page = st.radio("View", ["Live CV ↔ JD Analysis", "Research Results", "Methodology"])

if page == "Live CV ↔ JD Analysis":
    model = load_model()
    st.subheader("Analyze your CV against a job description")

    c1, c2 = st.columns(2)
    with c1:
        cv = st.text_area(
            "CV / candidate evidence",
            height=330,
            placeholder="Paste CV text, skills, education, work experience, and languages..."
        )
    with c2:
        jd = st.text_area(
            "Job description",
            height=330,
            placeholder="Paste the job description or requirement list..."
        )

    if st.button("Verify Job Fit", type="primary", use_container_width=True):
        reqs = extract_requirements(jd)
        if not cv.strip():
            st.error("Please paste candidate evidence.")
        elif not reqs:
            st.error("I could not extract any usable requirements from the job description.")
        else:
            live_text = [make_live_model_text(r, cv) for r in reqs]
            pred = model.predict(live_text)
            prob = model.predict_proba(live_text)
            conf = prob.max(axis=1)

            rows = []
            for r, p, c in zip(reqs, pred, conf):
                rows.append({
                    "Requirement": r,
                    "Verdict": LABEL_MAP[p],
                    "Model confidence": float(c),
                    "Confidence tier": confidence_tier(float(c)),
                })

            result = pd.DataFrame(rows)

            n = len(result)
            verified = int((result["Verdict"] == "Verified").sum())
            partial = int((result["Verdict"] == "Partially Supported").sum())
            not_verified = int((result["Verdict"] == "Not Verified").sum())
            insufficient = int((result["Verdict"] == "Insufficient Evidence").sum())

            a,b,c,d,e = st.columns(5)
            a.metric("Requirements", n)
            b.metric("Verified", verified)
            c.metric("Partial", partial)
            d.metric("Not Verified", not_verified)
            e.metric("Insufficient", insufficient)

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Model confidence": st.column_config.ProgressColumn(
                        "Model confidence", min_value=0.0, max_value=1.0, format="%.0f%%"
                    )
                }
            )

            evidence_score = (verified + 0.5 * partial) / max(n, 1)
            st.metric("Evidence coverage score", f"{100*evidence_score:.1f}%")
            st.caption(
                "This score is a portfolio-facing summary of requirement-level outputs; "
                "it is not a hiring recommendation."
            )

            st.warning(
                "Live free-form CV input differs from the structured candidate representation "
                "used in the benchmark. Treat live predictions as prototype outputs, not validated hiring decisions."
            )

elif page == "Research Results":
    st.subheader("Held-out provisional benchmark")

    results = pd.DataFrame([
        ["TF-IDF", 0.7844, 0.5944],
        ["SBERT", 0.6396, 0.4681],
        ["Zero-shot NLI", 0.4018, 0.2094],
        ["Hybrid", 0.7871, 0.5965],
    ], columns=["Model", "Test Accuracy", "Test Macro-F1"])

    st.dataframe(results, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Interpretation:** TF-IDF was the strongest standalone baseline.  
        SBERT and zero-shot NLI were weaker as standalone classifiers.  
        A selective hybrid rule produced a **small** improvement over TF-IDF:
        **0.5965 vs 0.5944 Macro-F1**.
        """
    )

    st.subheader("Per-class hybrid performance")
    per_class = pd.DataFrame([
        ["Verified", 0.895, 0.823, 0.857, 1669],
        ["Partially Supported", 0.278, 0.750, 0.406, 36],
        ["Not Verified", 0.328, 0.745, 0.455, 51],
        ["Insufficient Evidence", 0.663, 0.672, 0.667, 494],
    ], columns=["Class","Precision","Recall","F1","Support"])
    st.dataframe(per_class, use_container_width=True, hide_index=True)

    st.warning(
        "These are analytical-seed-label results, not final human-gold performance."
    )

else:
    st.subheader("Methodology")
    st.markdown(
        """
        **Pipeline**

        `Job Description → Atomic Requirements → Candidate Evidence → Baselines → Hybrid Decision → Confidence → Verdict`

        **Four verdicts**
        - **Verified**
        - **Partially Supported**
        - **Not Verified**
        - **Insufficient Evidence**

        **Experimental setup**
        - 15,000 requirement × candidate pairs
        - 228 atomic requirements
        - 100 synthetic/de-identified candidate profiles
        - grouped train/validation/test split by job
        - TF-IDF + logistic regression baseline
        - SBERT semantic baseline
        - zero-shot DeBERTa NLI baseline
        - selective hybrid decision rule
        - confidence calibration and error analysis

        **Important limitation**

        Current benchmark labels are deterministic analytical seed labels.
        A larger independently human-annotated benchmark is still needed for final real-world claims.
        """
    )

    st.subheader("Verdict definitions")
    for verdict, explanation in VERDICT_HELP.items():
        st.markdown(f"**{verdict}** — {explanation}")
