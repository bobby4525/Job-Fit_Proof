# CareerFit — Evidence-Grounded Job Requirement Verification

CareerFit is an NLP/ML portfolio project that asks a stricter question than a typical CV–job similarity score:

> **Can the candidate's CV actually support each requirement in a job description?**

Instead of producing one opaque “match percentage”, CareerFit breaks a job description into atomic requirements and assigns one of four evidence verdicts to each requirement:

- **Verified** — candidate evidence supports the requirement
- **Partially Supported** — some evidence exists, but the requirement is not fully supported
- **Not Verified** — available evidence conflicts with or clearly fails the requirement
- **Insufficient Evidence** — the CV does not contain enough explicit evidence to decide

This repository is intentionally presented as an **ML/NLP research project**, not as a production hiring platform. The earlier Streamlit shell has been removed from the portfolio-facing version so the repository focuses on the work that is actually validated: dataset design, modeling, evaluation, and limitations.

---

## Why this project exists

Most job-fit tools reduce a CV and a job description to keyword overlap or a single similarity score. That can hide an important distinction: a candidate may mention a technology without proving the level, duration, or context required by the role.

CareerFit reframes job fit as **requirement-level evidence verification**.

```text
Job Description
      ↓
Atomic Requirements
      ↓
Candidate Evidence Representation
      ↓
TF-IDF / SBERT / NLI Experiments
      ↓
Selective Hybrid Decision
      ↓
4-Class Evidence Verdict
```

---

## Dataset

The experimental dataset contains:

- **15,000** requirement × candidate pairs
- **228** atomic job requirements
- **100** synthetic/de-identified candidate profiles
- grouped train/validation/test splits by job to reduce leakage
- four analytical evidence labels

The current labels are **analytical seed labels**, not an independently human-annotated gold standard. All reported metrics below are therefore provisional research results.

---

## Models evaluated

| Model | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.7844 | 0.5944 |
| SBERT + Logistic Regression | 0.6396 | 0.4681 |
| Zero-shot DeBERTa NLI | 0.4018 | 0.2094 |
| Selective Hybrid | **0.7871** | **0.5965** |

### What the result means

The strongest standalone baseline was **TF-IDF + Logistic Regression**. SBERT and zero-shot NLI were weaker on this particular label construction and data representation. A selective hybrid rule produced only a **small** improvement over TF-IDF, so the project reports that result directly rather than overstating it.

The key lesson was not that “more advanced transformers always win.” The experiment showed that representation, label quality, class imbalance, and task formulation matter more than model complexity alone.

---

## Hybrid per-class performance

| Evidence class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Verified | 0.895 | 0.823 | 0.857 | 1,669 |
| Partially Supported | 0.278 | 0.750 | 0.406 | 36 |
| Not Verified | 0.328 | 0.745 | 0.455 | 51 |
| Insufficient Evidence | 0.663 | 0.672 | 0.667 | 494 |

The minority classes are much harder than the dominant Verified class. That imbalance is one of the most important limitations of the current benchmark.

---

## Repository structure

```text
Job-Fit_Proof/
├── README.md
├── requirements.txt
├── src/
│   ├── label_schema.py
│   ├── tfidf_baseline.py
│   └── evaluation.py
├── results/
│   ├── model_comparison.csv
│   └── hybrid_per_class.csv
└── docs/
    ├── methodology.md
    └── limitations.md
```

The code in `src/` is a cleaned portfolio-facing reference implementation of the evaluation setup used in the experiments.

---

## Technical stack

**Python · pandas · scikit-learn · sentence-transformers · transformers · PyTorch · NLP · text classification · model evaluation**

---

## Reproducible TF-IDF baseline

Install dependencies:

```bash
pip install -r requirements.txt
```

The reference baseline expects two tabular inputs:

- `requirement_candidate_pairs_15000.csv`
- `candidate_profiles_100.csv`

and preserves the predefined `train`, `validation`, and `test` split column.

Example:

```bash
python src/tfidf_baseline.py \
  --pairs data/requirement_candidate_pairs_15000.csv \
  --candidates data/candidate_profiles_100.csv
```

The private/full experimental data are not published in this repository. The script is provided to document the modeling pipeline and expected schema.

---

## Main research findings

1. **TF-IDF was a surprisingly strong baseline.** On the seed-label benchmark it outperformed both the SBERT and zero-shot NLI standalone experiments.
2. **A transformer model is not automatically better for a task.** The zero-shot NLI mapping also cannot directly represent the Partially Supported class.
3. **Macro-F1 matters more than accuracy here.** The four evidence classes are strongly imbalanced.
4. **Abstention is important.** “Insufficient Evidence” is an explicit outcome rather than forcing every requirement into a yes/no decision.
5. **The next bottleneck is data quality, not UI polish.** A larger independently human-annotated benchmark is required before making real-world hiring claims.

---

## Limitations

CareerFit is a research prototype and **must not be treated as a hiring decision system**.

Current limitations include:

- analytical seed labels rather than human-gold labels
- synthetic/de-identified candidate profiles
- strong class imbalance
- small support for Partially Supported and Not Verified
- free-form real CVs can differ from the structured representation used during experiments
- no fairness, subgroup, or real-world employer validation has yet been completed

See [`docs/limitations.md`](docs/limitations.md) for the full discussion.

---

## Next steps

The strongest next research step is **human annotation**, not another UI redesign:

1. annotate a representative subset with at least two independent raters
2. measure inter-rater agreement
3. retrain/evaluate against human-gold labels
4. improve requirement extraction and evidence retrieval
5. calibrate confidence and abstention thresholds
6. evaluate on real, consented CV–JD pairs

---

## Portfolio note

This project demonstrates experimentation rather than claiming a production-ready AI hiring product. The focus is on **problem formulation, leakage-aware evaluation, baseline comparison, error analysis, and honest reporting of limitations**.
