# Methodology

## Problem formulation

CareerFit treats job fit as a **requirement-level evidence verification problem**, not a document-similarity problem.

Each job description is decomposed into atomic requirements. Each requirement is paired with a structured candidate representation and assigned one of four evidence classes:

1. `supported`
2. `partially_supported`
3. `contradicted`
4. `insufficient_evidence`

The portfolio-facing names are **Verified**, **Partially Supported**, **Not Verified**, and **Insufficient Evidence**.

## Dataset design

The experimental benchmark contains 15,000 requirement × candidate pairs, 228 atomic requirements, and 100 synthetic/de-identified candidate profiles. Train, validation, and test partitions are predefined and grouped by job to reduce leakage across splits.

The current benchmark target is `analytical_seed_label`. These labels are deterministic analytical labels used to support model development; they are not a human-gold benchmark.

## Candidate representation

The baseline representation combines the requirement with structured candidate attributes including target role, years of experience, education, skills, language levels, domain experience, location, work authorization, and notes.

For the apples-to-apples baseline comparison, the same underlying rows and split assignments were reused while changing the text representation or decision model.

## Experiments

### 1. TF-IDF + Logistic Regression

A lexical baseline using TF-IDF features and class-balanced multinomial logistic regression.

### 2. SBERT + Logistic Regression

`sentence-transformers/all-MiniLM-L6-v2` was used to encode the same model text into dense semantic vectors, followed by class-balanced logistic regression.

### 3. Zero-shot NLI

A pretrained DeBERTa NLI model was used to classify candidate evidence against a requirement. Standard NLI outputs were mapped as:

- entailment → supported
- contradiction → contradicted
- neutral → insufficient evidence

This experiment cannot directly predict the Partially Supported class because standard NLI exposes three labels rather than four.

### 4. Selective hybrid

A selective hybrid combined the strongest signals from the baseline experiments. The hybrid improved test Macro-F1 from 0.5944 to 0.5965, a deliberately reported small gain rather than an exaggerated improvement claim.

## Evaluation

The main metrics are:

- accuracy
- Macro-F1
- weighted-F1
- per-class precision
- per-class recall
- per-class F1
- confusion matrices

Macro-F1 is emphasized because the four evidence classes are imbalanced.

## Interpretation

The main research result is that the simplest lexical baseline remained competitive and outperformed the standalone semantic and zero-shot NLI experiments on this provisional benchmark. This highlights the importance of task formulation and label construction rather than assuming a larger model will automatically perform better.
