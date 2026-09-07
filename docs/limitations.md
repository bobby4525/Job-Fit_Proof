# Limitations and responsible-use note

CareerFit is an experimental NLP/ML portfolio project. It is **not** validated for real hiring decisions and should not be used to accept, reject, rank, or screen candidates in employment settings.

## Current limitations

### 1. Seed labels are not human-gold labels

The current target labels are analytical seed labels. They are useful for controlled experimentation but do not replace independent human annotation.

### 2. Candidate data are synthetic/de-identified

The benchmark uses synthetic/de-identified candidate profiles. Performance on real CVs may differ substantially because real documents contain formatting variation, incomplete information, ambiguous language, and domain-specific phrasing.

### 3. Strong class imbalance

The minority classes, especially Partially Supported and Not Verified, have much lower support than Verified. Accuracy alone therefore overstates model quality; Macro-F1 and per-class metrics are more informative.

### 4. NLI class mismatch

Standard three-way NLI does not naturally represent CareerFit's four-way label set. Neutral can map to Insufficient Evidence, but there is no direct Partially Supported class.

### 5. Distribution shift in free-form CVs

The experiments use a structured candidate representation. Feeding an arbitrary CV directly into the trained baseline changes the input distribution and should not be treated as equivalent to the benchmark setup.

### 6. No fairness validation

No demographic subgroup, bias, disparate-impact, or fairness analysis has been completed. That is another reason the project must not be presented as a production hiring system.

## What would be required next

Before making stronger real-world claims, the project would need:

- independently annotated human-gold examples
- inter-rater agreement measurement
- more balanced class coverage
- evaluation on real, consented CV–JD pairs
- calibrated confidence and abstention thresholds
- robustness testing across industries and CV formats
- fairness and subgroup analysis
- external validation

The portfolio value of the current project is in the **experimental design, baseline comparison, evaluation discipline, and transparent reporting of limitations**.
