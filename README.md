# Job Fit Proof

Evidence-based CV ↔ job requirement verification research prototype.

## Live app engine
The deployed app uses the **saved trained TF-IDF baseline** for genuine live inference.

The evaluated hybrid model (TF-IDF + SBERT + NLI) is preserved in the research results, but it is intentionally not the live deployment engine because:
- its improvement over TF-IDF was only +0.002 Macro-F1;
- the NLI model is too heavy for a lightweight portfolio deployment;
- using TF-IDF keeps the live app reproducible and honest.

## Provisional benchmark

| Model | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| TF-IDF | 0.7844 | 0.5944 |
| SBERT | 0.6396 | 0.4681 |
| Zero-shot NLI | 0.4018 | 0.2094 |
| Hybrid | **0.7871** | **0.5965** |

Metrics use analytical seed labels, not final human-gold labels.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Important limitation

Live free-form CV input is a distribution shift from the structured candidate representation used in the research benchmark. Treat live predictions as prototype outputs, not validated hiring decisions.
