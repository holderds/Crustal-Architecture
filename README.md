# Crustal Architecture Shared Engine v0.7

This bundle gives you **both**:

1. `iogas_workflow.py` — a no-dashboard workflow script for ioGAS / CSV / Excel table enrichment.
2. `streamlit_app.py` — a lightweight dashboard using the same calculation engine.
3. `engine/` — the shared Python engine used by both.

## Why this structure?

The aim is to maintain one calculation engine and use it in two places:

```text
engine/
   ↓
ioGAS workflow       Streamlit dashboard
```

ioGAS is best for production table enrichment. Streamlit is best for interactive plots, QA, and model development.

## Quick start: Streamlit

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Quick start: ioGAS-style CSV workflow

Export a table from ioGAS as CSV, then run:

```bash
python iogas_workflow.py --input samples.csv --output samples_enriched.csv
```

Then import or append `samples_enriched.csv` back into ioGAS.

With a trained Guo model:

```bash
python iogas_workflow.py --input samples.csv --output samples_enriched.csv --guo-model models/guo_ert_model.joblib
```

## Training the Guo model once

```bash
python train_models.py --input guo_training.csv --target "Crustal thickness" --model guo --output models/guo_ert_model.joblib
```

The resulting `.joblib` can be used in both Streamlit and ioGAS workflows.

## Column mapping

The engine automatically recognises headers such as:

```text
Nd_ppm       -> Nd
La (ppm)     -> La
SiO2_wt%     -> SiO2
MgO_pct      -> MgO
sample_name  -> Sample_ID
```

Manual mapping is also supported in Python:

```python
from engine.pipeline import run_crustal_architecture_pipeline

mapping = {"Nd": "Nd_ppm", "SiO2": "Silica_wt_pct"}
out = run_crustal_architecture_pipeline(df, column_mapping=mapping)
```

## Important output columns

- `Sr_Y`, `La_Yb_N`, `Ce_Y`, `MnO_MgO`, `Dy_Yb`, `Gd_Yb`
- `H_Sundell2021_Paired_km`
- `H_Mantle2008_CeYmax_km`
- `H_Guo_ERT_km` if model is provided
- `Preferred_H_km`
- `Preferred_Method`
- `Confidence_Score`
- `Reliability_Flags`
- `System_Class`
- `Geological_Interpretation`

## Notes for ioGAS

The exact ioGAS Python Runner interface can vary by version/configuration. This bundle is therefore built around the most portable pattern:

```text
export active table -> run script -> import enriched table
```

If your ioGAS installation exposes the active table as a pandas dataframe, use this function directly:

```python
from iogas_workflow import run_iogas_dataframe

enriched = run_iogas_dataframe(active_dataframe, guo_model_path="models/guo_ert_model.joblib")
```

## Scientific caution

The proxy outputs are apparent thickness estimates. They must be interpreted with rock type, alteration, cumulate effects, sample grouping, and source/fractionation context.


## v0.7 ML options in Streamlit

The Streamlit dashboard now supports three ML modes for both Guo-style and Zou-style models:

```text
No ML
Upload .joblib
Upload training CSV/XLSX and train now
```

When you train from a CSV/XLSX inside Streamlit, the app also provides a button to download the trained `.joblib` model. Save that file in the `models/` folder and reuse it later in Streamlit or in the ioGAS workflow.

Typical Guo workflow:

```text
1. Open Streamlit
2. In the sidebar: Guo-style ERT ML mode → Upload training table and train now
3. Upload Guo training CSV/XLSX
4. Select target column, usually "Crustal thickness"
5. Click Train
6. Download guo_ert_model.joblib
7. Reuse the .joblib in Streamlit or ioGAS
```
