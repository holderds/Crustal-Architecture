# Mohometer

A single-file Streamlit application for estimating crustal thickness from arc-magma geochemistry. Mohometer wraps published proxies, the GAME mohometer of Luffi & Ducea (2022), CRUST1.0, and a configurable machine-learning stack into one workflow.

It is built so you can take a CSV of geochemistry, walk it through preparation → model training → validation → prediction → summary, and walk out with crustal-thickness estimates, uncertainty bands, and an Excel report. The same UI also lets a non-Python user reproduce the result with no command-line work.

---

## Contents

1. [Quick start](#quick-start)
2. [Workflow at a glance](#workflow-at-a-glance)
3. [Tab 1 — Data Prep](#tab-1--data-prep)
4. [Tab 2 — Model](#tab-2--model)
5. [Tab 3 — Validate](#tab-3--validate)
6. [Tab 4 — Predict](#tab-4--predict)
7. [Tab 5 — Summary](#tab-5--summary)
8. [Methods reference — proxies](#methods-reference--proxies)
9. [Methods reference — GAME](#methods-reference--game-luffi--ducea-2022)
10. [Methods reference — ML pipeline](#methods-reference--ml-pipeline)
11. [Uncertainty and error](#uncertainty-and-error)
12. [Pros and cons](#pros-and-cons)
13. [Limitations and caveats](#limitations-and-caveats)
14. [Output and reproducibility](#output-and-reproducibility)
15. [Deploying on Streamlit Community Cloud](#deploying-on-streamlit-community-cloud)
16. [Repository layout](#repository-layout)
17. [References](#references)

---

## Quick start

Local install:

```bash
pip install -r requirements.txt
streamlit run Mohometer.py
```

The app expects four data files alongside `Mohometer.py`:

| File | Purpose | Source |
| --- | --- | --- |
| `GuoYang_2023_Model.xlsx` | Default ML training table | Guo & Yang (2023) supplementary |
| `CRUST_1_0_excel.csv` | Reference crustal-thickness grid | Laske et al. (2013), reformatted |
| `LuffiDucea_2022_Calibration.csv` | GAME calibration table + alternative training set | Luffi & Ducea (2022) supplementary |
| `barrick_logo.png` *(optional)* | Header logo | — |

If a file is missing the corresponding feature is disabled but the rest of the app keeps running.

---

## Workflow at a glance

```
  ┌─────────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐
  │  Data Prep  │───▶│  Model  │───▶│ Validate │───▶│ Predict │───▶│ Summary │
  └─────────────┘    └─────────┘    └──────────┘    └─────────┘    └─────────┘
   load + clean      train ML       benchmark vs    apply to       export Excel,
   normalise         on Guo /        known H or      unknown        save session,
   compute ratios    Zou / Luffi /   CRUST1.0,       samples,        export models
   alteration        upload /        proxy curves,   proxy + GAME,
   screening         Data Prep       sample-size     local
                                     adequacy        estimates
```

Each tab consumes the output of the previous one. Routing between tabs is explicit — when you upload a file in Data Prep you tick which downstream tabs (`Model`, `Validate`, `Predict`) should receive the cleaned data.

---

## Tab 1 — Data Prep

**Purpose:** ingest geochemistry from any CSV/XLSX, normalise column headers to the app's controlled vocabulary, screen for alteration, and route the result to one or more downstream tabs.

### What it does

- **Smart column mapping.** Headers like `SiO2 (wt%)`, `La (ppm)`, `Longitude`, `MOHO depth`, `Rock type` are matched to canonical names (`SiO2`, `La`, `Lon`, `Crust_Thickness`, `Lithology_Type`) via a case-insensitive alias dictionary. Suffixes such as `(wt%)` and `(ppm)` are stripped automatically.
- **Anhydrous recalculation.** A toggle renormalises the major oxides to 100 % excluding LOI and H₂Ot — useful when comparing samples with different volatile contents.
- **Automatic ratio computation.** Sr/Y, La/Yb (raw and chondrite-normalised), Ce/Y, Zr/Y, Dy/Yb, Gd/Yb, Sm/Yb, Ce/Yb, Nd/Yb, Nb/Yb, Th/Yb, Ba/V, Ba/Sc, Ni/Sc, Ni/V, Cr/Sc, Cr/V, Lu/Hf, Zr/Ti, Rb/Sr — derived on the fly from the elements present.
- **Alteration screening.** Computes:
  - **CIA** (Nesbitt & Young 1982) — Chemical Index of Alteration
  - **AI** (Ishikawa et al. 1976) — Hydrothermal Alteration Index
  - **CCPI** (Large et al. 2001) — Chlorite-Carbonate-Pyrite Index
  - **ICV** (Cox et al. 1995) — Index of Compositional Variability
  - **K-Al-Na-Al ratios** for element-mobility plots
  - **Analytical totals & LOI flags** (totals outside 100 ± 2 %, LOI > 2 %)
- **Per-file routing.** Each upload has a multiselect (`Model (training)`, `Validate`, `Predict`) so a single file can fan out to all three, or split across them.

### How to use it

1. Drag one or more files onto the uploader.
2. Confirm the auto-mapping for each column. Unrecognised columns are listed for manual mapping or dismissal.
3. (Optional) toggle the anhydrous recalculation expander and review the alteration metrics.
4. Tick the destination tab(s). The cleaned table is then available everywhere downstream as `dp_training_df`, `dp_validation_df`, or `dp_prediction_df`.

### Common errors and how the app responds

| Symptom | Likely cause | What to check |
| --- | --- | --- |
| Column shows up red / "unmapped" | Header not in alias dict | Map manually or rename in source |
| Rows dropped silently | Missing required numeric value | Check the QA expander row counts |
| Ratios all NaN | Underlying elements missing | Verify both numerator and denominator are populated |
| "Anhydrous recalc skipped" | Less than half the major-oxide columns present | Add at least 6 of the 12 oxide columns |

---

## Tab 2 — Model

**Purpose:** train one or two ML models on a training dataset, with cross-validated metrics and feature-importance diagnostics.

### Training data sources

Selectable from a single dropdown:

- **Guo & Yang (2023)** — default. 32 features, ships with the repo.
- **Zou et al. (2021)** — alternative published training set. Falls back to Guo data restricted to Zou's feature list if the Zou file is not present.
- **Luffi & Ducea (2022)** — uses the GAME calibration table; the app derives `Crust_Thickness` from elevation via the paper's relationship `H = 6.79 × elevation + 26.40`.
- **Upload dataset** — your own table. Required columns: `Lon`, `Lat`, `Crust_Thickness`, `Age_Ma` plus any of the 32 Guo features.
- **From Data Prep tab** — whatever you routed in Tab 1.

### Algorithms

Wrapped in scikit-learn `Pipeline([StandardScaler, estimator])`:

| Algorithm | Strengths | Tree-CI | Notes |
| --- | --- | --- | --- |
| ExtraTrees | Variance reduction; calibrated like Guo & Yang's published model | ✅ | Default for Guo / Luffi sources |
| RandomForest | Same family, deeper smoothing | ✅ | Good for noisy data |
| GradientBoosting | Boosting, slow but precise | ❌ | Sequential trees |
| HistGradientBoosting | Histogram-based boosting; fast on large data | ❌ | Memory-friendly |
| XGBoost | Best benchmark on Zou et al. dataset | ❌ | Default for Zou source; needs `xgboost` package |

### Feature presets

Selectable per model from the *Element strategy → Preset* dropdown:

| Preset | Elements |
| --- | --- |
| **Guo & Yang (2023)** | major oxides + REE + Sr, Y, Rb, Ba, Hf, Nb, Ta, Th |
| **Zou et al. (2021)** | major oxides + REE + Rb, Sr, Y, Zr, Nb, Ba, Hf, Ta, Th, U |
| **Luffi & Ducea (2022)** | the elements that underpin every GAME sensor (33 features inc. Sc, V, Cr, Co, Ni, Ga, Pb) |
| **Full suite** | every numeric column in the dataset |
| **Immobile elements** | TiO₂, Al₂O₃, Y, Zr, Nb, REE, Hf, Ta, Th, U — the alteration-resistant subset |
| **Custom list** | your own multiselect |
| **Minimum RI** | drops features whose relative importance falls below a threshold |

### Comparison model

A second optional model can be added — either the primary model with a different element list, or a totally different training source/algorithm. Side-by-side metrics make it easy to see whether expanding the feature set actually helps.

### Cross-validation

`KFold` with `n_splits = min(10, n_samples)`. Reports per fold and overall:

- **R²** — fraction of variance explained
- **RMSE [km]** — root-mean-squared residual
- **MAE [km]** — mean absolute residual
- **Bias [km]** — mean signed residual (positive = model overestimates)

### How to use it

1. Pick a training source.
2. (Optional) restrict the training subset by age, lithology, tectonic setting, or a custom range on `Crust_Thickness`.
3. Pick an algorithm and feature preset (the defaults are sensible for each source).
4. Inspect the cross-validation summary tile and the *Element importance* expander.
5. (Optional) tick *Add comparison model* to compare a second configuration.

---

## Tab 3 — Validate

**Purpose:** test trained models against data with known crustal thickness that were not used for training, plus benchmark proxies, GAME, and CRUST1.0.

### Inputs

The validate tab accepts validation data from either:

- **Upload file** — CSV/XLSX with at minimum `Lon`, `Lat`, `Crust_Thickness`, and the model features.
- **From Data Prep tab** — anything you routed there.

The upload widget is always visible; if no model has been trained the app shows a hint instead of computing.

### What it produces

- **Validation summary table** — n, R², RMSE, MAE, bias for each (model, feature set) pair.
- **Blind-validation scatter** — predicted vs known thickness, optionally with:
  - Per-sample tree-CI error bars (90 % across the trees in ExtraTrees / RandomForest)
  - Best-fit line + 1:1 reference
  - ±10 km envelope band
  - Colour-by-residual / age / rock type / arc segment / dataset
- **Sample-size adequacy panel** *(new)*. Bootstraps the predicted-thickness median at log-spaced n from 2 to N, plots the 95 % CI band narrowing with n, and tells you exactly how many samples your dataset needs to hit ±5 km and ±10 km median precision. The numbers are entirely data-driven — a tight, coherent suite needs far fewer samples than a noisy or mixed one.
- **GAME diagnostics** — three sub-tabs:
  - **Consensus** — combined GAME estimate vs reference, with status counts.
  - **Reliability** — N, MAD, IQR, CI95 width, and the *Good / Caution / Low confidence* tier for each sample.
  - **Calibration** — view individual sensor LOWESS surfaces against the calibration table.
- **Proxy comparison** — every published proxy (Sr/Y, La/Yb_N, Ce/Y, paired, GAME) plotted against known thickness or model prediction, with optional moving median/mean and outlier clipping.
- **Local crustal-thickness estimate** — for each test sample, build a spatial-temporal neighbourhood from the rest of the validation data and report the local median, MAD, IQR, bootstrap CI95 on the median, and fallback metadata.

### Common questions answered here

- *Is my model unbiased?* — check the bias column and the residual histogram.
- *Are the proxies any better than ML?* — compare per-method RMSE in the proxy panel.
- *How many samples do I need from a target?* — sample-size panel.
- *Are my model errors systematic in age or lithology?* — colour-by on the scatter.

---

## Tab 4 — Predict

**Purpose:** apply trained models to new geochemistry where the crustal thickness is unknown.

### What it produces

- **Per-sample predicted thickness** for every trained model, plus tree-CI bounds when the algorithm supports them.
- **Prediction summary stats** — distribution of predictions per model (median, mean, SD, IQR).
- **GAME diagnostics** — same three sub-tabs as in Validate, but referenced against the model prediction rather than ground truth.
- **Sample-size adequacy** — same bootstrap panel, telling you whether your unknown-sample suite is large enough for a reliable median.
- **Proxy comparison** — every proxy plotted against the model prediction so you can see which proxies are tracking the ML and which are diverging.
- **Group splits** — split predictions by any categorical column (arc segment, lithology, dataset, etc.) and view group-level violins / box plots.
- **Local estimates** — same spatiotemporal neighbourhood logic as Validate.
- **Prediction map** — Lat/Lon scatter coloured by predicted thickness, CI width, age, or model.

### How to use it

1. Upload geochemistry without `Crust_Thickness`.
2. Confirm the column mapping (the same engine as Data Prep is used here for the upload widget).
3. Watch the prediction summary update.
4. Open the sample-size panel to confirm your sample count is adequate.
5. (Optional) split by group, plot the map, or export.

---

## Tab 5 — Summary

**Purpose:** assemble everything into a single Excel workbook with selectable column groups, plus session save/restore.

### Excel sheets

- **Predictions** — sample ID, location, age, prediction, CI, group label.
- **Group summary** — median / mean / SD per group × model.
- **Validation** — residuals, CRUST1.0 comparison, GAME outputs.
- **Input data** — the cleaned training data (without internal helper columns).
- **Feature importance** — per model.

### Column-group toggles

`All`, `Core`, `+ Metadata`, `+ Elements`, `+ Ratios`, `+ Used only` — combine them to control how wide the export is.

### Session save / restore

Click **Save session** to write `mohometer_session.pkl` containing every Data Prep, Result Summary, and cached state key. Click **Restore session** on a fresh app instance to load it back. Trained models can also be exported individually as `.pkl` files and re-imported on the Model tab.

---

## Methods reference — proxies

Every published proxy is hard-coded in the app's `enrich()` function and produces a column called `H_<Author><Year>_<Proxy>_km`. Inputs are checked for the right elements and missing rows are returned as NaN rather than guessed.

### Sr/Y proxies

| Method | Formula | Calibrated for | Output |
| --- | --- | --- | --- |
| Profeta et al. (2015) | H = (Sr/Y + 7.25) / 0.90 | Cordilleran arcs | `H_Profeta2015_SrY_km` |
| Sundell et al. (2021) | H = 19.6 × ln(Sr/Y) − 24 | Global arc dataset | `H_Sundell2021_SrY_km` |
| Zou et al. (2021) | H = 1.11 × Sr/Y + 8.05 | Continental arcs (SVR ML calibration) | `H_Zou2021_SrY_SVRE_km` |
| Hu et al. (2017) | H = 0.67 × Sr/Y + 28.21 | Collisional / post-collisional | `H_Hu2017_Collisional_SrY_km` |

### La/Yb (chondrite-normalised) proxies

| Method | Formula | Output |
| --- | --- | --- |
| Profeta et al. (2015) | H = ln(La/Yb_N / 0.98) / 0.047 | `H_Profeta2015_LaYbN_km` |
| Sundell et al. (2021) | H = 17 × ln(La/Yb_N) + 6.9 | `H_Sundell2021_LaYbN_km` |
| Zou et al. (2021) | H = 21.277 × ln(1.0204 × La/Yb_N) | `H_Zou2021_LaYbN_SVRE_km` |
| Hu et al. (2017) | H = 27.78 × ln(0.34 × La/Yb_N) | `H_Hu2017_Collisional_LaYbN_km` |
| Sundell et al. (2021), paired | H = 10.3 × ln(Sr/Y) + 8.8 × ln(La/Yb_N) − 10.6 | `H_Sundell2021_Paired_km` |

Chondrite normalisation uses McDonough & Sun (1995) CI values: La = 0.237 ppm, Yb = 0.161 ppm.

### Other proxies

| Method | Input | Formula | Output |
| --- | --- | --- | --- |
| Mantle & Collins (2008) | Ce/Y | H = ln(Ce/Y / 0.3029) / 0.0554 | `H_Mantle2008_CeY_sample_km` |
| Dhuime et al. (2015) | Rb/Sr | H = 426.8 × Rb/Sr + 4.1 | `H_Dhuime2015_RbSr_km` |
| Dhuime et al. (2015) | SiO₂ | H = 3.5 × SiO₂ − 157.8 | `H_Dhuime2015_SiO2_km` |
| Farner & Lee (2017) | SiO₂ + elev | h = 0.81 × SiO₂ − 45.82 ; H = 6.79h + 26.40 | `H_FarnerLee2017_SiO2_ElevMoho_km` |
| Luffi & Ducea (2022) | Elevation | H = 6.79 × elev + 26.40 | `H_LuffiDucea2022_Elevation_Moho_km` |

---

## Methods reference — GAME (Luffi & Ducea 2022)

GAME (*Geochemical Arc Moho Estimator*) combines up to **40 individual mohometers** into a single consensus estimate. The original MATLAB GAME app stores opaque fitted surfaces; Mohometer reconstructs the calibration directly from the published T2 table using local LOWESS-style interpolation.

### How a single mohometer estimate is built

For each sample × sensor pair (e.g. `La/Yb`):

1. Find the *k* nearest calibration points in normalised (MgO, sensor) space, where *k* = max(12, ⌈0.35 N⌉).
2. Apply tricube weights w = (1 − (d/d_max)³)³.
3. Fit a local bilinear surface β = [intercept, ∂/∂MgO, ∂/∂sensor, interaction].
4. Predict elevation at the sample's (MgO, sensor) point.
5. Convert to Moho depth via H = 6.79 × elev + 26.40.

### How the consensus is built

1. **Primary MAD filter** — keep estimates within ±5 km of the median.
2. **Secondary MAD filter** — if N drops below the minimum (default 3), widen to ±10 km.
3. **Fallback** — if still under-N, keep all valid estimates and flag as "high spread".
4. **Bootstrap CI95** — 250 bootstrap resamples of the kept estimates, percentile method, around the median.
5. **Reliability tier** — assigned from kept-N, MAD, CI95 width:

| Tier | Criteria |
| --- | --- |
| **Good** | N ≥ 15, MAD ≤ 5 km, CI95 width ≤ 8 km |
| **Caution** | N < 15, MAD > 5 km, CI95 width < 15 km |
| **Low confidence** | N < 8, MAD > 10 km, CI95 width > 15 km, or "all valid; high spread" |

### Output columns

`H_GAME_LuffiDucea2022_km`, `GAME_Luffi2022_N_mohometers`, `GAME_Luffi2022_MAD_km`, `GAME_Luffi2022_IQR_km`, `GAME_Luffi2022_CI95_Low_km`, `GAME_Luffi2022_CI95_High_km`, `GAME_Luffi2022_CI95_Width_km`, `GAME_Luffi2022_Reliability`, `GAME_Luffi2022_Status`.

---

## Methods reference — ML pipeline

### Training

`train_model(df, target, features, seed, algorithm)` strips rows with missing target or features, fits a `Pipeline([StandardScaler, estimator])`, and returns the fitted model plus the cleaned dataframe used for training.

### Cross-validation

`KFold` with `n_splits = min(10, n_samples)`. Per-fold metrics are aggregated into the validation summary table. The metrics use `sklearn.metrics.r2_score`, `mean_squared_error`, and `mean_absolute_error`.

### Feature importance

For tree ensembles: `model.feature_importances_` (Gini-style impurity decrease). For boosting: same field.
The *Element importance* panel shows relative importance, with a *Minimum RI %* slider that drops features below a threshold.

### Predicted-CI bars

Only available for ExtraTrees and RandomForest. The pipeline pulls `model.named_steps['estimator'].estimators_`, predicts with each individual tree, and reports the 5th and 95th percentiles across trees per sample. Boosting algorithms cannot produce per-tree CI in this way.

---

## Uncertainty and error

Mohometer reports several distinct uncertainty quantities — they answer different questions and should not be conflated.

| Quantity | What it measures | Where it shows |
| --- | --- | --- |
| **Cross-validation RMSE / MAE** | Average prediction error on held-out training rows | Model tab, Validate summary |
| **R²** | Fraction of variance explained | Same |
| **Bias** | Mean signed residual; positive = model overestimates | Same |
| **Tree CI90 (per sample)** | 90 % spread across individual trees in the ensemble — *epistemic* | Validate / Predict scatter, optional bars |
| **GAME MAD / IQR / CI95** | Spread across the ~40 mohometers for a single sample | GAME diagnostics tab |
| **Bootstrap median CI** | How confident you are in the *median* of n samples | Sample-size adequacy panel |
| **Local-N statistics** | Spread of nearby samples in space and time | Local estimate panel |

### Reading the sample-size panel

The panel sits inside Validate and Predict. It bootstraps the predicted-thickness median at log-spaced n values (2 → N), then plots:

- Blue line — the bootstrap median at each n.
- Blue band — the user-set CI on the median (default 95 %).
- Green / orange shaded zones — your precision targets (default ±5 km / ±10 km).
- Dotted vertical lines — exactly where the CI band crosses each precision target.
- Caption — "n samples for ±5 km · m samples for ±10 km", read directly from the curve.

The numbers are specific to *your* data. A coherent suite from one arc segment may converge by n = 8; a mixed or altered dataset may need 50+. Don't trust rules of thumb — read the curve.

### How to interpret tree-CI bars

The CI90 bars on the Validate scatter measure ensemble disagreement. They are *not* the same as the model's overall RMSE. A wide CI means individual trees disagreed (the sample sits in a sparse part of feature space, or near a decision boundary). A narrow CI just means the trees agree — it doesn't guarantee the prediction is correct, only that the ensemble is internally consistent.

---

## Pros and cons

### What Mohometer is good at

- **One-stop integration.** Six published proxies, GAME, CRUST1.0, and a configurable ML stack in one window — no spreadsheet juggling.
- **Honest uncertainty.** Tree CI, GAME MAD/CI, and the bootstrap median panel give three independent reads on confidence.
- **Reproducible.** Session pickle and per-model `.pkl` exports let you hand off a frozen state to a colleague.
- **Alteration-aware.** CIA / AI / CCPI / ICV computed up-front so altered samples can be triaged before they corrupt the model.
- **Non-Pythonic-friendly.** Streamlit UI means a colleague who has never seen Python can still drive it through a browser.

### Where it can mislead you

- **Proxies are calibrated on intermediate-felsic arc magmas.** Mafic samples (MgO > 8, low SiO₂) sit outside the calibration and will give nonsense Sr/Y or La/Yb numbers.
- **Cumulates fool Sr/Y.** A plagioclase-rich cumulate has high Sr/Y for petrographic reasons, not crustal-thickness reasons. The app flags `Sr/Y > 80 AND Sr > 1000` but cannot reject these for you.
- **CRUST1.0 is modern.** Comparing a 200 Ma sample to the present-day Moho assumes the crust hasn't changed — usually a poor assumption in orogens.
- **ML is only as good as its training set.** The Guo & Yang (2023) and Zou et al. (2021) tables are dominated by Cenozoic Cordilleran arc magmas. Predicting back into the Archean or for collision-zone magmas pushes the model outside its calibration domain.
- **"Sample-size adequate" ≠ "answer correct".** The sample-size panel only tells you the median is *stable*. It cannot detect systematic bias from a wrong proxy or a wrong training domain.

---

## Limitations and caveats

These are surfaced in the app via a `Reliability_Flags` column on each sample:

| Flag | Trigger | What it means |
| --- | --- | --- |
| `Mafic: intermediate Sr/Y-La/Yb proxies caution` | `Rock_Type_Model == 'mafic'` | Sample lies outside the proxy calibration range |
| `High LOI alteration caution` | `LOI > 3 wt%` | Volatile loss; mobile elements may be redistributed |
| `High Sr/Y + high Sr: possible cumulate` | `Sr/Y > 80 AND Sr > 1000` | Plagioclase accumulation rather than crustal-thickness signal |
| `Anomalous total` | `total < 98` or `total > 102` | Analytical issue or unusual mineral assemblage |
| `Low GAME N` | `N_mohometers < 8` | Not enough sensors survived MAD filtering |
| `High GAME spread` | `CI95_Width > 15 km` | Sensors disagree wildly; treat as low confidence |

Beyond the per-sample flags:

- The chondrite-normalisation constants are McDonough & Sun (1995). If you prefer Sun & McDonough (1989) or Boynton (1984), expect ±2–4 % differences in the La/Yb_N value.
- The `H = 6.79 × elev + 26.40` formula is the published Luffi & Ducea (2022) default. The α and β constants are exposed in code (`GAME_ALPHA_DEFAULT`, `GAME_BETA_DEFAULT`) for users who want to recalibrate against a regional dataset.
- CRUST1.0 nearest-neighbour matching uses cKDTree if available, else a haversine broadcast. Match distance is reported per sample — interpret estimates with multi-hundred-km matches with caution, especially in narrow arcs.

---

## Output and reproducibility

### Excel workbook

`mohometer_results.xlsx` — multi-sheet, with column-group toggles in the Summary tab. The `Predictions` sheet is the one most users want; `Validation`, `Group summary`, and `Feature importance` sit alongside it.

### Session pickle

`mohometer_session.pkl` — every Data Prep, Result Summary, and cached state key. Save it before closing the app, restore it on the next run to skip the upload-and-map cycle.

### Trained model export

Each trained model can be exported as a `.pkl` containing `{name, model, features, target}`. Re-import on the Model tab to skip retraining.

### Key configuration constants

If you need to tweak behaviour without editing the UI, these live near the top of `Mohometer.py`:

```python
GAME_ALPHA_DEFAULT = 6.79     # Moho = α × elev + β
GAME_BETA_DEFAULT  = 26.40
GUO_FEATURES       = [...]    # default 32-feature list
ZOU_2021_FEATURES  = [...]
LUFFI_FEATURES     = [...]    # 33 elements underpinning every GAME sensor
```

---

## Deploying on Streamlit Community Cloud

1. Push the repo to GitHub (already done at [holderds/Crustal-Architecture](https://github.com/holderds/Crustal-Architecture)).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
3. Click **New app**, choose:
   - Repository: `holderds/Crustal-Architecture`
   - Branch: `main`
   - Main file path: `Mohometer.py`
4. Click **Deploy**. First boot ≈ 2–3 min while `requirements.txt` resolves. Subsequent boots are fast.
5. The public URL will look like `https://<app-name>.streamlit.app/`.

### Restricting access

The Community Cloud free tier serves apps publicly by default. Two practical options for internal-only sharing:

- **Option A — private viewer list (recommended for small teams).** In the deployed app's settings, set Sharing to *Private* and add specific viewer emails (Google sign-in required).
- **Option B — host inside corporate infrastructure.** Containerise and deploy to Azure App Service, AWS, or an internal Streamlit instance behind SSO. Repo can stay private. Best long-term answer for organisation-wide rollouts.

Either way: **never commit credentials, API keys, or sensitive sample locations.** Community Cloud builds run from the public-facing branch.

### Resource limits

Community Cloud free tier provides ≈ 1 GB RAM and 1 vCPU. The default training tables fit comfortably. Heavy workloads (training on 50k+ rows, broad hyper-parameter sweeps) need a paid tier or self-hosting.

---

## Repository layout

```
Mohometer.py                              # The Streamlit app (single file, ~7600 lines)
requirements.txt                          # Python dependencies
README.md                                 # This document
.gitignore                                # Excludes pickles, logs, IDE files
GuoYang_2023_Model.xlsx                   # Default ML training table
Zou_2021_Model.xlsx                       # (optional) Zou et al. training set
LuffiDucea_2022_Calibration.csv           # GAME calibration / Luffi training source
CRUST_1_0_excel.csv                       # CRUST1.0 reference grid
barrick_logo.png                          # Header logo (optional)
engine/                                   # Reusable calculation modules
  classify.py                             # Rock / lithology classification
  ratios.py                               # Geochemical ratio computations
artifacts/feature_ablation/               # Feature-screening study outputs
.claude/launch.json                       # Local launcher config
```

---

## References

- **Guo, F. & Yang, X. (2023)** — Default ML training set; ExtraTrees feature-importance baseline.
- **Zou, Z. et al. (2021)** — SVR / ML calibration of Sr/Y and La/Yb_N proxies for continental arcs.
- **Sundell, K.E. et al. (2021)** — Global Sr/Y, La/Yb_N, and paired calibrations.
- **Profeta, L. et al. (2015)** — Cordilleran-arc Sr/Y and La/Yb_N regressions.
- **Hu, F. et al. (2017)** — Collisional / post-collisional Sr/Y calibration.
- **Mantle, G.W. & Collins, W.J. (2008)** — Ce/Y crustal-thickness relationship.
- **Luffi, P. & Ducea, M.N. (2022)** — GAME mohometer; H = 6.79 × elev + 26.40.
- **Dhuime, B. et al. (2015)** — Rb/Sr and SiO₂ proxies.
- **Farner, M.J. & Lee, C.-T.A. (2017)** — Elevation-SiO₂ Moho relationship.
- **McDonough, W.F. & Sun, S.-S. (1995)** — CI chondrite normalising values.
- **Nesbitt, H.W. & Young, G.M. (1982)** — Chemical Index of Alteration.
- **Ishikawa, Y. et al. (1976)** — Hydrothermal Alteration Index.
- **Large, R.R. et al. (2001)** — Chlorite-Carbonate-Pyrite Index.
- **Cox, R., Lowe, D.R. & Cullers, R.L. (1995)** — Index of Compositional Variability.
- **Laske, G. et al. (2013)** — CRUST1.0 model.

---

## Citation

Holder, D. (2026). *Mohometer: an integrated crustal-thickness estimation tool.* Internal Barrick application.

Underlying methods are credited to the publications above.
