# Mohometer

A Streamlit app for crustal-thickness estimation from arc-magma geochemistry. Mohometer wraps Guo & Yang (2023), Zou et al. (2021), Sundell et al. (2021), Profeta et al. (2015), Mantle & Collins (2008), Luffi & Ducea (2022) and CRUST1.0 into one single-file workflow:

- **Data Prep** — upload CSV/XLSX, map columns to a controlled vocabulary, run anhydrous recalculation, alteration screening (CIA / AI / CCPI / K-Al-Na-Al), and export cleaned data.
- **Model** — train and compare ExtraTrees, RandomForest, GradientBoosting, HistGradientBoosting and XGBoost models on the Guo & Yang (2023) table, the Zou et al. (2021) set, or your own training data.
- **Validate** — benchmark trained models against blind data with known thickness, plus CRUST1.0, proxy curves, GAME consensus, and spatial-temporal local estimates.
- **Predict** — apply trained models to unknown geochemistry, with proxy comparisons, GAME estimates, and grouping diagnostics.
- **Summary** — multi-sheet Excel export with selectable column groups.

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run Mohometer.py
```

The app expects these data files in the same folder as `Mohometer.py`:

| File | Purpose | Source |
| --- | --- | --- |
| `GuoYang_2023_Model.xlsx` | Default training table (Guo & Yang 2023) | Guo & Yang (2023) supplementary |
| `CRUST_1_0_excel.csv` | CRUST1.0 reference grid | Laske et al. (2013), reformatted |
| `luffi_ducea_2022_game_calibration.csv` | GAME mohometer calibration table | Luffi & Ducea (2022) supplementary |
| `barrick_logo.png` *(optional)* | Masthead logo | — |

If a file is missing, the corresponding feature is disabled but the rest of the app still runs.

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (already done at [holderds/Crustal-Architecture](https://github.com/holderds/Crustal-Architecture)).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. Click **New app** and select:
   - Repository: `holderds/Crustal-Architecture`
   - Branch: `main`
   - Main file path: `Mohometer.py`
4. Click **Deploy**. Streamlit will install `requirements.txt` and start the app. First boot takes ~2–3 minutes.
5. Once running, you'll get a public URL like `https://<app-name>.streamlit.app/`.

### Restricting access (for internal sharing)

The Community Cloud free tier serves apps publicly by default. To keep access internal:

- **Option A — private viewer list (recommended).** In the deployed app's settings, set Sharing to *Private* and add specific viewer emails (those users sign in with Google to view). Viewers must be added one at a time.
- **Option B — host inside corporate infrastructure.** Containerise the app and deploy to Azure App Service, AWS, or an internal Streamlit instance behind SSO. The repo can stay private. This is the durable answer for company-wide rollout.

Either way, **do not commit credentials, API keys, or sensitive sample locations** to the repo — Community Cloud builds run from the public-facing branch.

### Resource limits

Community Cloud free tier provides ~1 GB RAM and 1 vCPU. The default training tables fit well within this. Heavy workflows (training on tens of thousands of rows, broad hyperparameter sweeps) may need a paid tier or self-hosting.

## Repository layout

```
Mohometer.py                              # The Streamlit app (single file)
requirements.txt                          # Python dependencies
engine/                                   # Reusable calculation modules
  classify.py                             # Rock / lithology classification
  ratios.py                               # Geochemical ratio computations
artifacts/feature_ablation/               # Feature-screening study outputs
Table S1(1).xlsx                          # Guo & Yang 2023 training table
CRUST_1_0_excel.csv                       # CRUST1.0 grid
luffi_ducea_2022_game_calibration.csv     # Luffi & Ducea 2022 GAME calibration
barrick_logo.png                          # Masthead logo
```

## Notes

- Proxy outputs are *apparent* thickness estimates. Interpret them alongside rock type, alteration screening, cumulate effects, and source/fractionation context.
- CRUST1.0 is a modern (0 Ma) reference; the app does not assume it represents paleo-thickness for older samples unless you tell it to.
- The GAME implementation is a Python-native reconstruction from the published calibration table (the original MATLAB GAME app stores opaque fitted surfaces). It uses the paper defaults `Moho = 6.79 × elevation + 26.40` and the paper-style MAD filtering thresholds for combining mohometers.

## Citation

Holder, D. (2026). *Mohometer: an integrated crustal-thickness estimation tool.* Internal Barrick application.

Underlying methods are credited to the original publications referenced above.
