Guo feature ablation / reduced feature-set screening

Training data:
- Table S1(1).xlsx
- n complete training rows: 1480
- Target column: Crust_Thickness

Screening method:
- ExtraTreesRegressor
- 3-fold cross validation
- 50 trees
- random_state=42
- Purpose: fast relative screening, not a final publication-grade model assessment.

Full 32-feature baseline:
- R2 = 0.902
- RMSE = 4.91 km
- MAE = 2.81 km

Main conclusions:
1. No single feature is absolutely essential by itself because ExtraTrees can substitute correlated features.
2. Rb, K2O, Al2O3, Sr, Eu, TiO2, P2O5, and Ho were the most sensitive single-feature removals in this fast screen.
3. Global feature importance is dominated by incompatible enrichment and HREE depletion features: Rb, Yb, Lu, Tm, Th, Ho, Y, Er, K2O, Ba, Sr, Al2O3.
4. Several reduced models perform similarly to the full 32-feature model in internal CV.
5. External agreement with the full 32-feature model is best for Top_24 and Top_20. The Core_proxy_plus_major_context option is the best geologically interpretable compromise.

Recommended app options:
- Full_32: original Guo compatibility; use when all elements are available.
- Top_24: strongest reduced technical model in this screening.
- Top_20: practical reduced option with good agreement to full model.
- Core_proxy_plus_major_context: best geologically interpretable reduced model.
- Tiny_proxy_set: useful only for rapid screening and must be heavily caveated.

Feature interpretation:
- Rb, K2O, Th, Ba: incompatible enrichment / crustal or source signal.
- Yb, Lu, Tm, Ho, Er, Y: HREE depletion and garnet/amphibole depth signal.
- Sr, Y, Eu, Al2O3: plagioclase/amphibole/fractionation context.
- Major oxides: differentiation, rock type, and petrogenetic context.

Caveat:
This was a fast screening exercise. For publication-grade ranking, rerun with 500 trees, repeated K-fold CV, and external validation by arc/dataset rather than random sample splits.