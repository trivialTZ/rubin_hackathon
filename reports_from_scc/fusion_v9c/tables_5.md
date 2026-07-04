## Table 5 — Trust quality per expert (pooled fusion_v8 vs v6e2)

| expert | v8 raw AUC | v8 cal AUC | v8 Brier | v8 ECE | n_test | calibrator | fallback | v6e2 raw AUC | v6e2 cal AUC | v6e2 n_test |
|---|---|---|---|---|---|---|---|---|---|---|
| _pooled | — | — | — | — | — | — | no | — | — | — |
| alerce_lc | 0.841 | 0.840 | 0.1218 | 0.0196 | 17654 | isotonic | no | 0.879 | 0.879 | 30887 |
| ampel/snguess | 0.939 | 0.937 | 0.0361 | 0.0132 | 22613 | isotonic | no | — | — | — |
| fink/rf_ia | 0.862 | 0.862 | 0.1520 | 0.0257 | 23495 | isotonic | no | 0.814 | 0.813 | 23556 |
| fink/slsn | 0.937 | 0.932 | 0.0541 | 0.0373 | 3665 | isotonic | no | — | — | — |
| fink/snn | 0.940 | 0.939 | 0.0849 | 0.0139 | 23495 | isotonic | no | 0.921 | 0.920 | 23556 |
| fink_lsst/cats | — | — | — | — | 0 | isotonic | no | 0.849 | 0.848 | 8348 |
| fink_lsst/early_snia | — | — | — | — | 0 | global | no | 0.862 | 0.862 | 142 |
| fink_lsst/snn | — | — | — | — | 0 | isotonic | no | 0.836 | 0.835 | 9800 |
| lc_features_bv | 0.846 | 0.846 | 0.1474 | 0.0204 | 27086 | isotonic | no | 0.881 | 0.881 | 39998 |
| pittgoogle/supernnova_lsst | — | — | — | — | 0 | isotonic | no | 0.913 | 0.912 | 6552 |
| salt3_chi2 | 0.873 | 0.872 | 0.1458 | 0.0141 | 27170 | isotonic | no | 0.903 | 0.903 | 37675 |
| seq_v9 | 0.824 | 0.824 | 0.1602 | 0.0228 | 27323 | isotonic | no | — | — | — |
| supernnova | 0.847 | 0.847 | 0.1598 | 0.0209 | 22164 | isotonic | no | 0.907 | 0.907 | 34793 |
