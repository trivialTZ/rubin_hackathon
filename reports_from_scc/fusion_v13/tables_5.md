## Table 5 — Trust quality per expert (pooled fusion_v8 vs v6e2)

| expert | v8 raw AUC | v8 cal AUC | v8 Brier | v8 ECE | n_test | calibrator | fallback | v6e2 raw AUC | v6e2 cal AUC | v6e2 n_test |
|---|---|---|---|---|---|---|---|---|---|---|
| _pooled | — | — | — | — | — | — | no | — | — | — |
| alerce/stamp_classifier_rubin_beta | — | — | — | — | 0 | isotonic | no | — | — | — |
| alerce_lc | 0.833 | 0.833 | 0.1242 | 0.0185 | 17654 | isotonic | no | 0.879 | 0.879 | 30887 |
| ampel/snguess | 0.937 | 0.936 | 0.0366 | 0.0091 | 22613 | isotonic | no | — | — | — |
| fink/rf_ia | 0.862 | 0.862 | 0.1516 | 0.0222 | 23495 | isotonic | no | 0.814 | 0.813 | 23556 |
| fink/slsn | 0.933 | 0.933 | 0.0542 | 0.0283 | 3665 | isotonic | no | — | — | — |
| fink/snn | 0.939 | 0.939 | 0.0861 | 0.0181 | 23495 | isotonic | no | 0.921 | 0.920 | 23556 |
| fink_lsst/cats | — | — | — | — | 0 | isotonic | no | 0.849 | 0.848 | 8348 |
| fink_lsst/early_snia | — | — | — | — | 0 | global | no | 0.862 | 0.862 | 142 |
| fink_lsst/snn | — | — | — | — | 0 | isotonic | no | 0.836 | 0.835 | 9800 |
| lc_features_bv | 0.851 | 0.851 | 0.1448 | 0.0125 | 27086 | isotonic | no | 0.881 | 0.881 | 39998 |
| salt3_chi2 | 0.871 | 0.871 | 0.1468 | 0.0163 | 27170 | isotonic | no | 0.903 | 0.903 | 37675 |
| seq_v11 | 0.826 | 0.826 | 0.1657 | 0.0135 | 27323 | isotonic | no | — | — | — |
| seq_v9 | 0.816 | 0.815 | 0.1658 | 0.0112 | 27323 | isotonic | no | — | — | — |
| supernnova | 0.848 | 0.847 | 0.1599 | 0.0217 | 22164 | isotonic | no | 0.907 | 0.907 | 34793 |

### Table 5b — Trust at 3-5 detections (calibrated q, locked test, spec-only)

| expert | target | n_det | AUC | ECE(15) | pos/n |
|---|---|---|---|---|---|
| alerce/stamp_classifier_rubin_beta | is_sn | 3 | — | — | 0/0 |
| alerce/stamp_classifier_rubin_beta | is_sn | 4 | — | — | 0/0 |
| alerce/stamp_classifier_rubin_beta | is_sn | 5 | — | — | 0/0 |
| alerce/stamp_classifier_rubin_beta | is_sn | 3-5 | — | — | 0/0 |
| alerce_lc | is_topclass_correct | 3 | 0.694 | 0.052 | 595/777 |
| alerce_lc | is_topclass_correct | 4 | 0.692 | 0.054 | 576/770 |
| alerce_lc | is_topclass_correct | 5 | 0.712 | 0.045 | 618/765 |
| alerce_lc | is_topclass_correct | 3-5 | 0.700 | 0.039 | 1789/2312 |
| ampel/snguess | is_sn | 3 | — | 0.031 | 690/690 |
| ampel/snguess | is_sn | 4 | — | 0.028 | 684/684 |
| ampel/snguess | is_sn | 5 | — | 0.026 | 679/679 |
| ampel/snguess | is_sn | 3-5 | — | 0.028 | 2053/2053 |
| fink/rf_ia | is_topclass_correct | 3 | 0.690 | 0.225 | 274/670 |
| fink/rf_ia | is_topclass_correct | 4 | 0.721 | 0.130 | 277/683 |
| fink/rf_ia | is_topclass_correct | 5 | 0.771 | 0.111 | 288/702 |
| fink/rf_ia | is_topclass_correct | 3-5 | 0.725 | 0.151 | 839/2055 |
| fink/slsn | is_sn | 3 | — | 0.050 | 66/66 |
| fink/slsn | is_sn | 4 | — | 0.042 | 67/67 |
| fink/slsn | is_sn | 5 | — | 0.032 | 69/69 |
| fink/slsn | is_sn | 3-5 | — | 0.041 | 202/202 |
| fink/snn | is_topclass_correct | 3 | 0.993 | 0.096 | 12/670 |
| fink/snn | is_topclass_correct | 4 | 0.980 | 0.095 | 29/683 |
| fink/snn | is_topclass_correct | 5 | 0.972 | 0.091 | 64/702 |
| fink/snn | is_topclass_correct | 3-5 | 0.981 | 0.089 | 105/2055 |
| fink_lsst/cats | is_sn | 3 | — | — | 0/0 |
| fink_lsst/cats | is_sn | 4 | — | — | 0/0 |
| fink_lsst/cats | is_sn | 5 | — | — | 0/0 |
| fink_lsst/cats | is_sn | 3-5 | — | — | 0/0 |
| fink_lsst/early_snia | is_topclass_correct | 3 | — | — | 0/0 |
| fink_lsst/early_snia | is_topclass_correct | 4 | — | — | 0/0 |
| fink_lsst/early_snia | is_topclass_correct | 5 | — | — | 0/0 |
| fink_lsst/early_snia | is_topclass_correct | 3-5 | — | — | 0/0 |
| fink_lsst/snn | is_sn | 3 | — | — | 0/0 |
| fink_lsst/snn | is_sn | 4 | — | — | 0/0 |
| fink_lsst/snn | is_sn | 5 | — | — | 0/0 |
| fink_lsst/snn | is_sn | 3-5 | — | — | 0/0 |
| lc_features_bv | is_topclass_correct | 3 | 0.722 | 0.054 | 609/770 |
| lc_features_bv | is_topclass_correct | 4 | 0.726 | 0.074 | 463/770 |
| lc_features_bv | is_topclass_correct | 5 | 0.791 | 0.058 | 459/765 |
| lc_features_bv | is_topclass_correct | 3-5 | 0.766 | 0.050 | 1531/2305 |
| salt3_chi2 | is_topclass_correct | 3 | 0.701 | 0.052 | 436/777 |
| salt3_chi2 | is_topclass_correct | 4 | 0.723 | 0.092 | 459/770 |
| salt3_chi2 | is_topclass_correct | 5 | 0.791 | 0.057 | 446/765 |
| salt3_chi2 | is_topclass_correct | 3-5 | 0.741 | 0.054 | 1341/2312 |
| seq_v11 | is_topclass_correct | 3 | 0.704 | 0.051 | 406/777 |
| seq_v11 | is_topclass_correct | 4 | 0.741 | 0.049 | 428/770 |
| seq_v11 | is_topclass_correct | 5 | 0.782 | 0.039 | 442/765 |
| seq_v11 | is_topclass_correct | 3-5 | 0.744 | 0.033 | 1276/2312 |
| seq_v9 | is_topclass_correct | 3 | 0.692 | 0.058 | 450/777 |
| seq_v9 | is_topclass_correct | 4 | 0.726 | 0.038 | 465/770 |
| seq_v9 | is_topclass_correct | 5 | 0.766 | 0.028 | 479/765 |
| seq_v9 | is_topclass_correct | 3-5 | 0.730 | 0.030 | 1394/2312 |
| supernnova | is_topclass_correct | 3 | 0.684 | 0.065 | 377/685 |
| supernnova | is_topclass_correct | 4 | 0.730 | 0.061 | 372/678 |
| supernnova | is_topclass_correct | 5 | 0.784 | 0.054 | 376/673 |
| supernnova | is_topclass_correct | 3-5 | 0.735 | 0.051 | 1125/2036 |
