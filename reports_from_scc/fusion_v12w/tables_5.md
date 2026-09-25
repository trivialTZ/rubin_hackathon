## Table 5 — Trust quality per expert (pooled fusion_v8 vs v6e2)

| expert | v8 raw AUC | v8 cal AUC | v8 Brier | v8 ECE | n_test | calibrator | fallback | v6e2 raw AUC | v6e2 cal AUC | v6e2 n_test |
|---|---|---|---|---|---|---|---|---|---|---|
| _pooled | — | — | — | — | — | — | no | — | — | — |
| alerce/stamp_classifier_rubin_beta | — | — | — | — | 0 | isotonic | no | — | — | — |
| alerce_lc | 0.838 | 0.838 | 0.1222 | 0.0196 | 17654 | isotonic | no | 0.879 | 0.879 | 30887 |
| ampel/snguess | 0.936 | 0.936 | 0.0365 | 0.0122 | 22613 | isotonic | no | — | — | — |
| fink/rf_ia | 0.867 | 0.867 | 0.1488 | 0.0236 | 23495 | isotonic | no | 0.814 | 0.813 | 23556 |
| fink/slsn | 0.938 | 0.929 | 0.0509 | 0.0285 | 3665 | isotonic | no | — | — | — |
| fink/snn | 0.941 | 0.941 | 0.0838 | 0.0133 | 23495 | isotonic | no | 0.921 | 0.920 | 23556 |
| fink_lsst/cats | — | — | — | — | 0 | isotonic | no | 0.849 | 0.848 | 8348 |
| fink_lsst/early_snia | — | — | — | — | 0 | global | no | 0.862 | 0.862 | 142 |
| fink_lsst/snn | — | — | — | — | 0 | isotonic | no | 0.836 | 0.835 | 9800 |
| lc_features_bv | 0.853 | 0.853 | 0.1436 | 0.0119 | 27086 | isotonic | no | 0.881 | 0.881 | 39998 |
| parsnip | — | — | — | — | 0 | platt | no | — | — | — |
| salt3_chi2 | 0.873 | 0.873 | 0.1453 | 0.0187 | 27170 | isotonic | no | 0.903 | 0.903 | 37675 |
| seq_v11 | 0.829 | 0.829 | 0.1642 | 0.0127 | 27323 | isotonic | no | — | — | — |
| seq_v9 | 0.818 | 0.817 | 0.1647 | 0.0180 | 27323 | isotonic | no | — | — | — |
| supernnova | 0.849 | 0.849 | 0.1590 | 0.0187 | 22164 | isotonic | no | 0.907 | 0.907 | 34793 |

### Table 5b — Trust at 3-5 detections (calibrated q, locked test, spec-only)

| expert | target | n_det | AUC | ECE(15) | pos/n |
|---|---|---|---|---|---|
| alerce/LC_classifier_ATAT_forced_phot(beta) | is_topclass_correct | (no mapped_pred_class__alerce__LC_classifier_ATAT_forced_phot(beta) column in frame) | — | — | —/— |
| alerce/lc_classifier_BHRF_forced_phot_top | is_topclass_correct | (no mapped_pred_class__alerce__lc_classifier_BHRF_forced_phot_top column in frame) | — | — | —/— |
| alerce/lc_classifier_BHRF_forced_phot_transient | is_topclass_correct | (no mapped_pred_class__alerce__lc_classifier_BHRF_forced_phot_transient column in frame) | — | — | —/— |
| alerce/lc_classifier_transient | is_topclass_correct | (no mapped_pred_class__alerce__lc_classifier_transient column in frame) | — | — | —/— |
| alerce/stamp_classifier | is_topclass_correct | (no mapped_pred_class__alerce__stamp_classifier column in frame) | — | — | —/— |
| alerce/stamp_classifier_2025_beta | is_topclass_correct | (no mapped_pred_class__alerce__stamp_classifier_2025_beta column in frame) | — | — | —/— |
| alerce/stamp_classifier_rubin_beta | is_topclass_correct | 3 | — | — | 0/0 |
| alerce/stamp_classifier_rubin_beta | is_topclass_correct | 4 | — | — | 0/0 |
| alerce/stamp_classifier_rubin_beta | is_topclass_correct | 5 | — | — | 0/0 |
| alerce/stamp_classifier_rubin_beta | is_topclass_correct | 3-5 | — | — | 0/0 |
| alerce_lc | is_topclass_correct | 3 | 0.701 | 0.050 | 595/777 |
| alerce_lc | is_topclass_correct | 4 | 0.693 | 0.061 | 576/770 |
| alerce_lc | is_topclass_correct | 5 | 0.716 | 0.037 | 618/765 |
| alerce_lc | is_topclass_correct | 3-5 | 0.704 | 0.042 | 1789/2312 |
| ampel/parsnip_followme | is_topclass_correct | (no mapped_pred_class__ampel__parsnip_followme column in frame) | — | — | —/— |
| ampel/snguess | is_sn | 3 | — | 0.030 | 690/690 |
| ampel/snguess | is_sn | 4 | — | 0.025 | 684/684 |
| ampel/snguess | is_sn | 5 | — | 0.024 | 679/679 |
| ampel/snguess | is_sn | 3-5 | — | 0.027 | 2053/2053 |
| antares/oracle | is_topclass_correct | (no mapped_pred_class__antares__oracle column in frame) | — | — | —/— |
| antares/superphot_plus | is_topclass_correct | (no mapped_pred_class__antares__superphot_plus column in frame) | — | — | —/— |
| babamul | is_topclass_correct | (no mapped_pred_class__babamul column in frame) | — | — | —/— |
| fink/rf_ia | is_topclass_correct | 3 | 0.688 | 0.172 | 274/670 |
| fink/rf_ia | is_topclass_correct | 4 | 0.709 | 0.144 | 277/683 |
| fink/rf_ia | is_topclass_correct | 5 | 0.759 | 0.120 | 288/702 |
| fink/rf_ia | is_topclass_correct | 3-5 | 0.721 | 0.142 | 839/2055 |
| fink/slsn | is_sn | 3 | — | 0.051 | 66/66 |
| fink/slsn | is_sn | 4 | — | 0.046 | 67/67 |
| fink/slsn | is_sn | 5 | — | 0.036 | 69/69 |
| fink/slsn | is_sn | 3-5 | — | 0.045 | 202/202 |
| fink/snn | is_topclass_correct | 3 | 0.978 | 0.149 | 12/670 |
| fink/snn | is_topclass_correct | 4 | 0.978 | 0.139 | 29/683 |
| fink/snn | is_topclass_correct | 5 | 0.967 | 0.126 | 64/702 |
| fink/snn | is_topclass_correct | 3-5 | 0.974 | 0.134 | 105/2055 |
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
| lasair/sherlock | is_topclass_correct | (no mapped_pred_class__lasair__sherlock column in frame) | — | — | —/— |
| lc_features_bv | is_topclass_correct | 3 | 0.717 | 0.044 | 609/770 |
| lc_features_bv | is_topclass_correct | 4 | 0.736 | 0.068 | 463/770 |
| lc_features_bv | is_topclass_correct | 5 | 0.796 | 0.050 | 459/765 |
| lc_features_bv | is_topclass_correct | 3-5 | 0.769 | 0.038 | 1531/2305 |
| oracle_lsst | is_topclass_correct | (no mapped_pred_class__oracle_lsst column in frame) | — | — | —/— |
| parsnip | is_topclass_correct | 3 | — | — | 0/0 |
| parsnip | is_topclass_correct | 4 | — | — | 0/0 |
| parsnip | is_topclass_correct | 5 | — | — | 0/0 |
| parsnip | is_topclass_correct | 3-5 | — | — | 0/0 |
| pittgoogle/supernnova_lsst | is_topclass_correct | (no mapped_pred_class__pittgoogle__supernnova_lsst column in frame) | — | — | —/— |
| pittgoogle/supernnova_ztf | is_topclass_correct | (no mapped_pred_class__pittgoogle__supernnova_ztf column in frame) | — | — | —/— |
| pittgoogle/upsilon_lsst | is_topclass_correct | (no mapped_pred_class__pittgoogle__upsilon_lsst column in frame) | — | — | —/— |
| salt3_chi2 | is_topclass_correct | 3 | 0.707 | 0.041 | 436/777 |
| salt3_chi2 | is_topclass_correct | 4 | 0.724 | 0.082 | 459/770 |
| salt3_chi2 | is_topclass_correct | 5 | 0.796 | 0.052 | 446/765 |
| salt3_chi2 | is_topclass_correct | 3-5 | 0.745 | 0.050 | 1341/2312 |
| seq_v11 | is_topclass_correct | 3 | 0.705 | 0.058 | 406/777 |
| seq_v11 | is_topclass_correct | 4 | 0.727 | 0.055 | 428/770 |
| seq_v11 | is_topclass_correct | 5 | 0.778 | 0.057 | 442/765 |
| seq_v11 | is_topclass_correct | 3-5 | 0.739 | 0.047 | 1276/2312 |
| seq_v9 | is_topclass_correct | 3 | 0.696 | 0.051 | 450/777 |
| seq_v9 | is_topclass_correct | 4 | 0.719 | 0.065 | 465/770 |
| seq_v9 | is_topclass_correct | 5 | 0.767 | 0.031 | 479/765 |
| seq_v9 | is_topclass_correct | 3-5 | 0.730 | 0.036 | 1394/2312 |
| supernnova | is_topclass_correct | 3 | 0.679 | 0.068 | 377/685 |
| supernnova | is_topclass_correct | 4 | 0.730 | 0.075 | 372/678 |
| supernnova | is_topclass_correct | 5 | 0.778 | 0.060 | 376/673 |
| supernnova | is_topclass_correct | 3-5 | 0.732 | 0.056 | 1125/2036 |
