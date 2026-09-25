## Table 5 — Trust quality per expert (pooled fusion_v8 vs v6e2)

| expert | v8 raw AUC | v8 cal AUC | v8 Brier | v8 ECE | n_test | calibrator | fallback | v6e2 raw AUC | v6e2 cal AUC | v6e2 n_test |
|---|---|---|---|---|---|---|---|---|---|---|
| _pooled | — | — | — | — | — | — | no | — | — | — |
| alerce_lc | 0.841 | 0.840 | 0.1217 | 0.0160 | 17654 | isotonic | no | 0.879 | 0.879 | 30887 |
| ampel/snguess | 0.941 | 0.940 | 0.0361 | 0.0121 | 22613 | isotonic | no | — | — | — |
| fink/rf_ia | 0.860 | 0.860 | 0.1528 | 0.0260 | 23495 | isotonic | no | 0.814 | 0.813 | 23556 |
| fink/slsn | 0.939 | 0.937 | 0.0537 | 0.0320 | 3665 | isotonic | no | — | — | — |
| fink/snn | 0.940 | 0.940 | 0.0847 | 0.0140 | 23495 | isotonic | no | 0.921 | 0.920 | 23556 |
| fink_lsst/cats | — | — | — | — | 0 | isotonic | no | 0.849 | 0.848 | 8348 |
| fink_lsst/early_snia | — | — | — | — | 0 | platt | no | 0.862 | 0.862 | 142 |
| fink_lsst/snn | — | — | — | — | 0 | isotonic | no | 0.836 | 0.835 | 9800 |
| lc_features_bv | 0.850 | 0.849 | 0.1454 | 0.0169 | 27086 | isotonic | no | 0.881 | 0.881 | 39998 |
| pittgoogle/supernnova_lsst | — | — | — | — | 0 | isotonic | no | 0.913 | 0.912 | 6552 |
| salt3_chi2 | 0.873 | 0.873 | 0.1458 | 0.0156 | 27170 | isotonic | no | 0.903 | 0.903 | 37675 |
| seq_v11 | 0.826 | 0.826 | 0.1660 | 0.0197 | 27323 | isotonic | no | — | — | — |
| seq_v9 | 0.811 | 0.811 | 0.1680 | 0.0250 | 27323 | isotonic | no | — | — | — |
| supernnova | 0.847 | 0.847 | 0.1598 | 0.0164 | 22164 | isotonic | no | 0.907 | 0.907 | 34793 |

### Table 5b — Trust at 3-5 detections (calibrated q, locked test, spec-only)

| expert | target | n_det | AUC | ECE(15) | pos/n |
|---|---|---|---|---|---|
| alerce/LC_classifier_ATAT_forced_phot(beta) | is_topclass_correct | (no mapped_pred_class__alerce__LC_classifier_ATAT_forced_phot(beta) column in frame) | — | — | —/— |
| alerce/lc_classifier_BHRF_forced_phot_top | is_topclass_correct | (no mapped_pred_class__alerce__lc_classifier_BHRF_forced_phot_top column in frame) | — | — | —/— |
| alerce/lc_classifier_BHRF_forced_phot_transient | is_topclass_correct | (no mapped_pred_class__alerce__lc_classifier_BHRF_forced_phot_transient column in frame) | — | — | —/— |
| alerce/lc_classifier_transient | is_topclass_correct | (no mapped_pred_class__alerce__lc_classifier_transient column in frame) | — | — | —/— |
| alerce/stamp_classifier | is_topclass_correct | (no mapped_pred_class__alerce__stamp_classifier column in frame) | — | — | —/— |
| alerce/stamp_classifier_2025_beta | is_topclass_correct | (no mapped_pred_class__alerce__stamp_classifier_2025_beta column in frame) | — | — | —/— |
| alerce/stamp_classifier_rubin_beta | is_topclass_correct | (no mapped_pred_class__alerce__stamp_classifier_rubin_beta column in frame) | — | — | —/— |
| alerce_lc | is_topclass_correct | 3 | 0.702 | 0.054 | 595/777 |
| alerce_lc | is_topclass_correct | 4 | 0.703 | 0.045 | 576/770 |
| alerce_lc | is_topclass_correct | 5 | 0.718 | 0.033 | 618/765 |
| alerce_lc | is_topclass_correct | 3-5 | 0.708 | 0.033 | 1789/2312 |
| ampel/parsnip_followme | is_topclass_correct | (no mapped_pred_class__ampel__parsnip_followme column in frame) | — | — | —/— |
| ampel/snguess | is_sn | 3 | — | 0.024 | 690/690 |
| ampel/snguess | is_sn | 4 | — | 0.021 | 684/684 |
| ampel/snguess | is_sn | 5 | — | 0.019 | 679/679 |
| ampel/snguess | is_sn | 3-5 | — | 0.022 | 2053/2053 |
| antares/oracle | is_topclass_correct | (no mapped_pred_class__antares__oracle column in frame) | — | — | —/— |
| antares/superphot_plus | is_topclass_correct | (no mapped_pred_class__antares__superphot_plus column in frame) | — | — | —/— |
| babamul | is_topclass_correct | (no mapped_pred_class__babamul column in frame) | — | — | —/— |
| fink/rf_ia | is_topclass_correct | 3 | 0.690 | 0.159 | 274/670 |
| fink/rf_ia | is_topclass_correct | 4 | 0.719 | 0.122 | 277/683 |
| fink/rf_ia | is_topclass_correct | 5 | 0.773 | 0.107 | 288/702 |
| fink/rf_ia | is_topclass_correct | 3-5 | 0.729 | 0.122 | 839/2055 |
| fink/slsn | is_sn | 3 | — | 0.057 | 66/66 |
| fink/slsn | is_sn | 4 | — | 0.045 | 67/67 |
| fink/slsn | is_sn | 5 | — | 0.036 | 69/69 |
| fink/slsn | is_sn | 3-5 | — | 0.046 | 202/202 |
| fink/snn | is_topclass_correct | 3 | 0.987 | 0.163 | 12/670 |
| fink/snn | is_topclass_correct | 4 | 0.970 | 0.153 | 29/683 |
| fink/snn | is_topclass_correct | 5 | 0.960 | 0.131 | 64/702 |
| fink/snn | is_topclass_correct | 3-5 | 0.969 | 0.147 | 105/2055 |
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
| lc_features_bv | is_topclass_correct | 3 | 0.714 | 0.052 | 609/770 |
| lc_features_bv | is_topclass_correct | 4 | 0.736 | 0.068 | 463/770 |
| lc_features_bv | is_topclass_correct | 5 | 0.802 | 0.037 | 459/765 |
| lc_features_bv | is_topclass_correct | 3-5 | 0.771 | 0.038 | 1531/2305 |
| oracle_lsst | is_topclass_correct | (no mapped_pred_class__oracle_lsst column in frame) | — | — | —/— |
| parsnip | is_topclass_correct | (no mapped_pred_class__parsnip column in frame) | — | — | —/— |
| pittgoogle/supernnova_lsst | is_topclass_correct | 3 | — | — | 0/0 |
| pittgoogle/supernnova_lsst | is_topclass_correct | 4 | — | — | 0/0 |
| pittgoogle/supernnova_lsst | is_topclass_correct | 5 | — | — | 0/0 |
| pittgoogle/supernnova_lsst | is_topclass_correct | 3-5 | — | — | 0/0 |
| pittgoogle/supernnova_ztf | is_topclass_correct | (no mapped_pred_class__pittgoogle__supernnova_ztf column in frame) | — | — | —/— |
| pittgoogle/upsilon_lsst | is_topclass_correct | (no mapped_pred_class__pittgoogle__upsilon_lsst column in frame) | — | — | —/— |
| salt3_chi2 | is_topclass_correct | 3 | 0.704 | 0.043 | 436/777 |
| salt3_chi2 | is_topclass_correct | 4 | 0.724 | 0.076 | 459/770 |
| salt3_chi2 | is_topclass_correct | 5 | 0.795 | 0.053 | 446/765 |
| salt3_chi2 | is_topclass_correct | 3-5 | 0.744 | 0.049 | 1341/2312 |
| seq_v11 | is_topclass_correct | 3 | 0.700 | 0.064 | 406/777 |
| seq_v11 | is_topclass_correct | 4 | 0.728 | 0.061 | 428/770 |
| seq_v11 | is_topclass_correct | 5 | 0.777 | 0.040 | 442/765 |
| seq_v11 | is_topclass_correct | 3-5 | 0.737 | 0.048 | 1276/2312 |
| seq_v9 | is_topclass_correct | 3 | 0.691 | 0.063 | 450/777 |
| seq_v9 | is_topclass_correct | 4 | 0.720 | 0.062 | 465/770 |
| seq_v9 | is_topclass_correct | 5 | 0.763 | 0.050 | 479/765 |
| seq_v9 | is_topclass_correct | 3-5 | 0.727 | 0.052 | 1394/2312 |
| supernnova | is_topclass_correct | 3 | 0.673 | 0.063 | 377/685 |
| supernnova | is_topclass_correct | 4 | 0.729 | 0.060 | 372/678 |
| supernnova | is_topclass_correct | 5 | 0.784 | 0.042 | 376/673 |
| supernnova | is_topclass_correct | 3-5 | 0.733 | 0.047 | 1125/2036 |
