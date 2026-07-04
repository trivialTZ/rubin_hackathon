## Table 5 — Trust quality per expert (pooled fusion_v8 vs v6e2)

| expert | v8 raw AUC | v8 cal AUC | v8 Brier | v8 ECE | n_test | calibrator | fallback | v6e2 raw AUC | v6e2 cal AUC | v6e2 n_test |
|---|---|---|---|---|---|---|---|---|---|---|
| _pooled | — | — | — | — | — | — | no | — | — | — |
| alerce_lc | 0.841 | 0.840 | 0.1219 | 0.0197 | 17654 | isotonic | no | 0.879 | 0.879 | 30887 |
| ampel/snguess | 0.942 | 0.942 | 0.0365 | 0.0153 | 22613 | isotonic | no | — | — | — |
| fink/rf_ia | 0.865 | 0.865 | 0.1504 | 0.0232 | 23495 | isotonic | no | 0.814 | 0.813 | 23556 |
| fink/slsn | 0.934 | 0.921 | 0.0541 | 0.0251 | 3665 | isotonic | no | — | — | — |
| fink/snn | 0.941 | 0.941 | 0.0840 | 0.0134 | 23495 | isotonic | no | 0.921 | 0.920 | 23556 |
| fink_lsst/cats | — | — | — | — | 0 | isotonic | no | 0.849 | 0.848 | 8348 |
| fink_lsst/early_snia | — | — | — | — | 0 | global | no | 0.862 | 0.862 | 142 |
| fink_lsst/snn | — | — | — | — | 0 | isotonic | no | 0.836 | 0.835 | 9800 |
| lc_features_bv | 0.853 | 0.852 | 0.1444 | 0.0239 | 27086 | isotonic | no | 0.881 | 0.881 | 39998 |
| pittgoogle/supernnova_lsst | — | — | — | — | 0 | isotonic | no | 0.913 | 0.912 | 6552 |
| salt3_chi2 | 0.873 | 0.873 | 0.1454 | 0.0167 | 27170 | isotonic | no | 0.903 | 0.903 | 37675 |
| seq_v9 | 0.812 | 0.812 | 0.1676 | 0.0258 | 27323 | isotonic | no | — | — | — |
| supernnova | 0.848 | 0.848 | 0.1595 | 0.0209 | 22164 | isotonic | no | 0.907 | 0.907 | 34793 |

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
| alerce_lc | is_topclass_correct | 3 | 0.707 | 0.045 | 595/777 |
| alerce_lc | is_topclass_correct | 4 | 0.697 | 0.050 | 576/770 |
| alerce_lc | is_topclass_correct | 5 | 0.713 | 0.059 | 618/765 |
| alerce_lc | is_topclass_correct | 3-5 | 0.707 | 0.041 | 1789/2312 |
| ampel/parsnip_followme | is_topclass_correct | (no mapped_pred_class__ampel__parsnip_followme column in frame) | — | — | —/— |
| ampel/snguess | is_sn | 3 | — | 0.023 | 690/690 |
| ampel/snguess | is_sn | 4 | — | 0.020 | 684/684 |
| ampel/snguess | is_sn | 5 | — | 0.019 | 679/679 |
| ampel/snguess | is_sn | 3-5 | — | 0.021 | 2053/2053 |
| antares/oracle | is_topclass_correct | (no mapped_pred_class__antares__oracle column in frame) | — | — | —/— |
| antares/superphot_plus | is_topclass_correct | (no mapped_pred_class__antares__superphot_plus column in frame) | — | — | —/— |
| babamul | is_topclass_correct | (no mapped_pred_class__babamul column in frame) | — | — | —/— |
| fink/rf_ia | is_topclass_correct | 3 | 0.693 | 0.186 | 274/670 |
| fink/rf_ia | is_topclass_correct | 4 | 0.709 | 0.181 | 277/683 |
| fink/rf_ia | is_topclass_correct | 5 | 0.755 | 0.168 | 288/702 |
| fink/rf_ia | is_topclass_correct | 3-5 | 0.720 | 0.176 | 839/2055 |
| fink/slsn | is_sn | 3 | — | 0.053 | 66/66 |
| fink/slsn | is_sn | 4 | — | 0.045 | 67/67 |
| fink/slsn | is_sn | 5 | — | 0.039 | 69/69 |
| fink/slsn | is_sn | 3-5 | — | 0.045 | 202/202 |
| fink/snn | is_topclass_correct | 3 | 0.992 | 0.134 | 12/670 |
| fink/snn | is_topclass_correct | 4 | 0.987 | 0.122 | 29/683 |
| fink/snn | is_topclass_correct | 5 | 0.973 | 0.114 | 64/702 |
| fink/snn | is_topclass_correct | 3-5 | 0.983 | 0.121 | 105/2055 |
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
| lc_features_bv | is_topclass_correct | 3 | 0.720 | 0.037 | 609/770 |
| lc_features_bv | is_topclass_correct | 4 | 0.736 | 0.068 | 463/770 |
| lc_features_bv | is_topclass_correct | 5 | 0.792 | 0.052 | 459/765 |
| lc_features_bv | is_topclass_correct | 3-5 | 0.768 | 0.037 | 1531/2305 |
| oracle_lsst | is_topclass_correct | (no mapped_pred_class__oracle_lsst column in frame) | — | — | —/— |
| parsnip | is_topclass_correct | (no mapped_pred_class__parsnip column in frame) | — | — | —/— |
| pittgoogle/supernnova_lsst | is_topclass_correct | 3 | — | — | 0/0 |
| pittgoogle/supernnova_lsst | is_topclass_correct | 4 | — | — | 0/0 |
| pittgoogle/supernnova_lsst | is_topclass_correct | 5 | — | — | 0/0 |
| pittgoogle/supernnova_lsst | is_topclass_correct | 3-5 | — | — | 0/0 |
| pittgoogle/supernnova_ztf | is_topclass_correct | (no mapped_pred_class__pittgoogle__supernnova_ztf column in frame) | — | — | —/— |
| pittgoogle/upsilon_lsst | is_topclass_correct | (no mapped_pred_class__pittgoogle__upsilon_lsst column in frame) | — | — | —/— |
| salt3_chi2 | is_topclass_correct | 3 | 0.711 | 0.039 | 436/777 |
| salt3_chi2 | is_topclass_correct | 4 | 0.727 | 0.086 | 459/770 |
| salt3_chi2 | is_topclass_correct | 5 | 0.798 | 0.055 | 446/765 |
| salt3_chi2 | is_topclass_correct | 3-5 | 0.748 | 0.049 | 1341/2312 |
| seq_v9 | is_topclass_correct | 3 | 0.697 | 0.060 | 450/777 |
| seq_v9 | is_topclass_correct | 4 | 0.713 | 0.049 | 465/770 |
| seq_v9 | is_topclass_correct | 5 | 0.760 | 0.039 | 479/765 |
| seq_v9 | is_topclass_correct | 3-5 | 0.725 | 0.044 | 1394/2312 |
| supernnova | is_topclass_correct | 3 | 0.679 | 0.060 | 377/685 |
| supernnova | is_topclass_correct | 4 | 0.725 | 0.073 | 372/678 |
| supernnova | is_topclass_correct | 5 | 0.780 | 0.050 | 376/673 |
| supernnova | is_topclass_correct | 3-5 | 0.730 | 0.053 | 1125/2036 |
