## Table 5 — Trust quality per expert (pooled fusion_v8 vs v6e2)

| expert | v8 raw AUC | v8 cal AUC | v8 Brier | v8 ECE | n_test | calibrator | fallback | v6e2 raw AUC | v6e2 cal AUC | v6e2 n_test |
|---|---|---|---|---|---|---|---|---|---|---|
| _pooled | — | — | — | — | — | — | no | — | — | — |
| alerce/stamp_classifier_rubin_beta | — | — | — | — | 0 | isotonic | no | — | — | — |
| alerce_lc | 0.838 | 0.838 | 0.1229 | 0.0199 | 17654 | isotonic | no | 0.879 | 0.879 | 30887 |
| ampel/snguess | 0.935 | 0.933 | 0.0365 | 0.0086 | 22613 | isotonic | no | — | — | — |
| fink/rf_ia | 0.858 | 0.858 | 0.1542 | 0.0236 | 23495 | isotonic | no | 0.814 | 0.813 | 23556 |
| fink/slsn | 0.939 | 0.936 | 0.0522 | 0.0342 | 3665 | isotonic | no | — | — | — |
| fink/snn | 0.936 | 0.936 | 0.0871 | 0.0147 | 23495 | isotonic | no | 0.921 | 0.920 | 23556 |
| fink_lsst/cats | — | — | — | — | 0 | isotonic | no | 0.849 | 0.848 | 8348 |
| fink_lsst/early_snia | — | — | — | — | 0 | global | no | 0.862 | 0.862 | 142 |
| fink_lsst/snn | — | — | — | — | 0 | isotonic | no | 0.836 | 0.835 | 9800 |
| lc_features_bv | 0.845 | 0.845 | 0.1473 | 0.0097 | 27086 | isotonic | no | 0.881 | 0.881 | 39998 |
| parsnip | — | — | — | — | 0 | platt | no | — | — | — |
| salt3_chi2 | 0.870 | 0.870 | 0.1472 | 0.0160 | 27170 | isotonic | no | 0.903 | 0.903 | 37675 |
| seq_v11 | 0.825 | 0.825 | 0.1663 | 0.0126 | 27323 | isotonic | no | — | — | — |
| seq_v9 | 0.814 | 0.814 | 0.1661 | 0.0106 | 27323 | isotonic | no | — | — | — |
| supernnova | 0.848 | 0.848 | 0.1593 | 0.0181 | 22164 | isotonic | no | 0.907 | 0.907 | 34793 |

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
| alerce_lc | is_topclass_correct | 3 | 0.706 | 0.055 | 595/777 |
| alerce_lc | is_topclass_correct | 4 | 0.689 | 0.068 | 576/770 |
| alerce_lc | is_topclass_correct | 5 | 0.721 | 0.030 | 618/765 |
| alerce_lc | is_topclass_correct | 3-5 | 0.706 | 0.047 | 1789/2312 |
| ampel/parsnip_followme | is_topclass_correct | (no mapped_pred_class__ampel__parsnip_followme column in frame) | — | — | —/— |
| ampel/snguess | is_sn | 3 | — | 0.037 | 690/690 |
| ampel/snguess | is_sn | 4 | — | 0.033 | 684/684 |
| ampel/snguess | is_sn | 5 | — | 0.031 | 679/679 |
| ampel/snguess | is_sn | 3-5 | — | 0.034 | 2053/2053 |
| antares/oracle | is_topclass_correct | (no mapped_pred_class__antares__oracle column in frame) | — | — | —/— |
| antares/superphot_plus | is_topclass_correct | (no mapped_pred_class__antares__superphot_plus column in frame) | — | — | —/— |
| babamul | is_topclass_correct | (no mapped_pred_class__babamul column in frame) | — | — | —/— |
| fink/rf_ia | is_topclass_correct | 3 | 0.693 | 0.230 | 274/670 |
| fink/rf_ia | is_topclass_correct | 4 | 0.703 | 0.185 | 277/683 |
| fink/rf_ia | is_topclass_correct | 5 | 0.745 | 0.176 | 288/702 |
| fink/rf_ia | is_topclass_correct | 3-5 | 0.714 | 0.197 | 839/2055 |
| fink/slsn | is_sn | 3 | — | 0.047 | 66/66 |
| fink/slsn | is_sn | 4 | — | 0.043 | 67/67 |
| fink/slsn | is_sn | 5 | — | 0.033 | 69/69 |
| fink/slsn | is_sn | 3-5 | — | 0.041 | 202/202 |
| fink/snn | is_topclass_correct | 3 | 0.986 | 0.170 | 12/670 |
| fink/snn | is_topclass_correct | 4 | 0.981 | 0.164 | 29/683 |
| fink/snn | is_topclass_correct | 5 | 0.964 | 0.157 | 64/702 |
| fink/snn | is_topclass_correct | 3-5 | 0.975 | 0.161 | 105/2055 |
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
| lc_features_bv | is_topclass_correct | 3 | 0.728 | 0.050 | 609/770 |
| lc_features_bv | is_topclass_correct | 4 | 0.729 | 0.060 | 463/770 |
| lc_features_bv | is_topclass_correct | 5 | 0.796 | 0.044 | 459/765 |
| lc_features_bv | is_topclass_correct | 3-5 | 0.772 | 0.042 | 1531/2305 |
| oracle_lsst | is_topclass_correct | (no mapped_pred_class__oracle_lsst column in frame) | — | — | —/— |
| parsnip | is_topclass_correct | 3 | — | — | 0/0 |
| parsnip | is_topclass_correct | 4 | — | — | 0/0 |
| parsnip | is_topclass_correct | 5 | — | — | 0/0 |
| parsnip | is_topclass_correct | 3-5 | — | — | 0/0 |
| pittgoogle/supernnova_lsst | is_topclass_correct | (no mapped_pred_class__pittgoogle__supernnova_lsst column in frame) | — | — | —/— |
| pittgoogle/supernnova_ztf | is_topclass_correct | (no mapped_pred_class__pittgoogle__supernnova_ztf column in frame) | — | — | —/— |
| pittgoogle/upsilon_lsst | is_topclass_correct | (no mapped_pred_class__pittgoogle__upsilon_lsst column in frame) | — | — | —/— |
| salt3_chi2 | is_topclass_correct | 3 | 0.697 | 0.061 | 436/777 |
| salt3_chi2 | is_topclass_correct | 4 | 0.717 | 0.087 | 459/770 |
| salt3_chi2 | is_topclass_correct | 5 | 0.794 | 0.072 | 446/765 |
| salt3_chi2 | is_topclass_correct | 3-5 | 0.739 | 0.052 | 1341/2312 |
| seq_v11 | is_topclass_correct | 3 | 0.705 | 0.047 | 406/777 |
| seq_v11 | is_topclass_correct | 4 | 0.727 | 0.042 | 428/770 |
| seq_v11 | is_topclass_correct | 5 | 0.772 | 0.058 | 442/765 |
| seq_v11 | is_topclass_correct | 3-5 | 0.737 | 0.037 | 1276/2312 |
| seq_v9 | is_topclass_correct | 3 | 0.701 | 0.044 | 450/777 |
| seq_v9 | is_topclass_correct | 4 | 0.719 | 0.048 | 465/770 |
| seq_v9 | is_topclass_correct | 5 | 0.769 | 0.031 | 479/765 |
| seq_v9 | is_topclass_correct | 3-5 | 0.731 | 0.029 | 1394/2312 |
| supernnova | is_topclass_correct | 3 | 0.672 | 0.070 | 377/685 |
| supernnova | is_topclass_correct | 4 | 0.722 | 0.076 | 372/678 |
| supernnova | is_topclass_correct | 5 | 0.778 | 0.070 | 376/673 |
| supernnova | is_topclass_correct | 3-5 | 0.727 | 0.063 | 1125/2036 |
