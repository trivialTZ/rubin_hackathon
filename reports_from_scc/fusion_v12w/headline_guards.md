# fusion_v8 — pre-registered headline & guards

```json
{
  "preregistered": {
    "headline": "object-level spec-only macro OvR AUC @ n_det=5, locked test, fusion_v8 vs re-scored v6e2",
    "guards": [
      "LSST-spec slice non-regression",
      "DP1 EclBin+RRLyrae EF non-regression",
      "LightGBM bagging-seed spread on headline < 0.02 (5 seeds)"
    ]
  },
  "headline": {
    "n_objects": 765,
    "fusion_v8_macro_auc": {
      "value": 0.886394519703571,
      "lo": 0.8629034587408618,
      "hi": 0.9094199749063236,
      "n_boot_ok": 1000
    },
    "fusion_v8_auc_snia": {
      "value": 0.8880545788212879,
      "lo": 0.8638947584134109,
      "hi": 0.9109232463167607,
      "n_boot_ok": 1000
    },
    "vs_v6e2_snia_auc_delta": {
      "delta": 0.09458834736561927,
      "lo": 0.06782393546543687,
      "hi": 0.12144950657917919
    },
    "vs_v6e2_significant_win": true,
    "claim": "fusion_v8 beats re-scored v6e2 on snia OvR AUC @ n_det=5 (CI95 excludes 0)",
    "note": "macro OvR AUC has no v6e2 counterpart (binary head); the snia OvR axis is the comparable one"
  },
  "guards": [
    {
      "guard": "lsst_spec_non_regression",
      "n_objects": 0,
      "status": "N/A (no LSST spec test slice locally, or v6e2 unavailable)"
    },
    {
      "guard": "dp1_eclbin_rrlyrae_ef_non_regression",
      "fusion_v8_ef": 7.857788624899009,
      "v6e2_ef": 17.88,
      "v6e2_source": "documented v6e2 headline (17.88)",
      "pass": true,
      "rule": "pass iff fusion_v8 EF@top-1% <= v6e2 EF (lower = better suppression)"
    },
    {
      "guard": "seed_spread_lt_0.02",
      "status": "N/A (no seed variants recorded in fusion_v8_train.json)"
    }
  ]
}
```
