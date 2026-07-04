## Table 7 — Component gates (cal-only decisions) + headline & guards

| component | decision | Δ macro AUC (cal) | Δ CI95 | Δ purity@50 (cal) | note |
|---|---|---|---|---|---|
| seq_features | skipped | — | — | — | no seq_* columns in snapshot (v8 mode) |
| ext_features | kept | 0.008 | [0.001, 0.015] | 0.000 |  |
| seq_v9_expert | dropped | -0.000 | [-0.004, 0.004] | 0.000 |  |
| traj_features | dropped | 0.002 | [-0.002, 0.006] | 0.000 |  |
| pooled_trust_q | dropped | -0.046 | [-0.063, -0.031] | 0.000 |  |
| expert_dropout_aug | dropped | -0.001 | [-0.004, 0.003] | 0.000 |  |
| dirichlet_calibration | kept | — | — | — | fallback ladder internal to MulticlassFollowupArtifact |

### Pre-registered headline

```json
{
  "n_objects": 765,
  "fusion_v8_macro_auc": {
    "value": 0.9225690990852164,
    "lo": 0.9030066407566703,
    "hi": 0.9398508925633241,
    "n_boot_ok": 1000
  },
  "fusion_v8_auc_snia": {
    "value": 0.9244708123782974,
    "lo": 0.9049566468436294,
    "hi": 0.9420202617708251,
    "n_boot_ok": 1000
  },
  "vs_v6e2_snia_auc_delta": {
    "delta": 0.1310045809226288,
    "lo": 0.1037118732451912,
    "hi": 0.15735672916485685
  },
  "vs_v6e2_significant_win": true,
  "claim": "fusion_v8 beats re-scored v6e2 on snia OvR AUC @ n_det=5 (CI95 excludes 0)",
  "note": "macro OvR AUC has no v6e2 counterpart (binary head); the snia OvR axis is the comparable one"
}
```

### Guards

- **lsst_spec_non_regression**: N/A (no LSST spec test slice locally, or v6e2 unavailable)
- **dp1_eclbin_rrlyrae_ef_non_regression**: PASS
- **seed_spread_lt_0.02**: PASS
