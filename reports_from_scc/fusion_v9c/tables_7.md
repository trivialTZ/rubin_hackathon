## Table 7 — Component gates (cal-only decisions) + headline & guards

| component | decision | Δ macro AUC (cal) | Δ CI95 | Δ purity@50 (cal) | note |
|---|---|---|---|---|---|
| seq_features | skipped | — | — | — | no seq_* columns in snapshot (v8 mode) |
| ext_features | kept | 0.009 | [0.002, 0.016] | 0.000 |  |
| seq_v9_expert | dropped | -0.001 | [-0.005, 0.003] | 0.000 |  |
| traj_features | dropped | 0.001 | [-0.004, 0.005] | 0.000 |  |
| pooled_trust_q | dropped | -0.057 | [-0.073, -0.039] | -0.020 |  |
| expert_dropout_aug | dropped | -0.002 | [-0.006, 0.001] | 0.000 |  |
| dirichlet_calibration | kept | — | — | — | fallback ladder internal to MulticlassFollowupArtifact |

### Pre-registered headline

```json
{
  "n_objects": 765,
  "fusion_v8_macro_auc": {
    "value": 0.921031618172394,
    "lo": 0.9013927977781355,
    "hi": 0.9387488846870862,
    "n_boot_ok": 1000
  },
  "fusion_v8_auc_snia": {
    "value": 0.9228807979490916,
    "lo": 0.9037939446619274,
    "hi": 0.9404576926580372,
    "n_boot_ok": 1000
  },
  "vs_v6e2_snia_auc_delta": {
    "delta": 0.12941456649342298,
    "lo": 0.1022641140646323,
    "hi": 0.15599293093350622
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
