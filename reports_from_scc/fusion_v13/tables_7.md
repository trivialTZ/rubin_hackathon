## Table 7 — Component gates (cal-only decisions) + headline & guards

| component | decision | Δ macro AUC (cal) | Δ CI95 | Δ purity@50 (cal) | note |
|---|---|---|---|---|---|
| ? | pooled | — | — | — |  |
| ? | shared_head2 | — | — | — |  |
| ? | in | — | — | — |  |

### Pre-registered headline

```json
{
  "n_objects": 765,
  "fusion_v8_macro_auc": {
    "value": 0.9191929450989731,
    "lo": 0.9001688363973304,
    "hi": 0.9367970486930629,
    "n_boot_ok": 1000
  },
  "fusion_v8_auc_snia": {
    "value": 0.919739293669361,
    "lo": 0.901095432680172,
    "hi": 0.9376784355414076,
    "n_boot_ok": 1000
  },
  "vs_v6e2_snia_auc_delta": {
    "delta": 0.1262730622136924,
    "lo": 0.10120600039143361,
    "hi": 0.151328684934723
  },
  "vs_v6e2_significant_win": true,
  "claim": "fusion_v8 beats re-scored v6e2 on snia OvR AUC @ n_det=5 (CI95 excludes 0)",
  "note": "macro OvR AUC has no v6e2 counterpart (binary head); the snia OvR axis is the comparable one"
}
```

### Guards

- **lsst_spec_non_regression**: N/A (no LSST spec test slice locally, or v6e2 unavailable)
- **dp1_eclbin_rrlyrae_ef_non_regression**: PASS
- **seed_spread_lt_0.02**: N/A (no seed variants recorded in fusion_v8_train.json)
