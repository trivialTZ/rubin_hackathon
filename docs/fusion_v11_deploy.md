# fusion_v11 — SCC Deploy State + Run Instructions

Deploy-prep agent, 2026-07-05 (~01:15 EDT SCC time).
Target: `/project/pi-brout/rubin_hackathon` (the ACTIVE clone — never
`/projectnb/.../rubin_hackathon_tmp`, which is stale and has no `.env`).

## Status: SYNCED, NOT SUBMITTED

The v11 chain was **not** submitted. The workflow green flag at deploy time was
**false**: integration green=true, but local end-to-end smoke ok=false with
**4 serious findings**; the fixer phase subsequently reported 7 fixes applied with
the full test suite green. The overall flag was not re-raised, so per deploy policy
the chain stays unsubmitted until a human confirms the smoke findings are resolved.

Before submitting, confirm:

1. The 4 smoke findings are fixed or accepted (see the fixer's changes — all v11
   code now on SCC is the *post-fixer* working tree, `tests` green locally and on SCC).
2. Ideally re-run the local smoke (or accept SCC-side evidence below).

## What is on SCC now

- **Safety backup** (same ritual as v10): `~/pre_v11_sync_backup_20260704.tgz`
  on SCC — tar of the 13 pre-existing files the sync overwrote (68 KB).
  Rollback: `cd /project/pi-brout/rubin_hackathon && tar xzf ~/pre_v11_sync_backup_20260704.tgz`.
- **47-entry rsync** (checksum-verified identical after transfer, ~41 MB):
  - 13 modified files: `scripts/{build_snapshots_fusion,download_tns_bulk,train_seq_classifier,train_seq_encoder}.py`,
    `src/debass_meta/access/tns.py`, `src/debass_meta/experts/local/{__init__,seq_v9}.py`,
    `src/debass_meta/features/{detection,lightcurve,sequence_dataset}.py`,
    `src/debass_meta/models/{early_meta,multiclass_followup}.py`, `src/debass_meta/projectors/base.py`
  - New code: `src/debass_meta/models/{anchor_blend,hierarchical_followup}.py`,
    `scripts/{assert_locked_gold_identity,build_truth_lsst_live,fetch_elasticc2,harvest_ztf_lsst_associations,rederive_spec_truth,score_fusion_v11,train_fusion_v11}.py`
  - 8 new test files (`tests/test_{anchor_blend,gold_positive_only,hierarchical_followup,lightcurve_neg,score_v11,seq_v11,train_v11_smoke,truth_lsst_live}.py`)
  - 5 job scripts (`jobs/run_fusion_v11_{build,pretrain,expert}.sh`,
    `jobs/submit_fusion_v11_chain.sh`, `jobs/refresh_lsst_live.sh`)
  - 5 docs (`docs/fusion_v11_{spec,design,review_data,review_integration,review_ml}.md`)
  - Data the chain needs:
    - `data/truth/tns_public.parquet` (16 MB TNS bulk — build step 0 skips its fetch)
    - `data/crossmatch/lsst_to_ztf.csv` (association CSV; was MISSING on SCC —
      v10 ran with association guards degraded, v11 now gets them for real)
    - `data/gold/lsst_live_locked_test.json` (frozen 197-id LSST-live benchmark manifest;
      enables G6-at-build quarantine + the locked benchmark re-score)
    - `data/truth/object_truth_20260704_cleaned.parquet` (cleaned benchmark truth)
    - `data/live_eval_20260704/{cohort/labels.csv,lightcurves/ (379),silver/}` (benchmark
      cohort — without it the expert job's step-7 re-score self-skips with a WARN)
- **Deliberately NOT synced**: `data/truth/object_truth_v11.parquet` and
  `label_delta_v11.csv` (the build job must rederive them on SCC — its inputs
  `object_truth.parquet`, `ztf_bts.parquet`, `split_fusion_v10.json` are byte-identical
  on both sides, so the SCC rederive reproduces the local table with clean provenance);
  local gold/split v11 artifacts (idempotent build steps would wrongly skip);
  smoke/`.local` variants; `scripts/_verify_v11_probe*.py` scratch;
  the 379 `fixtures/raw/babamul` fixtures (no v11 test reads fixtures); SCC `.env`
  (never overwritten).

## SCC-side verification already done (2026-07-05)

- Import smoke in `.venv` (python 3.10): full `debass_meta` package incl. drifted-but-
  untouched modules (`projectors/{ampel,pittgoogle}`, `experts/local/{alerce_lc,parsnip,supernnova}`)
  — OK. `ALL_EXPERT_KEYS` = 30; `seq_v11` registered in `ALL_LOCAL_EXPERTS`.
- `--help` (module import + argparse) green for all 11 chain entry points:
  rederive_spec_truth, build_snapshots_fusion, train_fusion_v11, score_fusion_v11,
  assert_locked_gold_identity, train_seq_encoder, train_seq_classifier,
  build_truth_lsst_live, harvest_ztf_lsst_associations, download_tns_bulk, fetch_elasticc2.
- All 8 v11 test files pass **on SCC**: 110 passed (56 + 54), ~21 s total.
- Chain-external scripts the jobs call (`eval_fusion_v8.py`, `score_fusion_v8.py`,
  `local_infer.py`, `build_helpfulness_fusion.py`) verified checksum-identical to local.
- v10 dependencies present on SCC: `data/gold/split_fusion_v10.json` (rederive
  `--locked-split`), `object_epoch_snapshots_fusion_v10.parquet` (G5b reference),
  `models/{trust,followup,conformal}_fusion_v10` (benchmark v10 re-score).
- `qstat`: no running/queued jobs for tztang, no debass/fusion jobs from any user
  (checked 2026-07-05 01:14 EDT). `/project` quota: 45 G / 220 G used.

## Exactly what to run (once the green flag is cleared)

```bash
ssh scc
cd /project/pi-brout/rubin_hackathon

# 0. re-check nobody else is running a fusion job (user sometimes self-submits):
qstat -u "*" | grep -iE "debass|fusion"

# 1. submit the 3-job chain (build → GPU pretrain → expert, -hold_jid wired):
bash jobs/submit_fusion_v11_chain.sh
#    (optional: bash jobs/submit_fusion_v11_chain.sh <HOLD_JOB_ID> to hold the
#     build on an already-running job)
```

The chain (per `jobs/submit_fusion_v11_chain.sh`):

1. `debass_v11_build` (CPU, 16-core, 6 h): TNS bulk (skips — parquet synced) →
   rederive `object_truth_v11.parquet` → FULL v11 gold (positives-only epochs +
   neg features) + FINAL `split_fusion_v11.json` → G5b locked-ZTF-gold value identity.
2. `debass_v11_pretrain` (GPU `gpu_c=8.0` — never bare `-l gpus=1`, P100 crashes
   torch≥2.x; 12 h): seq_v11 SSL encoder + 5-fold OOF classifier, `--seq-schema v11`.
3. `debass_v11_expert` (CPU, 16-core, 10 h): seq_v11 OOF silver → gold rebuild with
   seq_v11 → Stage A/B hierarchical heads → anchored blend → post-blend conformal →
   guards/gates → score → eval (G1) → locked LSST-live benchmark re-score
   (v11 + v10 on the same cleaned truth; G4).

Watch: `tail -f logs/fusion_v11_{build,pretrain,expert}.qsub.{out,err}`;
results land in `reports/fusion_v11/` + `reports/lsst_live_bench/` +
`reports/metrics/fusion_v11_{build,train}.json`.

Knobs: `FUSION_V11_FORCE=1` rebuilds everything (jobs are otherwise idempotent —
safe to resubmit after a partial failure); `FUSION_V11_ELASTICC2=1` enables the
optional ELAsTiCC2 pretrain corpus (~3–4 GB fetch) for the pretrain job.
The expert job passes `--acknowledge-g2-unevaluable` (LSST spec-Ia pool < 10);
drop that flag in `jobs/run_fusion_v11_expert.sh` once the pool clears n≥10.
