# Bug: metaDEBASS reads Fink LSST CATS classes with the wrong codes

**Found:** 2026-09-24, while building the TNS × EDP2 Explorer classifier pages.
**Status:** code fixed 2026-09-24 (projector, trajectory mirror, comments, tests; 490 tests pass). The deployed v11
models still carry the old projection until the gold is rebuilt and the stack retrained.
**Affects:** `fink_lsst/cats` in fusion_v11 (and every earlier LSST build that used the same projector).

## Summary

`src/debass_meta/projectors/fink_lsst.py` assumes CATS returns these codes:

| code | assumed class | projected as |
|---|---|---|
| 11 | SN-like | SN (score split 50/50 between Ia and non-Ia) |
| 21 | Long | non-Ia SN, P = score |
| 31 | Fast | non-Ia SN, P = score |
| 41 | Periodic | not SN, P = 1 |
| 51 | Non-periodic | not SN, P = 1 |

Fink LSST actually returns the ELAsTiCC broad-class codes, which are different:

| code | ELAsTiCC broad class | members | projected today | should be |
|---|---|---|---|---|
| 11 | SN-like | SN Ia, Ib/c, II, Iax, 91bg | SN (50/50) | SN (50/50), unchanged |
| 12 | Fast | kilonova, M-dwarf flare, dwarf nova, microlensing | not SN, P = 1 | not SN |
| 13 | Long | SLSN, TDE, ILOT, CART, PISN | not SN, P = 1 | not SN, or SN (see below) |
| 21 | Periodic | Cepheid, RR Lyrae, δ Scuti, eclipsing binary, LPV | **non-Ia SN, P = score** | **not SN** |
| 22 | Non-periodic | AGN | not SN, P = 1 | not SN |

The serious error is code 21. A confident CATS call that the source is a **periodic variable star** becomes a
confident **non-Ia supernova** vote. Codes 31, 41 and 51 never occur in Fink LSST data.

A second, smaller problem: every class other than 11 and 21 is projected as P(not SN) = 1.0 whatever the
CATS score is, so a low-confidence non-SN call counts as fully certain.

## Evidence

**1. Codes in the data.** Counts of `clf_cats_class` events in silver `broker_events`:

| data | events | objects | 11 | 12 | 13 | 21 | 22 |
|---|---|---|---|---|---|---|---|
| training silver (`data/silver`) | 853,760 | 3,984 | 634,447 (74.3%) | 6,795 (0.8%) | 37,929 (4.4%) | 57,366 (6.7%) | 117,223 (13.7%) |
| live benchmark cohort (`data/live_eval_20260704`) | 20,779 | 340 | 16,309 | 157 | 992 | 2,261 | 1,060 |
| TNS × EDP2 run (`data/tnsx_eval_20260924`) | 20,215 | 466 | 16,231 | 309 | 410 | 2,410 | 855 |

Only 11, 12, 13, 21 and 22 appear, and never 31, 41 or 51. About a quarter of the CATS events the model trained on
fall in codes the projector gets wrong.

**2. The taxonomy.** CATS predicts the five ELAsTiCC broad classes: SN-like, Fast, Long, Periodic, Non-periodic
(Fraga et al. 2024, A&A 692, A208). The ELAsTiCC class IDs are hierarchical: 1 = Non-Recurring (11 SN-like,
12 Fast, 13 Long) and 2 = Recurring (21 Periodic, 22 Non-periodic). For example, SN Ia is class 111 under 11 and
1 ([ELAsTiCC challenge page](https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/)).

**3. The code** (`src/debass_meta/projectors/fink_lsst.py`):

```python
_CATS_SNIA_LIKE = {11}    # SN-like → could be Ia or non-Ia
_CATS_NONIA = {21, 31}    # Long/Fast transients → non-Ia SN-like
_CATS_OTHER = {41, 51}    # Periodic/Non-periodic → other
...
    elif cats_class in _CATS_NONIA:      # 21 = Periodic lands here
        p_snia = 0.0
        p_nonia = cats_score
        p_other = 1.0 - cats_score
    else:                                # 12, 13, 22 land here; score ignored
        p_snia = 0.0
        p_nonia = 0.0
        p_other = 1.0
```

The module docstring lists the same wrong codes and credits them to "Moller et al. 2024". The CATS paper is
Fraga et al. 2024.

## Where it propagates

- **Gold features:** `proj__fink_lsst__cats__{p_snia,p_nonIa_snlike,p_other,top1_prob,margin,entropy}`.
- **The v11 anchor** (`src/debass_meta/models/anchor_blend.py`, `compute_anchor`): `fink_lsst/cats` is not in
  `ANCHOR_EXCLUDED`, and with no `q__fink_lsst__cats` column its weight defaults to 1. The anchor's P(SN) is the
  trust-weighted mean of every firing expert's `p_snia + p_nonIa_snlike`, so each CATS "Periodic" call adds a vote
  of P(SN) = score.
  - In the 2026-09-24 TNS × EDP2 run, every LSST row fell back to α = 0 (`anchor_default`), so the P(supernova)
    metaDEBASS reports for Rubin alerts is exactly this anchor.
  - The 2026-07-07 live benchmark rows used α = 0.25.
- **Trust:** `fink_lsst/cats` is in `SN_FILTER_EXPERTS` (`models/expert_trust.py`), so its trust head is trained on
  `is_sn` using inputs from this projection.
- **Trajectory features:** `features/trajectory.py::_project_fink_lsst_cats` mirrors the projector, and its
  docstring repeats the wrong mapping. It uses p_snia only, and that is 0 for code 21 either way, so the numbers
  there do not change.
- **Comments and tests that repeat the wrong codes:**
  - `anchor_blend.py` line 69 says "classes 21/31 carry a negative Ia claim".
  - `tests/test_trajectory_asof.py::test_parity_fink_lsst_cats` uses code "41", which never occurs.

## Measured impact

Both measurements recompute the anchor with the corrected projection and nothing else changed. The
analysis script is `data/tnsx_eval_20260924/tools/cats_impact_live_bench.py`.

**TNS × EDP2 run** (168 Rubin alert objects scored by metaDEBASS; the recomputation reproduces the saved output
exactly):

- 55 of the 1,506 scored rows with CATS output have code 21.
- On those rows, the median deployed P(supernova) is 0.75; with the fix it is 0.55.
- The latest P(supernova) of 2 of the 168 objects moves by more than 0.1, and 1 crosses 0.5.

**Live LSST benchmark, 2026-07-07** (locked test set, 209 objects). This recomputation is approximate: the blended
P(SN Ia) differs from the saved predictions by up to 0.19, probably because some trust columns are not in the
saved file.

| SN-vs-other AUC | as deployed | fixed (12/13/21/22 → not SN) | fixed, Long counted as SN |
|---|---|---|---|
| n_det = 3 | 0.888 [0.803, 0.957] | 0.897 [0.813, 0.966] | 0.897 [0.813, 0.966] |
| n_det = 5 | 0.886 [0.800, 0.967] | 0.886 [0.800, 0.967] | 0.886 [0.800, 0.967] |
| latest | 0.905 [0.845, 0.953] | 0.912 [0.852, 0.958] | 0.911 [0.851, 0.957] |

- The 6 test objects whose latest CATS class is Periodic are all TNS "other". Their median P(supernova) drops
  from 0.25 to 0.16 with the fix.
- The benchmark effect is small because code 21 is only 2% of CATS rows in that test set (53 of 2,606). It is
  6.7% of the training events and 12% of this cohort's events.
- Not measured: the effect on the trained trust heads and follow-up heads, which saw the wrong projection
  during training.

## Fix (implemented 2026-09-24)

`clf_cats_score` is the probability of the predicted class: it never falls below ~0.24 in silver, just above the 1/5
floor of a five-class softmax. So for a non-SN call the remainder is spread evenly over the four other classes, and
the SN-like share is (1 − score) / 4. This replaces the (1 − score) proposed in the first version of this note.

```python
CATS_SN_LIKE = 11
CATS_NON_SN = frozenset({12, 13, 21, 22})
CATS_N_CLASSES = 5

def cats_ternary(cats_class, cats_score):
    if cats_class == CATS_SN_LIKE:
        return 0.5 * cats_score, 0.5 * cats_score, 1.0 - cats_score
    if cats_class in CATS_NON_SN:
        p_sn = (1.0 - cats_score) / (CATS_N_CLASSES - 1)
        return 0.5 * p_sn, 0.5 * p_sn, 1.0 - p_sn
    return None          # unknown code: no projection
```

- `src/debass_meta/projectors/fink_lsst.py`: `cats_ternary`, corrected docstring and citation.
- `src/debass_meta/features/trajectory.py::_project_fink_lsst_cats`: mirrors it (p_snia = (1 − score) / 8 for non-SN
  classes; unknown codes dropped).
- `src/debass_meta/models/anchor_blend.py`: comment corrected; `fink_lsst/cats` stays out of `IA_CAPABLE`.
- `tests/test_trajectory_asof.py`: parity over 11/12/13/21/22, unknown-code handling, and `test_cats_ternary_mapping`.
- **Long (13)** is treated as not SN-like, following the CATS taxonomy. `raw_cats_class` remains a feature, so the
  trust and follow-up heads can still learn how Long relates to our labels (SLSNe count as non-Ia SNe in the truth).

Still to do: rebuild the gold and retrain (the next fusion build), rerun the live benchmark, and re-export the
TNS × EDP2 Explorer.

## Reproduce

```bash
cd ~/Documents/GitHub/rubin_hackathon && source ~/.venvs/debass_py313/bin/activate
python - <<'PY'
import pandas as pd
s = pd.read_parquet("data/silver/broker_events.parquet", columns=["expert_key", "field", "class_name"])
s = s[(s.expert_key == "fink_lsst/cats") & (s.field == "clf_cats_class")]
print(s.class_name.value_counts())          # 11, 22, 21, 13, 12: no 31, 41, 51
PY
python data/tnsx_eval_20260924/tools/cats_impact_live_bench.py   # benchmark AUC, deployed vs fixed
```
