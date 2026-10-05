"""scripts/eval_rubin_sets.py on synthetic gold / prediction tables.

Covers: AUC with ties vs sklearn; slices (n_det 3/5/10, latest = max n_det per object);
SN-vs-other AUC, Brier, medians, fraction of others above 0.5; best single input
(absent expert = 0.5), paired bootstrap difference and the fixed seed; the
broker-called-SN subset; borrowed positives for a hard-negative set; call-trust
reliability / ECE; truth / ids / label-quality handling; string ids (no float64
corruption); the CLI outputs.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ev = importlib.import_module("scripts.eval_rubin_sets")

BASE_ID = 170028485647532096          # ~1e17: not representable in float64 beyond 16 digits
STAMP, SNN, OTHER = "alerce__stamp_classifier_rubin_beta", "fink_lsst__snn", "alerce_lc"


def make_set(n_sn=30, n_other=60, offset=0, seed=0, n_dets=(1, 3, 5, 10, 14), model_noise=0.15,
             with_call_trust=False, classes=("snia", "nonIa_snlike")):
    """Gold + predictions for one set.  The model separates well, STAMP less, SNN is weak;
    OTHER is absent (avail 0) everywhere."""
    rng = np.random.default_rng(seed)
    g, p = [], []
    for i in range(n_sn + n_other):
        oid = str(BASE_ID + offset + i)
        is_sn = i < n_sn
        cls = classes[i % len(classes)] if is_sn else "other"
        for nd in n_dets:
            def score(noise):
                return float(np.clip((0.8 if is_sn else 0.2) + rng.normal(0, noise), 0.01, 0.99))
            m = score(model_noise)
            row = {"object_id": oid, "n_det": float(nd), "alert_jd": 2461000.0 + nd}
            for san, noise in ((STAMP, 0.3), (SNN, 0.6)):
                s = score(noise)
                row[f"proj__{san}__p_snia"] = 0.0 if san == STAMP else s * 0.5
                row[f"proj__{san}__p_nonIa_snlike"] = s if san == STAMP else s * 0.5
                row[f"proj__{san}__p_other"] = 1 - s
                row[f"avail__{san}"] = 1.0
            row[f"proj__{OTHER}__p_snia"] = 0.9 if is_sn else 0.1       # informative but unavailable
            row[f"proj__{OTHER}__p_nonIa_snlike"] = 0.0
            row[f"avail__{OTHER}"] = 0.0
            g.append(row)
            q = {"object_id": oid, "n_det": float(nd), "target_class": cls,
                 "label_quality": "spectroscopic" if is_sn else "context",
                 "p_snia": m * 0.5, "p_nonia": m * 0.5, "p_other": 1 - m}
            if with_call_trust:
                for san in (STAMP, SNN):
                    p_sn = row[f"proj__{san}__p_snia"] + row[f"proj__{san}__p_nonIa_snlike"]
                    q[f"sn_call__{san}"] = float(p_sn >= 0.5)
                    q[f"call_trust__{san}"] = 0.9 if p_sn >= 0.5 else 0.6
            p.append(q)
    return pd.DataFrame(g), pd.DataFrame(p)


def write(tmp_path, name, gold, pred):
    gp, pp = tmp_path / f"{name}_gold.parquet", tmp_path / f"{name}_pred.parquet"
    gold.to_parquet(gp, index=False)
    pred.to_parquet(pp, index=False)
    return gp, pp


def load(tmp_path, name="a", **kw):
    gold, pred = make_set(**kw)
    gp, pp = write(tmp_path, name, gold, pred)
    return ev.load_set(name, gp, pp, [], None), gold, pred


def test_auc_columns_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    S = np.column_stack([np.round(rng.random(200), 1), rng.random(200), np.full(200, 0.5)])
    got = ev.auc_columns(S, y)
    for j in range(3):
        assert got[j] == pytest.approx(roc_auc_score(y, S[:, j]))
    assert np.isnan(ev.auc_columns(S, np.zeros(200, int))).all()


def test_string_ids_survive_and_float_ids_refused(tmp_path):
    rows, gold, pred = load(tmp_path)
    assert rows["object_id"].dtype == object and rows["object_id"].nunique() == 90
    assert str(BASE_ID + 1) in set(rows["object_id"])               # neighbouring 1e17 ids stay distinct
    assert len({np.float64(BASE_ID + i) for i in range(90)}) < 90   # the float64 trap this guards against
    bad = pred.assign(object_id=pred["object_id"].astype(float))
    with pytest.raises(ValueError, match="float"):
        ev.string_ids(bad["object_id"])
    gp, pp = write(tmp_path, "bad", gold, bad)
    with pytest.raises(ValueError, match="float"):
        ev.load_set("bad", gp, pp, [], None)


def test_slices_and_basic_metrics(tmp_path):
    rows, _, _ = load(tmp_path)
    res = ev.evaluate({"a": rows}, slices=["3", "5", "10", "latest"], called=[STAMP, SNN], donors={},
                      n_boot=200, seed=7)["a"]["slices"]
    for sl in ("3", "5", "10", "latest"):
        m = res[sl]["all"]
        assert m["n"] == 90 and m["n_sn"] == 30
    # latest = n_det 14 for every object here
    latest = ev.take_slice(rows, "latest")
    assert (latest["n_det"] == 14).all() and len(latest) == 90
    m = res["5"]["all"]
    s5 = rows[rows["n_det"] == 5]
    assert m["auc"] == pytest.approx(roc_auc_score(s5["y"], s5["psn"]), abs=1e-3)
    assert m["brier"] == pytest.approx(float(np.mean((s5["psn"] - s5["y"]) ** 2)), abs=1e-3)
    assert m["median_psn_sn"] == pytest.approx(float(np.median(s5.loc[s5.y == 1, "psn"])), abs=1e-3)
    assert m["median_psn_other"] == pytest.approx(float(np.median(s5.loc[s5.y == 0, "psn"])), abs=1e-3)
    assert m["frac_other_psn_gt_half"] == pytest.approx(float((s5.loc[s5.y == 0, "psn"] > 0.5).mean()), abs=1e-3)
    lo, hi = m["auc_ci"]
    assert lo <= m["auc"] <= hi


def test_latest_is_max_n_det_per_object(tmp_path):
    gold, pred = make_set(n_sn=4, n_other=6, n_dets=(1, 2, 7))
    pred.loc[pred["n_det"] == 7, "n_det"] = 7.0
    pred = pred[~((pred["n_det"] == 7) & pred["object_id"].isin(pred["object_id"].unique()[:3]))]   # 3 objects stop at 2
    gp, pp = write(tmp_path, "l", gold, pred)
    rows = ev.load_set("l", gp, pp, [], None)
    latest = ev.take_slice(rows, "latest").set_index("object_id")["n_det"]
    assert len(latest) == 10 and (latest.iloc[:3] == 2).all() and (latest.iloc[3:] == 7).all()


def test_best_single_input_baseline_and_paired_diff(tmp_path):
    rows, _, _ = load(tmp_path, model_noise=0.45)
    kw = dict(slices=["5"], called=[STAMP], donors={}, n_boot=300, seed=3)
    m = ev.evaluate({"a": rows}, **kw)["a"]["slices"]["5"]["all"]
    base = m["baseline"]
    assert "favours the baseline" in base["note"]
    s = rows[rows["n_det"] == 5]
    for san in (STAMP, SNN):
        want = roc_auc_score(s["y"], s[f"e__{san}"].fillna(0.5))
        assert base["per_expert"][san]["auc"] == pytest.approx(want, abs=1e-3)
        assert base["per_expert"][san]["n_available"] == 90
    # the unavailable (informative) expert is scored as 0.5 everywhere: AUC 0.5, nothing available
    assert base["per_expert"][OTHER] == {"auc": 0.5, "n_available": 0}
    assert base["best"] == STAMP and base["best_auc"] == pytest.approx(
        max(v["auc"] for v in base["per_expert"].values()))
    assert base["diff_model_minus_best"] == pytest.approx(m["auc"] - base["best_auc"], abs=2e-3)
    assert base["diff_ci"][0] <= base["diff_model_minus_best"] <= base["diff_ci"][1]
    # fixed seed: identical rerun
    assert ev.evaluate({"a": rows}, **kw) == ev.evaluate({"a": rows}, **kw)
    other_seed = ev.evaluate({"a": rows}, **{**kw, "seed": 4})["a"]["slices"]["5"]["all"]
    assert other_seed["auc_ci"] != m["auc_ci"] and other_seed["auc"] == m["auc"]


def test_broker_called_sn_subset(tmp_path):
    rows, _, _ = load(tmp_path)
    called = ev.called_sn(rows, [STAMP, SNN])
    want = ((rows[f"e__{STAMP}"] >= 0.5) | (rows[f"e__{SNN}"] >= 0.5)).to_numpy()
    assert (called == want).all() and 0 < called.sum() < len(rows)
    res = ev.evaluate({"a": rows}, slices=["latest"], called=["alerce/stamp_classifier_rubin_beta"], donors={},
                      n_boot=100, seed=1)["a"]["slices"]["latest"]
    s = ev.take_slice(rows, "latest")
    sub = s[s[f"e__{STAMP}"] >= 0.5]
    assert res["broker_called_sn"]["n"] == len(sub)               # slash form of the key accepted
    assert res["broker_called_sn"]["n_sn"] == int(sub["y"].sum())
    # an expert absent from the set calls nothing
    none = ev.evaluate({"a": rows}, slices=["latest"], called=["no/such_expert"], donors={}, n_boot=50,
                       seed=1)["a"]["slices"]["latest"]["broker_called_sn"]
    assert none["n"] == 0


def test_positives_from_for_hard_negative_set(tmp_path):
    bench, _, _ = load(tmp_path, "bench", n_sn=30, n_other=40, seed=1)
    hard, _, _ = load(tmp_path, "hard", n_sn=0, n_other=25, offset=10_000, seed=2)
    assert hard["y"].sum() == 0
    res = ev.evaluate({"bench": bench, "hard": hard}, slices=["3", "latest"], called=[STAMP], donors={"hard": "bench"},
                      n_boot=200, seed=5)
    m = res["hard"]["slices"]["3"]["all"]
    assert m["n_sn"] == 30 and m["n"] == 30 + 25 and m["positives_from"] == "bench" and m["n_own_sn_dropped"] == 0
    s_b, s_h = ev.take_slice(bench, "3"), ev.take_slice(hard, "3")
    want = pd.concat([s_b[s_b.y == 1], s_h[s_h.y == 0]])
    assert m["auc"] == pytest.approx(roc_auc_score(want["y"], want["psn"]), abs=1e-3)
    assert res["hard"]["positives_from"] == "bench" and res["bench"]["positives_from"] is None
    assert res["bench"]["slices"]["3"]["all"]["n"] == 70
    # a donor's own SNe in the target set are dropped, not scored
    mixed, _, _ = load(tmp_path, "mixed", n_sn=5, n_other=25, offset=20_000, seed=3)
    r2 = ev.evaluate({"bench": bench, "mixed": mixed}, slices=["3"], called=[STAMP], donors={"mixed": "bench"},
                     n_boot=50, seed=5)["mixed"]["slices"]["3"]["all"]
    assert r2["n_own_sn_dropped"] == 5 and r2["n_sn"] == 30 and r2["n"] == 55


def test_call_trust_reliability_and_ece(tmp_path):
    rows, _, _ = load(tmp_path, with_call_trust=True, model_noise=0.1)
    res = ev.evaluate({"a": rows}, slices=["latest"], called=[STAMP], donors={}, n_boot=20, seed=1)["a"]
    rel = res["call_trust_reliability"][STAMP]["latest"]
    s = ev.take_slice(rows, "latest")
    conf = s[f"call_trust__{STAMP}"].to_numpy(float)
    correct = (s[f"sn_call__{STAMP}"].to_numpy(float) == s["y"].to_numpy(float)).astype(float)
    ece = sum((conf == c).mean() * abs(correct[conf == c].mean() - c) for c in (0.6, 0.9))   # two populated bins
    assert rel["n"] == len(s) and rel["ece"] == pytest.approx(ece, abs=1e-3)
    assert sum(b["n"] for b in rel["bins"]) == rel["n"] and len(rel["bins"]) == 2
    assert {tuple(b["bin"]) for b in rel["bins"]} == {(0.6, 0.7), (0.9, 1.0)}
    # perfectly calibrated toy: confidence 0.8 with 80% right -> ECE 0
    r = ev.reliability(np.full(10, 0.8), np.array([1] * 8 + [0] * 2))
    assert r["ece"] == 0.0 and r["bins"][0]["bin"] == [0.8, 0.9]
    # confidence exactly 1.0 lands in the last bin
    assert ev.reliability(np.array([1.0]), np.array([1.0]))["bins"][0]["bin"] == [0.9, 1.0]
    # no call_trust columns -> no table
    plain, _, _ = load(tmp_path, "plain")
    assert ev.evaluate({"p": plain}, slices=["latest"], called=[STAMP], donors={}, n_boot=10, seed=1)["p"][
        "call_trust_reliability"] == {}


def test_truth_ids_and_quality_filters(tmp_path):
    gold, pred = make_set(n_sn=6, n_other=10, n_dets=(3,))
    gp, pp = write(tmp_path, "t", gold, pred)
    ids = sorted(pred["object_id"].unique())
    # truth flips two SNe to "other" and marks one object "weak"
    truth = pd.DataFrame({"object_id": ids, "final_class_ternary": ["snia"] * 6 + ["other"] * 10,
                          "label_quality": ["spectroscopic"] * 15 + ["weak"]})
    truth.loc[:1, "final_class_ternary"] = "other"
    tp = tmp_path / "truth.parquet"
    truth.to_parquet(tp, index=False)
    r = ev.load_set("t", gp, pp, [tp], {"spectroscopic", "context"})
    assert len(r) == 15 and r["y"].sum() == 4 and ids[-1] not in set(r["object_id"])
    assert len(ev.load_set("t", gp, pp, [tp], None)) == 16
    man = tmp_path / "ids.json"
    man.write_text(json.dumps({"test_ids": ids[:8]}))
    assert set(ev.load_set("t", gp, pp, [man], None)["object_id"]) == set(ids[:8])
    csvp = tmp_path / "ids.csv"
    pd.DataFrame({"object_id": ids[2:5]}).to_csv(csvp, index=False)
    assert set(ev.load_set("t", gp, pp, [csvp], None)["object_id"]) == set(ids[2:5])
    for order in ([tp, man], [man, tp]):                      # truth labels + id restriction, either order
        both = ev.load_set("t", gp, pp, order, None)
        assert set(both["object_id"]) == set(ids[:8]) and both["y"].sum() == 4   # ids[0:2] flipped to other


def test_cli_writes_json_and_markdown(tmp_path, capsys):
    bg, bp = write(tmp_path, "bench", *make_set(n_sn=20, n_other=30, seed=1, with_call_trust=True))
    hg, hp = write(tmp_path, "hard", *make_set(n_sn=0, n_other=20, offset=5000, seed=2))
    out = tmp_path / "out"
    rc = ev.main(["--set", f"bench={bg},{bp}", "--set", f"hard={hg},{hp}", "--positives-from", "bench",
                  "--slices", "3,latest", "--n-boot", "50", "--out", str(out)])
    assert rc == 0
    js = json.loads((out / "eval_rubin_sets.json").read_text())
    assert set(js["sets"]) == {"bench", "hard"} and js["meta"]["seed"] == 42
    assert js["sets"]["hard"]["positives_from"] == "bench"
    assert js["sets"]["hard"]["slices"]["latest"]["all"]["n_sn"] == 20
    md = (out / "eval_rubin_sets.md").read_text()
    assert "favours the baseline" in md and "SNe borrowed from `bench`" in md and "Call-trust reliability" in md
    # explicit path form and error handling
    assert ev.main(["--set", f"bench={bg},{bp}", "--slices", "3", "--n-boot", "10", "--out", str(tmp_path / "x.json")]) == 0
    assert (tmp_path / "x.json").exists() and (tmp_path / "x.md").exists()
    with pytest.raises(SystemExit):
        ev.main(["--set", f"bench={bg},{bp}", "--positives-from", "nope", "--out", str(out)])
    with pytest.raises(SystemExit):
        ev.main(["--set", "bench", "--out", str(out)])
