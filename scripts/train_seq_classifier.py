#!/usr/bin/env python
"""Fine-tune the v9 SSL encoder into the ``seq_v9`` 4-way classifier (v10).

Labels come from the fusion snapshot (object-level target_class /
label_quality — already demoted/cleaned), refined to the 4-way scheme
(snia / snii / other_sn / non_sn) by joining the truth table's raw type
strings (tns_type / bts_type / final_class_raw), and optionally extended with
weak LSST labels from data/lsst_candidates.csv (ALeRCE Rubin stamp classes;
label_source='alerce_self_label').

v10 protocol (vs v9c):
  * GROUPED WEAK-LABEL LOSS — weak stamp-SN rows AND every nonIa row without
    a parseable fine subtype supervise ONLY the grouped mass P(sn-like) vs
    P(non_sn), never which SN subclass, removing the structural
    "LSST ⇒ never Ia" bias.  "Definitely not Ia" supervision
    (P(snii)+P(other_sn)) is earned only by a raw type string that parses to
    a non-Ia fine class — quality tier alone never grants it (generic TNS
    "SN", legacy BTS '-', and consensus claims cannot exclude Ia).
  * HONEST EVAL — an inner 10% validation split (grouped by object, carved
    out of TRAIN) drives early stopping; cal is used ONLY for temperature
    calibration; headline standalone OvR AUCs are computed ONCE on the locked
    TEST split (behind ``--final-eval``, passed for the deployed artifact
    only), restricted to spectroscopic+tns labels, sliced per survey and per
    n_det bucket (3/5/10).  The cal diagnostics are kept (per-survey) but
    labelled as model-selection metrics — variant comparisons use those.
  * K-FOLD OOF ARTIFACTS — --oof-folds K partitions train object_ids into K
    deterministic seeded folds, trains K fold classifiers (each on K-1 folds,
    same early-stopping protocol) saved under fold_{k}/ plus fold_map.json,
    alongside the full-data root artifact.  experts/local/seq_v9.py routes
    each mapped object to the model that never saw it, so Stage A/B consume
    out-of-fold (not memorised) train-row projections.

Artifact: models/seq_classifier_v10/{classifier.pt, norm_stats.json,
config.json, train_log.json, fold_map.json, fold_{k}/…} — consumed by
experts/local/seq_v9.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.features.sequence_dataset import (  # noqa: E402
    NormStats,
    load_object_sequence,
    sequence_survey,
)
from debass_meta.models.seq_classifier import (  # noqa: E402
    CLASS_TO_TERNARY,
    CLASSES,
    TERNARY_CLASSES,
    SeqClassifier,
    SeqClassifierArtifact,
    classification_loss,
    fine_class_from_raw,
    fit_temperature,
    fold_probs_to_ternary,
    supervision_mask,
)
from debass_meta.models.seq_encoder import load_encoder, resolve_device  # noqa: E402

QUALITY_W = {"spectroscopic": 1.0, "tns_untyped": 0.6, "context": 0.15, "weak": 0.1}
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
TERN_IDX = {c: i for i, c in enumerate(TERNARY_CLASSES)}
# Label tiers admitted into the honest (headline) test diagnostics.
HEADLINE_QUALITIES = {"spectroscopic", "tns_untyped"}
DIAG_N_DETS = (3, 5, 10)


@dataclass
class RowLabel:
    ternary: str            # snia | nonIa_snlike | other
    quality: str
    fine: str | None = None      # 4-way class when the subtype is known
    grouped_sn: bool = False     # weak stamp-SN: supervise sn-like mass only


@dataclass
class SeqSet:
    """A collected split: sequences + 4-way supervision structure."""

    oids: list[str] = field(default_factory=list)
    seqs: list = field(default_factory=list)
    y3: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    y4: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # -1 = unknown
    masks: np.ndarray = field(default_factory=lambda: np.zeros((0, len(CLASSES)), dtype=bool))
    quals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    is_lsst: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def __len__(self) -> int:
        return len(self.oids)


def load_labels_from_snapshot(snap_path: Path) -> dict[str, RowLabel]:
    """object_id → RowLabel(ternary, quality), one row per object."""
    import pandas as pd

    cols = ["object_id", "target_class", "label_quality"]
    df = pd.read_parquet(snap_path, columns=cols).drop_duplicates("object_id")
    out: dict[str, RowLabel] = {}
    for oid, cls, qual in df.itertuples(index=False):
        if cls in TERN_IDX:
            out[str(oid)] = RowLabel(ternary=str(cls), quality=str(qual))
    return out


def load_fine_labels(truth_path: Path) -> dict[str, str]:
    """object_id → 4-way fine class from the truth table's raw type strings.

    Priority follows the truth tiers: tns_type (TNS spectroscopic), then
    bts_type (BTS spectroscopic), then final_class_raw.
    """
    import pandas as pd

    if not truth_path.exists():
        return {}
    cols = ["object_id", "tns_type", "bts_type", "final_class_raw"]
    df = pd.read_parquet(truth_path)
    cols = [c for c in cols if c in df.columns]
    out: dict[str, str] = {}
    for row in df[cols].itertuples(index=False):
        oid = str(row.object_id)
        for col in ("tns_type", "bts_type", "final_class_raw"):
            fine = fine_class_from_raw(getattr(row, col, None))
            if fine is not None:
                out[oid] = fine
                break
    return out


def attach_fine_labels(labels: dict[str, RowLabel], fine_map: dict[str, str]) -> tuple[int, int]:
    """Refine ternary labels in place; flag grouped-SN supervision rows.

    A fine label is attached only when it folds to the row's ternary class
    (contradictions keep the ternary group mask).  ANY nonIa row without a
    parseable subtype gets grouped sn-like supervision — regardless of
    quality tier: the ternary nonIa bucket admits raw types that cannot
    exclude Ia (generic TNS "SN", the legacy BTS ``'-'`` fabrication,
    consensus-tier claims), so "definitely not Ia" supervision is earned
    ONLY by a raw type string that parses to a non-Ia fine class (the v10
    fix for the structural anti-Ia bias).
    Returns (n_fine, n_grouped_sn).
    """
    n_fine = n_grouped = 0
    for oid, lab in labels.items():
        fine = fine_map.get(oid)
        if fine is not None and CLASS_TO_TERNARY.get(fine) == lab.ternary:
            lab.fine = fine
            n_fine += 1
        if lab.ternary == "nonIa_snlike" and lab.fine is None:
            lab.grouped_sn = True
        if lab.grouped_sn:
            n_grouped += 1
    return n_fine, n_grouped


def load_lsst_candidate_labels(path: Path) -> dict[str, RowLabel]:
    """Weak LSST labels from ALeRCE Rubin stamp discovery classes.

    SN → grouped sn-like supervision (the stamp cannot distinguish Ia — the
    row is penalised on P(snia)+P(snii)+P(other_sn) vs P(non_sn) only, never
    forced to a non-Ia subclass); AGN/VS/asteroid/bogus/satellite → non_sn.
    Quality 'weak'.
    """
    if not path.exists():
        return {}
    out: dict[str, RowLabel] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            stamp = str(row.get("alerce_stamp_class") or "").strip().upper()
            if not stamp:
                continue
            if stamp == "SN":
                out[str(row["object_id"])] = RowLabel(
                    ternary="nonIa_snlike", quality="weak", grouped_sn=True)
            else:
                out[str(row["object_id"])] = RowLabel(
                    ternary="other", quality="weak", fine="non_sn")
    return out


def load_split(split_path: Path) -> tuple[set[str], set[str], set[str], set[str]]:
    """(train, cal, test, quarantined) object IDs from the split manifest.

    ``quarantined_ids`` (v10) are association counterparts of locked-test
    objects — their photometry duplicates locked-test photometry, so they
    must never be trained on (not even via the weak-LSST ``allow_extra``
    path).
    """
    payload = json.loads(split_path.read_text())
    if "split" in payload and isinstance(payload["split"], dict):
        payload = payload["split"]
    g = lambda *keys: {str(o) for k in keys for o in payload.get(k, []) or []}
    return (g("train_ids", "train"), g("cal_ids", "cal"),
            g("test_ids", "test"), g("quarantined_ids"))


def split_inner_val(object_ids: list[str], frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic seeded object-grouped (fit, val) partition of TRAIN ids."""
    ids = sorted({str(o) for o in object_ids})
    perm = np.random.default_rng(seed).permutation(len(ids))
    shuffled = [ids[i] for i in perm]
    n_val = max(1, int(round(frac * len(ids))))
    return shuffled[n_val:], shuffled[:n_val]


def assign_folds(object_ids: list[str], n_folds: int, seed: int) -> dict[str, int]:
    """Deterministic seeded object-grouped fold assignment (object_id → k)."""
    ids = sorted({str(o) for o in object_ids})
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {ids[i]: int(pos % n_folds) for pos, i in enumerate(perm)}


def pad(seqs, idxs, stats: NormStats, device):
    chunk = [seqs[i] for i in idxs]
    lengths = [len(c) for c, _ in chunk]
    L = max(lengths)
    cont = np.zeros((len(chunk), L, chunk[0][0].shape[1]), dtype=np.float32)
    bands = np.zeros((len(chunk), L), dtype=np.int64)
    for j, (c, b) in enumerate(chunk):
        cont[j, : len(c)] = stats.apply(c)
        bands[j, : len(b)] = b
    return (torch.from_numpy(cont).to(device), torch.from_numpy(bands).to(device),
            torch.tensor(lengths, dtype=torch.long, device=device))


def collect(labels: dict[str, RowLabel], ids: set[str], *, lc_dir: Path, max_len: int,
            all_split_ids: set[str] | None = None, allow_extra: bool = False,
            quality_allow: set[str] | None = None, limit: int | None = None,
            surveys: str = "both") -> SeqSet:
    """Load sequences + supervision structure for one split.

    ``allow_extra`` admits labelled objects absent from every split (the weak
    LSST cohort) — train only.  ``quality_allow`` restricts label tiers (used
    for the honest test diagnostics).  ``surveys`` filters by the loaded
    sequence's survey (staged ZTF→LSST recipes; applied to TRAIN only).
    """
    pool = [o for o in labels
            if (o in ids) or (allow_extra and all_split_ids is not None and o not in all_split_ids)]
    out = SeqSet()
    y3, y4, masks, quals, is_lsst = [], [], [], [], []
    for oid in sorted(pool):
        if limit and len(out.oids) >= limit:
            break
        lab = labels[oid]
        if quality_allow is not None and lab.quality not in quality_allow:
            continue
        loaded = load_object_sequence(lc_dir, oid, max_len=max_len)
        if loaded is None or len(loaded[0]) < 1:
            continue
        if surveys != "both" and sequence_survey(loaded[0]) != surveys:
            continue
        mask = supervision_mask(lab.ternary, lab.fine, lab.grouped_sn)
        if lab.grouped_sn or (lab.fine is None and lab.ternary == "nonIa_snlike"):
            y4_val = -1
        elif lab.fine is not None:
            y4_val = CLS_IDX[lab.fine]
        elif lab.ternary == "snia":
            y4_val = CLS_IDX["snia"]
        else:  # ternary "other"
            y4_val = CLS_IDX["non_sn"]
        out.oids.append(oid)
        out.seqs.append(loaded)
        y3.append(TERN_IDX[lab.ternary])
        y4.append(y4_val)
        masks.append(mask)
        quals.append(QUALITY_W.get(lab.quality, 0.1))
        is_lsst.append(bool(np.max(loaded[0][:, 8]) > 0.5))  # any LSST detection
    out.y3 = np.array(y3, dtype=np.int64)
    out.y4 = np.array(y4, dtype=np.int64)
    out.masks = (np.array(masks, dtype=bool) if masks
                 else np.zeros((0, len(CLASSES)), dtype=bool))
    out.quals = np.array(quals, dtype=np.float64)
    out.is_lsst = np.array(is_lsst, dtype=bool)
    return out


def class_balance_weights(tr: SeqSet) -> tuple[np.ndarray, np.ndarray]:
    """(cls_w over CLASSES, per-object weight = quality × mask-mean cls_w).

    Grouped rows spread their quality mass uniformly over their mask classes.
    """
    mass = tr.masks.astype(np.float64)
    mass /= np.clip(mass.sum(axis=1, keepdims=True), 1e-8, None)
    eff = (mass * tr.quals[:, None]).sum(axis=0)
    cls_w = eff.sum() / (len(CLASSES) * np.clip(eff, 1e-8, None))
    obj_w = tr.quals * (mass * cls_w[None, :]).sum(axis=1)
    return cls_w, obj_w


def fit_classifier(tr: SeqSet, obj_w: np.ndarray, fit_idx: np.ndarray, val_idx: np.ndarray,
                   *, args, stats: NormStats, device, encoder_dir: Path,
                   seed: int, tag: str,
                   ) -> tuple[SeqClassifier, float, list[dict]]:
    """One full early-stopped fine-tune (used for the root and every fold).

    Early stopping monitors the inner-val grouped NLL (quality-weighted);
    cal is never touched here (temperature only, done by the caller).
    """
    torch.manual_seed(seed)
    if (encoder_dir / "encoder.pt").exists():
        encoder, _ = load_encoder(encoder_dir)  # fresh warm start per model
    else:
        from debass_meta.models.seq_encoder import SeqEncoder, SeqEncoderConfig
        encoder = SeqEncoder(SeqEncoderConfig())
    clf = SeqClassifier(encoder)
    if getattr(args, "init_from", None):
        # Warm start the WHOLE classifier (encoder + head) from an existing
        # artifact — the staged ZTF→LSST transfer recipe.  Fresh load per
        # model so folds never share mutable state.
        from debass_meta.models.seq_classifier import SeqClassifierArtifact
        init_art = SeqClassifierArtifact.load(Path(args.init_from))
        if init_art.classes != CLASSES:
            raise SystemExit(
                f"--init-from artifact classes {init_art.classes} != {CLASSES} — "
                f"cannot warm-start across class schemes")
        clf.load_state_dict(init_art.classifier.state_dict())
        print(f"  [{tag}] warm start from {args.init_from}", flush=True)
    clf = clf.to(device)

    opt = torch.optim.AdamW(
        [{"params": list(clf.cls_head.parameters()), "lr": args.lr},
         {"params": list(clf.encoder.parameters()), "lr": 0.0}],  # frozen until freeze-epochs pass
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(seed)
    masks_t = torch.from_numpy(tr.masks).to(device)
    w_fit_t = torch.from_numpy(obj_w.astype(np.float32)).to(device)
    q_t = torch.from_numpy(tr.quals.astype(np.float32)).to(device)

    def eval_nll(idxs: np.ndarray) -> float:
        clf.eval()
        nll_sum, w_sum = 0.0, 0.0
        with torch.no_grad():
            for s in range(0, len(idxs), args.batch):
                sub = idxs[s: s + args.batch]
                cont, bands, lengths = pad(tr.seqs, sub, stats, device)
                sub_t = torch.from_numpy(sub).to(device)
                loss, _ = classification_loss(
                    clf(cont, bands), None, lengths, q_t[sub_t], class_masks=masks_t[sub_t])
                bw = float(q_t[sub_t].sum())
                nll_sum += float(loss) * bw
                w_sum += bw
        return nll_sum / max(w_sum, 1e-8)

    best_nll, best_state, bad, log = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            opt.param_groups[1]["lr"] = args.encoder_lr
            print(f"  [{tag}] epoch {epoch}: encoder unfrozen (lr {args.encoder_lr})", flush=True)
        clf.train()
        order = fit_idx[rng.permutation(len(fit_idx))]
        tot, wtot = 0.0, 0.0
        for s in range(0, len(order), args.batch):
            sub = order[s: s + args.batch]
            cont, bands, lengths = pad(tr.seqs, sub, stats, device)
            sub_t = torch.from_numpy(sub).to(device)
            opt.zero_grad()
            loss, _ = classification_loss(
                clf(cont, bands), None, lengths, w_fit_t[sub_t], class_masks=masks_t[sub_t])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 5.0)
            opt.step()
            bw = float(w_fit_t[sub_t].sum())
            tot += float(loss) * bw
            wtot += bw
        val_nll = eval_nll(val_idx)
        log.append({"epoch": epoch, "train_nll": tot / max(wtot, 1e-8), "val_nll": val_nll})
        marker = ""
        if val_nll < best_nll - 1e-4:
            best_nll, bad = val_nll, 0
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
            marker = "  *best*"
        else:
            bad += 1
        print(f"  [{tag}] epoch {epoch:3d}  train {log[-1]['train_nll']:.4f}  "
              f"inner-val {val_nll:.4f}{marker}", flush=True)
        if bad >= args.patience:
            print(f"  [{tag}] early stop (patience {args.patience})", flush=True)
            break

    if best_state is not None:
        clf.load_state_dict(best_state)
    clf.to(device)
    return clf, best_nll, log


def last_prefix_logits(clf: SeqClassifier, s: SeqSet, stats: NormStats, device,
                       batch: int) -> np.ndarray:
    clf.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(s), batch):
            idxs = np.arange(start, min(start + batch, len(s)))
            cont, bands, lengths = pad(s.seqs, idxs, stats, device)
            logits = clf(cont, bands)
            out.append(logits[torch.arange(len(idxs)), lengths - 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def prefix_probs_by_ndet(clf: SeqClassifier, s: SeqSet, stats: NormStats, device,
                         batch: int, temperature: float,
                         ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """n_det → (object indices into ``s``, calibrated probs at that prefix)."""
    clf.eval()
    idx_by_n: dict[int, list[int]] = {n: [] for n in DIAG_N_DETS}
    proba_by_n: dict[int, list[np.ndarray]] = {n: [] for n in DIAG_N_DETS}
    with torch.no_grad():
        for start in range(0, len(s), batch):
            idxs = np.arange(start, min(start + batch, len(s)))
            cont, bands, lengths = pad(s.seqs, idxs, stats, device)
            probs = torch.softmax(clf(cont, bands) / max(temperature, 1e-3), dim=-1).cpu().numpy()
            for j, gi in enumerate(idxs):
                L = int(lengths[j])
                for n in DIAG_N_DETS:
                    if L >= n:
                        idx_by_n[n].append(int(gi))
                        proba_by_n[n].append(probs[j, n - 1])
    return {n: (np.array(idx_by_n[n], dtype=np.int64),
                np.array(proba_by_n[n]) if proba_by_n[n] else np.zeros((0, len(CLASSES))))
            for n in DIAG_N_DETS}


def ovr_auc_block(P: np.ndarray, idx: np.ndarray, s: SeqSet) -> dict:
    """Ternary (folded) + 4-way OvR AUCs for one slice of objects.

    Membership is determinate for a row when its label pins a single class,
    or when the class is outside its supervision mask (a certain negative) —
    rows that cannot decide membership are excluded from that class's AUC.
    Ternary AUCs apply the same rule on the mask folded to ternary buckets:
    a grouped-SN row (could be Ia OR nonIa) is excluded from the snia and
    nonIa AUCs but kept as a certain negative for 'other'.
    """
    from sklearn.metrics import roc_auc_score

    entry: dict = {"n_objects": int(len(idx))}
    if len(idx) < 10:
        return entry
    y3, y4, masks = s.y3[idx], s.y4[idx], s.masks[idx]
    p3 = fold_probs_to_ternary(P, CLASSES)
    tern_mask = np.zeros((len(idx), len(TERNARY_CLASSES)), dtype=bool)
    for c, cname in enumerate(CLASSES):
        tern_mask[:, TERN_IDX[CLASS_TO_TERNARY[cname]]] |= masks[:, c]
    single_bucket = tern_mask.sum(axis=1) == 1
    for c, cname in enumerate(TERNARY_CLASSES):
        det3 = single_bucket | (~tern_mask[:, c])
        yc = (y3[det3] == c).astype(int)
        if 0 < yc.sum() < len(yc):
            entry[f"auc_ternary_{cname}"] = float(roc_auc_score(yc, p3[det3][:, c]))
    for c, cname in enumerate(CLASSES):
        det = (y4 >= 0) | (~masks[:, c])
        yc = (y4[det] == c).astype(int)
        if 0 < yc.sum() < len(yc):
            entry[f"auc_{cname}"] = float(roc_auc_score(yc, P[det][:, c]))
    return entry


def diagnostics(clf: SeqClassifier, s: SeqSet, stats: NormStats, device,
                batch: int, temperature: float, *, per_survey: bool) -> dict:
    diag: dict[str, dict] = {}
    by_n = prefix_probs_by_ndet(clf, s, stats, device, batch, temperature)
    for n in DIAG_N_DETS:
        idx, P = by_n[n]
        if len(idx) == 0:
            continue
        entry = {"all": ovr_auc_block(P, idx, s)}
        if per_survey:
            lsst = s.is_lsst[idx]
            entry["ztf"] = ovr_auc_block(P[~lsst], idx[~lsst], s)
            entry["lsst"] = ovr_auc_block(P[lsst], idx[lsst], s)
        diag[f"n_det={n}"] = entry
    return diag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_fusion_v10.parquet")
    ap.add_argument("--split", default="data/gold/split_fusion_v10.json")
    ap.add_argument("--truth-table", default="data/truth/object_truth.parquet",
                    help="Truth parquet with raw type strings (tns_type/bts_type/final_class_raw) for 4-way fine labels")
    ap.add_argument("--lc-dir", default="data/lightcurves")
    ap.add_argument("--encoder", default="models/seq_encoder_v9", help="SSL warm start (strongly recommended)")
    ap.add_argument("--lsst-candidates", default="data/lsst_candidates.csv",
                    help="Weak LSST labels (objects must have cached lightcurves to contribute)")
    ap.add_argument("--no-lsst", action="store_true")
    ap.add_argument("--out", default="models/seq_classifier_v10")
    ap.add_argument("--surveys", choices=("ztf", "lsst", "both"), default="both",
                    help="Restrict TRAIN objects by the loaded sequence's survey "
                         "(staged ZTF→LSST recipes); cal/test stay unfiltered")
    ap.add_argument("--init-from", default=None,
                    help="Warm-start the classifier (encoder+head) from an existing "
                         "artifact dir (e.g. the ZTF stage for LSST adaptation)")
    ap.add_argument("--per-survey-norm", action="store_true",
                    help="Survey-keyed NormStats when fitting stats from scratch "
                         "(no effect when an encoder/init-from artifact supplies them)")
    ap.add_argument("--oof-folds", type=int, default=5,
                    help="K-fold OOF artifacts for train rows (0/1 disables)")
    ap.add_argument("--final-eval", action="store_true",
                    help="Compute the locked-TEST headline diagnostics.  Pass "
                         "this for the DEPLOYED artifact ONLY: reading the "
                         "locked test for every staged/diagnostic variant "
                         "turns the pre-registered headline set into a "
                         "model-selection set.  Variant comparisons use the "
                         "cal (model-selection) diagnostics instead.")
    ap.add_argument("--inner-val-frac", type=float, default=0.10,
                    help="Object-grouped early-stopping split carved OUT OF TRAIN")
    ap.add_argument("--max-len", type=int, default=20, help="Gold epoch cap")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze-epochs", type=int, default=3, help="Head-only epochs before unfreezing the encoder")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    lc_dir = Path(args.lc_dir)
    out_dir = Path(args.out)
    print(f"seq_v9 classifier fine-tune (v10) — device={device}", flush=True)

    train_ids, cal_ids, test_ids, quarantined_ids = load_split(Path(args.split))
    # Quarantined ids join all_split_ids so allow_extra can never admit them.
    all_split_ids = train_ids | cal_ids | test_ids | quarantined_ids
    labels = load_labels_from_snapshot(Path(args.snapshots))
    fine_map = load_fine_labels(Path(args.truth_table))
    n_fine, n_grouped = attach_fine_labels(labels, fine_map)
    print(f"  labels: {len(labels):,} from snapshot; {n_fine:,} refined to 4-way, "
          f"{n_grouped:,} grouped-SN supervised", flush=True)
    if not args.no_lsst:
        lsst = load_lsst_candidate_labels(Path(args.lsst_candidates))
        added = 0
        for oid, lab in lsst.items():
            if (oid not in labels and oid not in cal_ids
                    and oid not in test_ids and oid not in quarantined_ids):
                labels[oid] = lab
                added += 1
        print(f"  + {added:,} weak LSST candidates (grouped sn-like supervision)", flush=True)
    # LSST-candidate objects are NEW objects → train-only (never cal/test).

    tr = collect(labels, train_ids, lc_dir=lc_dir, max_len=args.max_len,
                 all_split_ids=all_split_ids, allow_extra=True, limit=args.limit,
                 surveys=args.surveys)
    ca = collect(labels, cal_ids, lc_dir=lc_dir, max_len=args.max_len, limit=args.limit)
    print(f"  train objects {len(tr):,} (surveys={args.surveys}; "
          f"4-way known {np.bincount(tr.y4[tr.y4 >= 0], minlength=4)}, "
          f"grouped {int((tr.y4 < 0).sum()):,}), cal {len(ca):,}", flush=True)
    if len(tr) < 100 or len(ca) < 30:
        raise SystemExit("Too few labeled objects with lightcurves")

    cls_w, obj_w = class_balance_weights(tr)
    print(f"  class weights ({'/'.join(CLASSES)}): {np.round(cls_w, 3)}", flush=True)

    enc_dir = Path(args.encoder)
    if (enc_dir / "encoder.pt").exists():
        _, stats = load_encoder(enc_dir)
        print(f"  warm start: {enc_dir} (SSL)", flush=True)
    elif args.init_from and (Path(args.init_from) / "norm_stats.json").exists():
        from debass_meta.features.sequence_dataset import load_norm_stats
        stats = load_norm_stats(Path(args.init_from) / "norm_stats.json")
        print(f"  norm stats inherited from --init-from {args.init_from}", flush=True)
    else:
        stats = NormStats.fit([c for c, _ in tr.seqs], per_survey=args.per_survey_norm)
        print("  WARNING: no SSL artifact — training from scratch", flush=True)

    pos = {oid: i for i, oid in enumerate(tr.oids)}

    def fit_with_inner_val(pool_oids: list[str], seed: int, tag: str):
        fit_oids, val_oids = split_inner_val(pool_oids, args.inner_val_frac, seed)
        fit_idx = np.array([pos[o] for o in fit_oids], dtype=np.int64)
        val_idx = np.array([pos[o] for o in val_oids], dtype=np.int64)
        clf, best_nll, log = fit_classifier(
            tr, obj_w, fit_idx, val_idx, args=args, stats=stats, device=device,
            encoder_dir=enc_dir, seed=seed, tag=tag)
        # Temperature on cal (last-prefix logits, grouped-mass NLL) — cal's
        # ONLY role in v10.
        logits_last = last_prefix_logits(clf, ca, stats, device, args.batch)
        temperature = fit_temperature(logits_last, None, ca.quals, class_masks=ca.masks)
        return clf, temperature, best_nll, log, len(fit_oids), len(val_oids)

    # ── Full-data root artifact ──────────────────────────────────────────
    clf, temperature, best_nll, log, n_fit, n_val = fit_with_inner_val(
        tr.oids, args.seed, "full")
    print(f"  [full] temperature: {temperature:.3f}", flush=True)

    # ── K-fold OOF artifacts ─────────────────────────────────────────────
    fold_assign: dict[str, int] = {}
    fold_meta: list[dict] = []
    if args.oof_folds and args.oof_folds >= 2:
        fold_assign = assign_folds(tr.oids, args.oof_folds, args.seed)
        for k in range(args.oof_folds):
            pool = [o for o in tr.oids if fold_assign[o] != k]
            excluded = [o for o in tr.oids if fold_assign[o] == k]
            fclf, ftemp, fnll, flog, fn_fit, fn_val = fit_with_inner_val(
                pool, args.seed + 101 * (k + 1), f"fold_{k}")
            fdir = out_dir / f"fold_{k}"
            fart = SeqClassifierArtifact(
                classifier=fclf.to("cpu"), norm_stats=stats, temperature=ftemp,
                meta={
                    "fold_index": k, "n_folds": args.oof_folds,
                    "n_excluded_objects": len(excluded),
                    "n_fit_objects": fn_fit, "n_inner_val_objects": fn_val,
                    "best_inner_val_nll": fnll,
                    "encoder_warm_start": str(enc_dir), "seed": args.seed,
                    "split_manifest": str(args.split),
                },
            )
            fart.save(fdir)
            (fdir / "train_log.json").write_text(json.dumps(flog, indent=1))
            fold_meta.append({"fold": k, "n_excluded": len(excluded),
                              "temperature": ftemp, "best_inner_val_nll": fnll})
            print(f"  [fold_{k}] saved → {fdir} (excludes {len(excluded):,} objects, "
                  f"T={ftemp:.3f})", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "fold_map.json").write_text(json.dumps({
            "n_folds": args.oof_folds, "seed": args.seed,
            "assignments": fold_assign,
        }, indent=1))
        print(f"  fold_map.json → {out_dir} ({len(fold_assign):,} train objects)", flush=True)

    # ── Diagnostics ──────────────────────────────────────────────────────
    # Cal diagnostics: MODEL-SELECTION metrics only (cal fits the
    # temperature; weak/context labels included) — NOT deployment estimates.
    # Per-survey so staged/variant artifacts can be compared WITHOUT ever
    # reading the locked test.
    cal_diag = diagnostics(clf, ca, stats, device, args.batch, temperature, per_survey=True)
    print("  cal diagnostics (MODEL-SELECTION ONLY — temperature was fit on cal):",
          json.dumps(cal_diag, indent=1), flush=True)

    # Honest headline: locked TEST split, spectroscopic+tns labels only,
    # sliced per survey and per n_det bucket.  Gated behind --final-eval so
    # only the DEPLOYED artifact ever reads test_ids — evaluating every
    # variant on the locked test would turn it into a model-selection set.
    if args.final_eval:
        te = collect(labels, test_ids, lc_dir=lc_dir, max_len=args.max_len,
                     quality_allow=HEADLINE_QUALITIES, limit=args.limit)
        test_diag: dict = {"n_test_objects": len(te)}
        if len(te) >= 10:
            test_diag.update(diagnostics(clf, te, stats, device, args.batch, temperature,
                                         per_survey=True))
        print(f"  TEST diagnostics (HEADLINE — locked test, {'/'.join(sorted(HEADLINE_QUALITIES))} "
              f"labels only, {len(te):,} objects):", json.dumps(test_diag, indent=1), flush=True)
    else:
        test_diag = {"skipped": "--final-eval not set — locked test untouched"}
        print("  TEST diagnostics SKIPPED (--final-eval not set) — locked test untouched",
              flush=True)

    clf.to("cpu")
    art = SeqClassifierArtifact(
        classifier=clf, norm_stats=stats, temperature=temperature,
        meta={
            "n_train_objects": len(tr), "n_cal_objects": len(ca),
            "n_inner_fit_objects": n_fit, "n_inner_val_objects": n_val,
            "inner_val_frac": args.inner_val_frac,
            "train_y4_counts": np.bincount(tr.y4[tr.y4 >= 0], minlength=len(CLASSES)).tolist(),
            "train_grouped_rows": int((tr.y4 < 0).sum()),
            "best_inner_val_nll": best_nll,
            "eval_protocol": {
                "early_stopping": "inner 10% object-grouped val carved out of TRAIN",
                "cal_role": "temperature calibration ONLY",
                "headline": "locked TEST split, spectroscopic+tns labels, "
                            "per-survey and per-n_det slices — computed only "
                            "with --final-eval (deployed artifact)",
            },
            "final_eval": bool(args.final_eval),
            "test_diagnostics_headline": test_diag,
            "cal_diagnostics_model_selection": {
                "note": "cal fit the temperature; weak labels included — "
                        "model-selection metrics, NOT deployment estimates",
                **cal_diag,
            },
            "oof_folds": args.oof_folds if fold_assign else 0,
            "oof_fold_summary": fold_meta,
            "surveys": args.surveys,
            "init_from": str(args.init_from) if args.init_from else None,
            "encoder_warm_start": str(enc_dir), "seed": args.seed,
            "lsst_weak_included": not args.no_lsst,
            "split_manifest": str(args.split),
            "truth_table": str(args.truth_table),
        },
    )
    art.save(out_dir)
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"Saved seq_v9 classifier (v10) → {out_dir} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
