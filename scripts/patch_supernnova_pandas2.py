#!/usr/bin/env python3
"""Patch installed SuperNNova to work with pandas 2.x + numpy 2.x.

SuperNNova pip package (<=3.0.36 on Python 3.10) uses deprecated pandas APIs:
  - DataFrame.drop(col, 1) → DataFrame.drop(columns=col)
  - Series.append(other)   → pd.concat([series, other])
  - Object-dtype arrays from pivot operations break np.log

Run once after `pip install supernnova`:
    python3 scripts/patch_supernnova_pandas2.py
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path


def _find_snn_dir() -> Path:
    """Locate the installed supernnova package directory."""
    try:
        import supernnova
        return Path(supernnova.__file__).parent
    except ImportError:
        print("ERROR: supernnova is not installed")
        sys.exit(1)


def _patch_file(path: Path, replacements: list[tuple[str, str]]) -> int:
    """Apply string replacements to a file. Returns count of changes."""
    text = path.read_text()
    original = text
    for old, new in replacements:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text)
        return sum(1 for old, _ in replacements if old in original)
    return 0


def main():
    snn = _find_snn_dir()
    total_fixes = 0

    # --- make_dataset.py: .drop(x, 1) → .drop(columns=x) ---
    f = snn / "data" / "make_dataset.py"
    if f.exists():
        n = _patch_file(f, [
            ('df.drop("PEAKMJDNORM", 1)', 'df.drop(columns="PEAKMJDNORM")'),
            ('df.drop(["MJD", "delta_time"], 1)', 'df.drop(columns=["MJD", "delta_time"])'),
            ('df.drop(list_filters, 1)', 'df.drop(columns=list_filters)'),
            ('df.drop("MJD", 1)', 'df.drop(columns="MJD")'),
        ])
        total_fixes += n
        print(f"  make_dataset.py: {n} fixes")

    # --- validate_onthefly.py: Series.append → pd.concat ---
    f = snn / "validation" / "validate_onthefly.py"
    if f.exists():
        n = _patch_file(f, [
            (
                'pd.Series(settings.list_filters_combination).append(df["FLT"])',
                'pd.concat([pd.Series(settings.list_filters_combination), df["FLT"]])',
            ),
        ])
        total_fixes += n
        print(f"  validate_onthefly.py: {n} fixes")

    # --- data_utils.py: Series.append → pd.concat ---
    f = snn / "utils" / "data_utils.py"
    if f.exists():
        n = _patch_file(f, [
            (
                'pd.Series(settings.list_filters_combination).append(df["FLT"])',
                'pd.concat([pd.Series(settings.list_filters_combination), df["FLT"]])',
            ),
        ])
        total_fixes += n
        print(f"  data_utils.py: {n} fixes")

    # --- training_utils.py: ensure float64 before np.log ---
    f = snn / "utils" / "training_utils.py"
    if f.exists():
        n = _patch_file(f, [
            # The array from pivot may be object-dtype; cast to float64
            (
                'arr_to_norm = arr[:, settings.idx_features_to_normalize]',
                'arr_to_norm = arr[:, settings.idx_features_to_normalize].astype(np.float64)',
            ),
        ])
        total_fixes += n
        print(f"  training_utils.py: {n} fixes")

    # --- Also patch get_data_batch FloatTensor conversion for safety ---
    f = snn / "utils" / "training_utils.py"
    if f.exists():
        text = f.read_text()
        # Ensure X is float32 before torch conversion
        old = "X_tensor[: X.shape[0], i, :] = torch.FloatTensor(X)"
        new = "X_tensor[: X.shape[0], i, :] = torch.FloatTensor(np.asarray(X, dtype=np.float32))"
        if old in text and new not in text:
            text = text.replace(old, new)
            f.write_text(text)
            total_fixes += 1
            print(f"  training_utils.py: +1 FloatTensor fix")

    print(f"\nTotal: {total_fixes} fixes applied")
    if total_fixes == 0:
        print("(Already patched or patterns not found)")

    # --- Verify ---
    print("\n--- Verification ---")
    try:
        importlib.invalidate_caches()
        from supernnova.validation.validate_onthefly import classify_lcs
        print("  classify_lcs imports OK")
    except Exception as e:
        print(f"  classify_lcs import FAILED: {e}")


if __name__ == "__main__":
    main()
