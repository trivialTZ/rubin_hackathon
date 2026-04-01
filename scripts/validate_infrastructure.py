#!/usr/bin/env python3
"""Validate DEBASS infrastructure readiness for SCC deployment.

Checks:
- Python dependencies installed
- Broker adapters importable
- Expert projectors working
- Feature extraction working
- Required scripts present
- Data directories structure
"""
import sys
from pathlib import Path

def check_dependencies():
    """Check core Python dependencies."""
    print("Checking Python dependencies...")
    missing = []

    deps = [
        ("pandas", "pandas"),
        ("pyarrow", "pyarrow"),
        ("numpy", "numpy"),
        ("lightgbm", "lightgbm"),
        ("sklearn", "scikit-learn"),
        ("alerce", "alerce (broker client)"),
        ("lasair", "lasair (broker client)"),
    ]

    for module, name in deps:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} MISSING")
            missing.append(name)

    # Fink is optional but recommended
    try:
        __import__("fink_client")
        print(f"  ✓ fink-client (broker client)")
    except ImportError:
        print(f"  ⚠ fink-client MISSING (optional)")

    return missing

def check_adapters():
    """Check broker adapters are importable."""
    print("\nChecking broker adapters...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    try:
        from debass_meta.access.fink import FinkAdapter
        print("  ✓ Fink adapter")
    except Exception as e:
        print(f"  ✗ Fink adapter: {e}")
        return False

    try:
        from debass_meta.access.alerce import AlerceAdapter
        print("  ✓ ALeRCE adapter")
    except Exception as e:
        print(f"  ✗ ALeRCE adapter: {e}")
        return False

    try:
        from debass_meta.access.lasair import LasairAdapter
        print("  ✓ Lasair adapter")
    except Exception as e:
        print(f"  ✗ Lasair adapter: {e}")
        return False

    try:
        from debass_meta.access.tns import TNSClient, load_tns_credentials
        print("  ✓ TNS adapter")
    except Exception as e:
        print(f"  ✗ TNS adapter: {e}")
        return False

    return True

def check_projectors():
    """Check expert projectors are working."""
    print("\nChecking expert projectors...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    try:
        from debass_meta.projectors.base import PHASE1_EXPERT_KEYS, project_expert_events
        print(f"  ✓ {len(PHASE1_EXPERT_KEYS)} Phase-1 experts defined")

        for expert_key in PHASE1_EXPERT_KEYS:
            result = project_expert_events(expert_key, [])
            if "prediction_type" in result:
                print(f"    ✓ {expert_key}")
            else:
                print(f"    ✗ {expert_key} (invalid projector)")
                return False
    except Exception as e:
        print(f"  ✗ Projectors: {e}")
        return False

    return True

def check_features():
    """Check lightcurve feature extraction."""
    print("\nChecking lightcurve features...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    try:
        from debass_meta.features.lightcurve import FEATURE_NAMES, extract_features
        print(f"  ✓ {len(FEATURE_NAMES)} features defined")

        # Test extraction on empty detections
        feats = extract_features([])
        if len(feats) == len(FEATURE_NAMES):
            print(f"  ✓ Feature extraction working")
        else:
            print(f"  ✗ Feature count mismatch")
            return False
    except Exception as e:
        print(f"  ✗ Features: {e}")
        return False

    return True

def check_scripts():
    """Check required scripts are present."""
    print("\nChecking pipeline scripts...")
    repo_root = Path(__file__).parent.parent

    required = [
        "scripts/download_alerce_training.py",
        "scripts/backfill.py",
        "scripts/normalize.py",
        "scripts/crossmatch_tns.py",
        "scripts/download_tns_bulk.py",
        "scripts/build_object_epoch_snapshots.py",
        "scripts/build_expert_helpfulness.py",
        "scripts/train_expert_trust.py",
        "scripts/train_followup.py",
        "scripts/score_nightly.py",
        "scripts/run_preml_pipeline.py",
        "scripts/check_preml_readiness.py",
    ]

    missing = []
    for script in required:
        path = repo_root / script
        if path.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} MISSING")
            missing.append(script)

    return missing

def check_data_structure():
    """Check data directory structure."""
    print("\nChecking data structure...")
    repo_root = Path(__file__).parent.parent

    # Check if we have any data
    data_dir = repo_root / "data"
    if not data_dir.exists():
        print("  ⚠ data/ directory not found (expected for fresh checkout)")
        return True

    # Check for key files
    labels = data_dir / "labels.csv"
    if labels.exists():
        import csv
        with open(labels) as f:
            n_objects = sum(1 for _ in csv.DictReader(f))
        print(f"  ✓ labels.csv ({n_objects} objects)")
    else:
        print("  ⚠ labels.csv not found")

    lc_dir = data_dir / "lightcurves"
    if lc_dir.exists():
        n_lc = len(list(lc_dir.glob("*.json")))
        print(f"  ✓ lightcurves/ ({n_lc} files)")
    else:
        print("  ⚠ lightcurves/ not found")

    tns_csv = data_dir / "tns_public_objects.csv"
    if tns_csv.exists():
        print(f"  ✓ tns_public_objects.csv (bulk CSV ready)")
    else:
        print(f"  ⚠ tns_public_objects.csv not found (run scripts/download_tns_bulk.py on SCC)")

    return True

def check_credentials():
    """Check if credentials are configured."""
    print("\nChecking credentials...")
    import os

    # Check .env file
    repo_root = Path(__file__).parent.parent
    env_file = repo_root / ".env"
    if env_file.exists():
        print("  ✓ .env file exists")
    else:
        print("  ⚠ .env file not found (copy .env.example)")

    # Check TNS credentials
    tns_vars = ["TNS_API_KEY", "TNS_TNS_ID", "TNS_MARKER_NAME"]
    tns_ok = all(os.getenv(var) for var in tns_vars)
    if tns_ok:
        print("  ✓ TNS credentials configured")
    else:
        print("  ⚠ TNS credentials not configured (needed for bulk CSV download)")

    # Check Lasair
    if os.getenv("LASAIR_API_TOKEN"):
        print("  ✓ Lasair API token configured")
    else:
        print("  ⚠ Lasair API token not configured")

    return True

def main():
    print("=" * 60)
    print("DEBASS Infrastructure Validation")
    print("=" * 60)

    issues = []

    # Run checks
    missing_deps = check_dependencies()
    if missing_deps:
        issues.append(f"Missing dependencies: {', '.join(missing_deps)}")

    if not check_adapters():
        issues.append("Broker adapters not working")

    if not check_projectors():
        issues.append("Expert projectors not working")

    if not check_features():
        issues.append("Feature extraction not working")

    missing_scripts = check_scripts()
    if missing_scripts:
        issues.append(f"Missing scripts: {', '.join(missing_scripts)}")

    check_data_structure()
    check_credentials()

    # Summary
    print("\n" + "=" * 60)
    if not issues:
        print("✓ ALL CHECKS PASSED")
        print("\nInfrastructure is ready for SCC deployment.")
        print("\nNext steps:")
        print("  1. Copy to SCC: scp -r . <scc>:/projectnb/<project>/rubin_hackathon")
        print("  2. Set up environment: python3 -m venv ~/debass_env && pip install -r env/requirements.txt")
        print("  3. Configure credentials: cp .env.example .env && nano .env")
        print("  4. Download TNS CSV: python3 scripts/download_tns_bulk.py")
        print("  5. Run smoke test: python3 scripts/run_preml_pipeline.py --root tmp/test --labels data/labels.csv --lightcurves-dir data/lightcurves --skip-local-infer")
    else:
        print("✗ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nFix these issues before deploying to SCC.")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
