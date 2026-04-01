#!/usr/bin/env python3
"""Download TNS bulk CSV catalog for local crossmatching.

Run this on SCC where TNS credentials are configured.
The CSV is ~160K objects, regenerated daily by TNS.
"""
import sys
from pathlib import Path
import zipfile

import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from debass_meta.access.tns import load_tns_credentials


def download_tns_bulk_csv(output_dir: Path) -> Path:
    """Download and extract TNS public objects CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    creds = load_tns_credentials()
    url = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip"

    zip_path = output_dir / "tns_public_objects.csv.zip"
    csv_path = output_dir / "tns_public_objects.csv"

    print(f"Downloading TNS bulk CSV from {url}")
    headers = {"User-Agent": creds.user_agent}
    r = requests.get(url, headers=headers, timeout=300)
    r.raise_for_status()

    print(f"Writing {len(r.content)} bytes to {zip_path}")
    zip_path.write_bytes(r.content)

    print(f"Extracting to {csv_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    print(f"✓ Downloaded {csv_path}")
    return csv_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    csv_path = download_tns_bulk_csv(args.output_dir)
    print(f"\nTo use: python scripts/crossmatch_tns.py --bulk-csv {csv_path}")
