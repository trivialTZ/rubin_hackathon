#!/bin/bash -l
# Download external model files for local expert inference.
#
# Run this ONCE on SCC before training with local experts.
#
# Usage:
#   bash jobs/setup_local_experts.sh
#   bash jobs/setup_local_experts.sh --snn-only
#   bash jobs/setup_local_experts.sh --all

set -euo pipefail

: "${DEBASS_ROOT:?DEBASS_ROOT must be set}"

ARTIFACTS="${DEBASS_ROOT}/artifacts/local_experts"
mkdir -p "${ARTIFACTS}"

INSTALL_SNN=false
INSTALL_RAPID=false
INSTALL_ORACLE=false
INSTALL_ALL=false

if [ $# -eq 0 ]; then
    INSTALL_SNN=true  # default: just SNN
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --snn-only)   INSTALL_SNN=true; shift ;;
        --rapid)      INSTALL_RAPID=true; shift ;;
        --oracle)     INSTALL_ORACLE=true; shift ;;
        --all)        INSTALL_ALL=true; shift ;;
        *)            echo "Unknown: $1"; exit 1 ;;
    esac
done
if ${INSTALL_ALL}; then INSTALL_SNN=true; INSTALL_RAPID=true; INSTALL_ORACLE=true; fi

echo "=== DEBASS Local Expert Setup ==="
echo "DEBASS_ROOT: ${DEBASS_ROOT}"
echo "Artifacts:   ${ARTIFACTS}"
echo ""

# -----------------------------------------------------------------------
# SuperNNova LSST weights (from Fink's ELAsTiCC training)
# -----------------------------------------------------------------------
if ${INSTALL_SNN}; then
    SNN_DIR="${ARTIFACTS}/supernnova"
    if [ -f "${SNN_DIR}/model.pt" ]; then
        echo "[SNN] Already installed at ${SNN_DIR}"
    else
        echo "[SNN] Downloading LSST-trained weights from fink-science..."
        mkdir -p "${SNN_DIR}"

        # Sparse clone just the SNN model directory
        TMPDIR=$(mktemp -d)
        cd "${TMPDIR}"
        git clone --depth 1 --filter=blob:none --sparse \
            https://github.com/astrolabsoftware/fink-science.git 2>/dev/null || {
            echo "[SNN] git clone failed. Try manual download:"
            echo "  git clone https://github.com/astrolabsoftware/fink-science.git"
            echo "  cp -r fink-science/fink_science/data/models/snn_models/* ${SNN_DIR}/"
        }
        if [ -d "fink-science" ]; then
            cd fink-science
            git sparse-checkout set fink_science/data/models/snn_models 2>/dev/null || true

            # Find and copy model files
            if [ -d "fink_science/data/models/snn_models" ]; then
                # Look for the snia_vs_nonia model
                find fink_science/data/models/snn_models -name "*.pt" -o -name "*.json" | head -20
                cp -r fink_science/data/models/snn_models/* "${SNN_DIR}/" 2>/dev/null || true
                echo "[SNN] Installed to ${SNN_DIR}"
                ls -la "${SNN_DIR}/"
            else
                echo "[SNN] Model directory not found in sparse checkout."
                echo "  Try: pip install fink-science && python -c 'import fink_science; print(fink_science.__path__)'"
            fi
        fi
        rm -rf "${TMPDIR}"
    fi

    # Also ensure supernnova Python package is installed
    pip install supernnova 2>/dev/null && echo "[SNN] Python package installed" || \
        echo "[SNN] pip install supernnova failed (may need torch first)"
    echo ""
fi

# -----------------------------------------------------------------------
# RAPID (pre-trained model included in pip package)
# -----------------------------------------------------------------------
if ${INSTALL_RAPID}; then
    echo "[RAPID] Installing from pip..."
    pip install astrorapid 2>/dev/null && {
        python3 -c "import astrorapid; print(f'[RAPID] Installed: {astrorapid.__file__}')" 2>/dev/null || true
    } || echo "[RAPID] pip install failed"
    echo ""
fi

# -----------------------------------------------------------------------
# ORACLE
# -----------------------------------------------------------------------
if ${INSTALL_ORACLE}; then
    echo "[ORACLE] Installing..."
    pip install astro-oracle 2>/dev/null || {
        echo "[ORACLE] pip install failed. Try from source:"
        echo "  pip install git+https://github.com/uiucsn/Astro-ORACLE.git"
    }
    echo ""
fi

echo "=== Setup Complete ==="
echo ""
echo "Available local experts:"
[ -f "${ARTIFACTS}/supernnova/model.pt" ] && echo "  ✓ SuperNNova (LSST)" || echo "  ✗ SuperNNova"
python3 -c "import astrorapid" 2>/dev/null && echo "  ✓ RAPID" || echo "  ✗ RAPID (pip install astrorapid)"
python3 -c "import astro_oracle" 2>/dev/null && echo "  ✓ ORACLE" || echo "  ✗ ORACLE (pip install astro-oracle)"
echo ""
echo "To use in training pipeline:"
echo "  bash jobs/submit_retrain_v2.sh --discover-lsst 400 --local"
