"""DEBASS Dashboard — colour palette and CSS helpers."""

# ── Semantic colours ──────────────────────────────────────────────
CLASS_COLORS = {
    "snia":           "#e74c3c",   # Red
    "nonIa_snlike":   "#3498db",   # Blue
    "other":          "#95a5a6",   # Gray
}

STATUS_COLORS = {
    "recommended":    "#2ecc71",   # Green
    "borderline":     "#f39c12",   # Orange
    "not_recommended":"#636e72",   # Dark gray
    "unavailable":    "#bdc3c7",   # Light gray
}

# Band colours (astronomy convention)
BAND_COLORS = {
    "u": "#7b2fbe",
    "g": "#00b894",
    "r": "#e74c3c",
    "i": "#fdcb6e",
    "z": "#e17055",
    "y": "#636e72",
}

BAND_ORDER = list("ugrizy")

# ── Expert display names ──────────────────────────────────────────
EXPERT_DISPLAY = {
    "fink/snn":       "Fink SNN",
    "fink/rf_ia":     "Fink RF",
    "fink_lsst/snn":  "Fink-LSST SNN",
    "fink_lsst/cats": "Fink-LSST CATS",
    "fink_lsst/early_snia": "Fink-LSST EarlySNIa",
    "parsnip":        "ParSNIP",
    "supernnova":     "SuperNNova",
    "alerce_lc":      "ALeRCE LC",
    "alerce/lc_classifier_transient": "ALeRCE LC Trans",
    "alerce/stamp_classifier": "ALeRCE Stamp",
    "alerce/lc_classifier_BHRF_forced_phot_transient": "ALeRCE BHRF",
    "alerce/lc_classifier_BHRF_forced_phot_top": "ALeRCE BHRF Top",
    "alerce/LC_classifier_ATAT_forced_phot(beta)": "ALeRCE ATAT",
    "alerce/stamp_classifier_2025_beta": "ALeRCE Stamp 2025",
    "alerce/stamp_classifier_rubin_beta": "ALeRCE Rubin Stamp",
    "lasair/sherlock":  "Lasair Sherlock",
}

def expert_name(key: str) -> str:
    return EXPERT_DISPLAY.get(key, key)

# ── Recommendation colour helper ──────────────────────────────────
def rec_color(p_follow: float) -> str:
    if p_follow >= 0.5:
        return STATUS_COLORS["recommended"]
    elif p_follow >= 0.3:
        return STATUS_COLORS["borderline"]
    return STATUS_COLORS["not_recommended"]

def rec_label(p_follow: float) -> str:
    if p_follow >= 0.5:
        return "Recommended"
    elif p_follow >= 0.3:
        return "Borderline"
    return "Not recommended"

# ── Global CSS injected via st.markdown ───────────────────────────
CUSTOM_CSS = """
<style>
    /* Metric cards — subtle border, no forced background */
    [data-testid="stMetric"] {
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    [data-testid="stMetricLabel"] p {
        font-weight: 600 !important;
    }
    /* Expert card — light background with accent border */
    .expert-card {
        background: #f7f8fa;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 8px;
        border-left: 4px solid #3498db;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        color: #1a1a2e;
    }
    .expert-card.agree { border-left-color: #2ecc71; }
    .expert-card.disagree { border-left-color: #e74c3c; }
    .expert-card.unavailable { border-left-color: #636e72; opacity: 0.6; }
</style>
"""
