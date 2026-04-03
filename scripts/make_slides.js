/**
 * DEBASS Rubin Hackathon – Slide Deck
 * Run: node scripts/make_slides.js
 * Output: docs/DEBASS_slides.pptx
 */

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const FIGS = path.join(ROOT, "docs", "slides_figures");
const OUT  = path.join(ROOT, "docs", "DEBASS_slides.pptx");

// ── Palette (deep-space astronomy theme) ──────────────────────────────────
const C = {
  bg:       "080D1A",   // deep space navy
  panel:    "0F1929",   // card background
  panelAlt: "131E2E",   // slightly lighter panel
  border:   "1F3050",   // subtle border
  text:     "E8EDF3",   // near-white
  muted:    "7B8EA8",   // secondary text
  blue:     "4F9CF9",   // accent blue (ZTF / ALeRCE)
  teal:     "2DD4BF",   // teal  (Fink)
  orange:   "F59E0B",   // orange/gold (non-Ia)
  green:    "34D399",   // green (SuperNNova / good AUC)
  purple:   "A78BFA",   // purple (meta ensemble)
  red:      "F87171",   // red (alert)
  yellow:   "FDE68A",   // soft yellow
  white:    "FFFFFF",
};

// ── Helpers ────────────────────────────────────────────────────────────────
const figPath = (name) => path.join(FIGS, name);

function headerBar(slide, title, subtitle) {
  // slim top accent bar
  slide.addShape(RECT, {
    x: 0, y: 0, w: 10, h: 0.06,
    fill: { color: C.blue }, line: { color: C.blue },
  });
  slide.addText(title, {
    x: 0.45, y: 0.18, w: 9.1, h: 0.52,
    fontSize: 22, bold: true, color: C.text, fontFace: "Calibri",
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.45, y: 0.72, w: 9.1, h: 0.3,
      fontSize: 12, color: C.muted, fontFace: "Calibri", margin: 0,
    });
  }
  // divider
  slide.addShape(RECT, {
    x: 0.45, y: 1.06, w: 9.1, h: 0.02,
    fill: { color: C.border }, line: { color: C.border },
  });
}

function card(slide, x, y, w, h, col) {
  slide.addShape(RECT, {
    x, y, w, h,
    fill: { color: col || C.panel },
    line: { color: C.border, width: 0.75 },
    shadow: { type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.35 },
  });
}

function statBlock(slide, x, y, w, h, value, label, col) {
  card(slide, x, y, w, h, C.panel);
  // accent left bar
  slide.addShape(RECT, {
    x, y: y + 0.04, w: 0.06, h: h - 0.08,
    fill: { color: col }, line: { color: col },
  });
  slide.addText(value, {
    x: x + 0.14, y: y + 0.05, w: w - 0.2, h: h * 0.58,
    fontSize: 28, bold: true, color: col, fontFace: "Calibri", margin: 0, valign: "bottom",
  });
  slide.addText(label, {
    x: x + 0.14, y: y + h * 0.58, w: w - 0.2, h: h * 0.38,
    fontSize: 9.5, color: C.muted, fontFace: "Calibri", margin: 0, valign: "top",
  });
}

function footer(slide, text) {
  slide.addText(text || "DEBASS · Rubin Hackathon 2026", {
    x: 0, y: 5.35, w: 10, h: 0.28,
    fontSize: 8, color: C.muted, fontFace: "Calibri",
    align: "center", margin: 0,
  });
}

// ── Presentation ──────────────────────────────────────────────────────────
const pres = new pptxgen();
const RECT = pres.shapes.RECTANGLE;
pres.layout = "LAYOUT_16x9";
pres.author = "DEBASS Team";
pres.title  = "DEBASS: Real-Time Transient Classification for Rubin/LSST";

// ═══════════════════════════════════════════════════════════════════════════
// Slide 1 – Title
// ═══════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: C.bg };

  // large accent block on left
  sl.addShape(RECT, {
    x: 0, y: 0, w: 0.18, h: 5.625,
    fill: { color: C.blue }, line: { color: C.blue },
  });
  // top bar
  sl.addShape(RECT, {
    x: 0, y: 0, w: 10, h: 0.08,
    fill: { color: C.blue }, line: { color: C.blue },
  });

  sl.addText("DEBASS", {
    x: 0.5, y: 0.9, w: 9, h: 1.1,
    fontSize: 64, bold: true, color: C.blue, fontFace: "Calibri",
    charSpacing: 6, margin: 0,
  });
  sl.addText("Real-Time Transient Classification\nfor Rubin / LSST", {
    x: 0.5, y: 1.95, w: 9, h: 1.1,
    fontSize: 26, color: C.text, fontFace: "Calibri", margin: 0,
  });
  sl.addText(
    "Broker Fusion · Expert Trust · Follow-up Prioritisation\n" +
    "3–5 detections → spectroscopic priority score",
    {
      x: 0.5, y: 3.1, w: 9, h: 0.8,
      fontSize: 14, color: C.muted, fontFace: "Calibri", margin: 0,
    }
  );

  // stat pills bottom
  const pills = [
    { v: "6,958",    l: "training objects", c: C.teal },
    { v: "18",       l: "expert classifiers", c: C.blue },
    { v: "AUC 0.995",l: "follow-up head", c: C.green },
    { v: "51",       l: "LC features", c: C.orange },
  ];
  pills.forEach((p, i) => {
    const x = 0.5 + i * 2.35;
    sl.addShape(RECT, {
      x, y: 4.3, w: 2.1, h: 0.9,
      fill: { color: C.panel }, line: { color: p.c, width: 1.2 },
    });
    sl.addText(p.v, {
      x: x + 0.05, y: 4.33, w: 2.0, h: 0.46,
      fontSize: 18, bold: true, color: p.c, fontFace: "Calibri",
      align: "center", margin: 0,
    });
    sl.addText(p.l, {
      x: x + 0.05, y: 4.77, w: 2.0, h: 0.35,
      fontSize: 9, color: C.muted, fontFace: "Calibri",
      align: "center", margin: 0,
    });
  });

  sl.addText("Rubin Hackathon · April 2026", {
    x: 0, y: 5.32, w: 10, h: 0.3,
    fontSize: 8.5, color: C.muted, fontFace: "Calibri",
    align: "center", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Slide 2 – The Problem
// ═══════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: C.bg };
  headerBar(sl, "The Challenge: Classify Early, Trigger Fast", "Rubin/LSST will deliver millions of alerts per night — most are not supernovae");

  const challenges = [
    { icon: "★", title: "Only 3–5 detections", body: "Spectroscopic follow-up must be requested before the target fades — decisions made in hours, not days.", col: C.blue },
    { icon: "⊕", title: "Heterogeneous brokers", body: "ALeRCE, Fink, Lasair, Pitt-Google each provide different classifiers, cadences, and coverage — none complete alone.", col: C.teal },
    { icon: "◈", title: "Mixed ZTF + LSST", body: "Training data spans both surveys (g,r vs u,g,r,i,z,y). Missing bands are the rule, not the exception.", col: C.orange },
    { icon: "✦", title: "Noisy labels", body: "Most objects lack spectroscopic confirmation. Truth comes from TNS, host context, and broker consensus.", col: C.purple },
  ];

  challenges.forEach((ch, i) => {
    const col = i % 2 === 0 ? 0 : 4.85;
    const row = Math.floor(i / 2);
    const x = col === 0 ? 0.45 : 5.1;
    const y = 1.2 + row * 1.85;
    card(sl, x, y, 4.45, 1.65, C.panel);
    // top color bar
    sl.addShape(RECT, {
      x, y, w: 4.45, h: 0.06,
      fill: { color: ch.col }, line: { color: ch.col },
    });
    sl.addText(ch.icon + "  " + ch.title, {
      x: x + 0.18, y: y + 0.12, w: 4.05, h: 0.42,
      fontSize: 13, bold: true, color: ch.col, fontFace: "Calibri", margin: 0,
    });
    sl.addText(ch.body, {
      x: x + 0.18, y: y + 0.55, w: 4.1, h: 1.0,
      fontSize: 10.5, color: C.text, fontFace: "Calibri", margin: 0,
    });
  });

  footer(sl);
}

// ═══════════════════════════════════════════════════════════════════════════
// Slide 3 – Pipeline Architecture
// ═══════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: C.bg };
  headerBar(sl, "DEBASS Pipeline Architecture", "From raw broker alerts to spectroscopic priority score");

  sl.addImage({
    path: figPath("fig5_architecture.png"),
    x: 0.35, y: 1.15, w: 9.3, h: 4.1,
  });

  footer(sl);
}

// ═══════════════════════════════════════════════════════════════════════════
// Slide 4 – Dataset Composition
// ═══════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: C.bg };
  headerBar(sl, "Training Dataset", "Multi-source truth with five quality tiers");

  // top stat row
  const stats = [
    { v: "6,958",   l: "total objects",    c: C.blue },
    { v: "111,597", l: "epoch rows (1–20 det)", c: C.teal },
    { v: "508",     l: "spectroscopic labels", c: C.green },
    { v: "3",       l: "target classes",   c: C.orange },
  ];
  stats.forEach((s, i) => {
    statBlock(sl, 0.45 + i * 2.32, 1.15, 2.1, 1.0, s.v, s.l, s.c);
  });

  // figure
  sl.addImage({
    path: figPath("fig2_dataset.png"),
    x: 0.45, y: 2.28, w: 9.1, h: 3.0,
  });

  footer(sl);
}

// ═══════════════════════════════════════════════════════════════════════════
// Slide 5 – Expert Trust Models
// ═══════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: C.bg };
  headerBar(sl, "Expert Trust Heads", "LightGBM binary classifier per broker · predicts whether the expert is correct");

  sl.addImage({
    path: figPath("fig1_expert_auc.png"),
    x: 0.45, y: 1.15, w: 5.8, h: 4.1,
  });

  // Right side: explanation bullets
  const bullets = [
    { t: "What is a trust head?", b: "Each broker expert gets its own LightGBM binary model that predicts whether that expert's classification will be correct on this object at this epoch.", c: C.blue },
    { t: "Training signal", b: "Ground truth: is the expert's top class the same as the object's true class? Grouped train/cal/test split prevents object-level leakage.", c: C.teal },
    { t: "Native NaN support", b: "Missing broker scores are passed directly to LightGBM — no imputation. Absence of a score is itself informative.", c: C.orange },
    { t: "Phase-calibrated", b: "Trust heads are trained on 6 experts in current phase; 18 registered experts targeted for Rubin LSST full deployment.", c: C.purple },
  ];

  bullets.forEach((b, i) => {
    const y = 1.2 + i * 1.05;
    card(sl, 6.45, y, 3.3, 0.95, C.panel);
    sl.addShape(RECT, {
      x: 6.45, y, w: 0.05, h: 0.95,
      fill: { color: b.c }, line: { color: b.c },
    });
    sl.addText(b.t, {
      x: 6.58, y: y + 0.06, w: 3.1, h: 0.3,
      fontSize: 10.5, bold: true, color: b.c, fontFace: "Calibri", margin: 0,
    });
    sl.addText(b.b, {
      x: 6.58, y: y + 0.36, w: 3.1, h: 0.56,
      fontSize: 8.5, color: C.text, fontFace: "Calibri", margin: 0,
    });
  });

  footer(sl);
}

// ═══════════════════════════════════════════════════════════════════════════
// Slide 6 – Coverage & Trust vs Detections
// ═══════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: C.bg };
  headerBar(sl, "Expert Behaviour vs Detection Count", "Coverage grows and trust calibration sharpens as more detections arrive");

  sl.addImage({
    path: figPath("fig3_coverage.png"),
    x: 0.35, y: 1.1, w: 9.3, h: 4.15,
  });

  // highlight callout
  card(sl, 0.45, 4.75, 4.5, 0.6, C.panelAlt);
  sl.addText("Key insight: ", {
    x: 0.6, y: 4.79, w: 1.05, h: 0.3,
    fontSize: 9.5, bold: true, color: C.orange, fontFace: "Calibri", margin: 0,
  });
  sl.addText("SuperNNova is available on 100% of objects from detection 1, making it the anchor expert for ultra-early classification.", {
    x: 1.63, y: 4.79, w: 3.2, h: 0.48,
    fontSize: 9, color: C.text, fontFace: "Calibri", margin: 0,
  });

  card(sl, 5.15, 4.75, 4.6, 0.6, C.panelAlt);
  sl.addText("Key insight: ", {
    x: 5.3, y: 4.79, w: 1.05, h: 0.3,
    fontSize: 9.5, bold: true, color: C.blue, fontFace: "Calibri", margin: 0,
  });
  sl.addText("Fink SNN trust rises from 0.12→0.44 (n_det 1→20) as the model accumulates photometric evidence to calibrate confidence.", {
    x: 6.33, y: 4.79, w: 3.3, h: 0.48,
    fontSize: 9, color: C.text, fontFace: "Calibri", margin: 0,
  });

  footer(sl);
}

// ═══════════════════════════════════════════════════════════════════════════
// Slide 7 – Follow-up Model Performance
// ═══════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: C.bg };
  headerBar(sl, "Follow-up Priority Head", "Trust-weighted ensemble · LightGBM · 165 features");

  // large AUC callout
  card(sl, 0.45, 1.18, 3.4, 2.1, C.panel);
  sl.addShape(RECT, {
    x: 0.45, y: 1.18, w: 3.4, h: 0.06,
    fill: { color: C.green }, line: { color: C.green },
  });
  sl.addText("ROC-AUC", {
    x: 0.6, y: 1.3, w: 3.1, h: 0.38,
    fontSize: 12, color: C.muted, fontFace: "Calibri", align: "center", margin: 0,
  });
  sl.addText("0.9949", {
    x: 0.6, y: 1.62, w: 3.1, h: 0.95,
    fontSize: 52, bold: true, color: C.green, fontFace: "Calibri",
    align: "center", margin: 0,
  });
  sl.addText("held-out test set (n=5,972)", {
    x: 0.6, y: 2.55, w: 3.1, h: 0.35,
    fontSize: 9, color: C.muted, fontFace: "Calibri", align: "center", margin: 0,
  });

  // secondary metrics
  const secondaryStats = [
    { v: "0.0305", l: "Brier score",         c: C.teal },
    { v: "39.2%",  l: "SN Ia positive rate", c: C.orange },
    { v: "5,918",  l: "training objects",    c: C.blue },
    { v: "165",    l: "total features",      c: C.purple },
  ];
  secondaryStats.forEach((s, i) => {
    const x = 0.45 + (i % 2) * 1.82;
    const y = 3.42 + Math.floor(i / 2) * 0.82;
    statBlock(sl, x, y, 1.68, 0.72, s.v, s.l, s.c);
  });

  // metric card 2x2 for bottom left
  sl.addImage({
    path: figPath("fig6_followup_card.png"),
    x: 4.1, y: 1.18, w: 5.65, h: 3.5,
  });

  footer(sl);
}

// ═══════════════════════════════════════════════════════════════════════════
// Slide 8 – Phase 1 → Current Improvement
// ═══════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: C.bg };
  headerBar(sl, "Model Improvement: Phase 1 → Current", "Expanded training set, additional experts, and dual-survey support");

  sl.addImage({
    path: figPath("fig4_improvement.png"),
    x: 0.45, y: 1.15, w: 5.7, h: 4.1,
  });

  // right: what changed
  const changes = [
    { t: "5,918 → 6,958 objects", d: "+18% more training data including LSST candidates from ALeRCE Rubin beta.", c: C.green },
    { t: "6 → 18 registered experts", d: "Added Pitt-Google BigQuery (ZTF + LSST), ALeRCE 2025β stamp, Fink LSST SNN/CATS/EarlySNIa.", c: C.blue },
    { t: "Dual-survey support", d: "LSST objects query both native LSST and ZTF-supplement brokers via association table (≤2 arcsec).", c: C.teal },
    { t: "51 LC features (was 32)", d: "Added per-band u/g/r/i/z/y statistics, colour slopes, and survey_is_lsst flag.", c: C.orange },
    { t: "Multi-source truth", d: "TNS spectroscopic + Fink crossmatch + host galaxy context (SIMBAD + photo-z + Gaia).", c: C.purple },
  ];

  changes.forEach((ch, i) => {
    const y = 1.2 + i * 0.82;
    card(sl, 6.35, y, 3.4, 0.72, C.panel);
    sl.addShape(RECT, {
      x: 6.35, y: y + 0.06, w: 0.05, h: 0.6,
      fill: { color: ch.c }, line: { color: ch.c },
    });
    sl.addText(ch.t, {
      x: 6.48, y: y + 0.06, w: 3.2, h: 0.28,
      fontSize: 10, bold: true, color: ch.c, fontFace: "Calibri", margin: 0,
    });
    sl.addText(ch.d, {
      x: 6.48, y: y + 0.35, w: 3.2, h: 0.4,
      fontSize: 8.5, color: C.text, fontFace: "Calibri", margin: 0,
    });
  });

  footer(sl);
}

// ═══════════════════════════════════════════════════════════════════════════
// Slide 9 – Expert Registry (18 experts table)
// ═══════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: C.bg };
  headerBar(sl, "18 Registered Expert Classifiers", "Heterogeneous brokers across ZTF and LSST surveys");

  const expertRows = [
    [{ text: "Expert Key", options: { bold: true, color: C.blue } },
     { text: "Survey", options: { bold: true, color: C.blue } },
     { text: "Temporal", options: { bold: true, color: C.blue } },
     { text: "Source", options: { bold: true, color: C.blue } }],
    ["fink/snn",                   "ZTF",  "exact_alert",         "Fink SuperNNova"],
    ["fink/rf_ia",                 "ZTF",  "exact_alert",         "Fink Random Forest"],
    ["fink_lsst/snn",              "LSST", "exact_alert",         "Fink LSST SuperNNova"],
    ["fink_lsst/cats",             "LSST", "exact_alert",         "Fink LSST CATS"],
    ["fink_lsst/early_snia",       "LSST", "exact_alert",         "Fink LSST EarlySNIa"],
    ["alerce/stamp_classifier",    "ZTF",  "static_safe",         "ALeRCE Stamp (ZTF)"],
    ["alerce/stamp_2025_beta",     "ZTF",  "static_safe",         "ALeRCE Stamp 2025β"],
    ["alerce/stamp_rubin_beta",    "LSST", "static_safe",         "ALeRCE Rubin Stamp"],
    ["alerce/lc_classifier",       "ZTF",  "latest_unsafe",       "ALeRCE Legacy LC"],
    ["alerce/BHRF_transient",      "ZTF",  "latest_unsafe",       "ALeRCE BHRF"],
    ["alerce/ATAT (beta)",         "ZTF",  "latest_unsafe",       "ALeRCE Transformer"],
    ["lasair/sherlock",            "any",  "static_safe",         "Lasair Host Context"],
    ["pittgoogle/snn_lsst",        "LSST", "exact_alert",         "PG BigQuery LSST SNN"],
    ["pittgoogle/snn_ztf",         "ZTF",  "latest_unsafe",       "PG BigQuery ZTF SNN"],
    ["parsnip",                    "any",  "rerun_exact",         "Local ParSNIP"],
    ["supernnova",                 "any",  "rerun_exact",         "Local SuperNNova"],
    ["alerce_lc",                  "any",  "rerun_exact",         "Local ALeRCE LC"],
  ];

  // colour-code survey column
  const surveyColor = (s) => {
    if (s === "ZTF")  return C.orange;
    if (s === "LSST") return C.blue;
    return C.muted;
  };

  const tableData = expertRows.map((row, ri) => {
    if (ri === 0) return row;
    return row.map((cell, ci) => {
      const colr = ci === 1 ? surveyColor(cell) : C.text;
      return { text: String(cell), options: { color: colr, fontSize: 9 } };
    });
  });

  sl.addTable(tableData, {
    x: 0.45, y: 1.15, w: 9.1, h: 4.1,
    fontSize: 9,
    fontFace: "Calibri",
    border: { pt: 0.5, color: C.border },
    fill: { color: C.panel },
    color: C.text,
    rowH: 0.228,
    colW: [2.8, 0.75, 1.4, 2.0],
  });

  footer(sl);
}

// ═══════════════════════════════════════════════════════════════════════════
// Slide 10 – What's Next
// ═══════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: C.bg };

  // accent top + left bars
  sl.addShape(RECT, {
    x: 0, y: 0, w: 10, h: 0.08,
    fill: { color: C.teal }, line: { color: C.teal },
  });
  sl.addShape(RECT, {
    x: 0, y: 0, w: 0.18, h: 5.625,
    fill: { color: C.teal }, line: { color: C.teal },
  });

  sl.addText("What's Next", {
    x: 0.5, y: 0.55, w: 9, h: 0.7,
    fontSize: 32, bold: true, color: C.teal, fontFace: "Calibri", margin: 0,
  });
  sl.addText("Roadmap toward full Rubin LSST deployment", {
    x: 0.5, y: 1.22, w: 9, h: 0.35,
    fontSize: 13, color: C.muted, fontFace: "Calibri", margin: 0,
  });

  const nexts = [
    { n: "01", t: "Activate all 18 experts", d: "Train trust heads for all registered experts once LSST Fink, Pitt-Google LSST, and ALeRCE Rubin stamp data volume is sufficient.", c: C.blue },
    { n: "02", t: "LSST-trained local experts", d: "Retrain SuperNNova & ParSNIP on ELAsTiCC simulations for native LSST lightcurve inference.", c: C.teal },
    { n: "03", t: "Real-time nightly pipeline", d: "Deploy submit_nightly.sh on SCC, streaming broker alerts → priority scores within 30 min of alert release.", c: C.orange },
    { n: "04", t: "Calibration study", d: "Formal reliability diagrams for p_follow scores. Brier decomposition: resolution vs. calibration contributions.", c: C.purple },
    { n: "05", t: "TOO integration", d: "Feed p_follow scores into Target-of-Opportunity queue for 4-m spectroscopic telescopes (NTT, P200, Keck).", c: C.green },
    { n: "06", t: "Association quality upgrade", d: "Replace 2-arcsec heuristic with proper Bayesian cross-match using Gaia proper motion and host photo-z priors.", c: C.red },
  ];

  nexts.forEach((item, i) => {
    const col = i % 2;
    const row = Math.floor(i / 3);
    const x = 0.45 + col * 4.75;
    const y = 1.7 + row * 1.88 + (i % 3 >= 3 ? 0 : 0) - Math.floor(i / 3) * 0 + (i % 3) * 1.0;

    // recalculate layout: 3 rows × 2 cols
    const ri = Math.floor(i / 2);
    const ci = i % 2;
    const cx = 0.45 + ci * 4.75;
    const cy = 1.7 + ri * 1.15;

    card(sl, cx, cy, 4.45, 1.05, C.panel);
    sl.addText(item.n, {
      x: cx + 0.12, y: cy + 0.08, w: 0.42, h: 0.42,
      fontSize: 20, bold: true, color: item.c, fontFace: "Calibri",
      align: "center", margin: 0,
    });
    sl.addText(item.t, {
      x: cx + 0.62, y: cy + 0.1, w: 3.7, h: 0.3,
      fontSize: 11, bold: true, color: item.c, fontFace: "Calibri", margin: 0,
    });
    sl.addText(item.d, {
      x: cx + 0.62, y: cy + 0.42, w: 3.7, h: 0.58,
      fontSize: 8.5, color: C.text, fontFace: "Calibri", margin: 0,
    });
  });

  footer(sl, "DEBASS · Rubin Hackathon 2026 · github.com/rubin_hackathon");
}

// ── Write output ───────────────────────────────────────────────────────────
pres.writeFile({ fileName: OUT }).then(() => {
  const kb = Math.round(fs.statSync(OUT).size / 1024);
  console.log(`\nSaved: ${OUT}  (${kb} KB)`);
}).catch(err => {
  console.error("Error:", err);
  process.exit(1);
});
