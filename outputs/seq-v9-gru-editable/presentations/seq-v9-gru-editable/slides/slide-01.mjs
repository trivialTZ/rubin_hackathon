import { C, arrow, box, footer, label, miniLightCurve, pill, title } from "./diagram-utils.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  title(
    slide,
    ctx,
    "seq_v9 as an editable teaching diagram",
    "Local v9 turns the first N detections into a causal sequence embedding, then reports a local expert probability and a trust-calibrated expert_confidence.",
  );

  box(slide, ctx, {
    x: 56,
    y: 118,
    w: 230,
    h: 196,
    text: "",
    fill: "#FFFFFF",
    stroke: C.grid,
  });
  label(slide, ctx, {
    x: 76,
    y: 130,
    w: 190,
    h: 24,
    text: "Causal prefix X1:N",
    size: 15,
    bold: true,
    color: C.ink,
  });
  miniLightCurve(slide, ctx, { x: 76, y: 162, w: 190, h: 106 });
  pill(slide, ctx, {
    x: 74,
    y: 278,
    w: 194,
    h: 24,
    text: "features use detections 1..N only",
    fill: C.tealLight,
    color: C.teal,
    bold: true,
  });

  arrow(slide, ctx, { x: 302, y: 202, w: 66, h: 22, fill: C.teal, labelText: "tokenize" });
  box(slide, ctx, {
    x: 390,
    y: 126,
    w: 178,
    h: 188,
    text: "per-detection tensor\n\ncontinuous: 9 LC features\nband: g/r embedding\nmask: prefix length N",
    fill: C.tealLight,
    stroke: C.teal,
    color: C.ink,
    size: 13,
    bold: false,
    align: "left",
    valign: "mid",
    insets: { left: 15, right: 12, top: 12, bottom: 12 },
  });

  arrow(slide, ctx, { x: 584, y: 202, w: 58, h: 22, fill: C.blue, labelText: "encode" });
  box(slide, ctx, {
    x: 664,
    y: 116,
    w: 270,
    h: 210,
    text: "",
    fill: "#FFFFFF",
    stroke: C.grid,
  });
  label(slide, ctx, {
    x: 688,
    y: 130,
    w: 210,
    h: 24,
    text: "2-layer GRU encoder",
    size: 16,
    bold: true,
    color: C.blue,
  });
  for (const [i, x] of [700, 780, 860].entries()) {
    box(slide, ctx, {
      x,
      y: 170,
      w: 58,
      h: 42,
      text: `GRU1\nt=${i + 1}`,
      fill: C.blueLight,
      stroke: C.blue,
      size: 10.5,
      bold: true,
    });
    box(slide, ctx, {
      x,
      y: 238,
      w: 58,
      h: 42,
      text: `GRU2\nt=${i + 1}`,
      fill: C.violetLight,
      stroke: C.violet,
      size: 10.5,
      bold: true,
    });
    if (i < 2) {
      arrow(slide, ctx, { x: x + 59, y: 183, w: 22, h: 12, fill: C.blue });
      arrow(slide, ctx, { x: x + 59, y: 251, w: 22, h: 12, fill: C.violet });
    }
    arrow(slide, ctx, { x: x + 21, y: 215, w: 16, h: 20, fill: "#9AA5B1", dir: "down" });
  }
  label(slide, ctx, {
    x: 689,
    y: 290,
    w: 210,
    h: 22,
    text: "final state hN becomes seq_emb",
    size: 11,
    color: C.muted,
    align: "center",
  });

  arrow(slide, ctx, { x: 952, y: 202, w: 58, h: 22, fill: C.violet, labelText: "heads" });
  box(slide, ctx, {
    x: 1032,
    y: 116,
    w: 190,
    h: 78,
    text: "seq_v9 class head\np(snia), p(nonIa-like), p(other)",
    fill: C.violetLight,
    stroke: C.violet,
    size: 12.5,
    bold: true,
  });
  box(slide, ctx, {
    x: 1032,
    y: 214,
    w: 190,
    h: 78,
    text: "trust head\nq_seq_v9 = P(helpful | prefix)",
    fill: C.greenLight,
    stroke: C.green,
    size: 12.5,
    bold: true,
  });
  pill(slide, ctx, {
    x: 1032,
    y: 310,
    w: 190,
    h: 30,
    text: "reported as expert_confidence",
    fill: C.orangeLight,
    color: C.orange,
    bold: true,
  });

  label(slide, ctx, {
    x: 70,
    y: 374,
    w: 260,
    h: 30,
    text: "Training story to explain on the slide",
    size: 16,
    bold: true,
    color: C.ink,
  });
  const steps = [
    ["1", "self-supervised pretrain", "predict next-detection dmag with Gaussian NLL", C.orangeLight, C.orange],
    ["2", "feature export", "freeze encoder; export seq_emb_00..15, surprisal, nll_mean", C.blueLight, C.blue],
    ["3", "supervised classifier", "calibrated ternary local expert: snia / nonIa-like / other", C.violetLight, C.violet],
    ["4", "trust calibration", "learn q_seq_v9 against helpfulness target; feed metaDEBASS", C.greenLight, C.green],
  ];
  let y = 414;
  for (const [num, head, body, fill, color] of steps) {
    pill(slide, ctx, {
      x: 76,
      y: y + 4,
      w: 34,
      h: 26,
      text: num,
      fill,
      color,
      bold: true,
      size: 12,
    });
    label(slide, ctx, { x: 124, y, w: 220, h: 20, text: head, size: 13, bold: true, color: C.ink });
    label(slide, ctx, { x: 124, y: y + 21, w: 420, h: 19, text: body, size: 10.2, color: C.muted });
    y += 56;
  }

  box(slide, ctx, {
    x: 590,
    y: 384,
    w: 294,
    h: 180,
    text: "What the audience should hear\n\n• The GRU is local-v9, not the final science posterior.\n• Each point on a causal-prefix plot means: rerun the model using only the first N detections.\n• q_seq_v9 is the model’s estimate that this expert is currently reliable.",
    fill: C.pale,
    stroke: C.grid,
    align: "left",
    valign: "mid",
    size: 13,
    insets: { left: 16, right: 14, top: 13, bottom: 13 },
  });

  box(slide, ctx, {
    x: 914,
    y: 384,
    w: 278,
    h: 180,
    text: "Numbers to annotate verbally\n\n• GRU hidden size: 64\n• 2 recurrent layers\n• sequence embedding: 16 dims\n• local train excludes cal/test objects\n• unsafe backfilled broker snapshots are excluded from trust training by default",
    fill: "#FFFFFF",
    stroke: C.grid,
    align: "left",
    valign: "mid",
    size: 12.5,
    insets: { left: 16, right: 14, top: 13, bottom: 13 },
  });

  footer(slide, ctx);
  return slide;
}
