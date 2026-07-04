import { C, arrow, box, bracket, footer, gruCell, label, line, pill, title } from "./diagram-utils.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  title(
    slide,
    ctx,
    "Paper-style diagram: 2-layer GRU encoder and v9 heads",
    "This version separates the recurrent encoder, self-supervised objective, frozen feature export, class head, and trust head.",
  );

  box(slide, ctx, {
    x: 58,
    y: 118,
    w: 170,
    h: 118,
    text: "Prefix input\nX1:N\n\nxt = [LC features,\nband embedding]",
    fill: C.tealLight,
    stroke: C.teal,
    size: 13,
    bold: true,
  });
  pill(slide, ctx, {
    x: 62,
    y: 248,
    w: 162,
    h: 24,
    text: "no future detections",
    fill: C.redLight,
    color: C.red,
    bold: true,
  });

  arrow(slide, ctx, { x: 244, y: 166, w: 56, h: 20, fill: C.teal });
  label(slide, ctx, { x: 252, y: 190, w: 42, h: 16, text: "x1..xN", size: 9.5, align: "center" });

  const xs = [330, 442, 554, 666];
  label(slide, ctx, { x: 386, y: 94, w: 300, h: 22, text: "unrolled recurrent encoder", size: 15, bold: true, color: C.ink, align: "center" });
  label(slide, ctx, { x: 262, y: 154, w: 52, h: 18, text: "layer 1", size: 10.5, bold: true, color: C.blue, align: "right" });
  label(slide, ctx, { x: 262, y: 242, w: 52, h: 18, text: "layer 2", size: 10.5, bold: true, color: C.violet, align: "right" });
  for (let i = 0; i < xs.length; i += 1) {
    const t = i === xs.length - 1 ? "N" : String(i + 1);
    gruCell(slide, ctx, { x: xs[i], y: 142, labelText: `GRU1\nh1_${t}`, fill: C.blueLight, stroke: C.blue });
    gruCell(slide, ctx, { x: xs[i], y: 230, labelText: `GRU2\nh2_${t}`, fill: C.violetLight, stroke: C.violet });
    arrow(slide, ctx, { x: xs[i] + 27, y: 190, w: 18, h: 24, fill: "#A3ACB8", dir: "down" });
    label(slide, ctx, { x: xs[i] + 11, y: 120, w: 50, h: 16, text: `t=${t}`, size: 10, color: C.muted, align: "center" });
    if (i < xs.length - 1) {
      arrow(slide, ctx, { x: xs[i] + 74, y: 157, w: 34, h: 14, fill: C.blue });
      arrow(slide, ctx, { x: xs[i] + 74, y: 245, w: 34, h: 14, fill: C.violet });
    }
  }
  bracket(slide, ctx, 316, 132, 458, 156, "#BFC7D2");
  label(slide, ctx, { x: 450, y: 300, w: 190, h: 18, text: "final recurrent state h2_N", size: 11, color: C.violet, bold: true, align: "center" });

  arrow(slide, ctx, { x: 792, y: 236, w: 54, h: 20, fill: C.violet });
  box(slide, ctx, {
    x: 866,
    y: 206,
    w: 144,
    h: 78,
    text: "projection\nseq_emb\n16 dims",
    fill: C.pale,
    stroke: C.grid,
    size: 12.5,
    bold: true,
  });
  arrow(slide, ctx, { x: 1028, y: 236, w: 54, h: 20, fill: C.ink });
  box(slide, ctx, {
    x: 1102,
    y: 206,
    w: 124,
    h: 78,
    text: "class logits\nsoftmax",
    fill: C.violetLight,
    stroke: C.violet,
    size: 12.5,
    bold: true,
  });

  arrow(slide, ctx, { x: 712, y: 318, w: 22, h: 40, fill: C.orange, dir: "down" });
  box(slide, ctx, {
    x: 560,
    y: 374,
    w: 332,
    h: 110,
    text: "self-supervised pretraining head\n\npredict next-detection dmag and uncertainty\nloss: heteroscedastic Gaussian NLL",
    fill: C.orangeLight,
    stroke: C.orange,
    align: "left",
    valign: "mid",
    size: 13,
    insets: { left: 16, right: 14, top: 12, bottom: 12 },
  });

  arrow(slide, ctx, { x: 922, y: 300, w: 18, h: 50, fill: C.green, dir: "down" });
  box(slide, ctx, {
    x: 918,
    y: 374,
    w: 306,
    h: 110,
    text: "frozen export + trust training\n\nseq_emb_00..15, seq_surprisal, seq_nll_mean\nq_seq_v9 = P(expert is helpful)",
    fill: C.greenLight,
    stroke: C.green,
    align: "left",
    valign: "mid",
    size: 13,
    insets: { left: 16, right: 14, top: 12, bottom: 12 },
  });

  box(slide, ctx, {
    x: 58,
    y: 374,
    w: 456,
    h: 124,
    text: "Training / evaluation contract\n\n• Train split fits encoder and classifier; calibration split handles early stopping / temperature.\n• Test split is not used during training.\n• Trust target is expert helpfulness, not a broker-derived truth shortcut.\n• unsafe latest_object_unsafe snapshots are excluded from trust training.",
    fill: "#FFFFFF",
    stroke: C.grid,
    align: "left",
    valign: "mid",
    size: 11.7,
    insets: { left: 16, right: 14, top: 12, bottom: 12 },
  });

  line(slide, ctx, 184, 540, 1072, 540, "#C9D0DA", 1.4);
  const bottom = [
    ["pretrain", "sequence dynamics", C.orangeLight, C.orange],
    ["freeze/export", "embeddings + surprisal", C.blueLight, C.blue],
    ["fine-tune", "local class probabilities", C.violetLight, C.violet],
    ["calibrate trust", "expert_confidence q", C.greenLight, C.green],
  ];
  let x = 182;
  for (const [head, sub, fill, color] of bottom) {
    arrow(slide, ctx, { x: x - 34, y: 529, w: 54, h: 22, fill: color });
    box(slide, ctx, {
      x,
      y: 514,
      w: 178,
      h: 64,
      text: `${head}\n${sub}`,
      fill,
      stroke: color,
      size: 11.5,
      bold: true,
    });
    x += 240;
  }

  pill(slide, ctx, {
    x: 1022,
    y: 602,
    w: 202,
    h: 34,
    text: "primary product: expert_confidence",
    fill: C.orangeLight,
    color: C.orange,
    bold: true,
    size: 12,
  });
  label(slide, ctx, {
    x: 60,
    y: 608,
    w: 640,
    h: 28,
    text: "Suggested caption: 2-layer causal GRU encoder for local v9; heads produce calibrated local sequence probabilities and trust-calibrated expert confidence.",
    size: 10.5,
    color: C.muted,
  });

  footer(slide, ctx, "metaDEBASS local v9/v9c | paper diagram, native editable shapes");
  return slide;
}
