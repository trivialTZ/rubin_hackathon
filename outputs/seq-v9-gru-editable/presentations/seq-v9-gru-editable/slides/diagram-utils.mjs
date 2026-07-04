export const C = {
  ink: "#17212B",
  muted: "#5B6470",
  grid: "#D9DEE7",
  pale: "#F6F8FB",
  panel: "#FFFFFF",
  teal: "#1C9A8A",
  tealLight: "#DFF4F0",
  blue: "#2B5C88",
  blueLight: "#E3EDF7",
  violet: "#6B5B95",
  violetLight: "#EFEAF8",
  orange: "#C97819",
  orangeLight: "#FAE8CE",
  green: "#2E7D52",
  greenLight: "#E2F3E9",
  red: "#B83A3A",
  redLight: "#F7DEDE",
  grayLight: "#EEF2F6",
};

export function title(slide, ctx, text, subtitle) {
  ctx.addText(slide, {
    x: 54,
    y: 32,
    width: 760,
    height: 40,
    text,
    fontSize: 27,
    bold: true,
    color: C.ink,
    typeface: ctx.fonts.title,
  });
  if (subtitle) {
    ctx.addText(slide, {
      x: 56,
      y: 72,
      width: 1110,
      height: 34,
      text: subtitle,
      fontSize: 14,
      color: C.muted,
      typeface: ctx.fonts.body,
    });
  }
}

export function footer(slide, ctx, text = "metaDEBASS local v9/v9c | editable diagram draft") {
  ctx.addText(slide, {
    x: 56,
    y: 690,
    width: 720,
    height: 20,
    text,
    fontSize: 9.5,
    color: "#7C8591",
    typeface: ctx.fonts.body,
  });
}

export function box(slide, ctx, opts) {
  const {
    x,
    y,
    w,
    h,
    text,
    fill = C.panel,
    stroke = C.grid,
    color = C.ink,
    size = 14,
    bold = false,
    align = "center",
    valign = "mid",
    name,
    typeface = ctx.fonts.body,
    lineWidth = 1.2,
    insets = { left: 8, right: 8, top: 5, bottom: 5 },
  } = opts;
  return ctx.addText(slide, {
    x,
    y,
    width: w,
    height: h,
    text,
    fontSize: size,
    bold,
    color,
    typeface,
    align,
    valign,
    fill,
    line: { fill: stroke, width: lineWidth },
    insets,
    name,
  });
}

export function label(slide, ctx, opts) {
  const {
    x,
    y,
    w,
    h,
    text,
    size = 11,
    color = C.muted,
    bold = false,
    align = "left",
    typeface = ctx.fonts.body,
  } = opts;
  return ctx.addText(slide, {
    x,
    y,
    width: w,
    height: h,
    text,
    fontSize: size,
    color,
    bold,
    typeface,
    align,
    valign: "mid",
  });
}

export function pill(slide, ctx, opts) {
  return box(slide, ctx, {
    ...opts,
    fill: opts.fill ?? C.grayLight,
    stroke: opts.stroke ?? "transparent",
    lineWidth: opts.lineWidth ?? 0,
    size: opts.size ?? 10.5,
    h: opts.h ?? 24,
    insets: { left: 8, right: 8, top: 3, bottom: 3 },
  });
}

export function arrow(slide, ctx, opts) {
  const {
    x,
    y,
    w,
    h,
    dir = "right",
    fill = C.ink,
    labelText,
    labelY,
    labelW,
    labelColor = C.muted,
    labelSize = 9.5,
  } = opts;
  const geometry = {
    right: "rightArrow",
    left: "leftArrow",
    down: "downArrow",
    up: "upArrow",
  }[dir];
  const shape = ctx.addShape(slide, {
    geometry,
    x,
    y,
    width: w,
    height: h,
    fill,
    line: { fill, width: 0 },
  });
  if (labelText) {
    label(slide, ctx, {
      x: x - 12,
      y: labelY ?? y + h + 4,
      w: labelW ?? w + 24,
      h: 18,
      text: labelText,
      size: labelSize,
      color: labelColor,
      align: "center",
    });
  }
  return shape;
}

export function dot(slide, ctx, x, y, color, size = 8, stroke = "#FFFFFF") {
  return ctx.addShape(slide, {
    geometry: "ellipse",
    x: x - size / 2,
    y: y - size / 2,
    width: size,
    height: size,
    fill: color,
    line: { fill: stroke, width: 1 },
  });
}

export function line(slide, ctx, x1, y1, x2, y2, color = C.grid, width = 2) {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  const w = Math.max(1, Math.abs(x2 - x1));
  const h = Math.max(1, Math.abs(y2 - y1));
  return ctx.addShape(slide, {
    geometry: "line",
    x: left,
    y: top,
    width: w,
    height: h,
    fill: "transparent",
    line: { fill: color, width },
  });
}

export function miniLightCurve(slide, ctx, opts = {}) {
  const x = opts.x ?? 74;
  const y = opts.y ?? 156;
  const w = opts.w ?? 184;
  const h = opts.h ?? 104;
  const points = [
    [0.02, 0.78, C.teal],
    [0.10, 0.55, C.orange],
    [0.20, 0.38, C.teal],
    [0.32, 0.25, C.orange],
    [0.44, 0.20, C.teal],
    [0.58, 0.28, C.orange],
    [0.74, 0.46, C.teal],
    [0.88, 0.66, C.orange],
  ];
  box(slide, ctx, {
    x,
    y,
    w,
    h,
    text: "",
    fill: C.pale,
    stroke: C.grid,
    lineWidth: 1,
  });
  line(slide, ctx, x + 24, y + h - 22, x + w - 14, y + h - 22, "#AAB3BE", 1);
  line(slide, ctx, x + 24, y + 12, x + 24, y + h - 22, "#AAB3BE", 1);
  label(slide, ctx, { x: x + 38, y: y + h - 18, w: 92, h: 14, text: "time", size: 8.5, color: C.muted });
  label(slide, ctx, { x: x + 3, y: y + 12, w: 28, h: 14, text: "mag", size: 8.5, color: C.muted });
  let previous = null;
  for (const [px, py, color] of points) {
    const cx = x + 28 + px * (w - 48);
    const cy = y + 18 + py * (h - 48);
    if (previous) line(slide, ctx, previous[0], previous[1], cx, cy, "#C4CBD5", 1.1);
    line(slide, ctx, cx, cy - 11, cx, cy + 11, color, 1);
    dot(slide, ctx, cx, cy, color, 7);
    previous = [cx, cy];
  }
}

export function gruCell(slide, ctx, opts) {
  const { x, y, labelText, fill = C.blueLight, stroke = C.blue, w = 72, h = 42 } = opts;
  return box(slide, ctx, {
    x,
    y,
    w,
    h,
    text: labelText,
    fill,
    stroke,
    color: C.ink,
    size: 12,
    bold: true,
    lineWidth: 1.3,
  });
}

export function bracket(slide, ctx, x, y, w, h, color = C.grid) {
  line(slide, ctx, x, y, x + w, y, color, 1.4);
  line(slide, ctx, x, y + h, x + w, y + h, color, 1.4);
  line(slide, ctx, x, y, x, y + h, color, 1.4);
  line(slide, ctx, x + w, y, x + w, y + h, color, 1.4);
}
