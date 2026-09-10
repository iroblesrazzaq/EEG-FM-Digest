(function (global) {
  const FONT =
    '"Avenir Next", "Segoe UI", "Helvetica Neue", Helvetica, sans-serif';
  const VIEW_W = 820;
  const CHASSIS_X = 168;
  const CHASSIS_W = 268;
  const PILL_W = 176;
  const PILL_H = 24;
  const ATTN_H = 28;
  const ADD_R = 9.5;
  const INNER_GAP = 16;
  const STACK_GAP = 18;
  const BLOCK_PAD_X = 30;
  const BLOCK_PAD_Y = 18;
  const CHASSIS_PAD_Y = 26;
  const CALLOUT_X = 500;
  const CALLOUT_W = 250;

  function cssVar(name, fallback) {
    if (typeof window === "undefined") {
      return fallback;
    }
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function palette() {
    return {
      ink: cssVar("--ink", "#1f1d1a"),
      muted: cssVar("--muted", "#6a645e"),
      line: cssVar("--line", "#e7e3dd"),
      accent: cssVar("--accent", "#7d6a52"),
      accentDeep: cssVar("--accent-deep", "#5f503f"),
      accentSoft: cssVar("--accent-soft", "#f2eee7"),
      surface: cssVar("--surface", "#ffffff"),
      chassis: "#e6e3de",
      block: "#cbb892",
      attention: cssVar("--ink", "#1f1d1a"),
      leader: "#8d8680",
      callout: "#9a948c",
    };
  }

  function asArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function commas(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) {
      return String(value);
    }
    return Math.round(num).toLocaleString("en-US");
  }

  function svgEl(name, attrs, text) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value != null && value !== "") {
        el.setAttribute(key, String(value));
      }
    }
    if (text != null) {
      el.textContent = text;
    }
    return el;
  }

  function wrapLines(text, maxChars) {
    const raw = String(text || "").trim();
    if (!raw) {
      return [];
    }
    if (raw.includes("\n")) {
      return raw.split(/\n/).map((line) => line.trim()).filter(Boolean);
    }
    if (raw.length <= maxChars) {
      return [raw];
    }
    const words = raw.split(/\s+/);
    const lines = [];
    let current = "";
    for (const word of words) {
      const next = current ? `${current} ${word}` : word;
      if (next.length > maxChars && current) {
        lines.push(current);
        current = word;
      } else {
        current = next;
      }
    }
    if (current) {
      lines.push(current);
    }
    return lines.length ? lines : [raw];
  }

  function galleryLabel(node) {
    const kind = node && node.kind;
    const label = String((node && (node.label || node.id)) || "");
    if (kind === "input" && /eeg/i.test(label)) {
      return "Sample EEG";
    }
    if ((kind === "stem" || kind === "embed") && /patch/i.test(label)) {
      return "Patch embedding layer";
    }
    if (kind === "attn" || kind === "attention") {
      if (/grouped|gqa/i.test(label)) {
        return "Masked grouped-query attention";
      }
      if (/mha|multi-head|attention/i.test(label)) {
        return "Multi-head attention";
      }
      return label || "Multi-head attention";
    }
    if (kind === "mlp" || kind === "ffn") {
      return "Feed forward";
    }
    if (kind === "output" || kind === "linear" || kind === "head") {
      if (/embed/i.test(label) || /output/i.test(label) || label === "Embedding") {
        return "Linear output layer";
      }
    }
    return label;
  }

  function activationFrom(node) {
    const blob = `${(node && node.label) || ""} ${(node && node.detail) || ""}`;
    if (/swiglu|silu|swish/i.test(blob)) {
      return "SiLU";
    }
    if (/geglu|gelu/i.test(blob)) {
      return "GELU";
    }
    return "GELU";
  }

  function parseHeads(node, sheet) {
    const fromSheet = Number(sheet && sheet.num_attention_heads);
    if (Number.isFinite(fromSheet) && fromSheet > 0) {
      return fromSheet;
    }
    const label = String((node && node.label) || "");
    const match = label.match(/(\d+)\s*[×x]/);
    if (match) {
      return Number(match[1]);
    }
    return null;
  }

  function hiddenFrom(node, sheet) {
    const embed = Number(sheet && sheet.hidden_size);
    const detail = String((node && node.detail) || "");
    const ratio = detail.match(/ratio\s+([0-9.]+)/i);
    if (Number.isFinite(embed) && embed > 0 && ratio) {
      return Math.max(1, Math.round(embed * Number(ratio[1])));
    }
    return null;
  }

  function ffnTitleFrom(node) {
    const blob = `${(node && node.label) || ""} ${(node && node.detail) || ""}`;
    if (/geglu/i.test(blob)) {
      return "FeedForward (GeGLU) module";
    }
    if (/swiglu/i.test(blob)) {
      return "FeedForward (SwiGLU) module";
    }
    return "FeedForward module";
  }

  function numberNorms(steps) {
    const norms = steps.filter((step) => step.kind === "norm");
    if (norms.length < 2) {
      return steps;
    }
    let index = 1;
    return steps.map((step) => {
      if (step.kind !== "norm") {
        return step;
      }
      const label = String(step.label || "RMSNorm");
      const numbered = /\d/.test(label) ? label : `${label} ${index}`;
      index += 1;
      return Object.assign({}, step, { label: numbered });
    });
  }

  function synthesizeDiagram(graph) {
    const nodes = asArray(graph.nodes);
    const sheet = graph.fact_sheet || {};
    const repeatNode = nodes.find((node) => node && (node.kind === "repeat" || Number(node.repeat) > 1));
    const idx = repeatNode ? nodes.indexOf(repeatNode) : -1;
    const before = idx >= 0 ? nodes.slice(0, idx) : [];
    const after = idx >= 0 ? nodes.slice(idx + 1) : nodes.filter((node) => node !== repeatNode);
    const below = [];
    const stem = [];
    const left = [];
    let attnChild = null;
    let mlpChild = null;

    for (const node of before) {
      if (!node) {
        continue;
      }
      if (node.kind === "input") {
        below.push({ id: node.id, label: galleryLabel(node), kind: "input" });
      } else if (node.kind === "posenc") {
        left.push({
          id: node.id,
          label: node.label || "Positional encoding",
          anchor: null,
        });
      } else {
        stem.push({ id: node.id, label: galleryLabel(node), kind: "embed" });
      }
    }

    let steps = [];
    if (repeatNode) {
      const children = asArray(repeatNode.children).filter((child) => child && child.kind !== "residual");
      steps = children.map((child) => {
        if (child.kind === "attn") {
          attnChild = child;
          return { id: child.id, label: galleryLabel(child), kind: "attention" };
        }
        if (child.kind === "mlp") {
          mlpChild = child;
          return { id: child.id, label: galleryLabel(child), kind: "ffn" };
        }
        return { id: child.id, label: child.label || child.id, kind: child.kind || "module" };
      });
      steps = numberNorms(steps);
      if (!steps.some((step) => step.kind === "add")) {
        steps.push({ id: `${repeatNode.id}.add`, label: "+", kind: "add" });
      }
    }

    if (left.length && !left[0].anchor) {
      left[0].anchor = (attnChild && attnChild.id) || (repeatNode && repeatNode.id) || (stem[0] && stem[0].id);
    }

    const head = after.map((node) => ({
      id: node.id,
      label: galleryLabel(node),
      kind: node.kind === "output" ? "linear" : node.kind || "module",
    }));

    const callouts = [];
    if (mlpChild) {
      callouts.push({
        id: "ffn-mod",
        kind: "ffn",
        anchor: mlpChild.id,
        title: ffnTitleFrom(mlpChild),
        activation: activationFrom(mlpChild),
        hidden_dim: hiddenFrom(mlpChild, sheet),
      });
    }
    const heads = parseHeads(attnChild, sheet);
    if (heads && attnChild) {
      callouts.push({
        id: "attn-heads",
        kind: "heads",
        anchor: attnChild.id,
        label: `${heads} heads`,
      });
    }

    const title = String(graph.label || graph.diagram?.title || "Model").split(":")[0].trim();
    const embed = Number(sheet.hidden_size);
    return {
      title,
      param_label: null,
      below,
      stem,
      repeat: repeatNode
        ? { id: repeatNode.id, count: Number(repeatNode.repeat) || 1, steps }
        : { id: "block", count: 1, steps },
      head,
      callouts,
      annotations: {
        embed_dim: Number.isFinite(embed) && embed > 0 ? embed : null,
        left,
      },
    };
  }

  function resolveDiagram(graph) {
    if (graph && graph.diagram && typeof graph.diagram === "object") {
      const diagram = graph.diagram;
      return {
        title: diagram.title || graph.label || "Model",
        param_label: diagram.param_label || null,
        below: asArray(diagram.below),
        stem: asArray(diagram.stem),
        repeat: diagram.repeat || { count: 1, steps: [] },
        head: asArray(diagram.head),
        callouts: asArray(diagram.callouts),
        annotations: diagram.annotations || {},
      };
    }
    return synthesizeDiagram(graph);
  }

  function stepSize(step) {
    if (!step) {
      return { w: PILL_W, h: PILL_H };
    }
    if (step.kind === "add") {
      return { w: ADD_R * 2, h: ADD_R * 2 };
    }
    if (step.kind === "attention") {
      return { w: PILL_W, h: ATTN_H };
    }
    return { w: PILL_W, h: PILL_H };
  }

  function stackHeight(items, gap, sizeFn) {
    let height = 0;
    items.forEach((item, index) => {
      height += sizeFn(item).h;
      if (index < items.length - 1) {
        height += gap;
      }
    });
    return height;
  }

  function placeUp(items, bottom, centerX, gap, sizeFn) {
    const placed = [];
    let cursor = bottom;
    for (let i = 0; i < items.length; i += 1) {
      const item = items[i];
      const size = sizeFn(item);
      const y = cursor - size.h;
      placed.push({
        id: item.id,
        label: item.label,
        kind: item.kind,
        x: centerX - size.w / 2,
        y,
        w: size.w,
        h: size.h,
        cx: centerX,
        cy: y + size.h / 2,
      });
      cursor = y - gap;
    }
    return placed;
  }

  function arrowUp(parent, x, yBottom, yTop, colors) {
    const head = 5;
    const tip = yTop;
    const base = yBottom;
    if (base - tip < 4) {
      return;
    }
    parent.appendChild(
      svgEl("line", {
        x1: x,
        y1: base,
        x2: x,
        y2: tip + head - 0.4,
        stroke: colors.ink,
        "stroke-width": 1,
      }),
    );
    parent.appendChild(
      svgEl("polygon", {
        points: `${x},${tip} ${x - 2.8},${tip + head} ${x + 2.8},${tip + head}`,
        fill: colors.ink,
      }),
    );
  }

  function connectColumn(parent, boxes, colors) {
    for (let i = 0; i < boxes.length - 1; i += 1) {
      const lower = boxes[i];
      const upper = boxes[i + 1];
      arrowUp(parent, lower.cx, lower.y, upper.y + upper.h, colors);
    }
  }

  function drawPill(parent, box, colors, opts) {
    const options = opts || {};
    const dark = Boolean(options.dark);
    const rx = options.rx != null ? options.rx : 8;
    const shape = svgEl("rect", {
      class: "arch-pill-shape",
      x: box.x,
      y: box.y,
      width: box.w,
      height: box.h,
      rx,
      ry: rx,
      fill: dark ? colors.attention : colors.surface,
      stroke: dark ? colors.attention : colors.ink,
      "stroke-width": dark ? 1 : 1.15,
    });
    parent.appendChild(shape);
    const lines = wrapLines(box.label, options.maxChars || (dark ? 16 : 22));
    const fontSize = options.fontSize || (dark ? 11 : 11);
    const lineH = fontSize + 2;
    const total = (lines.length - 1) * lineH;
    const startY = box.y + box.h / 2 - total / 2 + fontSize / 2.6;
    lines.forEach((line, index) => {
      parent.appendChild(
        svgEl(
          "text",
          {
            x: box.cx,
            y: startY + index * lineH,
            "text-anchor": "middle",
            fill: dark ? colors.surface : colors.ink,
            "font-size": fontSize,
            "font-weight": dark ? 600 : 500,
            "font-family": FONT,
          },
          line,
        ),
      );
    });
    return shape;
  }

  function drawAdd(parent, box, colors) {
    const cx = box.cx;
    const cy = box.cy;
    const r = Math.min(box.w, box.h) / 2;
    parent.appendChild(
      svgEl("circle", {
        cx,
        cy,
        r,
        fill: colors.surface,
        stroke: colors.ink,
        "stroke-width": 1.2,
      }),
    );
    const inset = r - 4;
    parent.appendChild(
      svgEl("line", {
        x1: cx - inset,
        y1: cy,
        x2: cx + inset,
        y2: cy,
        stroke: colors.ink,
        "stroke-width": 1.2,
        "stroke-linecap": "round",
      }),
    );
    parent.appendChild(
      svgEl("line", {
        x1: cx,
        y1: cy - inset,
        x2: cx,
        y2: cy + inset,
        stroke: colors.ink,
        "stroke-width": 1.2,
        "stroke-linecap": "round",
      }),
    );
  }

  function drawBox(parent, box, colors) {
    const group = svgEl("g", { class: `arch-step arch-step-${box.kind || "module"}` });
    if (box.id) {
      group.setAttribute("data-arch-id", box.id);
    }
    if (box.kind === "add") {
      drawAdd(group, box, colors);
    } else {
      drawPill(group, box, colors, { dark: box.kind === "attention", maxChars: box.kind === "attention" ? 22 : 28 });
    }
    parent.appendChild(group);
    return group;
  }

  function curlyBrace(parent, x, y1, y2, colors) {
    const mid = (y1 + y2) / 2;
    const depth = 9;
    const cusp = 7;
    const path = [
      `M ${x} ${y1}`,
      `C ${x - depth * 0.4} ${y1}, ${x - depth} ${y1 + 6}, ${x - depth} ${y1 + 16}`,
      `L ${x - depth} ${mid - 11}`,
      `C ${x - depth} ${mid - 3}, ${x - depth - cusp} ${mid}, ${x - depth - cusp - 2} ${mid}`,
      `C ${x - depth - cusp} ${mid}, ${x - depth} ${mid + 3}, ${x - depth} ${mid + 11}`,
      `L ${x - depth} ${y2 - 16}`,
      `C ${x - depth} ${y2 - 6}, ${x - depth * 0.4} ${y2}, ${x} ${y2}`,
    ].join(" ");
    parent.appendChild(
      svgEl("path", {
        d: path,
        fill: "none",
        stroke: colors.ink,
        "stroke-width": 1.2,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
      }),
    );
  }

  function leader(parent, x1, y1, x2, y2, colors) {
    parent.appendChild(
      svgEl("path", {
        d: `M ${x1} ${y1} L ${x2} ${y2}`,
        fill: "none",
        stroke: colors.leader,
        "stroke-width": 1,
        "stroke-dasharray": "2.2 2.4",
      }),
    );
  }

  function ffnIsGated(spec) {
    if (!spec) {
      return true;
    }
    if (spec.gated === false) {
      return false;
    }
    if (spec.gated === true) {
      return true;
    }
    return /glu/i.test(String(spec.title || ""));
  }

  function drawHiddenDim(group, spec, cx, y, h, colors) {
    if (!spec.hidden_dim) {
      return;
    }
    const lines = ["Hidden layer", `dimension of ${commas(spec.hidden_dim)}`];
    lines.forEach((line, index) => {
      group.appendChild(
        svgEl(
          "text",
          {
            x: cx,
            y: y + h + 34 + index * 15,
            "text-anchor": "middle",
            fill: colors.accent,
            "font-size": 12,
            "font-weight": 700,
            "font-family": FONT,
          },
          line,
        ),
      );
    });
  }

  function drawUngatedFfn(parent, spec, x, y, colors, focused) {
    const w = CALLOUT_W;
    const h = 118;
    const group = svgEl("g", { class: "arch-callout-ffn" });
    if (spec.title) {
      group.appendChild(
        svgEl(
          "text",
          {
            x,
            y: y - 8,
            fill: colors.accentDeep,
            "font-size": 12,
            "font-weight": 700,
            "font-family": FONT,
          },
          spec.title,
        ),
      );
    }
    group.appendChild(
      svgEl("rect", {
        x,
        y,
        width: w,
        height: h,
        rx: 16,
        ry: 16,
        fill: colors.surface,
        stroke: focused ? colors.accent : colors.callout,
        "stroke-width": focused ? 1.7 : 1.2,
        "stroke-dasharray": "5 3.5",
      }),
    );
    const pillW = 108;
    const pillH = 20;
    const cx = x + w / 2;
    const top = { x: cx - pillW / 2, y: y + 14, w: pillW, h: pillH, cx, cy: y + 14 + pillH / 2, label: "Linear layer" };
    const bot = { x: cx - pillW / 2, y: y + 84, w: pillW, h: pillH, cx, cy: y + 84 + pillH / 2, label: "Linear layer" };
    const act = {
      x: cx - 48,
      y: y + 48,
      w: 96,
      h: 22,
      cx,
      cy: y + 59,
      label: `${spec.activation || "GELU"} activation`,
    };
    drawPill(group, top, colors, { fontSize: 10, rx: 7, maxChars: 18 });
    drawPill(group, bot, colors, { fontSize: 10, rx: 7, maxChars: 18 });
    group.appendChild(
      svgEl("ellipse", {
        cx: act.cx,
        cy: act.cy,
        rx: act.w / 2,
        ry: act.h / 2,
        fill: colors.surface,
        stroke: colors.ink,
        "stroke-width": 1.1,
      }),
    );
    group.appendChild(
      svgEl(
        "text",
        {
          x: act.cx,
          y: act.cy + 3.5,
          "text-anchor": "middle",
          fill: colors.ink,
          "font-size": 8.5,
          "font-family": FONT,
        },
        act.label,
      ),
    );
    arrowUp(group, cx, top.y + top.h, act.cy - act.h / 2, colors);
    arrowUp(group, cx, act.cy + act.h / 2, bot.y, colors);
    drawHiddenDim(group, spec, cx, y, h, colors);
    parent.appendChild(group);
    return { x, y, w, h: h + (spec.hidden_dim ? 56 : 0), cx };
  }

  function drawGatedFfn(parent, spec, x, y, colors, focused) {
    const w = CALLOUT_W;
    const h = 148;
    const group = svgEl("g", { class: "arch-callout-ffn" });
    if (spec.title) {
      group.appendChild(
        svgEl(
          "text",
          {
            x,
            y: y - 8,
            fill: colors.accentDeep,
            "font-size": 12,
            "font-weight": 700,
            "font-family": FONT,
          },
          spec.title,
        ),
      );
    }
    group.appendChild(
      svgEl("rect", {
        x,
        y,
        width: w,
        height: h,
        rx: 16,
        ry: 16,
        fill: colors.surface,
        stroke: focused ? colors.accent : colors.callout,
        "stroke-width": focused ? 1.7 : 1.2,
        "stroke-dasharray": "5 3.5",
      }),
    );
    const pillW = 108;
    const pillH = 20;
    const cx = x + w / 2;
    const top = { x: cx - pillW / 2, y: y + 16, w: pillW, h: pillH, cx, cy: y + 16 + pillH / 2, label: "Linear layer" };
    const bot = { x: cx - pillW / 2, y: y + 112, w: pillW, h: pillH, cx, cy: y + 112 + pillH / 2, label: "Linear layer" };
    const mul = { cx, cy: y + 74, r: 8 };
    const actW = 74;
    const actH = 20;
    const act = {
      x: x + 14,
      y: mul.cy - actH / 2,
      w: actW,
      h: actH,
      cx: x + 14 + actW / 2,
      cy: mul.cy,
      label: `${spec.activation || "GELU"} activation`,
    };
    const sideW = 96;
    const side = {
      x: x + w - 14 - sideW,
      y: mul.cy - pillH / 2,
      w: sideW,
      h: pillH,
      cx: x + w - 14 - sideW / 2,
      cy: mul.cy,
      label: "Linear layer",
    };
    drawPill(group, top, colors, { fontSize: 10, rx: 7, maxChars: 18 });
    drawPill(group, bot, colors, { fontSize: 10, rx: 7, maxChars: 18 });
    drawPill(group, side, colors, { fontSize: 10, rx: 7, maxChars: 18 });
    group.appendChild(
      svgEl("ellipse", {
        cx: act.cx,
        cy: act.cy,
        rx: act.w / 2,
        ry: act.h / 2,
        fill: colors.surface,
        stroke: colors.ink,
        "stroke-width": 1.1,
      }),
    );
    group.appendChild(
      svgEl(
        "text",
        {
          x: act.cx,
          y: act.cy + 3.5,
          "text-anchor": "middle",
          fill: colors.ink,
          "font-size": 8.5,
          "font-family": FONT,
        },
        act.label,
      ),
    );
    group.appendChild(
      svgEl("circle", {
        cx: mul.cx,
        cy: mul.cy,
        r: mul.r,
        fill: colors.surface,
        stroke: colors.ink,
        "stroke-width": 1.15,
      }),
    );
    group.appendChild(
      svgEl(
        "text",
        {
          x: mul.cx,
          y: mul.cy + 3.6,
          "text-anchor": "middle",
          fill: colors.ink,
          "font-size": 11,
          "font-family": FONT,
        },
        "×",
      ),
    );
    arrowUp(group, cx, top.y + top.h, mul.cy - mul.r - 1, colors);
    arrowUp(group, cx, mul.cy + mul.r + 1, bot.y, colors);
    group.appendChild(
      svgEl("line", {
        x1: act.x + act.w,
        y1: act.cy,
        x2: mul.cx - mul.r,
        y2: mul.cy,
        stroke: colors.ink,
        "stroke-width": 1.05,
      }),
    );
    group.appendChild(
      svgEl("line", {
        x1: side.x,
        y1: side.cy,
        x2: mul.cx + mul.r,
        y2: mul.cy,
        stroke: colors.ink,
        "stroke-width": 1.05,
      }),
    );
    drawHiddenDim(group, spec, cx, y, h, colors);
    parent.appendChild(group);
    return { x, y, w, h: h + (spec.hidden_dim ? 56 : 0), cx };
  }

  function drawFfnCallout(parent, spec, x, y, colors, focused) {
    if (ffnIsGated(spec)) {
      return drawGatedFfn(parent, spec, x, y, colors, focused);
    }
    return drawUngatedFfn(parent, spec, x, y, colors, focused);
  }

  function draw(host, graph) {
    const colors = palette();
    const src = host.getAttribute("data-arch-src") || "";
    const focused = focusedBySrc.get(src) || "";
    const diagram = resolveDiagram(graph);
    const repeat = diagram.repeat || { count: 1, steps: [] };
    const steps = asArray(repeat.steps);
    const stem = asArray(diagram.stem);
    const head = asArray(diagram.head);
    const below = asArray(diagram.below);
    const annotations = diagram.annotations || {};
    const leftNotes = asArray(annotations.left);

    const cx = CHASSIS_X + CHASSIS_W / 2;
    const stemH = stackHeight(stem, INNER_GAP, stepSize);
    const headH = stackHeight(head, INNER_GAP, stepSize);
    const stepsH = stackHeight(steps, INNER_GAP, stepSize);
    const blockH = stepsH + BLOCK_PAD_Y * 2;
    const chassisInner =
      (stem.length ? stemH + STACK_GAP : 0) +
      (steps.length ? blockH : 0) +
      (head.length ? STACK_GAP + headH : 0);
    const chassisH = Math.max(220, chassisInner + CHASSIS_PAD_Y * 2);
    const belowH = below.length ? PILL_H : 0;
    const titleH = 52;
    const footerH = annotations.embed_dim ? 36 : 18;
    const viewH = titleH + chassisH + (below.length ? 28 + belowH : 12) + footerH;

    const chassisTop = titleH;
    const chassisBottom = chassisTop + chassisH;
    const innerBottom = chassisBottom - CHASSIS_PAD_Y;

    const svg = svgEl("svg", {
      class: "arch-graph-svg",
      viewBox: `0 0 ${VIEW_W} ${viewH}`,
      width: "100%",
      preserveAspectRatio: "xMidYMin meet",
      role: "img",
      "aria-label": `${diagram.title || "Model"} architecture`,
    });
    svg.appendChild(
      svgEl(
        "title",
        {},
        `${diagram.title || "Model"}${diagram.param_label ? ` ${diagram.param_label}` : ""} architecture`,
      ),
    );

    const titleText = [diagram.title, diagram.param_label].filter(Boolean).join(" ");
    svg.appendChild(
      svgEl(
        "text",
        {
          x: CHASSIS_X,
          y: 34,
          fill: colors.accent,
          "font-size": 24,
          "font-weight": 700,
          "font-family": FONT,
        },
        titleText,
      ),
    );

    svg.appendChild(
      svgEl("rect", {
        x: CHASSIS_X,
        y: chassisTop,
        width: CHASSIS_W,
        height: chassisH,
        rx: 42,
        ry: 42,
        fill: colors.chassis,
      }),
    );

    const blockBottom = innerBottom - (stem.length ? stemH + STACK_GAP : 0);
    const blockTop = blockBottom - (steps.length ? blockH : 0);
    const blockX = CHASSIS_X + BLOCK_PAD_X;
    const blockW = CHASSIS_W - BLOCK_PAD_X * 2;
    const headBottom = (steps.length ? blockTop : innerBottom - (stem.length ? stemH + STACK_GAP : 0)) - (head.length ? STACK_GAP : 0);
    const headPlaced = placeUp(head, headBottom, cx, INNER_GAP, stepSize);

    if (steps.length) {
      svg.appendChild(
        svgEl("rect", {
          class: "arch-repeat-block",
          x: blockX,
          y: blockTop,
          width: blockW,
          height: blockH,
          rx: 28,
          ry: 28,
          fill: colors.block,
        }),
      );
    }

    const stepPlaced = placeUp(steps, blockBottom - BLOCK_PAD_Y, cx, INNER_GAP, stepSize);
    const stemPlaced = placeUp(stem, innerBottom, cx, INNER_GAP, stepSize);

    const byId = {};
    [...stemPlaced, ...stepPlaced, ...headPlaced].forEach((box) => {
      if (box.id) {
        byId[box.id] = box;
      }
    });

    if (steps.length) {
      const braceX = blockX - 2;
      const attnBox = stepPlaced.find((box) => box.kind === "attention");
      const firstBox = stepPlaced[0];
      const braceTop = attnBox ? attnBox.y + attnBox.h + 2 : blockTop + 20;
      const braceBot = firstBox ? firstBox.y + firstBox.h + 8 : blockBottom - 10;
      curlyBrace(svg, braceX, Math.min(braceTop, braceBot), Math.max(braceTop, braceBot), colors);
      svg.appendChild(
        svgEl(
          "text",
          {
            x: braceX - 12,
            y: (Math.min(braceTop, braceBot) + Math.max(braceTop, braceBot)) / 2 + 4,
            "text-anchor": "end",
            fill: colors.ink,
            "font-size": 13,
            "font-weight": 600,
            "font-family": FONT,
          },
          `${Number(repeat.count) || 1} ×`,
        ),
      );
    }

    const flow = [...stemPlaced, ...stepPlaced, ...headPlaced];
    connectColumn(svg, flow, colors);
    if (below.length && stemPlaced.length) {
      const sample = {
        id: below[0].id,
        label: below[0].label,
        kind: below[0].kind || "input",
        w: 132,
        h: 24,
        x: cx - 66,
        y: chassisBottom + 18,
        cx,
        cy: chassisBottom + 18 + 12,
      };
      arrowUp(svg, cx, sample.y, stemPlaced[0].y + stemPlaced[0].h, colors);
      drawBox(svg, sample, colors);
      byId[sample.id] = sample;
    } else if (below.length) {
      const sample = {
        id: below[0].id,
        label: below[0].label,
        kind: "input",
        w: 132,
        h: 24,
        x: cx - 66,
        y: chassisBottom + 18,
        cx,
        cy: chassisBottom + 30,
      };
      drawBox(svg, sample, colors);
    }

    stemPlaced.forEach((box) => drawBox(svg, box, colors));
    stepPlaced.forEach((box) => {
      const group = drawBox(svg, box, colors);
      if (box.kind === "ffn" || box.kind === "attention") {
        group.setAttribute("tabindex", "0");
        group.setAttribute("role", "button");
        group.setAttribute("data-arch-toggle", box.id);
        group.style.cursor = "pointer";
        group.setAttribute(
          "aria-label",
          box.kind === "ffn" ? "Highlight feed-forward module" : "Highlight attention details",
        );
      }
    });
    headPlaced.forEach((box) => drawBox(svg, box, colors));

    const callouts = asArray(diagram.callouts);
    const ffnSpec = callouts.find((item) => item && item.kind === "ffn");
    const headsSpec = callouts.find((item) => item && item.kind === "heads");
    let ffnBox = null;
    if (ffnSpec) {
      const anchor = byId[ffnSpec.anchor] || stepPlaced.find((box) => box.kind === "ffn");
      const ffnY = Math.max(chassisTop - 2, (anchor ? anchor.y : blockTop) - 28);
      ffnBox = drawFfnCallout(svg, ffnSpec, CALLOUT_X, ffnY, colors, focused === (anchor && anchor.id));
      if (anchor) {
        leader(svg, anchor.x + anchor.w, anchor.cy, CALLOUT_X - 2, ffnY + 24, colors);
      }
    }
    if (headsSpec) {
      const anchor = byId[headsSpec.anchor] || stepPlaced.find((box) => box.kind === "attention");
      const hx = CALLOUT_X;
      const hy = ffnBox ? ffnBox.y + 148 + 16 : (anchor ? anchor.cy + 4 : blockTop + 80);
      svg.appendChild(
        svgEl(
          "text",
          {
            x: hx,
            y: hy,
            fill: colors.ink,
            "font-size": 12,
            "font-weight": 650,
            "font-family": FONT,
          },
          headsSpec.label || "",
        ),
      );
      if (anchor) {
        leader(svg, anchor.x + anchor.w, anchor.cy, hx - 4, hy - 4, colors);
      }
    }

    leftNotes.forEach((note) => {
      const anchor = byId[note.anchor] || stepPlaced.find((box) => box.kind === "attention") || stemPlaced[0];
      if (!anchor) {
        return;
      }
      const isShort = note.id === "pe" || /^(fourier|rope)/i.test(String(note.label || ""));
      const lines = wrapLines(note.label, isShort ? 16 : 18);
      const textX = isShort ? CHASSIS_X - 14 : CHASSIS_X - 72;
      const textY = anchor.cy - ((lines.length - 1) * 7);
      lines.forEach((line, index) => {
        svg.appendChild(
          svgEl(
            "text",
            {
              x: textX,
              y: textY + index * 14,
              "text-anchor": "end",
              fill: colors.ink,
              "font-size": 11,
              "font-weight": 600,
              "font-family": FONT,
            },
            line,
          ),
        );
      });
      leader(svg, textX + 6, anchor.cy, anchor.x - 2, anchor.cy, colors);
    });

    if (annotations.vocab_size) {
      const anchor =
        [...headPlaced].reverse().find((box) => box.kind === "linear") ||
        headPlaced[headPlaced.length - 1];
      const vx = VIEW_W - 16;
      const vy = 32;
      svg.appendChild(
        svgEl(
          "text",
          {
            x: vx,
            y: vy,
            "text-anchor": "end",
            fill: colors.ink,
            "font-size": 12,
            "font-weight": 600,
            "font-family": FONT,
          },
          `Vocabulary size of ${commas(annotations.vocab_size)}`,
        ),
      );
      if (anchor) {
        leader(svg, anchor.x + anchor.w, anchor.y + 6, vx - 8, vy + 2, colors);
      }
    }

    if (annotations.embed_dim) {
      const anchor = stemPlaced[0] || stepPlaced[0];
      const lines = ["Embedding", `dimension of ${commas(annotations.embed_dim)}`];
      const tx = CHASSIS_X + CHASSIS_W + 14;
      const ty = anchor ? anchor.cy + 6 : chassisBottom - 20;
      lines.forEach((line, index) => {
        svg.appendChild(
          svgEl(
            "text",
            {
              x: tx,
              y: ty + index * 15,
              fill: colors.accent,
              "font-size": 12,
              "font-weight": 700,
              "font-family": FONT,
            },
            line,
          ),
        );
      });
      if (anchor) {
        leader(svg, anchor.x + anchor.w, anchor.cy, tx - 4, ty - 6, colors);
      }
    }

    host.replaceChildren(svg);

    const toggle = (id) => {
      focusedBySrc.set(src, focusedBySrc.get(src) === id ? "" : id);
      draw(host, graph);
    };
    svg.querySelectorAll("[data-arch-toggle]").forEach((el) => {
      const id = el.getAttribute("data-arch-toggle");
      el.addEventListener("click", (event) => {
        event.preventDefault();
        toggle(id);
      });
      el.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          toggle(id);
        }
      });
    });
  }

  const graphCache = new Map();
  const focusedBySrc = new Map();

  function mount(host) {
    if (!host || host._archBound) {
      return;
    }
    const src = host.getAttribute("data-arch-src");
    if (!src) {
      return;
    }
    host._archBound = true;
    const apply = (graph) => {
      if (!graph || typeof graph !== "object") {
        host.textContent = "Architecture diagram unavailable.";
        host._archBound = false;
        return;
      }
      draw(host, graph);
    };
    const cached = graphCache.get(src);
    if (cached) {
      apply(cached);
      return;
    }
    host.innerHTML = "<p class='small arch-graph-loading'>Loading architecture…</p>";
    fetch(src)
      .then((response) => {
        if (!response.ok) {
          throw new Error(String(response.status));
        }
        return response.json();
      })
      .then((graph) => {
        graphCache.set(src, graph);
        apply(graph);
      })
      .catch(() => {
        host.innerHTML = "<p class='small'>Architecture diagram unavailable.</p>";
        host._archBound = false;
      });
  }

  function mountAll(root) {
    const scope = root && root.querySelectorAll ? root : document;
    scope.querySelectorAll(".arch-graph-host[data-arch-src]").forEach((host) => mount(host));
  }

  global.ArchGraph = { mount, mountAll };
})(typeof window !== "undefined" ? window : globalThis);
