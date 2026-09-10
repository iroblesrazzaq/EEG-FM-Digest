(function (global) {
  const KIND_FILL = {
    input: "accent-soft",
    stem: "surface",
    posenc: "surface",
    repeat: "accent-soft",
    attn: "surface",
    mlp: "surface",
    norm: "surface",
    residual: "surface",
    pool: "surface",
    head: "surface",
    output: "accent-soft",
    module: "surface",
  };

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
    };
  }

  function fillFor(kind, colors) {
    const key = KIND_FILL[kind] || "surface";
    return key === "accent-soft" ? colors.accentSoft : colors.surface;
  }

  function asArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function formatShape(shape) {
    if (shape == null || shape === "") {
      return "";
    }
    if (typeof shape === "string") {
      return shape;
    }
    if (Array.isArray(shape)) {
      return "(" + shape.map((dim) => (dim == null ? "B" : String(dim))).join(", ") + ")";
    }
    return String(shape);
  }

  function edgeShape(edges, fromId, toId) {
    const match = asArray(edges).find((edge) => edge && edge.from === fromId && edge.to === toId);
    return match ? formatShape(match.shape) : "";
  }

  function layout(graph, expanded) {
    const nodes = asArray(graph.nodes);
    const BOX_H = 52;
    const CHILD_H = 40;
    const GAP = 26;
    const INNER_GAP = 10;
    const INNER_PAD = 12;
    const HEADER_H = 36;
    let y = 8;
    const laid = [];
    for (const node of nodes) {
      if (!node || !node.id) {
        continue;
      }
      const children = asArray(node.children);
      const isExpanded = Boolean(node.repeat && expanded.has(node.id) && children.length);
      let height = BOX_H;
      let childLayouts = [];
      if (isExpanded) {
        height = HEADER_H + INNER_PAD * 2 + children.length * CHILD_H + Math.max(0, children.length - 1) * INNER_GAP;
        let cy = y + HEADER_H + INNER_PAD;
        childLayouts = children.map((child) => {
          const item = { node: child, y: cy, height: CHILD_H, expanded: false, children: [] };
          cy += CHILD_H + INNER_GAP;
          return item;
        });
      }
      laid.push({ node, y, height, expanded: isExpanded, children: childLayouts });
      y += height + GAP;
    }
    return { items: laid, height: y };
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

  function boxGroup(item, x, width, colors) {
    const g = svgEl("g", { class: `arch-node arch-node-${item.node.kind || "module"}` });
    const node = item.node;
    const repeat = Number(node.repeat) || 0;
    const clickable = repeat > 0 && asArray(node.children).length > 0;
    const rx = 10;
    const rect = svgEl("rect", {
      x,
      y: item.y,
      width,
      height: item.height,
      rx,
      ry: rx,
      fill: fillFor(node.kind, colors),
      stroke: clickable ? colors.accent : colors.line,
      "stroke-width": clickable ? 1.6 : 1,
      "stroke-dasharray": node.kind === "residual" ? "4 3" : "",
    });
    g.appendChild(rect);

    const label = String(node.label || node.id || "");
    const detail = String(node.detail || "");
    const titleY = item.expanded ? item.y + 22 : item.y + (detail ? 20 : 30);
    const title = svgEl(
      "text",
      {
        x: x + 14,
        y: titleY,
        fill: colors.ink,
        "font-size": 14,
        "font-weight": 650,
        "font-family": "var(--font-body, sans-serif)",
      },
      label,
    );
    g.appendChild(title);
    if (detail && !item.expanded) {
      g.appendChild(
        svgEl(
          "text",
          {
            x: x + 14,
            y: item.y + 38,
            fill: colors.muted,
            "font-size": 11,
            "font-family": "var(--font-body, sans-serif)",
          },
          detail,
        ),
      );
    }

    if (repeat > 0) {
      const badgeW = 40;
      const badgeX = x + width - badgeW - 12;
      const badgeY = item.y + 12;
      const badge = svgEl("rect", {
        x: badgeX,
        y: badgeY,
        width: badgeW,
        height: 22,
        rx: 11,
        fill: colors.surface,
        stroke: colors.accent,
      });
      g.appendChild(badge);
      g.appendChild(
        svgEl(
          "text",
          {
            x: badgeX + badgeW / 2,
            y: badgeY + 16,
            fill: colors.accentDeep,
            "font-size": 12,
            "font-weight": 700,
            "text-anchor": "middle",
            "font-family": "var(--font-body, sans-serif)",
          },
          `×${repeat}`,
        ),
      );
    }

    if (clickable) {
      g.setAttribute("tabindex", "0");
      g.setAttribute("role", "button");
      g.setAttribute("aria-expanded", item.expanded ? "true" : "false");
      g.setAttribute(
        "aria-label",
        item.expanded ? `Collapse ${label}` : `Expand ${label} ×${repeat}`,
      );
      g.style.cursor = "pointer";
      g.setAttribute("data-arch-toggle", node.id);
    }

    for (let i = 0; i < item.children.length; i += 1) {
      const child = item.children[i];
      const childWidth = width - 28;
      const childX = x + 14;
      const childGroup = boxGroup(child, childX, childWidth, colors);
      childGroup.addEventListener("click", (event) => event.stopPropagation());
      g.appendChild(childGroup);
      if (item.children[i + 1]) {
        const fromY = child.y + child.height;
        const toY = item.children[i + 1].y;
        g.appendChild(
          svgEl("line", {
            x1: childX + childWidth / 2,
            y1: fromY,
            x2: childX + childWidth / 2,
            y2: toY,
            stroke: colors.accent,
            "stroke-width": 1.2,
          }),
        );
      }
    }
    return g;
  }

  function drawConnector(fromItem, toItem, x, width, colors, edges) {
    const fromY = fromItem.y + fromItem.height;
    const toY = toItem.y;
    const mid = fromY + (toY - fromY) / 2;
    const cx = x + width / 2;
    const connector = svgEl("g", { class: "arch-edge" });
    connector.appendChild(
      svgEl("line", {
        x1: cx,
        y1: fromY,
        x2: cx,
        y2: toY,
        stroke: colors.accent,
        "stroke-width": 1.4,
      }),
    );
    connector.appendChild(
      svgEl("polygon", {
        points: `${cx - 4},${toY - 7} ${cx + 4},${toY - 7} ${cx},${toY - 1}`,
        fill: colors.accent,
      }),
    );
    const shape =
      edgeShape(edges, fromItem.node.id, toItem.node.id) || formatShape(toItem.node.shape);
    if (shape) {
      connector.appendChild(
        svgEl(
          "text",
          {
            x: cx + 10,
            y: mid + 4,
            fill: colors.muted,
            "font-size": 11,
            "font-family": "var(--font-code, monospace)",
          },
          shape,
        ),
      );
    }
    return connector;
  }

  function draw(host, graph) {
    const colors = palette();
    const expanded = host._archExpanded instanceof Set ? host._archExpanded : new Set();
    host._archExpanded = expanded;
    const laid = layout(graph, expanded);
    const width = Math.max(320, Math.min(host.clientWidth || 420, 560));
    const boxWidth = Math.min(360, width - 48);
    const x = Math.max(16, (width - boxWidth) / 2);
    const svg = svgEl("svg", {
      class: "arch-graph-svg",
      viewBox: `0 0 ${width} ${laid.height + 8}`,
      width: "100%",
      role: "img",
      "aria-label": graph.label || graph.title || "Model architecture",
    });
    const edges = asArray(graph.edges);
    for (let i = 0; i < laid.items.length; i += 1) {
      svg.appendChild(boxGroup(laid.items[i], x, boxWidth, colors));
      if (laid.items[i + 1]) {
        svg.appendChild(drawConnector(laid.items[i], laid.items[i + 1], x, boxWidth, colors, edges));
      }
    }
    host.replaceChildren(svg);

    const toggle = (id) => {
      if (expanded.has(id)) {
        expanded.delete(id);
      } else {
        expanded.add(id);
      }
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
