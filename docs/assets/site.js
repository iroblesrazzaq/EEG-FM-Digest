const TAG_ORDER = ["paper_type", "backbone", "objective", "tokenization", "topology"];

const TAG_LABELS = {
  paper_type: {
    "new-model": "New Model",
    "eeg-fm": "New Model",
    "post-training": "Post-Training",
    benchmark: "Benchmark",
    survey: "Survey",
  },
  backbone: {
    transformer: "Transformer",
    "mamba-ssm": "Mamba-SSM",
    moe: "MoE",
    diffusion: "Diffusion",
  },
  objective: {
    "masked-reconstruction": "Masked Reconstruction",
    autoregressive: "Autoregressive",
    contrastive: "Contrastive",
    "discrete-code-prediction": "Discrete Code Prediction",
  },
  tokenization: {
    "time-patch": "Time Patch",
    "latent-tokens": "Latent Tokens",
    "discrete-tokens": "Discrete Tokens",
  },
  topology: {
    "fixed-montage": "Fixed Montage",
    "channel-flexible": "Channel Flexible",
    "topology-agnostic": "Topology Agnostic",
  },
};

function norm(s) {
  return String(s || "").toLowerCase();
}

function esc(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function safeNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function tagValueLabel(category, value) {
  const mapping = TAG_LABELS[category] || {};
  if (mapping[value]) {
    return mapping[value];
  }
  return String(value || "").replaceAll("-", " ").replace(/\b\w/g, (m) => m.toUpperCase());
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`fetch_failed:${path}:${response.status}`);
  }
  return response.json();
}

function parseFallbackMonths(raw) {
  try {
    const parsed = JSON.parse(raw || "[]");
    return asArray(parsed).map((m) => String(m)).filter(Boolean);
  } catch (_err) {
    return [];
  }
}

function normalizeStats(raw, papers) {
  const stats = raw && typeof raw === "object" ? raw : {};
  const summarized = papers.filter((paper) => paper.summary).length;
  return {
    candidates: safeNumber(stats.candidates, papers.length),
    accepted: safeNumber(stats.accepted, papers.length),
    summarized: safeNumber(stats.summarized, summarized),
  };
}

function normalizePaper(raw, month) {
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const looksLikeLegacySummary = Boolean(raw.tags && raw.key_points && raw.unique_contribution);
  const summary =
    raw.summary && typeof raw.summary === "object"
      ? raw.summary
      : raw.paper_summary && typeof raw.paper_summary === "object"
      ? raw.paper_summary
      : looksLikeLegacySummary
      ? raw
      : null;

  const arxivIdBase = String(raw.arxiv_id_base || summary?.arxiv_id_base || "").trim();
  if (!arxivIdBase) {
    return null;
  }

  const links = raw.links && typeof raw.links === "object" ? raw.links : {};
  const absUrl = String(links.abs || "").trim() || `https://arxiv.org/abs/${arxivIdBase}`;
  const pdfUrl = String(links.pdf || "").trim();

  return {
    month: String(month || "").trim(),
    arxiv_id_base: arxivIdBase,
    arxiv_id: String(raw.arxiv_id || "").trim(),
    title: String(raw.title || summary?.title || "").trim(),
    published_date: String(raw.published_date || summary?.published_date || "").trim(),
    authors: asArray(raw.authors).map((author) => String(author)).filter(Boolean),
    categories: asArray(raw.categories || summary?.categories).map((cat) => String(cat)).filter(Boolean),
    links: { abs: absUrl, pdf: pdfUrl },
    triage:
      raw.triage && typeof raw.triage === "object"
        ? {
            decision: String(raw.triage.decision || "accept"),
            confidence: safeNumber(raw.triage.confidence, 0),
            reasons: asArray(raw.triage.reasons).map((item) => String(item)),
          }
        : { decision: "accept", confidence: 0, reasons: [] },
    summary,
    summary_failed_reason: String(raw.summary_failed_reason || "").trim(),
  };
}

function parseMonthPayload(payload, fallbackMonth) {
  if (Array.isArray(payload)) {
    const papers = payload
      .map((row) => normalizePaper(row, fallbackMonth))
      .filter((row) => row !== null);
    return {
      month: String(fallbackMonth || ""),
      stats: normalizeStats({}, papers),
      papers,
    };
  }
  if (!payload || typeof payload !== "object") {
    return {
      month: String(fallbackMonth || ""),
      stats: normalizeStats({}, []),
      papers: [],
    };
  }
  const month = String(payload.month || fallbackMonth || "");
  const rows = asArray(payload.papers);
  const papers = rows.map((row) => normalizePaper(row, month)).filter((row) => row !== null);
  return {
    month,
    stats: normalizeStats(payload.stats, papers),
    papers,
  };
}

function normalizeManifest(raw, fallbackMonths) {
  const fallback = {
    latest: fallbackMonths.length ? fallbackMonths[0] : null,
    months: fallbackMonths.map((month) => ({
      month,
      href: `digest/${month}/index.html`,
      json_path: `digest/${month}/papers.json`,
      stats: { candidates: 0, accepted: 0, summarized: 0 },
      empty_state: "unknown",
    })),
  };
  if (!raw || typeof raw !== "object" || !Array.isArray(raw.months)) {
    return fallback;
  }
  const monthRows = raw.months
    .map((row) => {
      if (!row || typeof row !== "object") {
        return null;
      }
      const month = String(row.month || "").trim();
      if (!month) {
        return null;
      }
      return {
        month,
        href: String(row.href || `digest/${month}/index.html`),
        json_path: String(row.json_path || `digest/${month}/papers.json`),
        stats: normalizeStats(row.stats, []),
        empty_state: String(row.empty_state || "unknown"),
      };
    })
    .filter((row) => row !== null);
  monthRows.sort((a, b) => b.month.localeCompare(a.month));
  return {
    latest: raw.latest ? String(raw.latest) : monthRows[0]?.month || null,
    months: monthRows,
  };
}

function collectTagOptions(papers) {
  const byCategory = {};
  for (const category of TAG_ORDER) {
    byCategory[category] = new Set();
  }
  for (const paper of papers) {
    const tags = paper.summary?.tags;
    if (!tags || typeof tags !== "object") {
      continue;
    }
    for (const category of TAG_ORDER) {
      for (const value of asArray(tags[category])) {
        const normalized = String(value || "").trim();
        if (normalized) {
          byCategory[category].add(normalized);
        }
      }
    }
  }
  const options = {};
  for (const category of TAG_ORDER) {
    options[category] = [...byCategory[category]].sort((a, b) =>
      tagValueLabel(category, a).localeCompare(tagValueLabel(category, b)),
    );
  }
  return options;
}

function hasTagFilters(state) {
  return TAG_ORDER.some((category) => state.selectedTags[category].size > 0);
}

function matchesTagFilters(paper, state) {
  const tags = paper.summary?.tags;
  for (const category of TAG_ORDER) {
    const selected = state.selectedTags[category];
    if (!selected || selected.size === 0) {
      continue;
    }
    const values = tags && typeof tags === "object" ? asArray(tags[category]) : [];
    let matched = false;
    for (const value of values) {
      if (selected.has(String(value))) {
        matched = true;
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }
  return true;
}

function paperHaystack(paper) {
  const summary = paper.summary || {};
  const tags = summary.tags && typeof summary.tags === "object" ? summary.tags : {};
  const tagValues = TAG_ORDER.flatMap((category) => asArray(tags[category]).map((value) => tagValueLabel(category, value)));
  return norm(
    [
      paper.title,
      paper.arxiv_id_base,
      paper.month,
      paper.published_date,
      ...paper.authors,
      summary.one_liner,
      summary.unique_contribution,
      summary.detailed_summary,
      ...asArray(summary.key_points),
      ...tagValues,
    ].join(" "),
  );
}

function sortPapers(papers, sortBy) {
  const copy = [...papers];
  copy.sort((a, b) => {
    if (sortBy === "published_asc") {
      return (
        a.published_date.localeCompare(b.published_date) ||
        a.month.localeCompare(b.month) ||
        a.arxiv_id_base.localeCompare(b.arxiv_id_base)
      );
    }
    if (sortBy === "title_asc") {
      return a.title.localeCompare(b.title) || a.arxiv_id_base.localeCompare(b.arxiv_id_base);
    }
    if (sortBy === "confidence_desc") {
      return (
        safeNumber(b.triage?.confidence, 0) - safeNumber(a.triage?.confidence, 0) ||
        b.published_date.localeCompare(a.published_date) ||
        a.arxiv_id_base.localeCompare(b.arxiv_id_base)
      );
    }
    return (
      b.published_date.localeCompare(a.published_date) ||
      b.month.localeCompare(a.month) ||
      a.arxiv_id_base.localeCompare(b.arxiv_id_base)
    );
  });
  return copy;
}

function monthBaseCount(state) {
  if (state.selectedMonth === "all") {
    return state.papers.length;
  }
  return state.papers.filter((paper) => paper.month === state.selectedMonth).length;
}

function monthEmptyMessage(month, stats) {
  const candidates = safeNumber(stats?.candidates, 0);
  const accepted = safeNumber(stats?.accepted, 0);
  const summarized = safeNumber(stats?.summarized, 0);
  if (candidates === 0) {
    return `No arXiv candidates were found for ${month}.`;
  }
  if (accepted === 0) {
    return `No papers were accepted by triage for ${month}.`;
  }
  if (summarized === 0) {
    return `Accepted papers exist for ${month}, but summaries are unavailable.`;
  }
  return `No papers match the current filters for ${month}.`;
}

function renderTagChips(summary) {
  const tags = summary?.tags;
  if (!tags || typeof tags !== "object") {
    return "";
  }
  const chips = [];
  for (const category of TAG_ORDER) {
    for (const rawValue of asArray(tags[category])) {
      const value = String(rawValue || "").trim();
      if (!value) {
        continue;
      }
      chips.push(
        `<span class="chip chip-${esc(category)}" title="${esc(category.replaceAll("_", " "))}">${esc(
          tagValueLabel(category, value),
        )}</span>`,
      );
    }
  }
  if (!chips.length) {
    return "";
  }
  return `<p class="chips">${chips.join(" ")}</p>`;
}

function renderPaperCard(paper, view) {
  const summary = paper.summary;
  const title = esc(paper.title || paper.arxiv_id_base);
  const absUrl = esc(paper.links?.abs || "#");
  const metaParts = [];
  if (view === "all") {
    metaParts.push(esc(paper.month));
  }
  if (paper.published_date) {
    metaParts.push(esc(paper.published_date));
  }
  if (paper.authors.length) {
    metaParts.push(esc(paper.authors.join(", ")));
  }
  const metaHtml = metaParts.length ? `<div class="meta">${metaParts.join(" · ")}</div>` : "";

  if (!summary) {
    const reason = paper.summary_failed_reason || "summary_unavailable";
    return `
      <article class="paper-card" id="${esc(paper.arxiv_id_base)}">
        <h3><a href="${absUrl}">${title}</a></h3>
        ${metaHtml}
        <p class="summary-failed"><strong>Summary unavailable.</strong> ${esc(reason)}</p>
      </article>
    `;
  }

  const points = asArray(summary.key_points)
    .map((point) => String(point || "").trim())
    .filter(Boolean)
    .slice(0, 3);
  const pointsHtml = points.length
    ? `<ul class="summary-points">${points.map((point) => `<li>${esc(point)}</li>`).join("")}</ul>`
    : "";
  const uniqueContribution = String(summary.unique_contribution || "").trim();
  const uniqueHtml = uniqueContribution
    ? `<p><strong>Unique contribution:</strong> ${esc(uniqueContribution)}</p>`
    : "";
  const detailed = String(summary.detailed_summary || summary.one_liner || "").trim();
  const detailHtml = detailed
    ? `<details class="summary-detail"><summary>Detailed summary</summary><p>${esc(detailed)}</p></details>`
    : "";
  const tagsHtml = renderTagChips(summary);

  const openSource = summary.open_source && typeof summary.open_source === "object" ? summary.open_source : {};
  const codeUrl = String(openSource.code_url || "").trim();
  const weightsUrl = String(openSource.weights_url || "").trim();
  const links = [];
  if (codeUrl) {
    links.push(`<a href="${esc(codeUrl)}">code</a>`);
  }
  if (weightsUrl) {
    links.push(`<a href="${esc(weightsUrl)}">weights</a>`);
  }
  const linksHtml = links.length ? `<p class="links">${links.join(" ")}</p>` : "";

  return `
    <article class="paper-card" id="${esc(paper.arxiv_id_base)}">
      <h3><a href="${absUrl}">${title}</a></h3>
      ${metaHtml}
      <p><strong>Summary Highlights:</strong></p>
      ${pointsHtml}
      ${uniqueHtml}
      ${detailHtml}
      ${tagsHtml}
      ${linksHtml}
    </article>
  `;
}

function renderControls(app, state) {
  const controls = app.querySelector("#controls");
  if (!controls) {
    return;
  }
  const monthControl =
    state.view === "all"
      ? `
      <label class="control" for="month-filter">
        <span>Month</span>
        <select id="month-filter">
          <option value="all"${state.selectedMonth === "all" ? " selected" : ""}>All months</option>
          ${state.months
            .map(
              (month) =>
                `<option value="${esc(month)}"${state.selectedMonth === month ? " selected" : ""}>${esc(month)}</option>`,
            )
            .join("")}
        </select>
      </label>`
      : "";

  const tagGroups = TAG_ORDER.map((category) => {
    const values = state.tagOptions[category] || [];
    if (!values.length) {
      return "";
    }
    const options = values
      .map((value) => {
        const checked = state.selectedTags[category].has(value) ? " checked" : "";
        return `
          <label class="tag-option">
            <input type="checkbox" data-tag-category="${esc(category)}" data-tag-value="${esc(value)}"${checked}>
            <span>${esc(tagValueLabel(category, value))}</span>
          </label>
        `;
      })
      .join("");
    return `
      <fieldset class="tag-filter">
        <legend>${esc(category.replaceAll("_", " "))}</legend>
        <div class="tag-options">${options}</div>
      </fieldset>
    `;
  })
    .filter(Boolean)
    .join("");

  controls.innerHTML = `
    <div class="control-row">
      <label class="control control-grow" for="search-input">
        <span>Search</span>
        <input id="search-input" type="text" value="${esc(state.queryRaw)}" placeholder="title, author, summary, tags">
      </label>
      ${monthControl}
      <label class="control" for="sort-select">
        <span>Sort</span>
        <select id="sort-select">
          <option value="published_desc"${state.sortBy === "published_desc" ? " selected" : ""}>Newest first</option>
          <option value="published_asc"${state.sortBy === "published_asc" ? " selected" : ""}>Oldest first</option>
          <option value="confidence_desc"${state.sortBy === "confidence_desc" ? " selected" : ""}>Triage confidence</option>
          <option value="title_asc"${state.sortBy === "title_asc" ? " selected" : ""}>Title A-Z</option>
        </select>
      </label>
      <button id="reset-filters" type="button">See all</button>
    </div>
    ${tagGroups ? `<div class="tag-filter-grid">${tagGroups}</div>` : ""}
  `;

  const searchInput = controls.querySelector("#search-input");
  if (searchInput) {
    searchInput.addEventListener("input", (event) => {
      state.queryRaw = event.target.value || "";
      state.query = norm(state.queryRaw);
      renderResults(app, state);
    });
  }
  const monthFilter = controls.querySelector("#month-filter");
  if (monthFilter) {
    monthFilter.addEventListener("change", (event) => {
      state.selectedMonth = String(event.target.value || "all");
      renderResults(app, state);
    });
  }
  const sortSelect = controls.querySelector("#sort-select");
  if (sortSelect) {
    sortSelect.addEventListener("change", (event) => {
      state.sortBy = String(event.target.value || "published_desc");
      renderResults(app, state);
    });
  }
  const resetBtn = controls.querySelector("#reset-filters");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      state.queryRaw = "";
      state.query = "";
      state.sortBy = "published_desc";
      if (state.view === "all") {
        state.selectedMonth = "all";
      }
      for (const category of TAG_ORDER) {
        state.selectedTags[category].clear();
      }
      renderControls(app, state);
      renderResults(app, state);
    });
  }
  controls.addEventListener("change", (event) => {
    const target = event.target;
    if (!target || target.tagName !== "INPUT") {
      return;
    }
    const category = target.getAttribute("data-tag-category");
    const value = target.getAttribute("data-tag-value");
    if (!category || !value) {
      return;
    }
    const selected = state.selectedTags[category];
    if (!selected) {
      return;
    }
    if (target.checked) {
      selected.add(value);
    } else {
      selected.delete(value);
    }
    renderResults(app, state);
  });
}

function renderResults(app, state) {
  const results = app.querySelector("#results");
  const meta = app.querySelector("#results-meta");
  if (!results || !meta) {
    return;
  }

  let filtered = state.papers;
  if (state.view === "all" && state.selectedMonth !== "all") {
    filtered = filtered.filter((paper) => paper.month === state.selectedMonth);
  }
  if (state.query) {
    filtered = filtered.filter((paper) => paperHaystack(paper).includes(state.query));
  }
  filtered = filtered.filter((paper) => matchesTagFilters(paper, state));
  filtered = sortPapers(filtered, state.sortBy);

  const baseCount = monthBaseCount(state);
  const scopeLabel =
    state.view === "all"
      ? state.selectedMonth === "all"
        ? "across all months"
        : `for ${state.selectedMonth}`
      : `for ${state.month}`;
  meta.textContent = `Showing ${filtered.length} of ${baseCount} accepted papers ${scopeLabel}.`;

  if (!filtered.length) {
    const monthKey = state.view === "month" ? state.month : state.selectedMonth;
    const noFilters = !state.query && !hasTagFilters(state);
    if (monthKey !== "all" && noFilters) {
      results.innerHTML = `<p class="empty-state">${esc(monthEmptyMessage(monthKey, state.monthStats[monthKey]))}</p>`;
    } else {
      results.innerHTML = "<p class='empty-state'>No papers match the current filters.</p>";
    }
    return;
  }

  results.innerHTML = filtered.map((paper) => renderPaperCard(paper, state.view)).join("\n");
}

async function setupDigestApp() {
  const app = document.getElementById("digest-app");
  if (!app) {
    return false;
  }
  const view = String(app.dataset.view || "all");
  const month = String(app.dataset.month || "");
  const manifestPath = String(app.dataset.manifestJson || "data/months.json");
  const monthJsonPath = String(app.dataset.monthJson || "");
  const fallbackMonths = parseFallbackMonths(app.dataset.fallbackMonths || "[]");

  let manifest = normalizeManifest(null, fallbackMonths);
  try {
    const rawManifest = await fetchJson(manifestPath);
    manifest = normalizeManifest(rawManifest, fallbackMonths);
  } catch (_err) {
    manifest = normalizeManifest(null, fallbackMonths);
  }

  const monthStats = {};
  const papers = [];
  if (view === "month") {
    let monthPayload = parseMonthPayload({}, month);
    if (monthJsonPath) {
      try {
        const raw = await fetchJson(monthJsonPath);
        monthPayload = parseMonthPayload(raw, month);
      } catch (_err) {
        monthPayload = parseMonthPayload({}, month);
      }
    }
    monthStats[monthPayload.month || month] = monthPayload.stats;
    papers.push(...monthPayload.papers);
  } else {
    const months = manifest.months;
    for (const item of months) {
      const monthKey = item.month;
      let payload = parseMonthPayload({}, monthKey);
      try {
        const raw = await fetchJson(item.json_path);
        payload = parseMonthPayload(raw, monthKey);
      } catch (_err) {
        payload = parseMonthPayload({}, monthKey);
      }
      monthStats[monthKey] = payload.stats;
      papers.push(...payload.papers);
    }
  }

  const state = {
    view,
    month,
    months: manifest.months.map((row) => row.month),
    monthStats,
    papers,
    queryRaw: "",
    query: "",
    sortBy: "published_desc",
    selectedMonth: view === "all" ? "all" : month,
    tagOptions: collectTagOptions(papers),
    selectedTags: Object.fromEntries(TAG_ORDER.map((category) => [category, new Set()])),
  };

  renderControls(app, state);
  renderResults(app, state);
  return true;
}

function setupLegacySearch() {
  const input = document.getElementById("searchBox");
  if (!input) {
    return;
  }
  input.addEventListener("input", () => {
    const q = norm(input.value);
    const cards = document.querySelectorAll(".card");
    for (const card of cards) {
      const hay = norm(card.getAttribute("data-hay"));
      card.style.display = hay.includes(q) ? "" : "none";
    }
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  const mounted = await setupDigestApp();
  if (!mounted) {
    setupLegacySearch();
  }
});
