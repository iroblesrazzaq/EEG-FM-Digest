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

function monthDisplayLabel(month) {
  if (!month) {
    return "";
  }
  const [year, mon] = String(month).split("-");
  const y = Number(year);
  const m = Number(mon);
  if (!Number.isFinite(y) || !Number.isFinite(m) || m < 1 || m > 12) {
    return String(month);
  }
  return new Date(Date.UTC(y, m - 1, 1)).toLocaleString("en-US", {
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });
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

function resolveMonthJsonPath(path, view) {
  const value = String(path || "");
  if (!value) {
    return value;
  }
  if (
    value.startsWith("http://") ||
    value.startsWith("https://") ||
    value.startsWith("/") ||
    value.startsWith("./") ||
    value.startsWith("../")
  ) {
    return value;
  }
  return view === "explore" ? `../${value}` : value;
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
      month_label: monthDisplayLabel(month),
      href: `digest/${month}/index.html`,
      json_path: `digest/${month}/papers.json`,
      stats: { candidates: 0, accepted: 0, summarized: 0 },
      empty_state: "unknown",
      featured: null,
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
        month_label: String(row.month_label || monthDisplayLabel(month)),
        href: String(row.href || `digest/${month}/index.html`),
        json_path: String(row.json_path || `digest/${month}/papers.json`),
        stats: normalizeStats(row.stats, []),
        empty_state: String(row.empty_state || "unknown"),
        featured:
          row.featured && typeof row.featured === "object"
            ? {
                arxiv_id_base: String(row.featured.arxiv_id_base || "").trim(),
                title: String(row.featured.title || "").trim(),
                one_liner: String(row.featured.one_liner || "").trim(),
                abs_url: String(row.featured.abs_url || "").trim(),
              }
            : null,
      };
    })
    .filter((row) => row !== null);
  monthRows.sort((a, b) => b.month.localeCompare(a.month));
  return {
    latest: raw.latest ? String(raw.latest) : monthRows[0]?.month || null,
    months: monthRows,
  };
}

function monthHasPapers(monthRow) {
  return safeNumber(monthRow?.stats?.accepted, 0) > 0;
}

function visibleMonthRows(state) {
  return state.monthRows.filter((row) => monthHasPapers(row));
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
  if (state.view === "month") {
    return state.papers.length;
  }
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
    return `No arXiv candidates were found for ${monthDisplayLabel(month)}.`;
  }
  if (accepted === 0) {
    return `No papers were accepted by triage for ${monthDisplayLabel(month)}.`;
  }
  if (summarized === 0) {
    return `Accepted papers exist for ${monthDisplayLabel(month)}, but summaries are unavailable.`;
  }
  return `No papers match the current filters for ${monthDisplayLabel(month)}.`;
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
  if (view === "explore") {
    metaParts.push(esc(monthDisplayLabel(paper.month)));
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
    links.push(`<a class="resource-btn resource-btn-code" href="${esc(codeUrl)}">Code Here</a>`);
  }
  if (weightsUrl) {
    links.push(`<a class="resource-btn resource-btn-weights" href="${esc(weightsUrl)}">Model Weights</a>`);
  }
  const linksHtml = links.length ? `<div class="resource-links">${links.join("")}</div>` : "";

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

function renderTagGroups(state, tagOptions, compact) {
  const groups = TAG_ORDER.map((category) => {
    const values = tagOptions[category] || [];
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
      <fieldset class="tag-filter tag-filter-${esc(category)}">
        <legend>${esc(category.replaceAll("_", " "))}</legend>
        <div class="tag-options">${options}</div>
      </fieldset>
    `;
  })
    .filter(Boolean)
    .join("");
  if (!groups) {
    return "";
  }
  const cls = compact ? "tag-filter-grid compact" : "tag-filter-grid";
  return `<div class="${cls}">${groups}</div>`;
}

function bindTagCheckboxes(controls, state, app) {
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

function renderExploreControls(app, state) {
  const controls = app.querySelector("#controls");
  if (!controls) {
    return;
  }
  const tagOptions = collectTagOptions(state.papers);
  const tagGroups = renderTagGroups(state, tagOptions, false);

  controls.innerHTML = `
    <div class="control-row search-only-row">
      <label class="control control-grow" for="search-input">
        <span>Search</span>
        <input id="search-input" type="text" value="${esc(state.queryRaw)}" placeholder="title, author, summary, tags">
      </label>
      <button id="reset-filters" type="button">Clear search</button>
    </div>
    <p class="small filter-help">Tag filters: OR within each category, AND across categories.</p>
    ${tagGroups}
  `;

  const searchInput = controls.querySelector("#search-input");
  if (searchInput) {
    searchInput.addEventListener("input", (event) => {
      state.queryRaw = event.target.value || "";
      state.query = norm(state.queryRaw);
      renderResults(app, state);
    });
  }
  const resetBtn = controls.querySelector("#reset-filters");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      state.queryRaw = "";
      state.query = "";
      for (const category of TAG_ORDER) {
        state.selectedTags[category].clear();
      }
      renderExploreControls(app, state);
      renderResults(app, state);
    });
  }
  bindTagCheckboxes(controls, state, app);
}

function renderMonthControls(app, state) {
  const controls = app.querySelector("#controls");
  if (!controls) {
    return;
  }
  const tagOptions = collectTagOptions(state.papers);
  const tagGroups = renderTagGroups(state, tagOptions, true);
  controls.innerHTML = `
    <div class="control-row month-controls-row">
      <label class="control" for="sort-select">
        <span>Sort</span>
        <select id="sort-select">
          <option value="published_desc"${state.sortBy === "published_desc" ? " selected" : ""}>Newest first</option>
          <option value="published_asc"${state.sortBy === "published_asc" ? " selected" : ""}>Oldest first</option>
          <option value="confidence_desc"${state.sortBy === "confidence_desc" ? " selected" : ""}>Triage confidence</option>
          <option value="title_asc"${state.sortBy === "title_asc" ? " selected" : ""}>Title A-Z</option>
        </select>
      </label>
      <button id="reset-filters" type="button">Clear filters</button>
    </div>
    ${
      tagGroups
        ? `<details class="filter-collapse"><summary>Filter by tags</summary><p class="small filter-help">OR within each category, AND across categories.</p>${tagGroups}</details>`
        : ""
    }
  `;

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
      state.sortBy = "published_desc";
      for (const category of TAG_ORDER) {
        state.selectedTags[category].clear();
      }
      renderMonthControls(app, state);
      renderResults(app, state);
    });
  }
  bindTagCheckboxes(controls, state, app);
}

function renderResults(app, state) {
  const results = app.querySelector("#results");
  const meta = app.querySelector("#results-meta");
  if (!results || !meta) {
    return;
  }

  let filtered = state.papers;
  if (state.view === "explore" && state.selectedMonth !== "all") {
    filtered = filtered.filter((paper) => paper.month === state.selectedMonth);
  }
  if (state.query) {
    filtered = filtered.filter((paper) => paperHaystack(paper).includes(state.query));
  }
  filtered = filtered.filter((paper) => matchesTagFilters(paper, state));
  filtered = sortPapers(filtered, state.sortBy);

  const baseCount = monthBaseCount(state);
  if (state.view === "month") {
    meta.textContent = `Showing ${filtered.length} of ${baseCount} accepted papers for ${monthDisplayLabel(state.month)}.`;
  } else {
    const scopeLabel = state.selectedMonth === "all" ? "across all months" : `for ${monthDisplayLabel(state.selectedMonth)}`;
    const loadingMeta =
      state.loading && state.loading.active
        ? ` Loading ${state.loading.loaded}/${state.loading.total} month digests...`
        : "";
    meta.textContent = `Showing ${filtered.length} of ${baseCount} accepted papers ${scopeLabel}.${loadingMeta}`;
  }

  if (!filtered.length) {
    const monthKey = state.view === "month" ? state.month : state.selectedMonth;
    const noFilters = !state.query && !hasTagFilters(state);
    if (state.view === "explore" && state.loading && state.loading.active && monthKey === "all" && noFilters) {
      results.innerHTML = "<p class='empty-state'>Loading accepted papers...</p>";
    } else if (monthKey === "all" && noFilters) {
      results.innerHTML = "<p class='empty-state'>No accepted papers are available for the selected month set.</p>";
    } else if (monthKey !== "all" && noFilters) {
      results.innerHTML = `<p class="empty-state">${esc(monthEmptyMessage(monthKey, state.monthStats[monthKey]))}</p>`;
    } else {
      results.innerHTML = "<p class='empty-state'>No papers match the current filters.</p>";
    }
    return;
  }

  results.innerHTML = filtered.map((paper) => renderPaperCard(paper, state.view)).join("\n");
}

function renderHome(app, state) {
  const controls = app.querySelector("#home-controls");
  const results = app.querySelector("#home-results");
  if (!controls || !results) {
    return;
  }
  controls.innerHTML = "";
  controls.style.display = "none";

  const rows = visibleMonthRows(state);
  if (!rows.length) {
    results.innerHTML = "<p class='empty-state'>No monthly digests to show yet.</p>";
    return;
  }

  const groups = {};
  for (const row of rows) {
    const year = String(row.month).slice(0, 4);
    if (!groups[year]) {
      groups[year] = [];
    }
    groups[year].push(row);
  }
  const years = Object.keys(groups).sort((a, b) => b.localeCompare(a));
  const newestYear = years[0];

  const yearBlocks = years
    .map((year) => {
      const yearPaperCount = groups[year].reduce(
        (total, row) => total + safeNumber(row?.stats?.accepted, 0),
        0,
      );
      const yearCountText = `${yearPaperCount} ${yearPaperCount === 1 ? "paper" : "papers"}`;
      const cards = groups[year]
        .map((row) => {
          const featured = row.featured;
          const stats = row.stats || {};
          const paperCount = safeNumber(stats.accepted, 0);
          const statsText = `${paperCount} ${paperCount === 1 ? "paper" : "papers"}`;
          const monthLabel = row.month_label || monthDisplayLabel(row.month);
          const monthHref = esc(row.href);
          const featuredHtml =
            featured && featured.title
              ? `
                <div class="featured-paper">
                  <p class="small">Featured paper</p>
                  <p class="featured-title"><a class="featured-paper-link" href="${esc(featured.abs_url || row.href)}">${esc(featured.title)}</a></p>
                  ${featured.one_liner ? `<p class="small">${esc(featured.one_liner)}</p>` : ""}
                </div>
              `
              : `<p class="small">Featured paper: not set.</p>`;
          return `
            <article class="month-card">
              <a class="month-card-link" href="${monthHref}" aria-label="Open ${esc(monthLabel)} Digest"></a>
              <div class="month-head">
                <h3>${esc(monthLabel)} Digest</h3>
                <p class="small month-stats">${esc(statsText)}</p>
              </div>
              ${featuredHtml}
            </article>
          `;
        })
        .join("");
      return `
        <details class="year-block"${year === newestYear ? " open" : ""}>
          <summary class="year-summary"><span>${esc(year)}</span><span class="year-summary-count">${esc(
            yearCountText,
          )}</span></summary>
          <div class="home-month-grid">${cards}</div>
        </details>
      `;
    })
    .join("");
  results.innerHTML = yearBlocks;
}

async function loadExploreMonthsLazy(app, state, monthRows, view) {
  const rows = Array.isArray(monthRows) ? [...monthRows] : [];
  const concurrency = Math.min(3, Math.max(1, rows.length));
  if (!rows.length) {
    state.loading.active = false;
    renderResults(app, state);
    return;
  }

  let refreshedTagControls = false;
  const refresh = () => {
    renderResults(app, state);
  };

  async function worker() {
    while (rows.length) {
      const item = rows.shift();
      if (!item || typeof item !== "object") {
        continue;
      }
      const monthKey = String(item.month || "");
      let payload = parseMonthPayload({}, monthKey);
      try {
        const raw = await fetchJson(resolveMonthJsonPath(item.json_path, view));
        payload = parseMonthPayload(raw, monthKey);
      } catch (_err) {
        payload = parseMonthPayload({}, monthKey);
        state.loading.failed += 1;
      }
      if (monthKey) {
        state.monthStats[monthKey] = payload.stats;
      }
      if (payload.papers.length) {
        state.papers.push(...payload.papers);
      }
      state.loading.loaded += 1;
      if (!refreshedTagControls && state.papers.length > 0) {
        refreshedTagControls = true;
        renderExploreControls(app, state);
      }
      refresh();
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  await Promise.all(Array.from({ length: concurrency }, () => worker()));
  state.loading.active = false;
  renderExploreControls(app, state);
  renderResults(app, state);
}

async function setupDigestApp() {
  const app = document.getElementById("digest-app");
  if (!app) {
    return false;
  }
  const view = String(app.dataset.view || "home");
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

  if (view === "home") {
    const state = {
      view,
      monthRows: manifest.months,
    };
    renderHome(app, state);
    return true;
  }

  const monthStats = {};
  for (const item of manifest.months) {
    if (item && typeof item === "object" && item.month) {
      monthStats[item.month] = normalizeStats(item.stats, []);
    }
  }

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
  }

  const state = {
    view: view === "all" ? "explore" : view,
    month,
    monthRows: manifest.months,
    monthStats,
    papers,
    queryRaw: "",
    query: "",
    sortBy: "published_desc",
    selectedMonth: view === "month" ? month : "all",
    selectedTags: Object.fromEntries(TAG_ORDER.map((category) => [category, new Set()])),
    loading:
      view === "month"
        ? null
        : {
            active: true,
            total: manifest.months.length,
            loaded: 0,
            failed: 0,
          },
  };

  if (state.view === "month") {
    renderMonthControls(app, state);
    renderResults(app, state);
  } else {
    renderExploreControls(app, state);
    renderResults(app, state);
    void loadExploreMonthsLazy(app, state, manifest.months, view);
  }
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
