"""Inline CSS/JS for the self-contained HTML report. No external requests, ever.

`CSS` styles the page shell (nav, header, sections, table) for both light and dark
viewers and includes a print stylesheet. `TABLE_JS` powers the sortable/filterable
trade table emitted by `figures.fig_trade_table`.
"""

from __future__ import annotations

CSS = """
:root {
  --bg: #ffffff;
  --bg-alt: #f4f5f7;
  --text: #1c1f26;
  --text-dim: #5a6270;
  --border: #e1e4ea;
  --accent: #4C78A8;
  --danger: #B23A48;
  --danger-bg: #fdeceb;
  --good: #1E7A3D;
  --nav-bg: #ffffff;
  --card-bg: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161b;
    --bg-alt: #1b1e25;
    --text: #e8eaf0;
    --text-dim: #9aa2b1;
    --border: #2b2f3a;
    --accent: #7fa9d8;
    --danger: #e57373;
    --danger-bg: #3a2020;
    --good: #7fcf9e;
    --nav-bg: #1b1e25;
    --card-bg: #1b1e25;
  }
}
:root[data-theme="dark"] {
  --bg: #14161b; --bg-alt: #1b1e25; --text: #e8eaf0; --text-dim: #9aa2b1;
  --border: #2b2f3a; --accent: #7fa9d8; --danger: #e57373; --danger-bg: #3a2020;
  --good: #7fcf9e; --nav-bg: #1b1e25; --card-bg: #1b1e25;
}
:root[data-theme="light"] {
  --bg: #ffffff; --bg-alt: #f4f5f7; --text: #1c1f26; --text-dim: #5a6270;
  --border: #e1e4ea; --accent: #4C78A8; --danger: #B23A48; --danger-bg: #fdeceb;
  --good: #1E7A3D; --nav-bg: #ffffff; --card-bg: #ffffff;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
  line-height: 1.45;
}
.odds-nav {
  position: sticky; top: 0; z-index: 50; display: flex; gap: 4px; flex-wrap: wrap;
  background: var(--nav-bg); border-bottom: 1px solid var(--border);
  padding: 10px 16px; align-items: center;
}
.odds-nav a {
  color: var(--text-dim); text-decoration: none; font-size: 0.85rem;
  padding: 6px 10px; border-radius: 6px;
}
.odds-nav a:hover { background: var(--bg-alt); color: var(--text); }
.odds-nav .brand { font-weight: 700; margin-right: 12px; color: var(--text); }

.synthetic-banner {
  background: repeating-linear-gradient(45deg, var(--danger-bg), var(--danger-bg) 12px, var(--danger) 12px, var(--danger) 13px);
  background-color: var(--danger-bg);
  border: 2px solid var(--danger);
  color: var(--danger);
  font-weight: 700;
  text-align: center;
  padding: 10px 16px;
  font-size: 0.95rem;
  letter-spacing: 0.02em;
}

.container { max-width: 1180px; margin: 0 auto; padding: 0 20px 60px; }

.stat-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px; margin: 18px 0 32px;
}
.stat-card {
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 14px;
}
.stat-card .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim); }
.stat-card .value { font-size: 1.35rem; font-weight: 700; margin-top: 4px; }
.stat-card .value.good { color: var(--good); }
.stat-card .value.bad { color: var(--danger); }

section.odds-section { padding-top: 28px; scroll-margin-top: 60px; }
section.odds-section h2 { border-bottom: 1px solid var(--border); padding-bottom: 8px; }
section.odds-section h3 { color: var(--text-dim); font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.04em; }

.fig-block { margin: 14px 0 30px; border: 1px solid var(--border); border-radius: 10px; padding: 6px; overflow-x: auto; }

.trade-filter {
  width: 100%; padding: 8px 10px; margin: 10px 0; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-alt); color: var(--text); font-size: 0.85rem;
}
.table-scroll { overflow-x: auto; }
table.trade-table { border-collapse: collapse; width: 100%; font-size: 0.78rem; }
table.trade-table th, table.trade-table td { padding: 5px 8px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
table.trade-table th:nth-child(1), table.trade-table th:nth-child(2), table.trade-table th:nth-child(3),
table.trade-table td:nth-child(1), table.trade-table td:nth-child(2), table.trade-table td:nth-child(3) { text-align: left; }
table.trade-table th { position: sticky; top: 0; background: var(--bg-alt); cursor: pointer; user-select: none; }
table.trade-table th:hover { color: var(--accent); }
table.trade-table tr.pnl-neg td:nth-last-child(8) { color: var(--danger); font-weight: 600; }
table.trade-table tr.pnl-pos td:nth-last-child(8) { color: var(--good); }

.manifest-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.manifest-grid pre { background: var(--bg-alt); border: 1px solid var(--border); border-radius: 8px; padding: 12px; overflow-x: auto; font-size: 0.78rem; }
.odds-empty { color: var(--text-dim); font-style: italic; padding: 20px 0; }
footer.odds-footer { color: var(--text-dim); font-size: 0.75rem; padding: 30px 0 10px; text-align: center; }

@media print {
  .odds-nav { position: static; }
  .fig-block { border: none; page-break-inside: avoid; }
  section.odds-section { page-break-before: always; }
  body { background: #fff; color: #000; }
}
"""

TABLE_JS = """
function oddsFilterTable(input, tableId) {
  const q = input.value.toLowerCase();
  const table = document.getElementById(tableId);
  if (!table) return;
  const rows = table.tBodies[0].rows;
  for (let i = 0; i < rows.length; i++) {
    const text = rows[i].innerText.toLowerCase();
    rows[i].style.display = text.includes(q) ? "" : "none";
  }
}

const oddsSortState = {};
function oddsSortTable(colIdx) {
  const table = document.getElementById("trade-table");
  if (!table) return;
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  const asc = !oddsSortState[colIdx];
  oddsSortState[colIdx] = asc;
  rows.sort((a, b) => {
    const av = a.cells[colIdx].innerText.trim();
    const bv = b.cells[colIdx].innerText.trim();
    const an = parseFloat(av), bn = parseFloat(bv);
    let cmp;
    if (!isNaN(an) && !isNaN(bn) && av !== "" && bv !== "") {
      cmp = an - bn;
    } else {
      cmp = av.localeCompare(bv);
    }
    return asc ? cmp : -cmp;
  });
  rows.forEach((r) => tbody.appendChild(r));
}
"""

THEME_JS = """
(function () {
  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
  }
  var saved = null;
  try { saved = localStorage.getItem("odds-lab-theme"); } catch (e) {}
  if (saved) apply(saved);
  window.oddsToggleTheme = function () {
    var cur = document.documentElement.getAttribute("data-theme");
    var next = cur === "dark" ? "light" : "dark";
    apply(next);
    try { localStorage.setItem("odds-lab-theme", next); } catch (e) {}
  };
})();
"""

__all__ = ["CSS", "TABLE_JS", "THEME_JS"]
