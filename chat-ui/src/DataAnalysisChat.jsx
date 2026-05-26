import { useState, useEffect, useRef, useCallback } from "react";
import "./DataAnalysisChat.css";
// ─── Config ──────────────────────────────────────────────────────────────────
const DEFAULT_API = `${window.location.protocol}//${window.location.host}`;
const SESSIONS_KEY = "dac_chat_sessions";
const MAX_SESSIONS = 20;

// ─── Query History Engine ─────────────────────────────────────────────────────
const QH_FREQ_KEY  = "dac_query_freq";   // {query: count}
const QH_RECENT_KEY = "dac_query_recent"; // [query, ...] last 10

function qhTrack(query) {
  const q = query.trim();
  if (!q || q.length < 4) return;
  try {
    // frequency
    const freq = JSON.parse(localStorage.getItem(QH_FREQ_KEY) || "{}");
    freq[q] = (freq[q] || 0) + 1;
    localStorage.setItem(QH_FREQ_KEY, JSON.stringify(freq));
    // recent
    const recent = JSON.parse(localStorage.getItem(QH_RECENT_KEY) || "[]");
    const filtered = recent.filter(r => r.toLowerCase() !== q.toLowerCase());
    filtered.unshift(q);
    localStorage.setItem(QH_RECENT_KEY, JSON.stringify(filtered.slice(0, 20)));
  } catch (error) {
    void error;
  }
}

function qhGetSuggestions() {
  try {
    const freq  = JSON.parse(localStorage.getItem(QH_FREQ_KEY)   || "{}");
    const recent = JSON.parse(localStorage.getItem(QH_RECENT_KEY) || "[]");
    // top queries by frequency (exclude what's already in recent top-3)
    const recentTop3 = recent.slice(0, 3);
    const topByFreq = Object.entries(freq)
      .sort((a, b) => b[1] - a[1])
      .map(([q]) => q)
      .filter(q => !recentTop3.some(r => r.toLowerCase() === q.toLowerCase()))
      .slice(0, 4);
    return { recent: recentTop3, top: topByFreq };
  } catch (error) {
    void error;
    return { recent: [], top: [] };
  }
}

const EXAMPLES = [
  "How many deals are there?",
  "Give me all closed won deals",
  "Show me total revenue by status",
  "Search for John",
  "What can you give me from the data?",
];

// ─── API Layer ───────────────────────────────────────────────────────────────
async function apiGet(base, path, timeout = 5000) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), timeout);
  try {
    const r = await fetch(`${base.replace(/\/+$/, "")}${path}`, { signal: ctrl.signal });
    clearTimeout(id);
    if (!r.ok) throw new Error(r.statusText);
    return await r.json();
  } catch { clearTimeout(id); return null; }
}

// signal is passed from outside (for stop button support)
// timeout=180000 (3 min) — complex agent reports can take up to 90s
async function apiPost(base, path, body, signal, timeout = 180000) {
  const timer = new AbortController();
  const id = setTimeout(() => timer.abort(), timeout);
  // Combine external stop signal + internal timeout signal when both exist
  const combined = signal || timer.signal;
  try {
    const r = await fetch(`${base.replace(/\/+$/, "")}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: combined,
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return await r.json();
  } finally {
    clearTimeout(id); // always cleans up the timer, even on abort/error
  }
}

// ─── Session helpers (localStorage) ──────────────────────────────────────────
function loadSessions() {
  try { return JSON.parse(localStorage.getItem(SESSIONS_KEY) || "[]"); }
  catch { return []; }
}
function saveSessions(sessions) {
  try { localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions.slice(0, MAX_SESSIONS))); }
  catch { /* storage full */ }
}
function sessionTitle(messages) {
  const first = messages.find(m => m.role === "user");
  if (!first) return "New chat";
  return first.content.length > 42 ? first.content.slice(0, 42) + "…" : first.content;
}
function fmtDate(ts) {
  const d = new Date(ts);
  const now = new Date();
  if (d.toDateString() === now.toDateString()) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

// ─── Tiny ID ─────────────────────────────────────────────────────────────────
let _id = 0;
const uid = () => `msg_${++_id}_${Date.now()}`;

// ─── Markdown-lite renderer ──────────────────────────────────────────────────
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function inlineMarkdown(str) {
  return escapeHtml(str)
    .replace(/`([^`]+)`/g, '<code class="md-code">$1</code>')
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" class="md-link">$1</a>');
}

function parseTableBlock(lines) {
  // lines[0] = header row, lines[1] = separator, lines[2..] = data rows
  const parseRow = line =>
    line.replace(/^\||\|$/g, "").split("|").map(c => c.trim());

  const headers = parseRow(lines[0]);
  const rows = lines.slice(2).map(parseRow);

  const th = headers.map(h => `<th>${inlineMarkdown(h)}</th>`).join("");
  const trs = rows.map(cells => {
    const tds = headers.map((_, i) =>
      `<td>${inlineMarkdown(cells[i] ?? "")}</td>`
    ).join("");
    return `<tr>${tds}</tr>`;
  }).join("");

  return `<div class="data-table-wrap"><table class="data-table"><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table></div>`;
}

function renderMarkdown(text) {
  if (!text) return "";
  const lines = text.split("\n");
  const output = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Markdown table
    if (
      line.trimStart().startsWith("|") &&
      lines[i + 1] &&
      /^\|[\s|:-]+\|/.test(lines[i + 1])
    ) {
      const tableLines = [];
      while (i < lines.length && lines[i].trimStart().startsWith("|")) {
        tableLines.push(lines[i]);
        i++;
      }
      if (tableLines.length >= 3) {
        output.push(parseTableBlock(tableLines));
        continue;
      } else {
        tableLines.forEach(l => output.push(`<span>${inlineMarkdown(l)}</span><br/>`));
        continue;
      }
    }

    // Bullet list — collect consecutive items, wrap in <ul>
    if (/^[\s]*[-*+]\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^[\s]*[-*+]\s/.test(lines[i])) {
        items.push(`<li>${inlineMarkdown(lines[i].replace(/^[\s]*[-*+]\s/, ""))}</li>`);
        i++;
      }
      output.push(`<ul class="md-list">${items.join("")}</ul>`);
      continue;
    }

    // Numbered list — collect consecutive items, wrap in <ol>
    if (/^[\s]*\d+[.)]\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^[\s]*\d+[.)]\s/.test(lines[i])) {
        items.push(`<li>${inlineMarkdown(lines[i].replace(/^[\s]*\d+[.)]\s/, ""))}</li>`);
        i++;
      }
      output.push(`<ol class="md-list">${items.join("")}</ol>`);
      continue;
    }

    // Heading
    const headMatch = line.match(/^(#{1,3})\s+(.*)/);
    if (headMatch) {
      const lvl = headMatch[1].length + 1;
      output.push(`<h${lvl} class="md-heading">${inlineMarkdown(headMatch[2])}</h${lvl}>`);
      i++;
      continue;
    }

    // Horizontal rule
    if (/^[-*_]{3,}$/.test(line.trim())) {
      output.push(`<hr class="md-hr"/>`);
      i++;
      continue;
    }

    // Empty line → paragraph break
    if (line.trim() === "") {
      output.push("<br/>");
      i++;
      continue;
    }

    output.push(`<span>${inlineMarkdown(line)}</span><br/>`);
    i++;
  }

  return output.join("");
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function Badge({ children, color = "blue" }) {
  return <span className={`badge badge--${color}`}>{children}</span>;
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <div className="metric-card__value">{value}</div>
      <div className="metric-card__label">{label.replace(/_/g, " ")}</div>
    </div>
  );
}

function DataTable({ rows }) {
  if (!rows?.length) return null;
  const keys = Object.keys(rows[0]);
  return (
    <div className="data-table-wrap">
      <table className="data-table">
        <thead><tr>{keys.map(k => <th key={k}>{k.replace(/_/g, " ")}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {keys.map(k => (
                <td key={k}>{typeof row[k] === "object" ? JSON.stringify(row[k]) : String(row[k] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Collapsible({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="collapsible">
      <button className="collapsible__trigger" onClick={() => setOpen(!open)}>
        <span>{title}</span>
        <span className={`collapsible__arrow${open ? " collapsible__arrow--open" : ""}`}>▾</span>
      </button>
      {open && <div className="collapsible__body">{children}</div>}
    </div>
  );
}

function JsonView({ data }) {
  return <pre className="json-view">{JSON.stringify(data, null, 2)}</pre>;
}

function StructuredData({ data, answerHasTable }) {
  if (!data) return null;

  // If answer already rendered a markdown table, skip rendering the raw array
  // to avoid duplicate display
  if (answerHasTable && Array.isArray(data)) return null;

  const parts = [];

  // Flat array of objects → table
  if (Array.isArray(data) && data.length) {
    if (typeof data[0] === "object" && data[0] !== null) {
      parts.push(
        <Collapsible key="raw-table" title={`Data (${data.length} records)`} defaultOpen={data.length <= 20}>
          <DataTable rows={data} />
        </Collapsible>
      );
    } else {
      // array of primitives
      parts.push(
        <div key="prim-list" className="insights-list">
          {data.map((v, i) => <div key={i} className="insight-item">{String(v)}</div>)}
        </div>
      );
    }
  }

  if (data && typeof data === "object" && !Array.isArray(data)) {
    // metrics block
    if (data.metrics && typeof data.metrics === "object") {
      parts.push(
        <div key="metrics" className="metrics-grid">
          {Object.entries(data.metrics).map(([k, v]) => <MetricCard key={k} label={k} value={v} />)}
        </div>
      );
    }
    // insights block
    if (Array.isArray(data.insights)) {
      parts.push(
        <div key="insights" className="insights-list">
          {data.insights.map((ins, i) => <div key={i} className="insight-item">{ins}</div>)}
        </div>
      );
    }
    // any key with array-of-objects value
    Object.entries(data).forEach(([key, val]) => {
      if (key === "metrics" || key === "insights") return;
      if (Array.isArray(val) && val.length && typeof val[0] === "object") {
        parts.push(
          <Collapsible key={key} title={`${key.replace(/_/g, " ")} (${val.length})`} defaultOpen={val.length <= 10}>
            <DataTable rows={val} />
          </Collapsible>
        );
      } else if (typeof val === "number" || typeof val === "string") {
        parts.push(<MetricCard key={key} label={key} value={val} />);
      }
    });
  }

  if (!parts.length) return null;
  return <div className="structured-data">{parts}</div>;
}

function QueryPlan({ plan }) {
  if (!plan || typeof plan !== "object") return null;
  const items = [];
  if (plan.intent) items.push({ label: `Intent: ${plan.intent}`, color: "amber" });
  if (plan.collection) items.push({ label: `Tables: ${plan.collection}`, color: "blue" });
  if (plan.filters && Object.keys(plan.filters).length) {
    items.push({ label: `Filters: ${Object.entries(plan.filters).map(([k, v]) => `${k}=${v}`).join(", ")}`, color: "purple" });
  }
  if (plan.output_format) items.push({ label: `Output: ${plan.output_format}`, color: "green" });
  if (!items.length) return null;
  return <div className="query-plan">{items.map((it, i) => <Badge key={i} color={it.color}>{it.label}</Badge>)}</div>;
}

// ─── FeedbackButtons ──────────────────────────────────────────────────────────
// Shows 👍 / 👎 after every bot response.
// 👍 → POST /feedback/positive (save golden example, show confirmation)
// 👎 → Show inline correction form → POST /feedback/correction → replace answer
function FeedbackButtons({ msg, apiUrl, onCorrectionApplied, onLearned }) {
  // "idle" | "liked" | "disliked" | "loading_like" | "loading_correct" | "corrected" | "error"
  const [state, setState] = useState("idle");
  const [correctionText, setCorrectionText] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  const query   = msg.resolved_query || msg._userQuery || "";
  const sql     = msg.query_used     || "";
  const summary = typeof msg.content === "string" ? msg.content.slice(0, 200) : "";

  // 👍 handler — save golden example
  const handleLike = async () => {
    if (!query || !sql) { setState("liked"); return; }
    setState("loading_like");
    try {
      await apiPost(apiUrl, "/feedback/positive", {
        query,
        sql,
        result_summary: summary,
      }, null);
      setState("liked");
      if (onLearned) onLearned();
    } catch {
      setState("liked"); // still show confirmation even if save fails
    }
  };

  // 👎 handler — show correction form
  const handleDislike = () => {
    setState("disliked");
    setCorrectionText("");
    setErrorMsg("");
  };

  // Correction submit → regenerate SQL + replace answer
  const handleCorrection = async () => {
    if (!correctionText.trim()) return;
    setState("loading_correct");
    setErrorMsg("");
    try {
      const resp = await apiPost(apiUrl, "/feedback/correction", {
        query,
        sql,
        user_feedback: correctionText.trim(),
        history: [],
      }, null, 120000);

      if (resp && resp.status === "ok" && resp.answer) {
        // Notify parent to update this message's content with the corrected answer
        if (onCorrectionApplied) {
          onCorrectionApplied(msg.id, resp.answer, resp.sql_used || sql);
        }
        if (onLearned) onLearned();
        setState("corrected");
      } else {
        setErrorMsg(resp?.answer || "Could not generate a corrected response. Try rephrasing.");
        setState("disliked");
      }
    } catch (err) {
      setErrorMsg("Request failed — please check the backend is running.");
      setState("disliked");
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────
  if (state === "liked") {
    return (
      <div style={STYLE.confirm}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2.5"><path d="M20 6 9 17l-5-5"/></svg>
        Got it! I&apos;ll remember this approach.
      </div>
    );
  }

  if (state === "corrected") {
    return (
      <div style={STYLE.confirm}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2.5"><path d="M20 6 9 17l-5-5"/></svg>
        Answer updated! Correction saved for future queries.
      </div>
    );
  }

  if (state === "disliked" || state === "loading_correct") {
    const isLoading = state === "loading_correct";
    return (
      <div style={STYLE.correctionBox}>
        <div style={STYLE.correctionLabel}>Tell me what you actually want:</div>
        <textarea
          style={STYLE.correctionTextarea}
          value={correctionText}
          onChange={e => setCorrectionText(e.target.value)}
          placeholder="E.g. 'Show results for 2024 instead' or 'Include company name in results'…"
          rows={3}
          disabled={isLoading}
          autoFocus
        />
        {errorMsg && <div style={STYLE.errorMsg}>{errorMsg}</div>}
        <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
          <button
            style={{...STYLE.proceedBtn, ...(isLoading ? STYLE.proceedBtnDisabled : {})}}
            onClick={handleCorrection}
            disabled={isLoading || !correctionText.trim()}
          >
            {isLoading
              ? <><span style={STYLE.spinner} />Regenerating…</>
              : "Proceed Again →"}
          </button>
          <button
            style={STYLE.cancelBtn}
            onClick={() => setState("idle")}
            disabled={isLoading}
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  if (state === "loading_like") {
    return <div style={STYLE.thumbRow}><span style={{...STYLE.thumbBtn, color:"#9ca3af"}}>Saving…</span></div>;
  }

  // Default: show 👍 / 👎
  return (
    <div style={STYLE.thumbRow}>
      <button style={STYLE.thumbBtn} onClick={handleLike} title="I like this response">
        👍
      </button>
      <button style={STYLE.thumbBtn} onClick={handleDislike} title="I don't like this response">
        👎
      </button>
    </div>
  );
}

// Inline styles — avoids touching the CSS file
const STYLE = {
  thumbRow: {
    display: "flex", gap: 4, marginTop: 6, alignItems: "center",
  },
  thumbBtn: {
    background: "none", border: "1px solid transparent", borderRadius: 6,
    cursor: "pointer", fontSize: "1em", padding: "2px 6px",
    color: "#9ca3af", transition: "all .15s",
    lineHeight: 1.4,
  },
  confirm: {
    display: "flex", alignItems: "center", gap: 5,
    fontSize: ".72em", color: "#6b7280", marginTop: 6,
  },
  correctionBox: {
    marginTop: 8, padding: 10,
    background: "rgba(99,102,241,.06)",
    border: "1px solid rgba(99,102,241,.2)",
    borderRadius: 8, display: "flex", flexDirection: "column", gap: 6,
  },
  correctionLabel: {
    fontSize: ".78em", fontWeight: 600, color: "#9ca3af",
  },
  correctionTextarea: {
    width: "100%", resize: "vertical", borderRadius: 6,
    border: "1px solid rgba(99,102,241,.3)",
    background: "rgba(255,255,255,.04)",
    color: "inherit", padding: "7px 9px", fontSize: ".82em",
    fontFamily: "inherit", outline: "none", boxSizing: "border-box",
  },
  proceedBtn: {
    background: "rgba(99,102,241,.85)", color: "#fff", border: "none",
    borderRadius: 6, padding: "5px 12px", fontSize: ".78em", cursor: "pointer",
    fontWeight: 600, display: "flex", alignItems: "center", gap: 5,
    transition: "background .15s",
  },
  proceedBtnDisabled: { background: "rgba(99,102,241,.35)", cursor: "not-allowed" },
  cancelBtn: {
    background: "none", color: "#9ca3af", border: "1px solid rgba(156,163,175,.2)",
    borderRadius: 6, padding: "5px 10px", fontSize: ".78em", cursor: "pointer",
  },
  errorMsg: {
    fontSize: ".72em", color: "#ef4444", marginTop: 2,
  },
  spinner: {
    display: "inline-block", width: 10, height: 10,
    border: "2px solid rgba(255,255,255,.3)", borderTopColor: "#fff",
    borderRadius: "50%", animation: "spin .7s linear infinite",
  },
};

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  };
  return (
    <button className="btn-copy" onClick={copy} title="Copy response">
      {copied
        ? <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2.5"><path d="M20 6 9 17l-5-5"/></svg>
        : <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
      }
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function TimingBadge({ ms }) {
  if (ms == null) return null;
  const sec = ms / 1000;
  const label = sec < 1 ? `${Math.round(ms)}ms` : `${sec.toFixed(1)}s`;
  const color = ms < 2000 ? "green" : ms < 8000 ? "amber" : "slate";
  return <Badge color={color}>⏱ {label}</Badge>;
}

// ─── PlotlyChart ─────────────────────────────────────────────────────────────
// Renders a Plotly chart inside the bot message bubble.
// Two-effect design: Effect 1 fetches the fig JSON; Effect 2 calls Plotly.newPlot
// after the div is in the DOM. This avoids rendering into a display:none element.
function PlotlyChart({ question, sql, chartData, apiUrl }) {
  const divRef  = useRef(null);
  const [figData, setFigData] = useState(null);  // parsed Plotly fig object
  const [failed, setFailed]   = useState(false);

  // Effect 1 — fetch chart JSON (runs once on mount; StrictMode-safe via cancelled flag)
  useEffect(() => {
    if (!chartData?.length || !window.Plotly) return;
    let cancelled = false;

    fetch(`${apiUrl}/api/chart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question || "Chart", sql: sql || "", data: chartData }),
    })
      .then(r => r.json())
      .then(result => {
        if (cancelled) return;       // StrictMode first-pass — ignore stale fetch
        if (!result.fig) { setFailed(true); return; }
        setFigData(JSON.parse(result.fig));
      })
      .catch(() => { if (!cancelled) setFailed(true); });

    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Effect 2 — call Plotly.newPlot once figData arrives and div is in DOM
  useEffect(() => {
    if (!figData || !divRef.current) return;
    window.Plotly.newPlot(divRef.current, figData.data, figData.layout, {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      toImageButtonOptions: {
        format: "png",
        filename: (question || "chart").slice(0, 40),
        height: 500,
        width: 900,
        scale: 2,
      },
    });
  }, [figData]); // eslint-disable-line react-hooks/exhaustive-deps

  // Nothing to chart
  if (!chartData?.length || failed) return null;

  return (
    <div className="chart-box">
      {!figData && <div className="chart-loading">Generating chart…</div>}
      {/* div always in DOM so Plotly has dimensions when it renders */}
      <div ref={divRef} style={{ width: "100%", minHeight: figData ? 380 : 0 }} />
    </div>
  );
}

function MessageBubble({ msg, apiUrl, onCorrectionApplied, onLearned }) {
  const isUser = msg.role === "user";
  const parsedContent = (() => {
    if (isUser || typeof msg.content !== "string") {
      return { display: msg.content, collection: null, appliedFilters: null };
    }
    const lines = msg.content.split("\n");
    let collection = null;
    let appliedFilters = null;
    const kept = [];
    for (const line of lines) {
      const cleanLine = line.trim();
      if (!collection && /^Collection:/i.test(cleanLine)) {
        collection = cleanLine.replace(/^Collection:\s*/i, "").trim();
        continue;
      }
      if (!appliedFilters && /^Applied filters:/i.test(cleanLine)) {
        appliedFilters = cleanLine.replace(/^Applied filters:\s*/i, "").trim();
        continue;
      }
      kept.push(line);
    }
    return { display: kept.join("\n").trim(), collection, appliedFilters };
  })();
  // Detect if the answer text already contains a rendered table (|...|) to avoid double render
  const answerHasTable = !isUser && typeof parsedContent.display === "string" && /\|.+\|/.test(parsedContent.display);

  return (
    <div className={`message-row message-row--${isUser ? "user" : "bot"}`}>
      <div className={`message-wrapper--${isUser ? "user" : "bot"}`}>
        <div className={`message-meta-row message-meta-row--${isUser ? "user" : "bot"}`}>
          <div className={`message-avatar message-avatar--${isUser ? "user" : "bot"}`}>
            {isUser ? "U" : <img src="/src/assets/elsner-logo.png" alt="AI" style={{width:16,height:16,objectFit:"contain"}}/>}
          </div>
          <span className="message-sender-name">{isUser ? "You" : "ECRM Agent"}</span>
        </div>
        <div className={`bubble bubble--${isUser ? "user" : "bot"}`}>
          <span dangerouslySetInnerHTML={{ __html: renderMarkdown(parsedContent.display || msg.content) }} />
        </div>
        {!isUser && (
          <div className="message-extras">
            {msg.chart_data?.length > 0 && (
              <PlotlyChart
                question={msg._userQuery || ""}
                sql={typeof msg.query_used === "string" ? msg.query_used : ""}
                chartData={msg.chart_data}
                apiUrl={apiUrl}
              />
            )}
            <StructuredData data={msg.data} answerHasTable={answerHasTable} />
            <div className="message-action-row">
              <CopyButton text={msg.content} />
            </div>
            <Collapsible title="Source & Details">
              {/* Timing + agent type + confidence — moved here from meta row */}
              <div style={{display:"flex",flexWrap:"wrap",gap:4,marginBottom:6}}>
                <TimingBadge ms={msg.processing_time_ms} />
                {msg.agent_type && (() => {
                  const at = (msg.agent_type || "").toLowerCase();
                  const label = at.includes("fast") ? "⚡ Fast Path"
                    : at.includes("complex") ? "🔬 Complex"
                    : at.includes("medium")  ? "🔧 Medium"
                    : at.includes("simple")  ? "✦ Simple"
                    : at.includes("guard")   ? "🛡 Guard"
                    : null;
                  const color = at.includes("fast") ? "green"
                    : at.includes("complex") ? "purple"
                    : at.includes("medium")  ? "amber"
                    : at.includes("simple")  ? "blue"
                    : "slate";
                  return label ? <Badge color={color}>{label}</Badge> : null;
                })()}
                {msg.confidence != null && msg.confidence < 1 && (
                  <Badge color={msg.confidence >= 0.7 ? "amber" : "red"}>
                    {Math.round(msg.confidence * 100)}% conf
                  </Badge>
                )}
              </div>
              {(parsedContent.collection || parsedContent.appliedFilters) && (
                <div className="message-perf">
                  {parsedContent.collection && <span>Collection: {parsedContent.collection}</span>}
                  {parsedContent.appliedFilters && <span>Applied filters: {parsedContent.appliedFilters}</span>}
                </div>
              )}
              <QueryPlan plan={msg.query_plan} />
              {msg.query_used && (
                <Collapsible title="SQL Query">
                  {Array.isArray(msg.query_used)
                    ? msg.query_used.map((q, idx) => (
                        <pre key={idx} className="json-view">
                          {q}
                        </pre>
                      ))
                    : (
                        <pre className="json-view">
                          {msg.query_used}
                        </pre>
                      )}
                </Collapsible>
              )}
              {msg.data && !answerHasTable && <Collapsible title="Raw JSON"><JsonView data={msg.data} /></Collapsible>}
              {msg.sources_used?.length > 0 && (
                <div className="message-sources">
                  {msg.sources_used.map((s, i) => <Badge key={i} color="slate">{s}</Badge>)}
                </div>
              )}
              <div className="message-perf">
                {msg.processing_time_ms != null && <span>Total: {Math.round(msg.processing_time_ms)}ms</span>}
                {msg.agent_time_ms != null && <span>Agent: {Math.round(msg.agent_time_ms)}ms</span>}
                {msg.confidence != null && <span>Confidence: {Math.round(msg.confidence * 100)}%</span>}
              </div>
            </Collapsible>
            <FeedbackButtons msg={msg} apiUrl={apiUrl} onCorrectionApplied={onCorrectionApplied} onLearned={onLearned} />
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Sidebar ─────────────────────────────────────────────────────────────────
function Sidebar({ open, onClose, apiUrl, setApiUrl, status, redisOn, sources, learnings,
                   sessions, activeSessionId, onLoadSession, onDeleteSession,
                   onClearChat, onClearCache, onExample,
                   onDeleteLearning, onRefreshLearnings }) {
  return (
    <>
      {open && <div className="sidebar-overlay" onClick={onClose} />}
      <aside className={`sidebar${open ? " sidebar--open" : ""}`}>

        {/* Settings */}
        <div className="sidebar__head">
          <div className="sidebar__header-row">
            <span className="sidebar__title">Settings</span>
            <button className="sidebar__close" onClick={onClose}>✕</button>
          </div>
          <label className="sidebar-field-label">API URL</label>
          <input className="sidebar-url-input" value={apiUrl} onChange={e => setApiUrl(e.target.value)} />
          <div className={`sidebar-status sidebar-status--${status ? "online" : "offline"}`}>
            {status ? "● Backend Online" : "● Backend Offline"}
          </div>
          <div className={`redis-status redis-status--${redisOn ? "on" : "off"}`}>
            <span className="redis-dot" />
            Redis {redisOn ? "Connected" : "Not connected"}
          </div>
          <div className="sidebar-btn-row">
            <button className="btn-small" onClick={onClearChat}>Clear Chat</button>
            <button className="btn-small" onClick={onClearCache}>Clear Cache</button>
          </div>
        </div>

        <hr className="sidebar-divider" />

        {/* Chat History */}
        <div className="sidebar-section">
          <span className="sidebar-section-title">Chat History</span>
          {sessions.length === 0
            ? <div className="sidebar-empty-text">No saved chats yet</div>
            : (
              <div className="session-list">
                {sessions.map(s => (
                  <div key={s.id} className={`session-item${s.id === activeSessionId ? " session-item--active" : ""}`}
                    onClick={() => { onLoadSession(s); onClose(); }}>
                    <div className="session-item__info">
                      <div className="session-item__title">{s.title}</div>
                      <div className="session-item__date">{fmtDate(s.savedAt)}</div>
                    </div>
                    <button className="session-item__delete"
                      onClick={e => { e.stopPropagation(); onDeleteSession(s.id); }}
                      title="Delete">✕</button>
                  </div>
                ))}
              </div>
            )
          }
        </div>

        <hr className="sidebar-divider" />

        {/* Data Sources */}
        <div className="sidebar-section">
          <span className="sidebar-section-title">Data Sources</span>
          {sources && Object.keys(sources).length > 0
            ? Object.entries(sources).map(([stype, items]) =>
                items?.length > 0 && (
                  <div key={stype}>
                    <div className="sidebar-source-type">{stype.replace(/_/g, " ")}</div>
                    {items.map((item, i) => (
                      <div key={i} className="sidebar-source-item">
                        • {typeof item === "object" ? item.name || "unknown" : item}
                      </div>
                    ))}
                  </div>
                ))
            : <div className="sidebar-empty-text">{status ? "No sources configured" : "Start backend to see sources"}</div>
          }
        </div>

        <hr className="sidebar-divider" />

        {/* Examples */}
        <div className="sidebar-section">
          <span className="sidebar-section-title">Try These</span>
          <div className="sidebar-examples">
            {EXAMPLES.map((ex, i) => (
              <button key={i} className="sidebar-example-btn" onClick={() => { onExample(ex); onClose(); }}>{ex}</button>
            ))}
          </div>
        </div>

        <hr className="sidebar-divider" />

        {/* AI Learnings */}
        <div className="sidebar-section sidebar-section--bottom">
          <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:6 }}>
            <span className="sidebar-section-title" style={{ marginBottom:0 }}>
              AI Learnings
              {learnings?.total > 0 && (
                <span style={{
                  marginLeft:7, background:"rgba(99,102,241,.15)", color:"#818cf8",
                  borderRadius:10, padding:"1px 7px", fontSize:".68em", fontWeight:700,
                }}>
                  {learnings.golden_count ?? 0} ✓ · {learnings.correction_count ?? 0} ✗
                </span>
              )}
            </span>
            {status && (
              <button onClick={onRefreshLearnings}
                style={{ background:"none", border:"none", cursor:"pointer", fontSize:".8em",
                         color:"#6b7280", padding:"1px 4px" }} title="Refresh">↻</button>
            )}
          </div>
          {learnings?.entries?.length > 0 ? (
            <div style={{ display:"flex", flexDirection:"column", gap:6, maxHeight:340, overflowY:"auto" }}>
              {learnings.entries.map((entry, idx) => {
                const isGolden = entry.type === "golden";
                const borderColor = isGolden ? "#16a34a" : "#dc2626";
                const bgColor     = isGolden ? "#16a34a0d" : "#dc26260d";
                const label       = isGolden ? "✓ Approved" : "✗ Corrected";
                const labelColor  = isGolden ? "#16a34a" : "#dc2626";
                const ts          = entry.ts ? new Date(entry.ts * 1000).toLocaleDateString() : "";
                return (
                  <div key={idx} style={{
                    borderLeft: `3px solid ${borderColor}`,
                    background: bgColor,
                    borderRadius: "0 6px 6px 0",
                    padding: "7px 9px",
                    fontSize: ".72em",
                    position: "relative",
                  }}>
                    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", gap:4 }}>
                      <span style={{ color: labelColor, fontWeight:700, fontSize:".9em" }}>{label}</span>
                      <div style={{ display:"flex", alignItems:"center", gap:4, flexShrink:0 }}>
                        {ts && <span style={{ color:"#9ca3af", fontSize:".85em" }}>{ts}</span>}
                        <button
                          onClick={() => onDeleteLearning(idx)}
                          style={{ background:"none", border:"none", cursor:"pointer",
                                   color:"#9ca3af", fontSize:"1em", padding:"0 2px",
                                   lineHeight:1, borderRadius:3 }}
                          title="Delete this learning">✕</button>
                      </div>
                    </div>
                    <div style={{ color:"#d1d5db", marginTop:3, fontStyle:"italic" }}>
                      &ldquo;{(entry.query || "").slice(0,120)}&rdquo;
                    </div>
                    {!isGolden && entry.user_feedback && (
                      <div style={{ color:"#fbbf24", marginTop:2 }}>
                        User said: {entry.user_feedback.slice(0,100)}
                      </div>
                    )}
                    {entry.sql && (
                      <details style={{ marginTop:4 }}>
                        <summary style={{ cursor:"pointer", color:"#818cf8" }}>SQL</summary>
                        <pre style={{ margin:"3px 0 0", fontSize:".9em", overflowX:"auto",
                                      whiteSpace:"pre-wrap", wordBreak:"break-all",
                                      color:"#a5b4fc" }}>
                          {entry.sql.slice(0,500)}
                        </pre>
                      </details>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="sidebar-empty-text">
              {status ? "No learnings yet. Rate responses to teach the AI!" : "Start backend to see learnings"}
            </div>
          )}
        </div>
      </aside>
    </>
  );
}

const THINKING_STEPS = [
  "Reading your query...",
  "Understanding the intent...",
  "Classifying query type...",
  "Identifying relevant data source...",
  "Scanning MongoDB collections...",
  "Matching collection schema...",
  "Checking available fields...",
  "Building database filter...",
  "Routing to the right collection...",
  "Running MongoDB query...",
  "Fetching matching documents...",
  "Counting results...",
  "Applying business rules...",
  "Filtering sensitive fields...",
  "Masking internal IDs...",
  "Formatting data for display...",
  "Generating response structure...",
  "Applying date formatting...",
  "Running safety checks...",
  "Preparing final response...",
  "Validating output quality...",
  "Almost there...",
  "Finalizing your answer...",
];

function TypingDots() {
  const [lines, setLines] = useState([THINKING_STEPS[0]]);
  const scrollRef = useRef(null);

  useEffect(() => {
    let idx = 1;
    const timer = setInterval(() => {
      setLines(prev => [...prev, THINKING_STEPS[idx % THINKING_STEPS.length]]);
      idx++;
    }, 1100);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines]);

  return (
    <div className="typing-row">
      <div className="typing-wrapper">
        <div className="typing-meta">
          <div className="typing-avatar">
            <img src="/src/assets/elsner-logo.png" alt="AI" style={{width:16,height:16,objectFit:"contain"}}/>
          </div>
          <span className="message-sender-name">Data Agent</span>
        </div>
        <div className="typing-bubble" style={{padding:"12px 16px",minWidth:280,maxWidth:400}}>
          <div ref={scrollRef} style={{maxHeight:160,overflowY:"hidden",display:"flex",flexDirection:"column",gap:5}}>
            {lines.map((line, i) => {
              const isLast = i === lines.length - 1;
              return (
                <div key={i} style={{
                  fontSize:".82em",
                  fontWeight: isLast ? 600 : 400,
                  color: isLast ? "#72b5e6" : "#b0bec5",
                  animation:"fadeUp .5s cubic-bezier(.22,1,.36,1)",
                  display:"flex", alignItems:"center", gap:7,
                  opacity: isLast ? 1 : Math.max(0.35, 1 - (lines.length - 1 - i) * 0.15),
                  transition:".4s ease, color .4s ease",
                }}>
                  <span style={{fontSize:".68em", color: isLast ? "#72b5e6" : "#cfd8dc", transition:"color .4s ease"}}>
                    {isLast ? "▶" : "✓"}
                  </span>
                  {line}
                  {isLast && <span style={{display:"inline-block",width:5,height:5,borderRadius:"50%",background:"#72b5e6",boxShadow:"0 0 6px #72b5e6",animation:"pulse 1.2s ease-in-out infinite",marginLeft:2}}/>}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onExample }) {
  const { recent, top } = qhGetSuggestions();
  const hasHistory = recent.length > 0 || top.length > 0;

  return (
    <div className="empty-state">
      <div className="empty-icon"><img src="/src/assets/elsner-logo.png" alt="Elsner Analytica AI" className="empty-logo-img" /></div>
      <p className="empty-desc">Ask questions about your data in natural language. I'll query your database and return structured results.</p>

      {hasHistory ? (
        <div style={{width:"100%",maxWidth:520,display:"flex",flexDirection:"column",gap:14}}>
          {recent.length > 0 && (
            <div>
              <div style={{fontSize:".72em",fontWeight:700,color:"#9ca3af",textTransform:"uppercase",letterSpacing:".06em",marginBottom:6}}>
                Recent
              </div>
              <div style={{display:"flex",flexDirection:"column",gap:5}}>
                {recent.map((q, i) => (
                  <button key={i} className="empty-example-btn" onClick={() => onExample(q)}
                    style={{textAlign:"left",display:"flex",alignItems:"center",gap:8}}>
                    <span style={{fontSize:".9em",opacity:.5}}>↩</span> {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          {top.length > 0 && (
            <div>
              <div style={{fontSize:".72em",fontWeight:700,color:"#9ca3af",textTransform:"uppercase",letterSpacing:".06em",marginBottom:6}}>
                Most Asked
              </div>
              <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
                {top.map((q, i) => (
                  <button key={i} className="empty-example-btn" onClick={() => onExample(q)}
                    style={{display:"flex",alignItems:"center",gap:6}}>
                    <span style={{fontSize:".85em",opacity:.5}}>🔥</span> {q}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="empty-examples">
          {EXAMPLES.slice(0, 4).map((ex, i) => (
            <button key={i} className="empty-example-btn" onClick={() => onExample(ex)}>{ex}</button>
          ))}
        </div>
      )}
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// ─── Main App ────────────────────────────────────────────────────────────────
// ═════════════════════════════════════════════════════════════════════════════
export default function DataAnalysisChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [apiUrl, setApiUrl] = useState(DEFAULT_API);
  const [backendUp, setBackendUp] = useState(null); // null=connecting, true=online, false=offline
  const [redisOn, setRedisOn] = useState(false);
  const [sources, setSources] = useState({});
  const [learnings, setLearnings] = useState({});

  // Session state
  const [sessions, setSessions] = useState(loadSessions);
  const [activeSessionId, setActiveSessionId] = useState(null);

  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const abortCtrlRef = useRef(null);   // ← stop button
  const pendingInputRef = useRef(""); // ← restores input on stop

  // ─── Voice ───────────────────────────────────────────────────────────────────
  const [voiceStatus, setVoiceStatus] = useState('idle'); // 'idle'|'listening'|'processing'
  const recognitionRef  = useRef(null);
  const wakeWordRef     = useRef(null);
  const silenceRef      = useRef(null);
  const transcriptBuf   = useRef('');
  const loadingRef      = useRef(false);
  const sendMsgRef      = useRef(null);
  // MediaRecorder fallback for Firefox / Safari / iOS
  const mediaRecRef     = useRef(null);
  const mediaChunksRef  = useRef([]);
  loadingRef.current    = loading; // always fresh

  // ── Health + Redis check ──
  const checkHealth = useCallback(async () => {
    const r = await apiGet(apiUrl, "/health");
    setBackendUp(!!r);
    if (r) {
      const cs = await apiGet(apiUrl, "/cache/status");
      if (cs) setRedisOn(cs.redis_connected);
    }
  }, [apiUrl]);

  useEffect(() => {
    checkHealth();
    const t = setInterval(checkHealth, 15000);
    return () => clearInterval(t);
  }, [checkHealth]);

  useEffect(() => {
    const onResize = () => {
      document.querySelectorAll(".chart-box .js-plotly-plot").forEach(el => {
        if (el.offsetParent !== null && window.Plotly) window.Plotly.Plots.resize(el);
      });
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const refreshLearnings = useCallback(() => {
    apiGet(apiUrl, "/feedback/learnings").then(r => r && setLearnings(r));
  }, [apiUrl]);

  useEffect(() => {
    if (backendUp) {
      apiGet(apiUrl, "/sources").then(r => r && setSources(r));
      refreshLearnings();
    }
  }, [backendUp, apiUrl, refreshLearnings]);

  const deleteLearning = useCallback(async (idx) => {
    try {
      await fetch(`${apiUrl}/feedback/learnings/${idx}`, { method: "DELETE" });
      refreshLearnings();
    } catch {
      // silently ignore — UI will refresh on next open
    }
  }, [apiUrl, refreshLearnings]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // ── Persist sessions to localStorage whenever they change ──
  useEffect(() => { saveSessions(sessions); }, [sessions]);

  // ── Auto-save current chat as a session ──
  const persistSession = useCallback((msgs) => {
    if (!msgs.length) return;
    const id = activeSessionId || `sess_${Date.now()}`;
    if (!activeSessionId) setActiveSessionId(id);
    setSessions(prev => {
      const filtered = prev.filter(s => s.id !== id);
      return [{ id, title: sessionTitle(msgs), messages: msgs, savedAt: Date.now() }, ...filtered];
    });
  }, [activeSessionId]);

  // ── Send message ──
  const sendMessage = async (text) => {
    const trimmed = (text || input).trim();
    if (!trimmed || loading) return;
    qhTrack(trimmed);

    const userMsg = { id: uid(), role: "user", content: trimmed };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setInput("");
    pendingInputRef.current = trimmed;
    setLoading(true);

    // Create abort controller for this request
    const ctrl = new AbortController();
    abortCtrlRef.current = ctrl;

    try {
      const histPayload = messages.slice(-20).map(m => ({ role: m.role, content: m.content || "" }));
      const resp = await apiPost(apiUrl, "/chat", { query: trimmed, history: histPayload }, ctrl.signal);

      const botMsg = {
        id: uid(), role: "assistant",
        content: resp.answer || "(No response)",
        data: resp.data,
        chart_data: resp.chart_data || null,
        sources_used: resp.sources_used || [],
        confidence: resp.confidence,
        processing_time_ms: resp.processing_time_ms,
        agent_time_ms: resp.agent_time_ms,
        query_used: resp.query_used,
        query_plan: resp.query_plan,
        agent_type: resp.agent_type,
        _userQuery: trimmed,
        // Memory resolution metadata — used by FeedbackButtons
        resolved_query: resp.resolved_query || trimmed,
        is_continuation: resp.is_continuation || false,
        memory_reasoning: resp.memory_reasoning || "",
      };
      const finalMessages = [...nextMessages, botMsg];
      setMessages(finalMessages);
      persistSession(finalMessages);
    } catch (err) {
      if (err.name === "AbortError") {
        // Stopped by user — remove the pending user message, restore input
        setMessages(prev => prev.filter(m => m.id !== userMsg.id));
        setInput(pendingInputRef.current);
      } else {
        setMessages(prev => [...prev, {
          id: uid(), role: "assistant",
          content: `Sorry, the backend is unavailable. ${err.message}`,
          _userQuery: trimmed,
        }]);
      }
    }

    abortCtrlRef.current = null;
    setLoading(false);
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  // Keep voice ref to sendMessage always fresh (runs every render after sendMessage is defined)
  sendMsgRef.current = sendMessage;

  // ─── Voice helpers ──────────────────────────────────────────────────────────
  const SR = typeof window !== 'undefined'
    ? (window.SpeechRecognition || window.webkitSpeechRecognition)
    : null;

  function stopWakeWord() {
    if (wakeWordRef.current) {
      try { wakeWordRef.current.abort(); } catch (error) { void error; }
      wakeWordRef.current = null;
    }
  }

  function startWakeWord() {
    if (!SR || wakeWordRef.current) return;
    const ww = new SR();
    ww.continuous = true;
    ww.interimResults = true;
    ww.lang = 'en-US';
    ww.onresult = (ev) => {
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const t = ev.results[i][0].transcript.toLowerCase();
        if (t.includes('hi elsner') || t.includes('hey elsner')) {
          stopWakeWord();
          startListening();
          return;
        }
      }
    };
    ww.onend = () => {
      wakeWordRef.current = null;
      if (!recognitionRef.current) setTimeout(startWakeWord, 400);
    };
    ww.onerror = (e) => {
      wakeWordRef.current = null;
      // If permission not yet granted, don't retry — the permissions watcher will handle it
      if (e.error === 'not-allowed') return;
      // For other errors (network, aborted) retry after a short wait
      if (!recognitionRef.current) setTimeout(startWakeWord, 1000);
    };
    try { ww.start(); wakeWordRef.current = ww; } catch (error) { void error; }
  }

  function startListening() {
    if (!SR) {
      alert('Speech Recognition is not supported.\nPlease use Google Chrome or Microsoft Edge.');
      return;
    }
    stopWakeWord();
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (error) { void error; }
      recognitionRef.current = null;
    }
    transcriptBuf.current = '';
    setVoiceStatus('listening');
    setInput('');

    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = 'en-US';

    rec.onresult = (ev) => {
      let interim = '';
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const t = ev.results[i][0].transcript;
        if (ev.results[i].isFinal) transcriptBuf.current += t + ' ';
        else interim = t;
      }
      setInput((transcriptBuf.current + interim).trim());
      clearTimeout(silenceRef.current);
      silenceRef.current = setTimeout(() => {
        if (recognitionRef.current) {
          try { recognitionRef.current.stop(); } catch (error) { void error; }
        }
      }, 1800);
    };

    rec.onend = () => {
      clearTimeout(silenceRef.current);
      recognitionRef.current = null;
      const final = transcriptBuf.current.trim();
      if (final && !loadingRef.current) {
        setVoiceStatus('processing');
        sendMsgRef.current(final);
      } else {
        setVoiceStatus('idle');
        if (final) setInput(final);
      }
      setTimeout(startWakeWord, 600);
    };

    rec.onerror = (e) => {
      if (e.error === 'no-speech') return;
      if (e.error === 'not-allowed') {
        alert('Microphone access was denied.\nPlease allow microphone access in your browser settings.');
      }
      clearTimeout(silenceRef.current);
      recognitionRef.current = null;
      setVoiceStatus('idle');
      setTimeout(startWakeWord, 600);
    };

    try { rec.start(); recognitionRef.current = rec; } catch { setVoiceStatus('idle'); }
  }

  function stopListening() {
    clearTimeout(silenceRef.current);
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (error) { void error; }
      recognitionRef.current = null;
    }
    setVoiceStatus('idle');
    setTimeout(startWakeWord, 600);
  }

  // ─── MediaRecorder fallback (Firefox / Safari / iOS) ────────────────────────
  function getBestMimeType() {
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
    return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
  }

  function startMediaRecorder() {
    if (!navigator.mediaDevices?.getUserMedia) {
      alert('Microphone access is not supported in this browser.');
      return;
    }
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const mimeType = getBestMimeType();
        const mr = new MediaRecorder(stream, mimeType ? { mimeType } : {});
        mediaChunksRef.current = [];

        mr.ondataavailable = e => { if (e.data.size > 0) mediaChunksRef.current.push(e.data); };

        mr.onstop = async () => {
          stream.getTracks().forEach(t => t.stop());
          const blob = new Blob(mediaChunksRef.current, { type: mimeType || 'audio/webm' });
          const ext  = mimeType.includes('mp4') ? 'mp4' : mimeType.includes('ogg') ? 'ogg' : 'webm';
          setVoiceStatus('processing');

          try {
            const form = new FormData();
            form.append('audio', blob, `recording.${ext}`);
            const res  = await fetch(`${apiUrl}/transcribe`, { method: 'POST', body: form });
            const data = await res.json();
            const text = (data.text || '').trim();
            if (text && !loadingRef.current) {
              setInput(text);
              sendMsgRef.current(text);
            } else {
              setInput(text);
              setVoiceStatus('idle');
            }
          } catch {
            setVoiceStatus('idle');
            alert('Transcription failed. Check that the backend is running.');
          }
        };

        mr.start();
        mediaRecRef.current = mr;
        setVoiceStatus('listening');
      })
      .catch(e => {
        if (e.name === 'NotAllowedError') {
          alert('Microphone access was denied.\nPlease allow microphone access in your browser settings.');
        }
        setVoiceStatus('idle');
      });
  }

  function stopMediaRecorder() {
    if (mediaRecRef.current && mediaRecRef.current.state !== 'inactive') {
      mediaRecRef.current.stop();
    }
    mediaRecRef.current = null;
  }

  function toggleVoice() {
    const hasWebSpeech = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
    const hasMediaRec  = !!(window.MediaRecorder && navigator.mediaDevices);

    if (!hasWebSpeech && !hasMediaRec) {
      alert('Voice input is not supported in this browser.');
      return;
    }

    if (voiceStatus === 'listening') {
      if (hasWebSpeech) stopListening();
      else stopMediaRecorder();
    } else {
      if (hasWebSpeech) startListening();
      else startMediaRecorder();
    }
  }

  // ── Stop request ──
  const stopMessage = () => {
    abortCtrlRef.current?.abort();
  };

  // ── New chat — save current session first ──
  const clearChat = () => {
    if (messages.length) persistSession(messages);
    setMessages([]);
    setActiveSessionId(null);
  };

  // ── Load a saved session ──
  const loadSession = (session) => {
    if (messages.length) persistSession(messages);
    setMessages(session.messages);
    setActiveSessionId(session.id);
  };

  // ── Delete a session ──
  const deleteSession = (id) => {
    setSessions(prev => prev.filter(s => s.id !== id));
    if (activeSessionId === id) {
      setMessages([]);
      setActiveSessionId(null);
    }
  };

  const clearCache = async () => {
    try { await apiPost(apiUrl, "/cache/clear", {}, null); } catch (error) { void error; }
  };

  // ── Correction applied callback ──
  // When user submits a 👎 correction, FeedbackButtons calls this to update
  // the message in-place with the corrected answer and new SQL.
  const handleCorrectionApplied = (msgId, newAnswer, newSql) => {
    setMessages(prev => prev.map(m =>
      m.id === msgId
        ? { ...m, content: newAnswer, query_used: newSql || m.query_used }
        : m
    ));
  };

  // ── Voice: reset 'processing' status when API call finishes ──
  useEffect(() => {
    if (!loading && voiceStatus === 'processing') setVoiceStatus('idle');
  }, [loading]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Voice: start wake-word listener once mic permission is granted ──
  // (Wake word only works with Web Speech API — Chrome/Edge)
  useEffect(() => {
    const hasWebSpeech = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
    if (!hasWebSpeech) return; // Firefox/Safari/iOS use the mic button + Whisper, no wake word

    let permStatus = null;

    function tryStartWakeWord() {
      setTimeout(startWakeWord, 300);
    }

    if (navigator.permissions) {
      navigator.permissions.query({ name: 'microphone' })
        .then(status => {
          permStatus = status;
          if (status.state === 'granted') tryStartWakeWord();
          status.onchange = () => {
            if (status.state === 'granted') tryStartWakeWord();
            else stopWakeWord();
          };
        })
        .catch(() => tryStartWakeWord());
    } else {
      tryStartWakeWord();
    }

    return () => {
      if (permStatus) permStatus.onchange = null;
      stopWakeWord();
      if (recognitionRef.current) {
        try { recognitionRef.current.stop(); } catch (error) { void error; }
      }
      clearTimeout(silenceRef.current);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="chat-root">
      <Sidebar
        open={sidebarOpen} onClose={() => setSidebarOpen(false)}
        apiUrl={apiUrl} setApiUrl={setApiUrl}
        status={backendUp} redisOn={redisOn}
        sources={sources} learnings={learnings}
        onDeleteLearning={deleteLearning} onRefreshLearnings={refreshLearnings}
        sessions={sessions} activeSessionId={activeSessionId}
        onLoadSession={loadSession} onDeleteSession={deleteSession}
        onClearChat={clearChat} onClearCache={clearCache}
        onExample={sendMessage}
      />

      <header className="chat-header">
        <button className="btn-menu" onClick={() => setSidebarOpen(true)}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <line x1="3" y1="6" x2="21" y2="6"/>
            <line x1="3" y1="12" x2="21" y2="12"/>
            <line x1="3" y1="18" x2="21" y2="18"/>
          </svg>
        </button>
        <div className="chat-header__title">
          Elsner Analytica AI
          <span className={`chat-status-badge chat-status-badge--${
            backendUp === null ? "connecting" : backendUp ? "online" : "offline"
          }`}>
            {backendUp === null ? "Connecting…" : backendUp ? "Online" : "Offline"}
          </span>
        </div>
        <button className="btn-new-chat" onClick={clearChat}>New Chat</button>
      </header>

      <div className="chat-messages-area">
        <div className="chat-messages-inner">
          {messages.length === 0 && !loading
            ? <EmptyState onExample={sendMessage} />
            : (
              <>
                {messages.map(m => (
                  <MessageBubble
                    key={m.id}
                    msg={m}
                    apiUrl={apiUrl}
                    onCorrectionApplied={handleCorrectionApplied}
                    onLearned={refreshLearnings}
                  />
                ))}
                {loading && <TypingDots />}
              </>
            )
          }
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="chat-input-bar">
        <div className={`chat-input-inner${voiceStatus === 'listening' ? ' chat-input-inner--listening' : ''}`}>
          <textarea
            ref={inputRef}
            className="chat-textarea"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
            placeholder={voiceStatus === 'listening' ? 'Listening… speak now' : 'Ask me anything about your data…'}
            rows={1}
            disabled={loading}
            onInput={e => {
              e.target.style.height = "auto";
              e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
            }}
          />

          {/* ── Mic Button ── */}
          {!loading && (
            <button
              className={`btn-mic${voiceStatus === 'listening' ? ' btn-mic--active' : ''}`}
              onClick={toggleVoice}
              title={voiceStatus === 'listening' ? 'Stop recording' : 'Voice input  (or say "Hi Elsner")'}
            >
              {voiceStatus === 'listening' ? (
                <span className="voice-wave">
                  <span /><span /><span />
                </span>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="9" y="2" width="6" height="11" rx="3"/>
                  <path d="M5 10a7 7 0 0 0 14 0"/>
                  <line x1="12" y1="19" x2="12" y2="22"/>
                  <line x1="8" y1="22" x2="16" y2="22"/>
                </svg>
              )}
            </button>
          )}

          {loading ? (
            <button className="btn-stop" onClick={stopMessage} title="Stop">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <rect x="4" y="4" width="16" height="16" rx="2"/>
              </svg>
            </button>
          ) : (
            <button className="btn-send" onClick={() => sendMessage()} disabled={!input.trim()}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
            </button>
          )}
        </div>

        {/* ── Voice status bar ── */}
        {voiceStatus !== 'idle' && (
          <div className="voice-status-bar">
            {voiceStatus === 'listening' && <span className="voice-status-dot" />}
            {voiceStatus === 'listening' ? 'Listening… speak now, auto-stops after silence'
              : voiceStatus === 'processing' ? 'Processing your voice query…'
              : null}
          </div>
        )}

        <div className="chat-hint">
          Press Enter to send · Shift+Enter for new line · Click 🎤 or say &quot;Hi Elsner&quot; for voice
        </div>
      </div>
    </div>
  );
}
