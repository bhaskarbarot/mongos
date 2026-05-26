# Plotly Chart Integration — Complete Implementation Guide
## How Vanna AI Does It + How to Clone It in Any Chatbot

---

## SECTION 1 — What Happens End-to-End in Vanna

```
User asks a question (natural language)
          │
          ▼
Vanna LLM generates SQL query
          │
          ▼
SQL runs against PostgreSQL → returns pandas DataFrame
          │
          ├─── Answer text sent to frontend
          │
          ▼
generate_plotly_code(question, sql, df.dtypes)
   └── LLM writes Python Plotly code as a string
          │
          ▼
get_plotly_figure(plotly_code, df)
   └── exec() runs the Python string → produces a Plotly `fig` object
          │
          ▼
fig.to_json()
   └── Converts Plotly figure → JSON string
          │
          ▼
Flask returns: { "type": "plotly_figure", "id": "...", "fig": "JSON_STRING" }
          │
          ▼
Frontend: JSON.parse(data.fig) → Plotly.newPlot(divEl, fig.data, fig.layout, config)
          │
          ▼
Chart renders in browser
```

---

## SECTION 2 — Vanna Backend Code (How It Actually Works)

### File: `vanna/legacy/base/base.py`

#### Step 1 — LLM Generates Python Plotly Code

```python
def generate_plotly_code(self, question, sql, df_metadata) -> str:
    system_msg = f"The following is a pandas DataFrame from: '{question}'"
    system_msg += f"\n\nSQL used: {sql}\n\n"
    system_msg += f"DataFrame info:\n{df_metadata}"

    message_log = [
        self.system_message(system_msg),
        self.user_message(
            "Generate Python plotly code to chart the dataframe. "
            "The dataframe is called 'df'. "
            "If only one value, use an Indicator. "
            "Respond with ONLY Python code."
        ),
    ]
    code = self.submit_prompt(message_log)
    return self._sanitize_plotly_code(code)  # strips fig.show()
```

Called with:
```python
code = vn.generate_plotly_code(
    question=question,
    sql=sql,
    df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
)
```

#### Step 2 — exec() Runs the Code, Extracts `fig`

```python
def get_plotly_figure(self, plotly_code: str, df: pd.DataFrame) -> go.Figure:
    ldict = {"df": df, "px": px, "go": go}
    exec(plotly_code, globals(), ldict)
    fig = ldict.get("fig", None)
    # if exec fails → fallback heuristics:
    #   2 numeric cols → scatter
    #   1 numeric + 1 categorical → bar
    #   1 categorical → pie
    return fig
```

#### Step 3 — Flask Endpoint

```python
@flask_app.route("/api/v0/generate_plotly_figure", methods=["GET"])
def generate_plotly_figure():
    id = request.args.get("id")
    chart_instructions = request.args.get("chart_instructions")  # optional user tweak

    df = cache.get(id=id, field="df")
    question = cache.get(id=id, field="question")
    sql = cache.get(id=id, field="sql")

    if not chart_instructions:
        code = cache.get(id=id, field="plotly_code")   # use cached code
    else:
        # user said "make it a pie chart" → re-run LLM with instruction
        question = f"{question}. Use these chart instructions: {chart_instructions}"
        code = vn.generate_plotly_code(question=question, sql=sql,
                                       df_metadata=f"{df.dtypes}")
        cache.set(id=id, field="plotly_code", value=code)

    fig = vn.get_plotly_figure(plotly_code=code, df=df, dark_mode=False)
    fig_json = fig.to_json()

    cache.set(id=id, field="fig_json", value=fig_json)

    return jsonify({
        "type": "plotly_figure",
        "id": id,
        "fig": fig_json,     # ← this is a JSON STRING, not an object
    })
```

API response shape:
```json
{
  "type": "plotly_figure",
  "id": "abc123",
  "fig": "{\"data\":[{\"type\":\"bar\",...}],\"layout\":{\"title\":...}}"
}
```

> **CRITICAL**: `fig` is a **JSON string** inside the JSON response.
> You MUST call `JSON.parse(data.fig)` before passing to Plotly.

---

## SECTION 3 — Heuristic Chart Generator (No LLM — Cheaper & Faster)

### File: `vanna/integrations/plotly/chart_generator.py`

This auto-picks the chart type based on DataFrame column types. No LLM call, no tokens spent.

| DataFrame Shape | Chart Chosen |
|---|---|
| 4+ columns | Plotly Table |
| 1 numeric column | Histogram |
| 1 categorical + 1 numeric | Bar chart |
| 2 numeric columns | Scatter plot |
| 3+ numeric columns | Correlation heatmap |
| Datetime column + numeric(s) | Time-series line chart |
| 2+ categorical columns | Grouped bar chart |

```python
from vanna.integrations.plotly.chart_generator import PlotlyChartGenerator

generator = PlotlyChartGenerator()
fig_dict = generator.generate_chart(df, title="Revenue by Customer")
# fig_dict is a plain dict (already parsed), pass directly to jsonify
```

---

## SECTION 4 — Vanna Frontend Rendering (How the UI Does It)

### HTML: Load Plotly CDN

```html
<!-- Latest (large, ~3.5MB) -->
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<!-- Pinned version (recommended for production) -->
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
```

### HTML: Chart Container

```html
<div id="chart-container" style="width:100%; height:400px;"></div>
```

### JS: First Render

```javascript
async function renderChart(queryId) {
  const resp = await fetch(`/api/v0/generate_plotly_figure?id=${queryId}`);
  const data = await resp.json();

  if (data.type !== "plotly_figure" || !data.fig) return;

  // fig IS A JSON STRING — must parse it first
  const fig = JSON.parse(data.fig);

  const el = document.getElementById("chart-container");

  Plotly.newPlot(el, fig.data, fig.layout, {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    toImageButtonOptions: {
      format: "png",
      filename: "chart",
      height: 500,
      width: 900,
      scale: 2,
    },
  });
}
```

### JS: Re-render When User Asks to Change Chart Type

```javascript
async function redrawChart(queryId, instructions) {
  const url = `/api/v0/generate_plotly_figure?id=${queryId}&chart_instructions=${encodeURIComponent(instructions)}`;
  const resp = await fetch(url);
  const data = await resp.json();
  const fig = JSON.parse(data.fig);
  // Use Plotly.react (not newPlot) to preserve zoom/pan state
  Plotly.react(document.getElementById("chart-container"), fig.data, fig.layout);
}
```

### JS: Handle Window Resize

```javascript
window.addEventListener("resize", () => {
  Plotly.Plots.resize(document.getElementById("chart-container"));
});
```

---

## SECTION 5 — HOW TO IMPLEMENT IN YOUR OWN CHATBOT (Step by Step)

Your chatbot already has a `/chat` endpoint that runs SQL and returns rows.
You need to add chart generation on top of it.

### STEP 1 — Install Dependencies

```bash
pip install plotly pandas
```

### STEP 2 — Add Chart Backend (Python/Flask)

Add this to your Flask app:

```python
import json
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from flask import request, jsonify


# ── Option A: Heuristic chart (no LLM, instant, free) ────────────────────────
def generate_chart_heuristic(df: pd.DataFrame, title: str = "Chart") -> str:
    """Returns fig.to_json() string. Use JSON.parse() on frontend."""
    if df.empty:
        return None

    numeric  = df.select_dtypes(include="number").columns.tolist()
    categ    = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime = df.select_dtypes(include=["datetime64"]).columns.tolist()

    fig = None

    # 4+ columns → table
    if len(df.columns) >= 4:
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns),
                        fill_color="#CC785C", font=dict(color="white", size=12), align="left"),
            cells=dict(values=[df[c].tolist() for c in df.columns],
                       fill_color=[["#FAF9F7" if i%2==0 else "white" for i in range(len(df))]],
                       font=dict(color="#1C1815", size=11), align="left")
        )])
        fig.update_layout(title=title)

    # Datetime + numeric → line chart
    elif datetime and numeric:
        fig = go.Figure()
        for col in numeric[:5]:
            fig.add_trace(go.Scatter(x=df[datetime[0]], y=df[col], mode="lines", name=col))
        fig.update_layout(title=title, xaxis_title=datetime[0], yaxis_title="Value", hovermode="x unified")

    # 1 numeric only → histogram
    elif len(numeric) == 1 and not categ:
        fig = px.histogram(df, x=numeric[0], title=title)

    # 1 categorical + 1 numeric → bar
    elif len(numeric) == 1 and len(categ) == 1:
        agg = df.groupby(categ[0])[numeric[0]].sum().reset_index()
        fig = px.bar(agg, x=categ[0], y=numeric[0], title=title)

    # 2 numeric → scatter
    elif len(numeric) == 2:
        fig = px.scatter(df, x=numeric[0], y=numeric[1], title=title)

    # 3+ numeric → correlation heatmap
    elif len(numeric) >= 3:
        corr = df[numeric].corr()
        fig = px.imshow(corr, title=title, zmin=-1, zmax=1,
                        color_continuous_scale=["#023d60", "#FAF9F7", "#CC785C"])

    # 2+ categorical → grouped bar
    elif len(categ) >= 2:
        grp = df.groupby(categ[:2]).size().reset_index(name="count")
        fig = px.bar(grp, x=categ[0], y="count", color=categ[1], barmode="group", title=title)

    # Fallback → bar of first 2 cols
    elif len(df.columns) >= 2:
        fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)

    if fig is None:
        return None

    # Apply brand styling
    fig.update_layout(
        paper_bgcolor="#FAF9F7",
        plot_bgcolor="#FAF9F7",
        font=dict(color="#1C1815", family="DM Sans, Segoe UI, sans-serif"),
        colorway=["#CC785C", "#B8624A", "#8B5E3C", "#5C554F", "#9B948D"],
    )

    return fig.to_json()   # returns a STRING


# ── Option B: LLM-generated chart (smarter, costs tokens) ───────────────────
def generate_chart_llm(question: str, sql: str, df: pd.DataFrame, llm_fn) -> str:
    """
    llm_fn: callable that takes a prompt string and returns code string.
    Returns fig.to_json() string.
    """
    df_info = str(df.dtypes)
    prompt = (
        f"Question: {question}\n"
        f"SQL: {sql}\n"
        f"DataFrame dtypes:\n{df_info}\n\n"
        "Write Python plotly code to chart this DataFrame. "
        "Import plotly.express as px and plotly.graph_objects as go. "
        "DataFrame is already loaded as variable 'df'. "
        "Store result in a variable called 'fig'. "
        "Do NOT call fig.show(). Respond with ONLY Python code."
    )
    code = llm_fn(prompt)
    code = code.replace("fig.show()", "").strip()
    # Strip markdown fences if LLM added them
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])

    ldict = {"df": df, "px": px, "go": go}
    try:
        exec(code, {}, ldict)
        fig = ldict.get("fig")
        if fig:
            return fig.to_json()
    except Exception:
        pass

    # Fallback to heuristic if exec fails
    return generate_chart_heuristic(df, title=question)


# ── Flask endpoint ────────────────────────────────────────────────────────────
@app.route("/api/chart", methods=["POST"])
def chart_endpoint():
    body     = request.get_json()
    question = body.get("question", "Chart")
    sql      = body.get("sql", "")
    rows     = body.get("data", [])          # list of dicts from your /chat endpoint

    if not rows:
        return jsonify({"error": "no data"}), 400

    df = pd.DataFrame(rows)

    # Use heuristic (swap generate_chart_llm for LLM-powered charts)
    fig_json = generate_chart_heuristic(df, title=question)

    if not fig_json:
        return jsonify({"error": "cannot visualize this data"}), 422

    return jsonify({"fig": fig_json})
```

### STEP 3 — Add Plotly to Your HTML

```html
<head>
  <!-- Add this ONE line inside <head> -->
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
```

### STEP 4 — Add Chart Container to Your Chat Message HTML

Inside the bot message bubble, add:

```html
<!-- Chart renders here — hidden until data is ready -->
<div class="chart-box" id="chart-{{msg_id}}" style="display:none; width:100%; height:380px; margin-top:12px;"></div>

<!-- Optional: button to toggle chart visibility -->
<button onclick="toggleChart('{{msg_id}}')">Show Chart</button>
```

Or dynamically in JS when building the message:
```javascript
const msgHtml = `
  <div class="bot-message">
    <p>${answer}</p>
    <div id="chart-${msgId}" style="width:100%;height:380px;margin-top:12px;display:none;"></div>
  </div>
`;
```

### STEP 5 — Call the Chart API and Render

After your `/chat` endpoint returns data, call this:

```javascript
async function renderChart(msgId, question, sql, rows) {
  // rows = the data array returned by your /chat endpoint

  const chartEl = document.getElementById(`chart-${msgId}`);
  if (!chartEl || !rows || rows.length === 0) return;

  // Show a loading state
  chartEl.style.display = "block";
  chartEl.innerHTML = '<div style="padding:20px;color:#9A8E84;font-size:.85rem;">Generating chart…</div>';

  try {
    const resp = await fetch("/api/chart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, sql, data: rows }),
    });

    const result = await resp.json();

    if (!result.fig) {
      chartEl.style.display = "none";
      return;
    }

    // IMPORTANT: fig is a JSON STRING — must parse before passing to Plotly
    const fig = JSON.parse(result.fig);

    chartEl.innerHTML = "";   // clear loading text

    Plotly.newPlot(chartEl, fig.data, fig.layout, {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      toImageButtonOptions: {
        format: "png",
        filename: question.slice(0, 40),
        height: 500,
        width: 900,
        scale: 2,
      },
    });

  } catch (err) {
    chartEl.style.display = "none";
    console.error("Chart error:", err);
  }
}
```

### STEP 6 — Wire It Into Your Chat Flow

In your existing send/receive flow, add the chart call after the response arrives:

```javascript
async function sendMessage(question) {
  // ... your existing chat code ...

  const resp = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await resp.json();

  const msgId = generateId();   // some unique ID per message

  // Render text answer as usual
  appendBotMessage(msgId, data.answer);

  // ← ADD THIS: render chart if data rows came back
  if (data.data && data.data.length > 0) {
    renderChart(msgId, question, data.sql || "", data.data);
  }
}
```

### STEP 7 — Handle Resize

Add once, globally:

```javascript
window.addEventListener("resize", () => {
  document.querySelectorAll(".chart-box").forEach(el => {
    if (el.offsetParent !== null) {   // only visible charts
      Plotly.Plots.resize(el);
    }
  });
});
```

---

## SECTION 6 — Apply Elsner Brand Colors to Plotly

### In Python (before fig.to_json()):

```python
fig.update_layout(
    paper_bgcolor="#FAF9F7",        # page background
    plot_bgcolor="#FAF9F7",         # chart area background
    font=dict(
        color="#1C1815",            # dark text
        family="DM Sans, Segoe UI, sans-serif"
    ),
    colorway=["#CC785C", "#B8624A", "#8B5E3C", "#5C554F", "#9B948D"],
    title=dict(font=dict(color="#1C1815")),
)
```

### In JavaScript (after Plotly.newPlot()):

```javascript
Plotly.relayout(chartEl, {
  "paper_bgcolor": "#FAF9F7",
  "plot_bgcolor": "#FAF9F7",
  "font.color": "#1C1815",
  "font.family": "DM Sans, Segoe UI, sans-serif",
  "colorway": ["#CC785C", "#B8624A", "#8B5E3C", "#5C554F", "#9B948D"],
});
```

---

## SECTION 7 — Your Chat Endpoint Must Return These Fields

For chart rendering to work, your `/chat` backend response must include:

```json
{
  "answer": "There are 42 deals in the pipeline.",
  "sql": "SELECT * FROM deals WHERE status = 'open'",
  "data": [
    {"deal_name": "ABC Corp", "value": 50000, "stage": "Proposal"},
    {"deal_name": "XYZ Ltd",  "value": 30000, "stage": "Negotiation"}
  ],
  "row_count": 2
}
```

- `data` — array of row dicts (from `df.to_dict(orient="records")`)
- `sql` — the SQL query used (optional, but good for the LLM chart method)
- `answer` — the human-readable answer text

If your `/chat` already returns `data` as a list of dicts, **you only need to add the `/api/chart` endpoint** — no other backend changes required.

---

## SECTION 8 — Key Rules and Gotchas

| # | Rule | Why |
|---|---|---|
| 1 | `fig` from API is a **JSON string** | Always `JSON.parse(data.fig)` — Plotly needs an object, not a string |
| 2 | Use `Plotly.newPlot` first time | Creates the chart with full config |
| 3 | Use `Plotly.react` to update | Preserves zoom/pan state when re-rendering |
| 4 | `responsive: true` in config | Chart auto-fills container width |
| 5 | Set container to `width:100%; height:380px` | Plotly needs explicit height |
| 6 | `exec()` is only safe on trusted backend | Never run on user-controlled input |
| 7 | Strip `fig.show()` from LLM code | It will crash the server process if not removed |
| 8 | Strip markdown fences from LLM output | LLM often wraps code in ```python ... ``` |
| 9 | Fallback heuristic if exec fails | LLM can write broken code; always have a fallback |
| 10 | CDN `plotly-latest.min.js` = ~3.5 MB | Use pinned version in production for speed |

---

## SECTION 9 — Minimal Working Example (Copy-Paste)

### Backend (add to your Flask app)

```python
import json, plotly.express as px, plotly.graph_objects as go
import pandas as pd
from flask import request, jsonify

@app.route("/api/chart", methods=["POST"])
def api_chart():
    body = request.get_json()
    rows = body.get("data", [])
    title = body.get("question", "Chart")

    if not rows:
        return jsonify({"error": "no data"}), 400

    df = pd.DataFrame(rows)
    numeric = df.select_dtypes(include="number").columns.tolist()
    categ   = df.select_dtypes(include=["object","category"]).columns.tolist()

    if len(df.columns) >= 4:
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), fill_color="#CC785C",
                        font=dict(color="white"), align="left"),
            cells=dict(values=[df[c].tolist() for c in df.columns], align="left")
        )])
    elif len(numeric)==1 and len(categ)==1:
        fig = px.bar(df.groupby(categ[0])[numeric[0]].sum().reset_index(),
                     x=categ[0], y=numeric[0], title=title)
    elif len(numeric)==2:
        fig = px.scatter(df, x=numeric[0], y=numeric[1], title=title)
    elif len(numeric)==1:
        fig = px.histogram(df, x=numeric[0], title=title)
    elif len(df.columns)>=2:
        fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
    else:
        return jsonify({"error": "cannot chart"}), 422

    fig.update_layout(paper_bgcolor="#FAF9F7", plot_bgcolor="#FAF9F7",
                      font=dict(color="#1C1815"),
                      colorway=["#CC785C","#B8624A","#8B5E3C"])
    return jsonify({"fig": fig.to_json()})
```

### Frontend (add to your HTML)

```html
<!-- In <head> -->
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

<!-- In each bot message -->
<div id="chart-MSGID" style="width:100%;height:380px;display:none;"></div>
```

```javascript
// Call this after your chat response arrives
async function showChart(msgId, question, sql, rows) {
  const el = document.getElementById("chart-" + msgId);
  if (!el || !rows || !rows.length) return;
  el.style.display = "block";

  const r = await fetch("/api/chart", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({question, sql, data: rows})
  });
  const d = await r.json();
  if (!d.fig) { el.style.display = "none"; return; }

  const fig = JSON.parse(d.fig);          // ← must parse
  Plotly.newPlot(el, fig.data, fig.layout, {responsive:true});
}
```

---

## SECTION 10 — LLM vs Heuristic: Which to Use?

| | LLM-Generated | Heuristic |
|---|---|---|
| **Accuracy** | Higher — LLM understands context | Rule-based, may miss nuance |
| **Speed** | ~2-5 seconds extra | Instant |
| **Cost** | Uses LLM tokens | Free |
| **Reliability** | May produce broken code | Always works |
| **Best for** | Complex data, custom charts | Standard results, dashboards |

**Recommendation**: Start with heuristic. Add LLM option as a "Regenerate chart" button.
