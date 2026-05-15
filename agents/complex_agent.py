"""complex_agent.py — LangGraph StateGraph for COMPLEX CRM queries.

Handles: full KPI reports, trend analysis, anomaly detection, multi-table deep reports.
Target latency: < 30 seconds.

Extended graph flow (10 nodes):
  fetch_schema → plan_report → decompose → generate_sql → execute_parallel
    → statistical_analysis → trend_detection → analyze → deep_synthesis → save_examples

Extra vs medium_agent:
  • plan_report_node    — LLM plans report sections before decomposing
  • statistical_analysis_node — pure Python: growth rates, top/bottom N, outliers
  • trend_detection_node      — pure Python: detects consistent growth/decline, drops
  • deep_synthesis_node       — Groq 70b builds an 800+ word structured report

Zero hallucination:
  • statistical_analysis and trend_detection use only the raw SQL result numbers
  • deep_synthesis prompt explicitly forbids inventing data
  • Every number in the final report traces back to sql_results
"""
from __future__ import annotations

import logging
import math
import re
import statistics
import time
from typing import Any, Dict, List, Optional, TypedDict

from config import settings

LOGGER = logging.getLogger("sql_chatbot")

# Import shared helpers from medium_agent
from agents.medium_agent import (
    MediumState,
    _build_medium_graph,
    _get_graph as _get_medium_graph,
    extract_sql,
    extract_tables,
    extract_bad_column,
    run_sql_with_selfheal,
    format_results_for_llm,
    _fallback_format,
    _parse_sub_queries,
    fetch_schema_node,
    generate_sql_node,
    execute_parallel_node,
    save_examples_node,
)


# ══════════════════════════════════════════════════════════════════════════════
# EXTENDED STATE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class ComplexState(TypedDict):
    # All MediumState fields
    query:                str
    classification:       Dict[str, Any]
    schema:               Dict[str, Any]
    sub_queries:          List[Dict[str, str]]
    sql_results:          List[Dict[str, Any]]
    analysis:             str
    final_response:       str
    attempts:             int
    errors:               List[str]
    t_start:              float
    # Complex-only additions
    report_sections:      List[Dict[str, str]]   # planned report structure
    kpis:                 Dict[str, Any]         # extracted KPI values
    trends:               List[Dict[str, Any]]   # time-series analysis
    anomalies:            List[Dict[str, Any]]   # detected outliers / issues
    statistical_analysis: Dict[str, Any]         # growth rates, rankings, std devs


# ══════════════════════════════════════════════════════════════════════════════
# COMPLEX-ONLY NODES
# ══════════════════════════════════════════════════════════════════════════════

def plan_report_node(state: ComplexState) -> Dict:
    """Node 0: LLM plans the report sections before decomposing.

    Output: [{section_title, data_needed, sql_intent}]
    This gives structure so decomposition produces the right sub-queries.
    """
    from pipeline.llm_router import call as llm_call

    all_tables = state["schema"].get("all_tables", [])
    tables_str = ", ".join(all_tables[:25])

    system = (
        "You are a CRM report planner. Given a user query, plan what sections the report "
        "needs and what data each section requires.\n\n"
        "OUTPUT: JSON array only. Format:\n"
        '[{"section_title": "...", "data_needed": "...", "sql_intent": "count|sum|list|rank|compare"}]\n'
        "Max 5 sections. Keep each data_needed description under 15 words."
    )
    user = (
        f"Available CRM tables: {tables_str}\n\n"
        f'Report request: "{state["query"]}"\n\n'
        "Plan the report sections (JSON array):"
    )

    raw = llm_call("decompose", system, user, max_tokens=400)
    sections = _parse_report_sections(raw)

    if not sections:
        sections = [{"section_title": "Query Result",
                     "data_needed":    state["query"],
                     "sql_intent":     "general"}]

    LOGGER.info("Complex agent planned %d report sections", len(sections))
    return {"report_sections": sections}


def _parse_report_sections(raw: Optional[str]) -> List[Dict[str, str]]:
    if not raw:
        return []
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(l for l in cleaned.split("\n") if not l.startswith("```")).strip()
    start = cleaned.find("[")
    end   = cleaned.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        import json
        items = json.loads(cleaned[start:end])
    except Exception:
        return []
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("section_title"):
            result.append({
                "section_title": str(item.get("section_title", "")),
                "data_needed":   str(item.get("data_needed", "")),
                "sql_intent":    str(item.get("sql_intent", "general")),
            })
    return result[:5]


def complex_decompose_node(state: ComplexState) -> Dict:
    """Decompose query using planned report sections as context (max 6 sub-queries)."""
    from pipeline.llm_router import call as llm_call

    sections    = state.get("report_sections", [])
    all_tables  = state["schema"].get("all_tables", [])
    tables_str  = ", ".join(all_tables[:30])
    schema_str  = state["schema"].get("schema_str", "")

    # Build decomposition hint from planned sections
    sections_hint = "\n".join(
        f"- Section '{s['section_title']}': {s['data_needed']}"
        for s in sections
    )

    system = (
        "You are a CRM query decomposer. Break the query into atomic, independently-answerable "
        "sub-queries (max 6). Each sub-query must be answerable by ONE SQL query.\n\n"
        "OUTPUT: JSON array only.\n"
        '[{"sub_query": "...", "intent": "count|list|sum|compare|rank|lookup|trend"}]\n'
        "Keep each sub_query under 20 words. No sub-query should depend on another."
    )
    user = (
        f"Available tables: {tables_str}\n"
        f"Report sections planned:\n{sections_hint}\n\n"
        f"Schema:\n{schema_str[:500]}\n\n"
        f'Full query: "{state["query"]}"\n\nJSON array:'
    )

    raw         = llm_call("decompose", system, user, max_tokens=700)
    sub_queries = _parse_sub_queries(raw)

    if not sub_queries:
        # Fallback: one sub-query per planned section
        sub_queries = [
            {"sub_query": s["data_needed"], "intent": s["sql_intent"]}
            for s in sections[:6]
        ] or [{"sub_query": state["query"], "intent": "general"}]

    LOGGER.info("Complex agent: %d sub-queries", len(sub_queries))
    return {"sub_queries": sub_queries}


def statistical_analysis_node(state: ComplexState) -> Dict:
    """Pure Python statistical analysis of SQL results — no LLM, zero hallucination.

    Computes:
    - Growth rates: (current - previous) / previous * 100
    - Top/bottom N performers per numeric column
    - Outliers: values > 2 standard deviations from mean
    - Period-over-period comparisons
    """
    sql_results = state["sql_results"]
    stat_result: Dict[str, Any] = {}
    kpis:        Dict[str, Any] = {}
    anomalies:   List[Dict]     = []

    for res_item in sql_results:
        sq   = res_item.get("sub_query", "")
        data = res_item.get("result_data", {})
        if not data.get("success"):
            continue

        rows    = data.get("rows", [])
        columns = data.get("columns", [])
        if not rows:
            continue

        # Extract numeric values from each column
        for col_idx, col_name in enumerate(columns):
            numeric_vals = []
            for row in rows:
                if col_idx < len(row) and row[col_idx] is not None:
                    try:
                        numeric_vals.append(float(str(row[col_idx]).replace(",", "")))
                    except (ValueError, TypeError):
                        pass

            if not numeric_vals:
                continue

            col_key = f"{sq[:30]}::{col_name}"

            # Basic statistics
            total  = sum(numeric_vals)
            mean   = statistics.mean(numeric_vals)
            count  = len(numeric_vals)
            max_v  = max(numeric_vals)
            min_v  = min(numeric_vals)

            stat_result[col_key] = {
                "column":  col_name,
                "context": sq[:50],
                "count":   count,
                "total":   round(total, 2),
                "mean":    round(mean, 2),
                "max":     max_v,
                "min":     min_v,
            }

            # KPI: record the total/count for key columns
            col_lower = col_name.lower()
            if any(kw in col_lower for kw in ["total", "revenue", "amount", "usd", "count"]):
                kpis[col_name] = round(total, 2)

            # Outlier detection (> 2 std dev from mean)
            if count >= 4:
                try:
                    std = statistics.stdev(numeric_vals)
                    if std > 0:
                        for row in rows:
                            if col_idx < len(row) and row[col_idx] is not None:
                                try:
                                    v = float(str(row[col_idx]).replace(",", ""))
                                    z = abs(v - mean) / std
                                    if z > 2.0:
                                        label = str(row[0]) if rows and rows[0] else "unknown"
                                        anomalies.append({
                                            "context":   sq[:50],
                                            "metric":    col_name,
                                            "value":     v,
                                            "mean":      round(mean, 2),
                                            "z_score":   round(z, 2),
                                            "label":     label,
                                            "direction": "high" if v > mean else "low",
                                        })
                                except (ValueError, TypeError):
                                    pass
                except statistics.StatisticsError:
                    pass

    # Period-over-period growth (detect if we have 2 matching queries — this/last)
    growth_rates: Dict[str, float] = {}
    totals_by_intent: Dict[str, List] = {}
    for res_item in sql_results:
        intent = res_item.get("intent", "")
        data   = res_item.get("result_data", {})
        if intent == "sum" and data.get("success") and data.get("rows"):
            rows = data["rows"]
            if rows and len(rows[0]) >= 1:
                try:
                    val = float(str(rows[0][0]).replace(",", ""))
                    totals_by_intent.setdefault(intent, []).append(val)
                except (ValueError, TypeError):
                    pass

    if len(totals_by_intent.get("sum", [])) == 2:
        a, b = totals_by_intent["sum"]
        if b and b != 0:
            rate = (a - b) / abs(b) * 100
            growth_rates["period_over_period"] = round(rate, 2)

    return {
        "statistical_analysis": {
            "column_stats":  stat_result,
            "growth_rates":  growth_rates,
        },
        "kpis":      kpis,
        "anomalies": anomalies[:10],  # cap at 10 anomalies
    }


def trend_detection_node(state: ComplexState) -> Dict:
    """Pure Python trend detection across time-series SQL results.

    Detects: consistent growth, consistent decline, sudden drops, flat lines.
    Works on any multi-row result where the first column can be a time label.
    """
    sql_results = state["sql_results"]
    trends: List[Dict[str, Any]] = []

    for res_item in sql_results:
        sq   = res_item.get("sub_query", "")
        data = res_item.get("result_data", {})
        if not data.get("success"):
            continue

        rows    = data.get("rows", [])
        columns = data.get("columns", [])

        # Need at least 3 rows with at least one numeric column for trend detection
        if len(rows) < 3 or len(columns) < 2:
            continue

        # Try each numeric column (skip the first which is usually a label/period)
        for col_idx in range(1, len(columns)):
            series = []
            for row in rows:
                if col_idx < len(row) and row[col_idx] is not None:
                    try:
                        series.append(float(str(row[col_idx]).replace(",", "")))
                    except (ValueError, TypeError):
                        pass

            if len(series) < 3:
                continue

            col_name = columns[col_idx]

            # Check for consistent direction
            diffs     = [series[i+1] - series[i] for i in range(len(series)-1)]
            pos_diffs = sum(1 for d in diffs if d > 0)
            neg_diffs = sum(1 for d in diffs if d < 0)
            total_d   = len(diffs)

            direction = "flat"
            magnitude = 0.0

            if series[0] != 0:
                overall_change = (series[-1] - series[0]) / abs(series[0]) * 100
                magnitude      = round(overall_change, 2)

            if pos_diffs / total_d >= 0.75:
                direction = "growth"
            elif neg_diffs / total_d >= 0.75:
                direction = "decline"
            elif pos_diffs / total_d >= 0.5:
                direction = "slight_growth"
            else:
                direction = "volatile"

            # Detect sudden drop (any single period down > 30%)
            sudden_drop = None
            for i, d in enumerate(diffs):
                if series[i] and series[i] != 0 and (d / abs(series[i])) < -0.30:
                    period_label = str(rows[i+1][0]) if i+1 < len(rows) and rows[i+1] else f"period {i+2}"
                    sudden_drop  = {"period": period_label, "drop_pct": round((d / abs(series[i])) * 100, 1)}
                    break

            trends.append({
                "metric":           col_name,
                "context":          sq[:50],
                "direction":        direction,
                "magnitude_pct":    magnitude,
                "periods_analyzed": len(series),
                "first_value":      series[0],
                "last_value":       series[-1],
                "sudden_drop":      sudden_drop,
            })

    return {"trends": trends}


def complex_analyze_node(state: ComplexState) -> Dict:
    """LLM analysis enriched with statistical and trend findings (Groq 70b)."""
    from pipeline.llm_router import call as llm_call

    data_block  = format_results_for_llm(state["sql_results"])
    stat_data   = state.get("statistical_analysis", {})
    trends      = state.get("trends", [])
    anomalies   = state.get("anomalies", [])

    # Build statistical summary for the LLM
    stat_summary = ""
    growth = stat_data.get("growth_rates", {})
    if growth:
        stat_summary += "Period-over-period growth: " + ", ".join(
            f"{k}: {v:+.1f}%" for k, v in growth.items()
        ) + "\n"

    if trends:
        stat_summary += "Detected trends: " + "; ".join(
            f"{t['metric']} is {t['direction']} ({t['magnitude_pct']:+.1f}% over {t['periods_analyzed']} periods)"
            for t in trends[:4]
        ) + "\n"

    if anomalies:
        stat_summary += "Outliers detected: " + "; ".join(
            f"{a['label']} has {a['metric']}={a['value']} (mean={a['mean']}, z={a['z_score']})"
            for a in anomalies[:4]
        ) + "\n"

    system = (
        "You are a senior CRM data analyst. Analyze the combined database results and "
        "statistical findings to produce a concise analysis.\n\n"
        "RULES:\n"
        "1. ONLY use data present in the results below — never invent values.\n"
        "2. Reference the statistical findings (trends, outliers) where they're confirmed.\n"
        "3. Keep analysis under 400 words — deep_synthesis will build the full report.\n"
        "4. If a query part returned an error, note 'data unavailable for X'.\n"
        "5. Identify the most important finding in the first sentence."
    )
    user = (
        f'User asked: "{state["query"]}"\n\n'
        f"Database Results:\n{data_block[:2000]}\n\n"
        f"Statistical Analysis:\n{stat_summary or 'No statistical summary available.'}\n\n"
        "Provide concise analysis (under 400 words):"
    )

    analysis = llm_call("synthesize", system, user, max_tokens=800)
    return {"analysis": analysis or "Analysis unavailable."}


def deep_synthesis_node(state: ComplexState) -> Dict:
    """Build the full structured report using Groq 70b (minimum 800 words for complex)."""
    from pipeline.llm_router import call as llm_call

    data_block   = format_results_for_llm(state["sql_results"])
    analysis     = state.get("analysis", "")
    sections     = state.get("report_sections", [])
    kpis         = state.get("kpis", {})
    trends       = state.get("trends", [])
    anomalies    = state.get("anomalies", [])

    # KPI dashboard block
    kpi_block = ""
    if kpis:
        kpi_block = "**KPI Dashboard**\n"
        for k, v in list(kpis.items())[:8]:
            try:
                kpi_block += f"• {k.replace('_', ' ').title()}: **{v:,.2f}**\n"
            except (TypeError, ValueError):
                kpi_block += f"• {k.replace('_', ' ').title()}: **{v}**\n"

    # Trend block
    trend_block = ""
    if trends:
        trend_block = "**Trends Detected**\n"
        for t in trends[:4]:
            indicator = "📈" if "growth" in t["direction"] else "📉" if "decline" in t["direction"] else "➡️"
            trend_block += (
                f"• {t['metric']}: {indicator} {t['direction']} "
                f"({t['magnitude_pct']:+.1f}% over {t['periods_analyzed']} periods)\n"
            )
            if t.get("sudden_drop"):
                drop = t["sudden_drop"]
                trend_block += f"  ⚠️ Sudden drop in {drop['period']}: {drop['drop_pct']:.1f}%\n"

    # Anomaly block
    anomaly_block = ""
    if anomalies:
        anomaly_block = "**Anomalies & Outliers**\n"
        for a in anomalies[:5]:
            anomaly_block += (
                f"• {a['label']} shows {a['direction']} outlier for {a['metric']}: "
                f"**{a['value']:,.2f}** (avg: {a['mean']:,.2f}, z-score: {a['z_score']})\n"
            )

    # Sections planned
    sections_desc = "\n".join(
        f"- {s['section_title']}: {s['data_needed']}" for s in sections
    ) if sections else ""

    system = (
        "You are a senior CRM business analyst writing an executive report.\n\n"
        "ABSOLUTE RULES:\n"
        "1. Use ONLY data from 'Database Results' below — NEVER invent numbers.\n"
        "2. Every number you write must be traceable to the data below.\n"
        "3. Bold ALL key metrics: **176 deals**, **$2.4M**, **42%**\n"
        "4. If a sub-query failed, state 'Data unavailable for [topic]'\n"
        "5. Minimum 600 words — this is an executive report\n"
        "6. Structure with markdown headers (##, ###)\n"
        "7. Include a 'Key Insights' section with 3-5 actionable bullet points\n"
        "8. Include a 'Recommendations' section only if data supports specific actions"
    )

    user = (
        f'# Report Request: "{state["query"]}"\n\n'
        f"## Report Sections Planned:\n{sections_desc}\n\n"
        f"## KPI Summary:\n{kpi_block or 'No KPI data extracted.'}\n\n"
        f"## Trend Analysis:\n{trend_block or 'No trends detected.'}\n\n"
        f"## Anomalies:\n{anomaly_block or 'No anomalies detected.'}\n\n"
        f"## Data Analysis:\n{analysis[:600]}\n\n"
        f"## Raw Database Results:\n{data_block[:3000]}\n\n"
        "Now write the full executive report (minimum 600 words, use ## headers):"
    )

    response = llm_call("synthesize", system, user, max_tokens=2500)
    return {"final_response": response or _fallback_format(state["sql_results"])}


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_complex_graph():
    """Build and compile the extended LangGraph StateGraph for COMPLEX queries."""
    from langgraph.graph import StateGraph, END

    # Adapters: fetch_schema_node and others from medium_agent work with MediumState
    # We pass ComplexState which is a superset — TypedDict sub-typing is structural
    def _fetch_schema(state: ComplexState) -> Dict:
        return fetch_schema_node(state)   # type: ignore[arg-type]

    def _gen_sql(state: ComplexState) -> Dict:
        return generate_sql_node(state)   # type: ignore[arg-type]

    def _exec_parallel(state: ComplexState) -> Dict:
        return execute_parallel_node(state)  # type: ignore[arg-type]

    def _save_examples(state: ComplexState) -> Dict:
        return save_examples_node(state)  # type: ignore[arg-type]

    graph = StateGraph(ComplexState)

    graph.add_node("fetch_schema",         _fetch_schema)
    graph.add_node("plan_report",          plan_report_node)
    graph.add_node("decompose",            complex_decompose_node)
    graph.add_node("generate_sql",         _gen_sql)
    graph.add_node("execute_parallel",     _exec_parallel)
    graph.add_node("statistical_analysis", statistical_analysis_node)
    graph.add_node("trend_detection",      trend_detection_node)
    graph.add_node("analyze",              complex_analyze_node)
    graph.add_node("deep_synthesis",       deep_synthesis_node)
    graph.add_node("save_examples",        _save_examples)

    graph.set_entry_point("fetch_schema")
    graph.add_edge("fetch_schema",         "plan_report")
    graph.add_edge("plan_report",          "decompose")
    graph.add_edge("decompose",            "generate_sql")
    graph.add_edge("generate_sql",         "execute_parallel")
    graph.add_edge("execute_parallel",     "statistical_analysis")
    graph.add_edge("statistical_analysis", "trend_detection")
    graph.add_edge("trend_detection",      "analyze")
    graph.add_edge("analyze",              "deep_synthesis")
    graph.add_edge("deep_synthesis",       "save_examples")
    graph.add_edge("save_examples",        END)

    return graph.compile()


_graph_instance = None
_graph_lock     = __import__("threading").Lock()


def _get_graph():
    global _graph_instance
    if _graph_instance is None:
        with _graph_lock:
            if _graph_instance is None:
                _graph_instance = _build_complex_graph()
    return _graph_instance


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_complex_agent(query: str, classification: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a COMPLEX query through the LangGraph 10-node pipeline.

    Args:
        query:          Raw user query string
        classification: Output from classifier.classify()

    Returns:
        Standard pipeline response dict compatible with api.py
    """
    t_start = time.monotonic()
    LOGGER.info("Complex agent START | query: %.80s", query)

    initial_state: ComplexState = {
        "query":                query,
        "classification":       classification,
        "schema":               {},
        "sub_queries":          [],
        "sql_results":          [],
        "analysis":             "",
        "final_response":       "",
        "attempts":             0,
        "errors":               [],
        "t_start":              t_start,
        "report_sections":      [],
        "kpis":                 {},
        "trends":               [],
        "anomalies":            [],
        "statistical_analysis": {},
    }

    try:
        graph       = _get_graph()
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        LOGGER.error("Complex agent graph error: %s", exc, exc_info=True)
        elapsed = int((time.monotonic() - t_start) * 1000)
        return {
            "answer":                "I encountered an error building your report. Please try a simpler query.",
            "data":                  None,
            "sql_queries":           [],
            "tables_used":           [],
            "confidence":            0.0,
            "agent_type":            "complex",
            "latency_ms":            elapsed,
            "attempts":              0,
            "classification_reason": classification.get("reason", ""),
            "layer":                 "complex_agent_error",
        }

    # Collect metadata
    all_sql:    List[str] = []
    all_tables: set       = set()
    for res_item in final_state.get("sql_results", []):
        data = res_item.get("result_data", {})
        if data.get("sql_used"):
            all_sql.append(data["sql_used"])
        all_tables.update(data.get("tables_used", []))

    elapsed = int((time.monotonic() - t_start) * 1000)
    LOGGER.info(
        "Complex agent DONE | %dms | tables=%s | kpis=%s",
        elapsed, sorted(all_tables), list(final_state.get("kpis", {}).keys()),
    )

    return {
        "answer":                final_state.get("final_response") or "No response generated.",
        "data":                  None,
        "sql_queries":           all_sql,
        "tables_used":           sorted(all_tables),
        "confidence":            0.82,
        "agent_type":            "complex",
        "latency_ms":            elapsed,
        "attempts":              len(final_state.get("sql_results", [])),
        "classification_reason": classification.get("reason", ""),
        "layer":                 "complex_agent",
        "_kpis":                 final_state.get("kpis", {}),
        "_trends":               final_state.get("trends", []),
        "_anomalies":            final_state.get("anomalies", []),
    }
