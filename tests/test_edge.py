"""
Edge-case diagnostic — verifies:
1. Complex queries break fast-path and use LLM pipeline
2. Semantic/vague queries are routed to COMPLEX + synthesis
3. Simple queries stay fast (<200ms)
"""
import logging, sys, time
from dotenv import load_dotenv; load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)], force=True,
)
for noisy in ["transformers","sentence_transformers","httpx","urllib3","sqlalchemy.engine","langchain"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

from db import get_database
from agent import get_sql_agent, run_agent_query, ConversationMemory
from pipeline.classifier import classify

QUERIES = [
    # ── Should break fast-path → COMPLEX pipeline ─────────────────────────────
    ("COMPLEX-1", "compare last 3 months revenue growth trend and highlight anomalies if any"),
    ("COMPLEX-2", "which salesperson improved most compared to previous quarter and who declined"),
    ("COMPLEX-3", "give me insights not just numbers about deals pipeline health and what needs attention"),
    # ── Semantic / vague → should route COMPLEX ───────────────────────────────
    ("SEMANTIC-1", "how are we doing lately?"),
    ("SEMANTIC-2", "is business improving?"),
    ("SEMANTIC-3", "any risks in pipeline?"),
    # ── Simple → should stay fast-path ────────────────────────────────────────
    ("SIMPLE-1",   "how many deals are there"),
    ("SIMPLE-2",   "give me all closed won deals"),
    ("SIMPLE-3",   "total revenue this month"),
]

def run():
    print("\n" + "="*80)
    print("  EDGE CASE DIAGNOSTIC — Routing + Understanding Quality")
    print("="*80 + "\n")

    print("── CLASSIFIER PRE-CHECK (0ms, rule-based) ──────────────────────────────────")
    for label, q in QUERIES:
        d = classify(q)
        badge = "🔴 COMPLEX" if d["type"] == "COMPLEX" else "🟢 SIMPLE "
        print(f"  {badge}  [{label}]")
        print(f"           Q: {q}")
        print(f"           R: {d['reason']}\n")

    print("\nLoading DB + agent (schema discovery once)...")
    db    = get_database()
    agent = get_sql_agent(db)
    mem   = ConversationMemory()
    print("Ready.\n")

    pass_count = fail_count = 0

    for label, query in QUERIES:
        expected_complex = label.startswith("COMPLEX") or label.startswith("SEMANTIC")
        print("\n" + "─"*80)
        print(f"[{label}]")
        print(f"Q: {query}")
        print("─"*80)

        t0     = time.perf_counter()
        result = run_agent_query(agent, query, memory=mem)
        ms     = round((time.perf_counter() - t0) * 1000)

        answer   = result.get("answer", "")
        sub_res  = result.get("_sub_results")
        tables   = result.get("tables_used", [])
        conf     = result.get("confidence", 0)

        used_complex = sub_res is not None
        routed_right = (expected_complex == used_complex) or \
                       (expected_complex and ms > 800)  # slow = complex path used

        if label.startswith("SIMPLE"):
            status = "PASS" if ms < 500 and not used_complex else "WARN"
        else:
            status = "PASS" if used_complex else "WARN (stayed in fast-path)"

        if "PASS" in status:
            pass_count += 1
        else:
            fail_count += 1

        layer = f"COMPLEX pipeline ({len(sub_res)} sub-queries)" if used_complex \
                else f"Fast-path / Text2SQL ({ms}ms)"

        print(f"  STATUS   : {status}")
        print(f"  LAYER    : {layer}")
        print(f"  LATENCY  : {ms} ms")
        print(f"  TABLES   : {tables}")
        print(f"  CONF     : {conf}")
        if sub_res:
            for i, s in enumerate(sub_res, 1):
                print(f"  Part {i}  : [{s['intent']}] {s['sub_query'][:75]}")

        # Trim executive summary wrapper for display
        display = answer
        for strip in ["Executive Summary:", "Result is generated from"]:
            if strip in display:
                idx = display.find(strip)
                if strip == "Executive Summary:":
                    display = display[idx + len(strip):].strip()
                else:
                    display = display[:idx].strip()

        print(f"\n  ANSWER (first 600 chars):\n{display[:600]}")
        if len(display) > 600:
            print("  ...[truncated]")

    print("\n" + "="*80)
    print(f"  ROUTING RESULTS: {pass_count}/{len(QUERIES)} correct | {fail_count} warnings")
    print("="*80)

if __name__ == "__main__":
    run()
