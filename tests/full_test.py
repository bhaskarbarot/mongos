"""
Full regression + new-feature test.
Covers every fix made in this session.
"""
import os, sys, logging, time
os.chdir('/home/elsner/Documents/mongos')
from dotenv import load_dotenv; load_dotenv()
logging.disable(logging.CRITICAL)

from db import get_database
from agent import get_sql_agent, run_agent_query, ConversationMemory

db    = get_database()
agent = get_sql_agent(db)
mem   = ConversationMemory()

PASS = 0; FAIL = 0

def check(label, query, must_contain=None, must_not_contain=None, max_ms=None):
    global PASS, FAIL
    t0 = time.perf_counter()
    r  = run_agent_query(agent, query, memory=mem)
    ms = round((time.perf_counter()-t0)*1000)
    ans = r.get('answer','')
    layer = r.get('layer','?')
    ok = True
    fail_reasons = []

    if must_contain:
        for kw in must_contain:
            if kw.lower() not in ans.lower():
                ok = False; fail_reasons.append(f'missing "{kw}"')
    if must_not_contain:
        for kw in must_not_contain:
            if kw.lower() in ans.lower():
                ok = False; fail_reasons.append(f'should NOT contain "{kw}"')
    if max_ms and ms > max_ms:
        ok = False; fail_reasons.append(f'too slow {ms}ms > {max_ms}ms')

    status = 'PASS' if ok else 'FAIL'
    if ok: PASS += 1
    else:  FAIL += 1

    reason = ' | ' + ', '.join(fail_reasons) if fail_reasons else ''
    print(f'  [{status}] {label} ({ms}ms, {layer}){reason}')
    if not ok:
        print(f'         Q: {query}')
        print(f'         A: {ans[:150]}')
    return ok

print('\n' + '='*70)
print('  FULL REGRESSION TEST SUITE')
print('='*70 + '\n')

# ── GROUP 1: Revenue with correct date filter ─────────────────────────────
print('── Revenue (invoice_date filter, NOT updated_at) ──')
check('Revenue April 2026',         'give me revenue of april 2026',          must_contain=['April 2026'], must_not_contain=['1,949,081'])
check('Revenue December 2025',      'revenue december 2025',                   must_contain=['December 2025'])
check('Revenue 2025',               'total revenue 2025',                      must_contain=['2025'])
check('Revenue last year',          'revenue last year',                       must_contain=['last year'])
check('Revenue this month',         'total revenue this month',                must_contain=['this month'])
check('Revenue last month',         'revenue last month',                      must_contain=['last month'])
check('Revenue last year december', 'revenue of last year december',           must_contain=['December 2025'])
check('Revenue previous year',      'previous year revenue',                   must_contain=['last year'])

# ── GROUP 2: Invoice filtering ────────────────────────────────────────────
print('\n── Invoice status filters ──')
check('Paid invoices only',         'give me list of paid invoices',           must_contain=['paid'], must_not_contain=['draft','confirmed'])
check('Draft invoices',             'show me draft invoices',                  must_contain=['draft'])
check('Last month invoices',        'give me last month invoices',             must_contain=['Apr 2026'], must_not_contain=['Mar 2026'])
check('USD invoices',               'give me invoices that currency is USD',   must_contain=['USD'])
check('Unpaid invoices',            'give me unpaid invoices',                 must_not_contain=['"paid"'])

# ── GROUP 3: Top / Sort / Limit ───────────────────────────────────────────
print('\n── Top + Sort + Limit ──')
check('Top 5 invoices',             'give me top 5 invoices',                  must_contain=['5'])
check('Top invoice (no N)',         'give me top invoice',                     max_ms=500)
check('Invoices highest to lowest', 'give me top invoices by amount highest to lower', max_ms=500)
check('Biggest invoice',            'biggest invoice',                         max_ms=500)
check('Latest invoices',            'latest invoices',                         max_ms=500)
check('First invoice',              'give me first invoice',                   max_ms=500)

# ── GROUP 4: Deals group_by (was routing to bills) ───────────────────────
print('\n── Deals group_by (correct table) ──')
check('Deals by stage',             'give me deals by stage',                  must_contain=['deals'], must_not_contain=['bills'])
check('Deals by status',            'deals by status',                         must_contain=['stage','deals'], must_not_contain=['bills'])
check('Deals by type',              'deals by type',                           must_contain=['deals'])
check('Deal stage count',           'give me deal stage count',                must_contain=['stage'])
check('Deals by categories',        'all deals give by categories',            must_not_contain=['categories table'])

# ── GROUP 5: Sales orders (not revenue) ──────────────────────────────────
print('\n── Sales orders routing ──')
check('Sales orders list',          'give me sales orders',                    must_contain=['sales'])
check('Top sales order',            'give me top sales order',                 must_contain=['sales'])

# ── GROUP 6: Financial year ───────────────────────────────────────────────
print('\n── Financial year ──')
check('This FY',                    'revenue this financial year',             must_contain=['FY 2026'])
check('Last FY',                    'revenue last financial year',             must_contain=['FY 2025'])
check('FY 2024',                    'revenue FY 2024',                         must_contain=['FY 2024'])

# ── GROUP 7: Simple queries still fast ───────────────────────────────────
print('\n── Fast-path still instant ──')
check('Count deals',                'how many deals are there',                must_contain=['172'], max_ms=200)
check('Closed won',                 'give me all closed won deals',            must_contain=['Closed Won'], max_ms=200)
check('Revenue this month fast',    'total revenue this month',                max_ms=200)
check('Invoice count',              'how many invoices',                       max_ms=200)

# ── GROUP 8: Complex queries still work ──────────────────────────────────
print('\n── Complex pipeline still works ──')
check('Pipeline KPI',               'give me deals KPI report with total count and win rate and pipeline value', must_contain=['deals'])
check('Business health',            'how are we doing lately',                 must_contain=['deals','revenue','Total'])

# ── SUMMARY ───────────────────────────────────────────────────────────────
total = PASS + FAIL
print(f'\n{"="*70}')
print(f'  RESULTS: {PASS}/{total} PASSED | {FAIL} FAILED')
print('='*70)
if FAIL == 0:
    print('\n  🟢 GREEN FLAG — All tests passed.\n')
else:
    print(f'\n  🔴 {FAIL} test(s) need attention.\n')
