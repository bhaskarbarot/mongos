import os, re, sys, time as _t
os.chdir('/home/elsner/Documents/mongos')
from dotenv import load_dotenv; load_dotenv()
from pipeline.fast_path import _parse_time_condition
from pipeline.utils import normalize_text

PASS = FAIL = 0
def check(label, got, expected):
    global PASS, FAIL
    ok = got == expected
    if ok: PASS += 1
    else:  FAIL += 1
    print(f'  [{"PASS" if ok else "FAIL"}] {label}' + (f'  got={got!r} want={expected!r}' if not ok else ''))

cy = _t.gmtime().tm_year; ly = cy-1
fy_now = cy if _t.gmtime().tm_mon >= 4 else cy-1

print('\n=== TIME PARSING (17 cases) ===')
cases = [
    ('april 2026',            'April 2026'),
    ('december 2025',         'December 2025'),
    ('revenue 2025',          '2025'),
    ('last year',             'last year'),
    ('this year',             'this year'),
    ('last month',            'last month'),
    ('this month',            'this month'),
    ('last 3 months',         'last 3 months'),
    ('last 30 days',          'last 30 days'),
    ('last year december',    f'December {ly}'),
    ('previous year march',   f'March {ly}'),
    ('this year february',    f'February {cy}'),
    ('previous year',         'last year'),
    ('this financial year',   f'FY {fy_now}-{str(fy_now+1)[-2:]}'),
    ('last financial year',   f'FY {fy_now-1}-{str(fy_now)[-2:]}'),
    ('last FY',               f'FY {fy_now-1}-{str(fy_now)[-2:]}'),
    ('FY 2024',               'FY 2024-25'),
]
for q, expected in cases:
    r = _parse_time_condition(q)
    check(q, r['label'] if r else 'NONE', expected)

print('\n=== STATUS ADJECTIVE (6 cases) ===')
_ADJ = {r'\bpaid\b':('payment_status','paid'),r'\bunpaid\b':('payment_status',None),
        r'\bconfirmed\b':('payment_status','confirmed'),r'\bdraft\b':('payment_status','draft'),
        r'\bpending\b':('status','Pending')}
def detect_status(q):
    t = normalize_text(q)
    for pat,(f,v) in _ADJ.items():
        if re.search(pat,t): return v or 'NOT_IN_PAID'
    return None

check('paid invoices',       detect_status('give me paid invoices'),    'paid')
check('draft invoices',      detect_status('show draft invoices'),      'draft')
check('confirmed invoices',  detect_status('confirmed invoices'),       'confirmed')
check('unpaid invoices',     detect_status('unpaid invoices'),          'NOT_IN_PAID')
check('all invoices (none)', detect_status('give me all invoices'),     None)
check('pending tasks',       detect_status('pending tasks'),            'Pending')

print('\n=== TOP / LIMIT (6 cases) ===')
def detect_limit(q):
    t = normalize_text(q)
    tm = re.search(r'\btop\s+(\d+)\b',t)
    bt = bool(re.search(r'\btop\b',t)) and not tm
    fm = re.search(r'\b(1st|first|oldest|earliest)\b',t)
    if fm: return 1
    elif tm: return min(int(tm.group(1)),100)
    elif bt and re.search(r'\b(invoice|deal|sales|order|customer)\b',t): return 10
    return 20

check('top invoice (no N)',  detect_limit('give me top invoice'),      10)
check('top 5 invoices',      detect_limit('give me top 5 invoices'),   5)
check('top 20 deals',        detect_limit('top 20 deals'),             20)
check('first invoice',       detect_limit('give me first invoice'),    1)
check('default',             detect_limit('give me all invoices'),     20)
check('top deal',            detect_limit('top deal'),                 10)

print('\n=== SORT SYNONYMS (8 cases) ===')
def detect_sort(q):
    t = normalize_text(q)
    d = re.search(r'\b(highest\s+to\s+(lower|lowest)|descend\w*|by\s+amount|biggest|largest|most\b|recent|latest|newest)\b',t)
    a = re.search(r'\b(lowest\s+to\s+(higher|highest)|ascend\w*|smallest|cheapest|oldest)\b',t)
    return 'DESC' if d else ('ASC' if a else None)

check('highest to lower → DESC', detect_sort('highest to lower'),     'DESC')
check('biggest → DESC',          detect_sort('biggest invoice'),       'DESC')
check('latest → DESC',           detect_sort('latest invoices'),       'DESC')
check('recent → DESC',           detect_sort('recent invoices'),       'DESC')
check('smallest → ASC',          detect_sort('smallest invoice'),      'ASC')
check('cheapest → ASC',          detect_sort('cheapest deal'),         'ASC')
check('oldest → ASC',            detect_sort('oldest deal'),           'ASC')
check('no signal → None',        detect_sort('give me all invoices'),  None)

print('\n=== SALES ORDER BLOCK (4 cases) ===')
def revenue_match(q):
    t = normalize_text(q)
    if re.search(r'\bsales\s+order',t): return False
    return any(kw in t for kw in ['revenue','income','earning','billing']) or \
           bool(re.search(r'\b(total|show|give)\b.{0,15}\bsales\b',t))

check('top sales order BLOCKED',  revenue_match('give me top sales order'),  False)
check('give sales orders BLOCKED',revenue_match('give me sales orders'),     False)
check('revenue still works',      revenue_match('total revenue this month'), True)
check('billing still works',      revenue_match('billing this year'),        True)

print(f'\n{"="*50}')
total = PASS+FAIL
print(f'RESULTS: {PASS}/{total} PASSED | {FAIL} FAILED')
if FAIL==0: print('\n  ✅ ALL LOGIC TESTS PASSED\n')
else: print(f'\n  ❌ {FAIL} FAILED\n')
