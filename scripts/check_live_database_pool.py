from concurrent.futures import ThreadPoolExecutor
import json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.api.database import DatabasePool, get_database_url, normalize_database_url
pool=DatabasePool(normalize_database_url(get_database_url()),max_size=3)
def query(i):
    with pool.connection() as conn:
        return conn.execute('SELECT %s::integer',(i,)).fetchone()[0]
start=time.monotonic()
with ThreadPoolExecutor(max_workers=8) as workers:
    values=list(workers.map(query,range(40)))
assert values==list(range(40)) and pool.created_count<=3
try:
    with pool.connection() as conn:
        conn.execute("SET LOCAL statement_timeout='150ms'")
        conn.execute('SELECT pg_sleep(1)')
    raise AssertionError('timeout did not fire')
except Exception as exc:
    assert type(exc).__name__=='QueryCanceled',type(exc).__name__
assert query(99)==99
report={'concurrent_queries':40,'threads':8,'max_connections':pool.created_count,
        'query_timeout_cancels':True,'connection_usable_after_rollback':True,'seconds':round(time.monotonic()-start,2)}
Path('outputs/live_pool_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report));pool.closeall()
