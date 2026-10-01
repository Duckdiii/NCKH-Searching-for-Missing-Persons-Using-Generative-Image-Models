import json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
path=Path(sys.argv[1])
report=json.loads(path.read_text(encoding='utf-8'))
from backend.api.main import app
from fastapi.testclient import TestClient
client=TestClient(app)
if os.environ.get('FACE_MEDIA_API_KEY'):
    client.headers['X-API-Key']=os.environ['FACE_MEDIA_API_KEY']
print('REBUILD',flush=True)
r=client.post('/api/gallery/rebuild')
assert r.status_code==200, r.text
print('QUERY_GENERATED',flush=True)
generated_id=report['checks']['generation_db']['generated_ids'][0][0]
r=client.post('/api/gallery/query',json={'generated_image_id':generated_id})
assert r.status_code==200, r.text
report['checks']['generated_query']=r.json()
report['checks']['query_retry_note']='Initial test process waited after BEGIN; request rerun with fresh process/DB connections. Root cause of wait not established.'
report['checks']['result']='PASS_WITH_QUERY_RETRY'
path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS',r.json()['run_id'],len(r.json()['results']),flush=True)
