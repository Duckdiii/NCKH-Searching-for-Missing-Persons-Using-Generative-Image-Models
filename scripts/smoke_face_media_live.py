"""Explicit live smoke test: real models, API and DB; retains identified fixtures.
Run from repo root: .venv/Scripts/python scripts/smoke_face_media_live.py --generation
"""
import argparse, hashlib, json, os, sys, time, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--generation', action='store_true')
parser.add_argument('--resume', type=Path, help='Resume a prior report without duplicating uploads')
args = parser.parse_args()
prior = json.loads(args.resume.read_text(encoding='utf-8')) if args.resume else None
run = prior['run_id'] if prior else str(uuid.uuid4())
out = ROOT / 'outputs' / 'live_storage_tests' / run
out.mkdir(parents=True, exist_ok=True)
report = prior or {'run_id': run, 'checks': {}, 'fixture_note': 'Video synthesized from existing sample; not live camera.'}
def record(name, value):
    report['checks'][name] = value
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(name, json.dumps(value, ensure_ascii=False, default=str), flush=True)
from backend.api.database import get_pool
with get_pool().connection() as conn:
    rows=conn.execute("SELECT tablename FROM pg_tables WHERE schemaname='face_media' ORDER BY tablename").fetchall()
record('tables', [r[0] for r in rows])
from fastapi.testclient import TestClient
from backend.api.main import app
# No lifespan: do not reconcile another application's active jobs.
client = TestClient(app)
if os.environ.get('FACE_MEDIA_API_KEY'):
    client.headers['X-API-Key'] = os.environ['FACE_MEDIA_API_KEY']
def request(method,path,**kwargs):
    r=getattr(client,method)(path,**kwargs)
    if r.status_code>=400:
        raise RuntimeError(f'{method} {path}: HTTP {r.status_code}: {r.text[:800]}')
    return r.json()
try:
    if prior:
        uploaded=report['checks']['reference_upload']
        selected=report['checks']['reference_crop']
        observed=report['checks']['observed_image']
        started=report['checks']['video_started']
        sid=uploaded['session_id']
        restored=request('get',f'/api/sessions/{sid}')
        assert restored['crop_id']==selected['crop_id']
    else:
        sample=ROOT/'outputs'/'aligned_input.png'
        data=sample.read_bytes()
        uploaded=request('post','/api/sessions',files={'file':(f'live-smoke-{run}.png',data,'image/png')})
        record('reference_upload',uploaded)
        assert uploaded['source_id'] and uploaded['detection_ids']
        sid=uploaded['session_id']
        selected=request('post',f'/api/sessions/{sid}/select-face',json={'selected_idx':0})
        record('reference_crop',selected)
        assert selected['crop_id']
        request('post',f'/api/sessions/{sid}/resolve-age',json={'mode':'manual','manual_age':30,'photo_year':2020})
        from backend.api.session_store import sessions
        sessions.pop(sid,None)
        restored=request('get',f'/api/sessions/{sid}')
        record('session_restore',restored)
        observed=request('post','/api/search-sources/images',files=[('files',(f'live-smoke-{run}.png',data,'image/png'))])
        record('observed_image',observed)
        assert observed['items'][0]['status']=='done'
        import cv2
        frame=cv2.imread(str(sample)); h,w=frame.shape[:2]
        video=out/'sample.mp4'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),4,(w,h))
        assert writer.isOpened()
        for _ in range(8): writer.write(frame)
        writer.release()
        started=request('post','/api/search-sources/videos?fps_target=1&max_frames=3',files={'file':('sample.mp4',video.read_bytes(),'video/mp4')})
        record('video_started',started)
        for _ in range(180):
            state=request('get',f"/api/ingestion-runs/{started['run_id']}")
            if state['status'] in ('done','error','canceled'): break
            time.sleep(1)
        record('video_finished',state)
        assert state['status']=='done' and state['faces_found']>0
    record('gallery',request('post','/api/gallery/rebuild'))
    result=request('post','/api/gallery/query',json={'crop_id':selected['crop_id'],'top_k':5})
    record('reference_query',result)
    assert result['results']
    ids=[uploaded['source_id'],observed['items'][0]['source_id'],started['source_id']]
    with get_pool().connection() as conn:
        assets=conn.execute('''SELECT DISTINCT a.storage_key,a.sha256,a.byte_size FROM face_media.assets a WHERE a.id IN (
        SELECT original_asset_id FROM face_media.sources WHERE id=ANY(%s::uuid[])
        UNION SELECT asset_id FROM face_media.frames WHERE source_id=ANY(%s::uuid[])
        UNION SELECT c.asset_id FROM face_media.face_crops c JOIN face_media.face_detections d ON d.id=c.detection_id JOIN face_media.frames f ON f.id=d.frame_id WHERE f.source_id=ANY(%s::uuid[]))''',(ids,ids,ids)).fetchall()
    from backend.api.storage import get_storage
    for key,digest,size in assets:
        with get_storage().open(key) as handle: stored=handle.read()
        assert len(stored)==size and hashlib.sha256(stored).hexdigest()==digest
    record('asset_integrity',{'verified':len(assets),'sha256_and_size':True,'storage_backend':type(get_storage()).__name__})
    if args.generation:
        job=request('post',f'/api/sessions/{sid}/run',json={})
        record('generation_started',job)
        deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            status=request('get',f"/api/jobs/{job['job_id']}")
            if status.get('status') in ('done','error'): break
            time.sleep(3)
        record('generation_finished',status)
        with get_pool().connection() as conn:
            db_job=conn.execute('SELECT status FROM face_media.generation_jobs WHERE id=%s',(job['job_id'],)).fetchone()
            generated=conn.execute('SELECT id FROM face_media.generated_images WHERE job_id=%s',(job['job_id'],)).fetchall()
        record('generation_db',{'status':db_job,'generated_ids':generated})
        automatic_run = status.get('pipeline_params', {}).get('search_run_id')
        assert automatic_run, 'Job did not persist its own search run'
        with get_pool().connection() as conn:
            history = conn.execute('SELECT count(*) FROM face_media.search_runs WHERE id=%s AND generation_job_id=%s', (automatic_run, job['job_id'])).fetchone()[0]
            hits = conn.execute('SELECT count(*) FROM face_media.search_results WHERE run_id=%s', (automatic_run,)).fetchone()[0]
        assert history == 1 and hits > 0
        record('automatic_job_search', {'run_id': automatic_run, 'results': hits})
        assert db_job and db_job[0]=='done' and generated
        record('generated_query',request('post','/api/gallery/query',json={'generated_image_id':str(generated[0][0])}))
    record('result','PASS')
except Exception as exc:
    record('result',{'status':'FAIL','type':type(exc).__name__,'message':str(exc)[:1200]})
    raise
finally:
    print('REPORT',out/'report.json',flush=True)
