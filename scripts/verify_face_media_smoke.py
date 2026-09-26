"""Read-only validation of persisted assets/lineage for a completed live smoke report."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from backend.api.database import get_pool
from backend.api.storage import get_storage

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding='utf-8'))
checks = report['checks']
job = checks['generation_started']['job_id']
with get_pool().connection() as conn:
    rows = conn.execute('''SELECT g.id, j.input_crop_id, a.storage_key, a.sha256,
        a.byte_size, a.width, a.height FROM face_media.generated_images g
        JOIN face_media.generation_jobs j ON j.id=g.job_id
        JOIN face_media.assets a ON a.id=g.asset_id WHERE j.id=%s''', (job,)).fetchall()
    assert rows, 'No generated images persisted'
    for row in rows:
        assert row[1] == checks['reference_crop']['crop_id']
    invalid_candidates = conn.execute('''SELECT count(*) FROM face_media.search_results r
        JOIN face_media.face_crops c ON c.id=r.candidate_crop_id
        WHERE r.run_id=ANY(%s::uuid[]) AND c.purpose <> 'search' ''',
        ([checks['reference_query']['run_id'], checks['generated_query']['run_id']],)).fetchone()[0]
    assert invalid_candidates == 0
    embeddings = conn.execute('''SELECT e.dimensions, e."values" FROM face_media.face_embeddings e
        JOIN face_media.face_crops c ON c.id=e.crop_id
        WHERE c.id=%s''', (checks['reference_crop']['crop_id'],)).fetchall()
    assert embeddings and all(d == len(v) == 512 and abs(sum(x*x for x in v)-1) < .001 for d,v in embeddings)
import cv2
import numpy as np
for _,_,key,digest,size,width,height in rows:
    with get_storage().open(key) as f:
        data=f.read()
    assert len(data)==size and hashlib.sha256(data).hexdigest()==digest
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    assert image is not None and image.shape[:2]==(height,width)
checks['postflight']={'generated_assets_verified':len(rows),'lineage_matches_input_crop':True,
    'gallery_candidates_only_search':True,'reference_embedding_512_normalized':True,
    'generated_sha256_size_dimensions_match':True}
path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks['postflight'],indent=2))
