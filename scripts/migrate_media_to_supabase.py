"""Copy DB-referenced assets to a private bucket. Default is inventory only.
--apply uploads and verifies checksums; --enable switches src/.env after success.
Local originals are never deleted. Run while ingestion/generation is stopped.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from backend.api.database import get_pool
parser=argparse.ArgumentParser()
parser.add_argument('--apply',action='store_true')
parser.add_argument('--enable',action='store_true')
args=parser.parse_args()
with get_pool().connection() as conn:
    assets=conn.execute('SELECT storage_key,mime_type,sha256,byte_size FROM face_media.assets ORDER BY created_at').fetchall()
print('ASSETS',len(assets),'BYTES',sum(row[3] for row in assets),flush=True)
if not args.apply:
    print('INVENTORY_ONLY_NO_UPLOAD');sys.exit(0)
from backend.api.supabase_storage import SupabaseMediaStorage
from backend.api.storage import LocalMediaStorage
remote=SupabaseMediaStorage();remote.ensure_bucket();local=LocalMediaStorage()
results=[]
for key,mime,digest,size in assets:
    if not remote.exists(key):
        with local.open(key) as stream:
            remote.put_stream(stream,key,mime_type=mime)
    data=remote._request('GET','object/authenticated/'+remote._path(key)).content
    assert len(data)==size and hashlib.sha256(data).hexdigest()==digest,key
    results.append({'storage_key':key,'verified':True})
    Path('outputs/storage_cloud_migration.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('REMOTE_ASSETS_VERIFIED',len(results))
if args.enable:
    from dotenv import set_key
    set_key('src/.env','STORAGE_BACKEND','supabase')
    set_key('src/.env','SUPABASE_BUCKET',remote.bucket)
    print('SUPABASE_BACKEND_ENABLED_RESTART_BACKEND')
