import hashlib, json, sys, uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.api import database
from backend.api.supabase_storage import SupabaseMediaStorage
import httpx
store=SupabaseMediaStorage(); store.ensure_bucket()
key='diagnostics/'+str(uuid.uuid4())+'.txt'
payload=b'Non-sensitive storage connectivity test'
store.put_bytes(payload,key,mime_type='text/plain')
try:
    assert store.exists(key)
    with store.open(key) as handle: assert handle.read()==payload
    try:
        store.put_bytes(b'overwrite',key,mime_type='text/plain')
        raise AssertionError('Duplicate object overwritten')
    except FileExistsError: pass
    signed=store.get_access_url(key)
    response=httpx.get(signed,timeout=15)
    assert response.status_code==200 and response.content==payload
    print('PRIVATE_STORAGE_UPLOAD_DOWNLOAD_SIGNED_URL_DUPLICATE_PASS')
finally:
    store.delete(key)
assert not store.exists(key)
print('DELETE_PASS')
