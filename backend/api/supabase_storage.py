"""Private Supabase Storage adapter; local fallback reads pre-migration assets."""
import hashlib
import io
import os
import tempfile
from urllib.parse import quote

import httpx

from backend.api.storage import MediaStorage, LocalMediaStorage, validate_storage_key, _default_max_bytes


class SupabaseMediaStorage(MediaStorage):
    def __init__(self, *, client=None):
        self.url = os.environ.get('SUPABASE_URL', '').rstrip('/')
        secret = os.environ.get('SUPABASE_SECRET_KEY') or os.environ.get('SUPABASE_SERVICE_KEY')
        self.bucket = os.environ.get('SUPABASE_BUCKET', 'face-media')
        if not self.url.startswith('https://') or not secret:
            raise ValueError('Cần SUPABASE_URL và SUPABASE_SECRET_KEY/SUPABASE_SERVICE_KEY ở backend.')
        headers = {'apikey': secret}
        # New sb_secret keys are API keys, not JWT bearer tokens.
        if not secret.startswith('sb_secret_'):
            headers['Authorization'] = 'Bearer ' + secret
        self.client = client or httpx.Client(headers=headers, timeout=httpx.Timeout(60, connect=10))
        self.max_bytes = _default_max_bytes()
        self.legacy = LocalMediaStorage()

    def _path(self, key):
        return quote(self.bucket, safe='') + '/' + quote(validate_storage_key(key), safe='/')

    def _request(self, method, path, **kwargs):
        try:
            response = self.client.request(method, self.url + '/storage/v1/' + path, **kwargs)
        except httpx.HTTPError as exc:
            raise IOError('Supabase Storage network request failed: ' + type(exc).__name__) from None
        if response.is_success:
            return response
        try:
            body = response.json()
            code = str(body.get('statusCode', response.status_code))
            error = str(body.get('error', '')).lower()
        except ValueError:
            code, error = str(response.status_code), ''
        if code == '404' or error in ('not_found', 'notfound'):
            raise FileNotFoundError('Supabase object not found')
        if code in ('409',) or error in ('duplicate', 'resourcealreadyexists'):
            raise FileExistsError('Supabase object already exists')
        raise IOError(f'Supabase Storage HTTP {response.status_code} (code={code})')

    def ensure_bucket(self):
        try:
            bucket = self._request('GET', 'bucket/' + quote(self.bucket, safe='')).json()
        except FileNotFoundError:
            self._request('POST', 'bucket', json={'id': self.bucket, 'name': self.bucket, 'public': False})
            return
        if bucket.get('public'):
            raise ValueError('Face-media bucket phải private.')

    def put_bytes(self, data, key, *, mime_type):
        self.put_stream(io.BytesIO(data), key, mime_type=mime_type)
        return validate_storage_key(key)

    def put_stream(self, stream, key, *, mime_type, max_bytes=None):
        clean = validate_storage_key(key)
        limit = self.max_bytes if max_bytes is None else min(max_bytes, self.max_bytes)
        if not mime_type or '/' not in mime_type:
            raise ValueError('mime_type không hợp lệ')
        digest, size = hashlib.sha256(), 0
        with tempfile.TemporaryFile() as staged:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise ValueError('File vượt giới hạn dung lượng')
                digest.update(chunk)
                staged.write(chunk)
            staged.seek(0)
            self._request('POST', 'object/' + self._path(clean),
                          content=iter(lambda: staged.read(1024 * 1024), b''),
                          headers={'Content-Type': mime_type, 'Content-Length': str(size), 'x-upsert': 'false'})
        return clean, size, digest.hexdigest()

    def open(self, key):
        try:
            response = self._request('GET', 'object/authenticated/' + self._path(key))
            return io.BytesIO(response.content)
        except FileNotFoundError:
            return self.legacy.open(key)

    def get_access_url(self, key):
        try:
            result = self._request('POST', 'object/sign/' + self._path(key), json={'expiresIn': 3600}).json()
        except FileNotFoundError:
            if self.legacy.exists(key):
                return self.legacy.get_access_url(key)
            raise
        signed = result['signedURL']
        return self.url + '/storage/v1' + signed if signed.startswith('/') else signed

    def delete(self, key):
        # Delete only the remote object; do not delete legacy originals implicitly.
        if not self._exists(key):
            raise FileNotFoundError(key)
        self._request('DELETE', 'object/' + quote(self.bucket, safe=''), json={'prefixes': [validate_storage_key(key)]})

    def _exists(self, key):
        try:
            self._request('GET', 'object/info/authenticated/' + self._path(key))
            return True
        except FileNotFoundError:
            return False
