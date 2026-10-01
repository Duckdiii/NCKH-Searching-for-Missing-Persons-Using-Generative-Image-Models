import io
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import httpx
import pytest
from PIL import Image
from backend.api.database import DatabasePool
from backend.api.storage import LocalMediaStorage
from backend.api.supabase_storage import SupabaseMediaStorage
from backend.api.persistence import persist_generated_outputs
from backend.api.ingest import sample_video_frames
from backend.tests.test_search_ingest import _dummy_video


def test_video_duration_is_source_not_last_sample(tmp_path):
    path=tmp_path/'v.mp4'
    _dummy_video(str(path),frames=8,fps=4)
    frames,meta=sample_video_frames(str(path),fps_target=1,max_frames=1)
    assert len(frames)==1 and meta['truncated']
    assert meta['duration_sec']==2.0


def test_pool_reuses_and_replaces_dead_connection():
    pool=DatabasePool('unused',max_size=1)
    dead=MagicMock();dead.closed=True
    pool._available.put(dead);pool._created=1
    good=MagicMock();good.closed=False
    with patch.object(pool,'_new_connection',return_value=good):
        assert pool.getconn(timeout=1) is good
        pool.putconn(good)
        assert pool.getconn(timeout=1) is good
    assert pool.created_count==1
    dead.close.assert_called_once()
    good.cursor.return_value.__enter__.return_value.execute.assert_called_with('SELECT 1')


def test_pool_checkout_exhaustion_is_bounded():
    pool=DatabasePool('unused',max_size=1);pool._created=1
    with pytest.raises(TimeoutError): pool.getconn(timeout=.02)


@pytest.mark.parametrize('commit_failure',[False,True])
def test_generated_batch_failure_removes_all_files(tmp_path,commit_failure):
    storage=LocalMediaStorage(tmp_path/'media')
    images={}
    for age in (30,50):
        path=tmp_path/f'{age}.png';Image.new('RGB',(24,32)).save(path);images[age]=str(path)
    @contextmanager
    def transaction():
        yield MagicMock()
        if commit_failure: raise ConnectionError('commit failed')
    with patch('backend.api.persistence.get_pool_conn',transaction), \
         patch('backend.api.storage.get_storage',return_value=storage), \
         patch('backend.api.repositories.create_asset'), \
         patch('backend.api.repositories.create_generated_image',side_effect=None if commit_failure else [None,ValueError('second insert failed')]):
        assert persist_generated_outputs(job_id='job',edited_images=images)==[]
    assert len(list((tmp_path/'media').rglob('*.png'))) == (2 if commit_failure else 0)


def test_private_storage_roundtrip_duplicate_limit_and_signed_url(monkeypatch,tmp_path):
    monkeypatch.setenv('SUPABASE_URL','https://example.supabase.co')
    monkeypatch.setenv('SUPABASE_SECRET_KEY','sb_secret_test')
    monkeypatch.setenv('MEDIA_ROOT',str(tmp_path))
    objects={}
    def handle(req):
        path=req.url.path
        if '/object/sign/' in path:
            return httpx.Response(200,json={'signedURL':'/object/sign/face-media/a.png?token=TEST'})
        if req.method=='POST':
            if path in objects: return httpx.Response(409,json={'statusCode':409})
            objects[path]=req.read();return httpx.Response(200,json={})
        if req.method=='GET':
            return httpx.Response(200,content=objects[path.replace('/authenticated','')])
        return httpx.Response(200)
    store=SupabaseMediaStorage(client=httpx.Client(transport=httpx.MockTransport(handle)))
    store.put_bytes(b'payload','a.png',mime_type='image/png')
    with store.open('a.png') as f: assert f.read()==b'payload'
    with pytest.raises(FileExistsError): store.put_bytes(b'other','a.png',mime_type='image/png')
    with pytest.raises(ValueError): store.put_stream(io.BytesIO(b'123'),'big.png',mime_type='image/png',max_bytes=2)
    assert store.get_access_url('a.png').startswith('https://example.supabase.co/storage/v1/object/sign/')


def test_storage_network_error_has_no_secret(monkeypatch):
    monkeypatch.setenv('SUPABASE_URL','https://example.supabase.co')
    monkeypatch.setenv('SUPABASE_SECRET_KEY','sb_secret_PRIVATE')
    client=MagicMock();client.request.side_effect=httpx.ReadTimeout('secret-looking URL')
    with pytest.raises(IOError) as exc:
        SupabaseMediaStorage(client=client).put_bytes(b'a','x.png',mime_type='image/png')
    assert 'secret-looking' not in str(exc.value) and 'PRIVATE' not in str(exc.value)


def test_generated_job_ensembles_and_persists_lineage():
    import numpy as np
    from backend.api.job_search import search_generated_job
    pool=MagicMock()
    variants=[{'id':'g30','target_age':30},{'id':'g50','target_age':50}]
    with patch('backend.api.job_search.get_pool',return_value=pool), \
         patch('backend.api.job_search.get_embedder') as emb, \
         patch('backend.api.job_search.gallery.rebuild_gallery',return_value={'key':'snapshot'}), \
         patch('backend.api.job_search.gallery.search_gallery',side_effect=[
             [{'crop_id':'c1','score':.7},{'crop_id':'c2','score':.5}],
             [{'crop_id':'c1','score':.9},{'crop_id':'c2','score':.4}]]), \
         patch('backend.api.job_search.repo.create_embedding') as embed_save, \
         patch('backend.api.job_search.repo.create_search_run',return_value='run') as run_save, \
         patch('backend.api.job_search.repo.add_search_result') as result_save, \
         patch('backend.api.job_search.repo.get_search_run_detail',return_value={'results':[{'crop_key':'search/crops/a.png'}]}), \
         patch('backend.api.job_search.get_storage'):
        emb.return_value.embed.return_value=np.array([3.,4.])
        result=search_generated_job('job',variants,{30:'a.png',50:'b.png'})
    assert result['search_run_id']=='run' and result['best_age']==50
    assert result['final_scores']=={'c1':.9,'c2':.5}
    assert result_save.call_args_list[0].kwargs['best_generated_image_id']=='g50'
    assert run_save.call_args.kwargs['generation_job_id']=='job'
    assert embed_save.call_count==2


def test_db_job_worker_uses_persisted_search_not_local_gallery():
    from backend.tests.test_job_runner import create_mock_session
    from backend.api.job_runner import PIPELINE_LOCK, run_pipeline_job
    from backend.api.session_store import JobState, save_job, get_job
    sess=create_mock_session('db-worker-session')
    save_job(JobState(job_id='db-worker-job',session_id=sess.session_id))
    summary={'search_run_id':'search-run','final_scores':{'crop':.8},'accepted':True,
        'top_identity':'crop','top_score':.8,'best_age':30,'age_scores':{30:.8},'matched_gallery_image':'/crop.png'}
    with patch('main.run_specialization',return_value='ckpt'), \
         patch('main.run_inversion',return_value=(None,None,None)), \
         patch('main.run_editing',return_value={30:'image.png'}), \
         patch('main.run_embedding_and_search') as legacy, \
         patch('backend.api.job_runner.persist_generated_outputs',return_value=[{'id':'g','target_age':30}]), \
         patch('backend.api.job_runner.update_job_record'), \
         patch('backend.api.job_search.search_generated_job',return_value=summary):
        PIPELINE_LOCK.acquire()
        run_pipeline_job('db-worker-job',sess,{'paths':{}},db_job_id='db-worker-job',input_crop_id='crop')
    assert get_job('db-worker-job').status=='done'
    assert get_job('db-worker-job').result['pipeline_params']['search_run_id']=='search-run'
    legacy.assert_not_called()
