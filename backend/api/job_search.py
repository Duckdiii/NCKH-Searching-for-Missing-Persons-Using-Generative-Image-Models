"""Search generated variants against observed DB gallery and commit one history run."""
import numpy as np
from backend.api import gallery, repositories as repo
from backend.api.database import get_pool
from backend.api.dependencies import get_embedder
from backend.api.storage import get_storage
from src.utils.cancellation import checkpoint


def search_generated_job(job_id, variants, edited_images, *, top_k=5, threshold=0.6):
    model, version, preproc = gallery.current_embed_triple()
    snapshot = gallery.rebuild_gallery()
    best = {}
    age_scores = {}
    embeddings = []
    for variant in variants:
        checkpoint()
        age = variant['target_age']
        path = edited_images.get(age, edited_images.get(str(age)))
        vector = np.asarray(get_embedder().embed(path), dtype=np.float64)
        norm = np.linalg.norm(vector)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)) or norm == 0:
            raise ValueError("Embedding ảnh tạo sinh không hợp lệ")
        vector = vector / norm
        embeddings.append((variant['id'], vector.tolist()))
        hits = gallery.search_gallery(vector, k=top_k)
        age_scores[age] = max((h['score'] for h in hits), default=0.0)
        for hit in hits:
            score = float(np.clip(hit['score'], -1, 1))
            old = best.get(hit['crop_id'])
            if old is None or score > old['score']:
                best[hit['crop_id']] = {**hit, 'score': score,
                    'generated_id': variant['id'], 'age': age}
    ranked = sorted(best.values(), key=lambda h: (-h['score'], h['crop_id']))[:top_k]
    checkpoint()
    with get_pool().connection() as conn:
        for generated_id, values in embeddings:
            repo.create_embedding(conn, model_name=model, model_version=version,
                preprocessing_version=preproc, dimensions=len(values), values=values,
                normalized=True, generated_image_id=generated_id)
        run_id = repo.create_search_run(conn, query_kind='generated_set',
            generation_job_id=job_id, model_name=model, model_version=version,
            preprocessing_version=preproc, index_version=snapshot['key'],
            threshold=threshold, parameters={'aggregation': 'max_cosine_per_observed_crop', 'top_k': top_k})
        for rank, hit in enumerate(ranked, 1):
            repo.add_search_result(conn, run_id=run_id, candidate_crop_id=hit['crop_id'],
                best_generated_image_id=hit['generated_id'], score=hit['score'],
                rank=rank, accepted_by_threshold=hit['score'] >= threshold)
        detail = repo.get_search_run_detail(conn, run_id)
    top = ranked[0] if ranked else None
    return {'search_run_id': run_id,
        'final_scores': {h['crop_id']: h['score'] for h in ranked},
        'accepted': bool(top and top['score'] >= threshold),
        'top_identity': top['crop_id'] if top else '',
        'top_score': top['score'] if top else 0.0,
        'best_age': top['age'] if top else None, 'age_scores': age_scores,
        'matched_gallery_image': get_storage().get_access_url(detail['results'][0]['crop_key']) if top else None}
