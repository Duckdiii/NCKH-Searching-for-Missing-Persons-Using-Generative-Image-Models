"""P2 — Liên kết tracklet xuyên camera thành ID giả thuyết (doc §5).

Nguyên tắc:
- Global ID là GIẢ THUYẾT (kết hợp tương đồng khuôn mặt + không-thời gian),
  cho phép unresolved và sửa gán nhầm; TÁCH KHỎI xác nhận người mất tích
  (human_confirmed trong search_results).
- Mỗi tracklet tạo descriptor từ các mẫu tốt; giữ vector đơn lẻ đã chọn để
  giải thích/sửa gán nhầm, không chỉ centroid.
- Topology thu hẹp vùng tìm; khi chưa đầy đủ, dùng như gợi ý mềm + tìm lịch
  sử rộng hơn (không ép gộp hai người chỉ để giảm dung lượng).
- Cần ngưỡng chấp nhận + margin top-1/top-2; HIỆU CHỈNH trên cặp khác người
  khó — KHÔNG lấy cố định cosine 0.7/0.9. Mặc định dưới đây là giá trị
  "chưa hiệu chuẩn", caller phải truyền config đã hiệu chuẩn; mọi quyết định
  accept đều ghi lại ngưỡng đã dùng để truy vết.
- Chặn gán chung khi hai vị trí đồng thời không thể di chuyển tới nhau và
  timestamp đáng tin. Camera chồng lấn có thể cùng thấy một người → không cấm
  trùng thời gian cho mọi cặp.
- Một writer/tracklet (UNIQUE partial index) + transaction kiểm tra version
  để tránh gán xung đột.
- Chỉ cập nhật exemplar ID từ association đủ tin cậy (score >= exemplar_min),
  tối đa 3 slot/không gian vector; tính lại khi sửa/tách/gộp.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Giá trị "chưa hiệu chuẩn" — phải hiệu chuẩn lại trên dữ liệu triển khai.
# Đặt qua env LINK_ACCEPT_THRESHOLD / LINK_MARGIN khi đã đo.
UNCALIBRATED_ACCEPT = float(os.environ.get("LINK_ACCEPT_THRESHOLD", "0.55"))
UNCALIBRATED_MARGIN = float(os.environ.get("LINK_MARGIN", "0.05"))
EXEMPLAR_MIN_SCORE = float(os.environ.get("LINK_EXEMPLAR_MIN", "0.65"))
TIME_WINDOW_SEC = float(os.environ.get("LINK_TIME_WINDOW_SEC", "3600"))
MAX_CANDIDATES = int(os.environ.get("LINK_MAX_CANDIDATES", "50"))
W_COSINE = 0.7
W_QUALITY = 0.2
W_TIME = 0.1


@dataclass
class LinkConfig:
    accept_threshold: float = UNCALIBRATED_ACCEPT
    margin: float = UNCALIBRATED_MARGIN
    exemplar_min: float = EXEMPLAR_MIN_SCORE
    time_window_sec: float = TIME_WINDOW_SEC
    max_candidates: int = MAX_CANDIDATES
    topology_version: Optional[int] = None
    calibrated: bool = False  # True khi ngưỡng đã hiệu chuẩn trên cặp khó
    model_name: str = "insightface"
    model_version: str = "buffalo_l"
    preprocessing_version: str = ""

    def warn_if_uncalibrated(self) -> None:
        if not self.calibrated:
            logger.warning(
                "LinkConfig chưa hiệu chuẩn (accept=%.3f margin=%.3f) — "
                "chỉ dùng thử nghiệm, phải hiệu chuẩn trên cặp khác người khó.",
                self.accept_threshold, self.margin)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def tracklet_descriptor(members: list[dict]) -> dict:
    """Descriptor tracklet: centroid trọng số quality + giữ members đơn lẻ.

    members: [{embedding (array-like), quality, crop_id, embedding_id}].
    Trả {centroid (đã normalize), members:[...], quality_mean}.
    """
    vecs, weights, kept = [], [], []
    for m in members:
        try:
            v = np.asarray(m["embedding"], dtype=np.float64)
            if v.ndim != 1 or not np.all(np.isfinite(v)):
                continue
            n = float(np.linalg.norm(v))
            if n == 0:
                continue
            w = max(0.0, float(m.get("quality", 0.5)))
            vecs.append(v / n)
            weights.append(w)
            kept.append(m)
        except Exception:
            continue
    if not vecs:
        raise ValueError("Tracklet không có embedding hợp lệ để tạo descriptor.")
    mat = np.stack(vecs)
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones_like(w)
    centroid = (mat * w[:, None]).sum(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    return {"centroid": centroid, "members": kept,
            "quality_mean": float(w.mean())}


def _topo_between(topology: list[dict], cam_a: Optional[str],
                  cam_b: Optional[str]) -> Optional[dict]:
    if not cam_a or not cam_b:
        return None
    for t in topology or []:
        a, b = t.get("camera_a"), t.get("camera_b")
        if (a == cam_a and b == cam_b) or (a == cam_b and b == cam_a):
            return t
    return None


def travel_feasible(cam_a: Optional[str], cam_b: Optional[str],
                    dt_sec: Optional[float],
                    topology: list[dict]) -> tuple[bool, str]:
    """Chặn gán chung khi đồng thời nhưng không thể di chuyển tới nhau.

    - Camera chồng lấn (overlapping) → luôn feasible (có thể cùng thấy 1 người).
    - Thiếu topology/timestamp → feasible với lý do "thiếu bằng chứng" (giữ
      unresolved ở bước decide, không chặn cứng).
    """
    if cam_a is None or cam_b is None or cam_a == cam_b:
        return True, "same_or_unknown_camera"
    if dt_sec is None or dt_sec < 0:
        return True, "missing_timestamp"
    link = _topo_between(topology, cam_a, cam_b)
    if link is None:
        return True, "no_topology_soft"
    if link.get("overlapping"):
        return True, "overlapping_cameras"
    tmin = link.get("travel_sec_min")
    tmax = link.get("travel_sec_max")
    if tmax is not None and dt_sec > float(tmax) * 3:
        # Vượt xa thời gian di chuyển tối đa → vẫn có thể là cùng người quay
        # lại sau, nên không chặn cứng ở đây; caller dùng time penalty.
        return True, "beyond_typical_travel"
    if tmin is not None and dt_sec < float(tmin) and abs(dt_sec) < 1.0:
        # Cùng thời điểm (<1s) ở hai camera xa nhau, không chồng lấn →
        # không thể là cùng người.
        return False, "simultaneous_distant_cameras"
    return True, "topology_ok"


def rerank(descriptor: dict, candidates: list[dict],
           embeddings_by_id: dict[str, np.ndarray],
           *, now_camera: Optional[str] = None,
           now_time: Optional[datetime] = None,
           topology: Optional[list[dict]] = None,
           config: Optional[LinkConfig] = None) -> list[dict]:
    """Re-rank ứng viên bằng vector gốc + quality + thời gian di chuyển."""
    cfg = config or LinkConfig()
    centroid = np.asarray(descriptor["centroid"], dtype=np.float64)
    qmean = float(descriptor.get("quality_mean", 0.5))
    out = []
    for cand in candidates:
        gid = str(cand.get("global_id"))
        evec = embeddings_by_id.get(gid)
        if evec is None:
            continue
        cos = cosine(centroid, np.asarray(evec, dtype=np.float64))
        # Thời gian: dt từ valid_from của ứng viên tới now.
        dt = None
        try:
            vf = cand.get("valid_from")
            if vf and now_time is not None:
                t0 = vf if isinstance(vf, datetime) else datetime.fromisoformat(str(vf))
                dt = abs((now_time - t0).total_seconds())
        except Exception:
            dt = None
        feasible, feas_reason = travel_feasible(
            now_camera, cand.get("camera_id") or cand.get("src_camera"),
            dt, topology or [])
        if not feasible:
            out.append({"global_id": gid, "score": -1.0, "cosine": cos,
                        "feasible": False, "reason": feas_reason,
                        "evidence": {"cosine": cos, "blocked": feas_reason}})
            continue
        time_term = 1.0
        if dt is not None and cfg.time_window_sec > 0:
            time_term = max(0.0, 1.0 - min(1.0, dt / (cfg.time_window_sec * 3)))
        cqual = float(cand.get("score", 0.5) or 0.5)
        score = (W_COSINE * cos
                 + W_QUALITY * (0.5 * qmean + 0.5 * min(1.0, max(0.0, cqual)))
                 + W_TIME * time_term)
        out.append({"global_id": gid, "score": float(score), "cosine": float(cos),
                    "feasible": True,
                    "reason": f"cos={cos:.3f} q={qmean:.2f} dt={dt} {feas_reason}",
                    "evidence": {"cosine": float(cos), "quality_mean": qmean,
                                 "dt_sec": dt, "feasibility": feas_reason,
                                 "weights": {"cos": W_COSINE, "q": W_QUALITY,
                                             "t": W_TIME}}})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def decide(ranked: list[dict], config: Optional[LinkConfig] = None) -> dict:
    """Ngưỡng chấp nhận + margin top-1/top-2; thiếu bằng chứng → unresolved."""
    cfg = config or LinkConfig()
    cfg.warn_if_uncalibrated()
    if not ranked:
        return {"action": "unresolved", "global_id": None,
                "reason": "no_candidates",
                "threshold": cfg.accept_threshold, "margin": cfg.margin,
                "calibrated": cfg.calibrated}
    feas = [r for r in ranked if r.get("feasible", True) and r["score"] >= 0]
    if not feas:
        return {"action": "unresolved", "global_id": None,
                "reason": "all_blocked_by_spacetime",
                "threshold": cfg.accept_threshold, "margin": cfg.margin,
                "calibrated": cfg.calibrated}
    top = feas[0]
    second = feas[1] if len(feas) > 1 else None
    margin = (top["score"] - second["score"]) if second else 1.0
    if top["score"] < cfg.accept_threshold:
        return {"action": "unresolved", "global_id": None,
                "reason": f"below_threshold top={top['score']:.3f}",
                "top": top, "threshold": cfg.accept_threshold,
                "margin": cfg.margin, "calibrated": cfg.calibrated}
    if margin < cfg.margin:
        return {"action": "unresolved", "global_id": None,
                "reason": f"ambiguous_margin {margin:.3f}",
                "top": top, "second": second,
                "threshold": cfg.accept_threshold, "margin": cfg.margin,
                "calibrated": cfg.calibrated}
    return {"action": "assign", "global_id": top["global_id"], "top": top,
            "second": second, "margin": float(margin),
            "threshold": cfg.accept_threshold, "calibrated": cfg.calibrated,
            "reason": top.get("reason", "")}


def embedding_space_key(model_name: str, model_version: str,
                        preprocessing_version: str, dimensions: int) -> str:
    """Không gian vector gồm weights fingerprint (doc §6).

    Hậu tố /w<hash> chỉ thêm khi đo được fingerprint; "unknown" → giữ key
    legacy để không vỡ exemplar đã lưu.
    """
    base = f"{model_name}/{model_version}/{preprocessing_version}/d{dimensions}"
    try:
        from backend.api.gallery import weights_fingerprint
        fp = weights_fingerprint(model_version)
    except Exception:
        fp = "unknown"
    if not fp or fp == "unknown":
        return base
    return f"{base}/w{fp}"


def candidate_spaces(model_name: str, model_version: str,
                     preprocessing_version: str, dimensions: int) -> list[str]:
    """Các space cần tìm (mới trước, legacy sau) để exemplar cũ vẫn dùng được."""
    base = f"{model_name}/{model_version}/{preprocessing_version}/d{dimensions}"
    new = embedding_space_key(model_name, model_version, preprocessing_version,
                              dimensions)
    return [new] if new == base else [new, base]


def link_tracklet(conn, *, tracklet_id: str, descriptor: dict,
                  camera_id: Optional[str] = None,
                  at_time: Optional[datetime] = None,
                  config: Optional[LinkConfig] = None,
                  site: str = "default") -> dict:
    """Liên kết 1 tracklet → global ID (hoặc unresolved / ID tạm riêng).

    Transaction: đọc ứng viên → re-rank → decide → ghi assignment + exemplars.
    Một writer/tracklet (UNIQUE partial) — xung đột → trả conflict để caller
    đọc lại thay vì ghi đè.
    """
    from backend.api import repositories as repo

    cfg = config or LinkConfig()
    if not cfg.preprocessing_version:
        try:
            from backend.api.gallery import current_embed_triple
            m, mv, pv = current_embed_triple()
            cfg.model_name, cfg.model_version, cfg.preprocessing_version = m, mv, pv
        except Exception:
            pass
    now = at_time or datetime.now(timezone.utc)
    try:
        topology = repo.get_topology(conn, site=site)
    except Exception:
        topology = []
    try:
        spaces = candidate_spaces(cfg.model_name, cfg.model_version,
                                  cfg.preprocessing_version,
                                  len(descriptor["centroid"]))
    except Exception:
        spaces = [f"{cfg.model_name}/{cfg.model_version}/{cfg.preprocessing_version}"]
    space = spaces[0]
    try:
        cands: list[dict] = []
        seen_ids: set[str] = set()
        per_space = max(1, cfg.max_candidates // len(spaces))
        for sp in spaces:
            try:
                part = repo.recent_candidates(
                    conn, embedding_space=sp, site=site, camera_id=camera_id,
                    limit=per_space)
            except TypeError:
                # Tương thích repo cũ (chưa có site/camera_id).
                part = repo.recent_candidates(conn, embedding_space=sp,
                                              limit=per_space)
            for c in part:
                key = (str(c.get("global_id")), str(c.get("slot")))
                if key not in seen_ids:
                    seen_ids.add(key)
                    cands.append({**c, "space": sp})
        cands = cands[:cfg.max_candidates]
    except Exception:
        cands = []
    # Tải vector exemplar cho re-rank (giữ vector đơn lẻ để giải thích).
    embeddings_by_id: dict[str, np.ndarray] = {}
    try:
        with conn.cursor() as cur:
            for c in cands:
                eid = c.get("embedding_id")
                if not eid:
                    continue
                cur.execute(
                    'SELECT "values" FROM face_media.face_embeddings WHERE id = %s',
                    (eid,))
                row = cur.fetchone()
                if row:
                    embeddings_by_id[str(c["global_id"])] = np.asarray(
                        row[0], dtype=np.float64)
    except Exception:
        pass
    # Fallback: ứng viên không có vector exemplar → dùng score cũ làm cosine proxy.
    if not embeddings_by_id:
        for c in cands:
            embeddings_by_id[str(c["global_id"])] = np.asarray(
                descriptor["centroid"], dtype=np.float64) * 0.0 + 0.5
    ranked = rerank(descriptor, cands, embeddings_by_id, now_camera=camera_id,
                    now_time=now, topology=topology, config=cfg)
    decision = decide(ranked, cfg)
    evidence = {"ranked": [{k: r.get(k) for k in ("global_id", "score", "cosine",
                                                 "reason", "feasible")} for r in ranked[:5]],
                "topology_version": cfg.topology_version,
                "threshold": cfg.accept_threshold, "margin": cfg.margin,
                "calibrated": cfg.calibrated}
    if decision["action"] != "assign":
        # Giữ unresolved hoặc ID tạm riêng (status unresolved).
        try:
            gid = repo.create_identity(conn, status="unresolved",
                                       last_seen_at=now.isoformat())
            aid = repo.add_assignment(
                conn, tracklet_id=tracklet_id, global_id=gid,
                score=0.0, margin=None, evidence={**evidence, "unresolved": decision["reason"]},
                reason=f"unresolved: {decision['reason']}",
                model_name=cfg.model_name, model_version=cfg.model_version,
                preprocessing_version=cfg.preprocessing_version,
                topology_version=cfg.topology_version)
            return {"action": "unresolved", "global_id": gid,
                    "assignment_id": aid, **decision}
        except Exception as exc:
            from backend.api.repositories import ConflictError
            if isinstance(exc, ConflictError):
                return {"action": "conflict", "global_id": None,
                        "reason": str(exc)}
            # DB chưa migrate 006 → unresolved trong RAM.
            return {"action": "unresolved", "global_id": None,
                    "reason": f"{decision['reason']} (no_identity_table: {exc})"}
    top = decision["top"]
    try:
        aid = repo.add_assignment(
            conn, tracklet_id=tracklet_id, global_id=str(top["global_id"]),
            score=float(top["score"]),
            margin=float(decision.get("margin") or 0.0),
            evidence={**evidence, "cosine": top.get("cosine")},
            reason=str(top.get("reason", ""))[:500],
            model_name=cfg.model_name, model_version=cfg.model_version,
            preprocessing_version=cfg.preprocessing_version,
            topology_version=cfg.topology_version)
    except Exception as exc:
        from backend.api.repositories import ConflictError
        if isinstance(exc, ConflictError):
            return {"action": "conflict", "global_id": str(top["global_id"]),
                    "reason": str(exc)}
        raise
    try:
        repo.bump_revision(conn, str(top["global_id"]),
                           last_seen_at=now.isoformat())
    except Exception:
        pass
    # Chỉ cập nhật exemplar từ association đủ tin cậy, tối đa 3 slot.
    try:
        if float(top["score"]) >= cfg.exemplar_min:
            members = sorted(descriptor.get("members", []),
                             key=lambda m: float(m.get("quality", 0)), reverse=True)
            existing = []
            try:
                existing = repo.get_exemplars(conn, str(top["global_id"]), space)
            except Exception:
                existing = []
            used_crops = {e.get("crop_id") for e in existing}
            slot = len(existing)
            for m in members:
                if slot >= 3 or float(m.get("quality", 0)) < 0.4:
                    break
                if m.get("crop_id") in used_crops:
                    continue
                try:
                    repo.set_exemplar(
                        conn, global_id=str(top["global_id"]),
                        embedding_space=space, slot=slot,
                        crop_id=m.get("crop_id"),
                        embedding_id=m.get("embedding_id"),
                        weight=float(m.get("quality", 1.0)))
                    slot += 1
                except Exception:
                    break
    except Exception as exc:
        logger.debug("bỏ qua exemplar update: %s", exc)
    return {"action": "assign", "global_id": str(top["global_id"]),
            "assignment_id": aid, **decision}


def merge_identities(conn, *, winner_id: str, loser_id: str,
                     reason: str = "manual_merge") -> dict:
    """Gộp loser → winner: đóng assignment loser, mở lại về winner + bump revision.

    Tính lại exemplar (doc §5.7): lấp slot trống của winner bằng exemplar tốt
    nhất của loser (theo từng không gian vector, giữ tối đa 3 slot), rồi xóa
    exemplar loser để không còn ID nào viện dẫn cùng bằng chứng.
    """
    from backend.api import repositories as repo
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM face_media.identity_assignments
            WHERE global_id = %s AND valid_to IS NULL
            """,
            (loser_id,),
        )
        rows = [r[0] for r in cur.fetchall()]
    moved = 0
    for aid in rows:
        try:
            repo.supersede_assignment(conn, aid, new_global_id=winner_id,
                                      reason=reason)
            moved += 1
        except Exception:
            continue
    exemplars_merged = 0
    try:
        loser_ex = repo.get_exemplars(conn, loser_id)
        by_space: dict[str, list[dict]] = {}
        for e in loser_ex:
            by_space.setdefault(str(e.get("embedding_space")), []).append(e)
        for space, items in by_space.items():
            try:
                taken = {int(e.get("slot")) for e in
                         repo.get_exemplars(conn, winner_id, space)}
            except Exception:
                taken = set()
            for e in sorted(items, key=lambda x: float(x.get("weight", 0) or 0),
                            reverse=True):
                free = next((s for s in (0, 1, 2) if s not in taken), None)
                if free is None:
                    break
                try:
                    repo.set_exemplar(
                        conn, global_id=winner_id, embedding_space=space,
                        slot=free, crop_id=e.get("crop_id"),
                        embedding_id=e.get("embedding_id"),
                        weight=float(e.get("weight", 1.0) or 1.0))
                    taken.add(free)
                    exemplars_merged += 1
                except Exception:
                    continue
        try:
            repo.clear_exemplars(conn, loser_id)
        except Exception:
            pass
    except Exception as exc:
        logger.debug("merge exemplar recompute bỏ qua: %s", exc)
    try:
        rev = repo.bump_revision(conn, winner_id)
    except Exception:
        rev = -1
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE face_media.global_identities SET status='merged' WHERE id=%s",
                (loser_id,))
    except Exception:
        pass
    return {"winner": winner_id, "loser": loser_id, "moved": moved,
            "exemplars_merged": exemplars_merged, "revision": rev}


def split_identity(conn, *, assignment_id: str,
                   reason: str = "manual_split") -> dict:
    """Tách 1 tracklet khỏi ID hiện tại thành ID riêng (revision mới).

    Tính lại exemplar (doc §5.7): exemplar của ID cũ mà crop thuộc tracklet
    vừa tách được CHUYỂN sang ID mới (slot trống cùng không gian), không để ID
    cũ tiếp tục viện dẫn bằng chứng không còn thuộc về nó.
    """
    from backend.api import repositories as repo
    with conn.cursor() as cur:
        cur.execute(
            "SELECT global_id, tracklet_id FROM face_media.identity_assignments"
            " WHERE id = %s",
            (assignment_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Assignment không tồn tại: {assignment_id}")
        old_global, tracklet_id = row[0], row[1]
    new_global = repo.create_identity(conn, status="open")
    new_aid = repo.supersede_assignment(conn, assignment_id,
                                        new_global_id=new_global, reason=reason)
    exemplars_moved = 0
    try:
        moved_crops = set(repo.crops_of_tracklet(conn, str(tracklet_id)))
        if moved_crops:
            for e in repo.get_exemplars(conn, str(old_global)):
                if str(e.get("crop_id")) not in moved_crops:
                    continue
                space = str(e.get("embedding_space"))
                try:
                    taken = {int(x.get("slot")) for x in
                             repo.get_exemplars(conn, new_global, space)}
                except Exception:
                    taken = set()
                free = next((s for s in (0, 1, 2) if s not in taken), None)
                try:
                    repo.delete_exemplar(conn, global_id=str(old_global),
                                         embedding_space=space,
                                         slot=int(e.get("slot")))
                except Exception:
                    continue
                if free is not None:
                    try:
                        repo.set_exemplar(
                            conn, global_id=new_global, embedding_space=space,
                            slot=free, crop_id=e.get("crop_id"),
                            embedding_id=e.get("embedding_id"),
                            weight=float(e.get("weight", 1.0) or 1.0))
                        exemplars_moved += 1
                    except Exception:
                        continue
    except Exception as exc:
        logger.debug("split exemplar recompute bỏ qua: %s", exc)
    try:
        repo.bump_revision(conn, old_global)
    except Exception:
        pass
    return {"old_global": old_global, "new_global": new_global,
            "new_assignment": new_aid, "exemplars_moved": exemplars_moved}
