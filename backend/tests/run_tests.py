import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 70)
print("RUNNING BACKEND UNIT TESTS")
print("=" * 70)

from backend.tests.test_session_router import (
    test_upload_image_one_face,
    test_upload_image_multiple_faces,
    test_upload_image_no_face_fails,
    test_select_face_success_and_invalid_index,
    test_resolve_age_manual_and_validation,
    clean_sessions,
)
from backend.tests.test_job_runner import (
    test_job_runner_success,
    test_job_runner_failure_releases_lock,
    test_pipeline_mutex_prevents_concurrent_runs,
    test_config_isolation_between_jobs,
    clean_stores,
)

from backend.api.session_store import sessions, jobs
from backend.api.job_runner import PIPELINE_LOCK

session_tests = [
    ("test_upload_image_one_face", test_upload_image_one_face),
    ("test_upload_image_multiple_faces", test_upload_image_multiple_faces),
    ("test_upload_image_no_face_fails", test_upload_image_no_face_fails),
    ("test_select_face_success_and_invalid_index", test_select_face_success_and_invalid_index),
    ("test_resolve_age_manual_and_validation", test_resolve_age_manual_and_validation),
]

job_tests = [
    ("test_job_runner_success", test_job_runner_success),
    ("test_job_runner_failure_releases_lock", test_job_runner_failure_releases_lock),
    ("test_pipeline_mutex_prevents_concurrent_runs", test_pipeline_mutex_prevents_concurrent_runs),
    ("test_config_isolation_between_jobs", test_config_isolation_between_jobs),
]

passed = 0
failed = 0

print("\n--- 1. Session Router Tests ---")
for name, fn in session_tests:
    sessions.clear()
    t0 = time.time()
    try:
        fn()
        print(f"  ✅ {name} PASSED ({time.time()-t0:.3f}s)")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name} FAILED: {e}")
        failed += 1
    finally:
        sessions.clear()

print("\n--- 2. Job Runner & Mutex Tests ---")
for name, fn in job_tests:
    sessions.clear()
    jobs.clear()
    if PIPELINE_LOCK.locked():
        try:
            PIPELINE_LOCK.release()
        except RuntimeError:
            pass
    t0 = time.time()
    try:
        fn()
        print(f"  ✅ {name} PASSED ({time.time()-t0:.3f}s)")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name} FAILED: {e}")
        failed += 1
    finally:
        sessions.clear()
        jobs.clear()
        if PIPELINE_LOCK.locked():
            try:
                PIPELINE_LOCK.release()
            except RuntimeError:
                pass

print("\n" + "=" * 70)
print(f"TEST SUMMARY: {passed} PASSED, {failed} FAILED (Total: {passed + failed})")
print("=" * 70)

if failed > 0:
    sys.exit(1)
else:
    sys.exit(0)
