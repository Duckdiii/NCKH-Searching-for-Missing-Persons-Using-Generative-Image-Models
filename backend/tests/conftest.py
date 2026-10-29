"""Offline backend tests must never inherit a developer's live DB credentials.
Live smoke tests belong under scripts/ and are explicitly invoked separately.
"""
import os
from backend.api import database

database.close_pool()
os.environ.pop('DATABASE_URL', None)
os.environ.pop('DATABASE_URL_TEST', None)

import pytest

@pytest.fixture(autouse=True)
def offline_database_environment(monkeypatch):
    database.close_pool()
    monkeypatch.setenv('DATABASE_URL', '')
    monkeypatch.setenv('DATABASE_URL_TEST', '')
    monkeypatch.setenv('STORAGE_BACKEND', 'local')
    from backend.api.storage import reset_storage_cache
    reset_storage_cache()
    yield
    database.close_pool()
