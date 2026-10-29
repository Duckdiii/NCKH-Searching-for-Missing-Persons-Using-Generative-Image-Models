from backend.api import database


def test_environment_legacy_path_and_precedence(tmp_path, monkeypatch):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / '.env').write_text('DATABASE_URL=postgresql://legacy/test\n', encoding='utf-8')
    monkeypatch.setattr(database, '__file__', str(tmp_path / 'backend' / 'api' / 'database.py'))
    monkeypatch.delenv('DATABASE_URL', raising=False)
    monkeypatch.chdir(tmp_path / 'src')
    database._load_environment()
    assert database.get_database_url() == 'postgresql://legacy/test'
    monkeypatch.delenv('DATABASE_URL')
    (tmp_path / '.env').write_text('DATABASE_URL=postgresql://root/test\n', encoding='utf-8')
    database._load_environment()
    assert database.get_database_url() == 'postgresql://root/test'
    monkeypatch.setenv('DATABASE_URL', 'postgresql://process/test')
    database._load_environment()
    assert database.get_database_url() == 'postgresql://process/test'


def test_pool_decodes_uuid_as_api_string():
    from unittest.mock import MagicMock, patch
    from psycopg.types.string import TextLoader
    fake = MagicMock()
    with patch('psycopg.connect', return_value=fake):
        conn = database.DatabasePool('postgresql://unused/test')._new_connection()
    assert conn is fake
    fake.adapters.register_loader.assert_called_once_with('uuid', TextLoader)
    assert TextLoader(2950).load(b'00000000-0000-0000-0000-000000000001') == '00000000-0000-0000-0000-000000000001'
