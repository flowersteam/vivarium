"""Tests for update checking functionality."""

import hashlib
import json
import urllib.error
import pytest
from unittest.mock import patch, MagicMock

from vivarium.runtime.updater import (
    check_for_updates,
    get_defaults_update_info,
    find_latest_backup_dir,
    perform_post_update_merge,
    DEFAULTS_MANIFEST_FILE,
)


class TestCheckForUpdates:
    """Tests for GitHub update checking."""

    def test_check_for_updates_returns_none_when_current(self):
        """Should return None when already on latest version."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "tag_name": "v0.2.0",  # Same as current
            "html_url": "https://github.com/...",
            "assets": [],
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock()

        with patch('vivarium.runtime.updater.get_version', return_value='0.2.0'):
            with patch('urllib.request.urlopen', return_value=mock_response):
                result = check_for_updates()

        assert result is None

    def test_check_for_updates_returns_info_when_newer(self):
        """Should return update info when newer version available."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "tag_name": "v1.0.0",
            "html_url": "https://github.com/flowersteam/vivarium/releases/tag/v1.0.0",
            "body": "Release notes here",
            "assets": [
                {"name": "vivarium-macos-arm64.tar.gz", "browser_download_url": "https://..."}
            ],
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock()

        with patch('vivarium.runtime.updater.get_version', return_value='0.2.0'):
            with patch('urllib.request.urlopen', return_value=mock_response):
                with patch('sys.platform', 'darwin'):
                    result = check_for_updates()

        assert result is not None
        assert result['latest_version'] == '1.0.0'
        assert result['current_version'] == '0.2.0'
        assert 'release_url' in result

    def test_check_for_updates_handles_network_error(self):
        """Should return None on network errors (not raise)."""
        with patch('vivarium.runtime.updater.get_version', return_value='0.2.0'):
            with patch('urllib.request.urlopen', side_effect=urllib.error.URLError('Network error')):
                result = check_for_updates()

        assert result is None

    def test_check_for_updates_handles_timeout(self):
        """Should return None on timeout (not raise)."""
        with patch('vivarium.runtime.updater.get_version', return_value='0.2.0'):
            with patch('urllib.request.urlopen', side_effect=TimeoutError()):
                result = check_for_updates()

        assert result is None


class TestGetDefaultsUpdateInfo:
    """Tests for defaults update detection."""

    def test_returns_none_in_dev_mode(self):
        """Should return None when not in frozen mode."""
        assert get_defaults_update_info() is None

    def test_detects_new_files_with_manifest(self, tmp_path):
        """Should detect truly new files that weren't in old defaults."""
        # Create user conf (empty)
        (tmp_path / 'conf').mkdir()

        # Create defaults with a new file
        defaults = tmp_path / '_defaults'
        (defaults / 'conf').mkdir(parents=True)
        (defaults / 'conf' / 'new_scene.yaml').write_text('new: true')

        # Create manifest that doesn't include the new file (simulating old version)
        manifest = {}  # Empty = file is new in this version

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(tmp_path)):
                with patch('vivarium.runtime.updater.get_defaults_dir', return_value=str(defaults)):
                    with patch('vivarium.runtime.updater.load_defaults_manifest', return_value=manifest):
                        with patch('vivarium.runtime.updater.clear_defaults_manifest'):
                            result = get_defaults_update_info()

        assert result is not None
        assert 'new_scene.yaml' in result['conf']['new_files']

    def test_detects_new_files_without_manifest(self, tmp_path):
        """Should report new files when no manifest exists (legacy/first update)."""
        # Create user conf (empty)
        (tmp_path / 'conf').mkdir()

        # Create defaults with a file user doesn't have
        defaults = tmp_path / '_defaults'
        (defaults / 'conf').mkdir(parents=True)
        (defaults / 'conf' / 'new_scene.yaml').write_text('new: true')

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(tmp_path)):
                with patch('vivarium.runtime.updater.get_defaults_dir', return_value=str(defaults)):
                    with patch('vivarium.runtime.updater.load_defaults_manifest', return_value=None):
                        result = get_defaults_update_info()

        assert result is not None
        assert 'new_scene.yaml' in result['conf']['new_files']

    def test_detects_conflicts(self, tmp_path):
        """Should detect when a file changed in update AND user has local modifications."""
        import hashlib

        # Create user conf with user modifications
        (tmp_path / 'conf').mkdir()
        (tmp_path / 'conf' / 'config.yaml').write_text('user: modifications')

        # Create defaults with NEW content (different from old default)
        defaults = tmp_path / '_defaults'
        (defaults / 'conf').mkdir(parents=True)
        (defaults / 'conf' / 'config.yaml').write_text('new: default content')

        # Create manifest representing OLD defaults (before update)
        # The old default was different from both user's file AND new default
        old_content = 'old: default content'
        old_hash = hashlib.md5(old_content.encode()).hexdigest()

        manifest = {
            'conf/config.yaml': old_hash
        }

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(tmp_path)):
                with patch('vivarium.runtime.updater.get_defaults_dir', return_value=str(defaults)):
                    with patch('vivarium.runtime.updater.load_defaults_manifest', return_value=manifest):
                        with patch('vivarium.runtime.updater.clear_defaults_manifest'):
                            result = get_defaults_update_info()

        assert result is not None
        assert 'config.yaml' in result['conf']['conflicts']

    def test_no_conflict_when_only_user_modified(self, tmp_path):
        """Should NOT report conflict when user modified but defaults unchanged."""
        from vivarium.runtime.updater import _compute_file_hash
        import hashlib

        # Create user conf with modifications
        (tmp_path / 'conf').mkdir()
        (tmp_path / 'conf' / 'config.yaml').write_text('user: modifications')

        # Create defaults - same as old defaults (no change in update)
        defaults = tmp_path / '_defaults'
        (defaults / 'conf').mkdir(parents=True)
        default_content = 'default: content'
        (defaults / 'conf' / 'config.yaml').write_text(default_content)

        # Manifest shows defaults unchanged (same hash as current defaults)
        old_hash = hashlib.md5(default_content.encode()).hexdigest()
        manifest = {
            'conf/config.yaml': old_hash
        }

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(tmp_path)):
                with patch('vivarium.runtime.updater.get_defaults_dir', return_value=str(defaults)):
                    with patch('vivarium.runtime.updater.load_defaults_manifest', return_value=manifest):
                        with patch('vivarium.runtime.updater.clear_defaults_manifest'):
                            result = get_defaults_update_info()

        # No conflict because defaults didn't change - user's modification is fine
        assert result is None

    def test_returns_none_when_no_changes(self, tmp_path):
        """Should return None when user files match defaults and nothing changed."""
        import hashlib
        content = 'same: content'
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Create identical conf in both locations
        (tmp_path / 'conf').mkdir()
        (tmp_path / 'conf' / 'config.yaml').write_text(content)

        defaults = tmp_path / '_defaults'
        (defaults / 'conf').mkdir(parents=True)
        (defaults / 'conf' / 'config.yaml').write_text(content)

        # Manifest shows same content (no change)
        manifest = {
            'conf/config.yaml': content_hash
        }

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(tmp_path)):
                with patch('vivarium.runtime.updater.get_defaults_dir', return_value=str(defaults)):
                    with patch('vivarium.runtime.updater.load_defaults_manifest', return_value=manifest):
                        with patch('vivarium.runtime.updater.clear_defaults_manifest'):
                            result = get_defaults_update_info()

        assert result is None


class TestFindLatestBackupDir:
    """Tests for find_latest_backup_dir()."""

    def test_returns_none_when_not_frozen(self):
        """In dev mode (not frozen), should return None."""
        with patch('vivarium.runtime.updater.is_frozen', return_value=False):
            result = find_latest_backup_dir()
            assert result is None

    def test_returns_none_when_no_backups(self, tmp_path):
        """When no backup directories exist, should return None."""
        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(tmp_path)):
                result = find_latest_backup_dir()
                assert result is None

    def test_finds_single_backup(self, tmp_path):
        """When one backup exists, should return it."""
        backup_dir = tmp_path / 'update_backup_20250101_120000'
        backup_dir.mkdir()

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(tmp_path)):
                result = find_latest_backup_dir()
                assert result == str(backup_dir)

    def test_finds_latest_backup(self, tmp_path):
        """When multiple backups exist, should return the most recent."""
        (tmp_path / 'update_backup_20250101_120000').mkdir()
        (tmp_path / 'update_backup_20250102_120000').mkdir()
        latest = tmp_path / 'update_backup_20250103_120000'
        latest.mkdir()

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(tmp_path)):
                result = find_latest_backup_dir()
                assert result == str(latest)

    def test_ignores_non_backup_directories(self, tmp_path):
        """Should only consider directories starting with update_backup_."""
        (tmp_path / 'conf').mkdir()
        (tmp_path / 'notebooks').mkdir()
        (tmp_path / '.other_backup').mkdir()
        backup_dir = tmp_path / 'update_backup_20250101_120000'
        backup_dir.mkdir()

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(tmp_path)):
                result = find_latest_backup_dir()
                assert result == str(backup_dir)


def _compute_hash_from_content(content: str) -> str:
    """Helper to compute MD5 hash from string content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


class TestPerformPostUpdateMerge:
    """Tests for perform_post_update_merge() - smart merge after updates."""

    def test_returns_none_when_not_frozen(self):
        """In dev mode (not frozen), should return None."""
        with patch('vivarium.runtime.updater.is_frozen', return_value=False):
            result = perform_post_update_merge()
            assert result is None

    def test_returns_none_when_no_backup(self, tmp_path):
        """When no backup directory exists, should return None."""
        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(tmp_path)):
                result = perform_post_update_merge()
                assert result is None

    def test_returns_none_when_no_manifest(self, tmp_path):
        """When no manifest exists, should return None."""
        backup_dir = tmp_path / 'update_backup_20250101_120000'
        backup_dir.mkdir()
        (backup_dir / 'conf').mkdir()
        (backup_dir / 'conf' / 'test.yaml').write_text('user content')

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(tmp_path)):
                with patch('vivarium.runtime.updater.get_defaults_dir', return_value=str(tmp_path / '_defaults')):
                    result = perform_post_update_merge()
                    assert result is None

    def test_restores_user_modified_file_when_update_unchanged(self, tmp_path):
        """
        When user modified a file but update didn't change it,
        should restore from backup.
        """
        # Setup directory structure
        app_root = tmp_path / 'app'
        app_root.mkdir()
        defaults_dir = app_root / '_defaults'
        defaults_dir.mkdir()
        (defaults_dir / 'conf').mkdir(parents=True)
        (app_root / 'conf').mkdir()

        # Create backup with user's modified content
        backup_dir = app_root / 'update_backup_20250101_120000'
        backup_dir.mkdir()
        (backup_dir / 'conf').mkdir()
        (backup_dir / 'conf' / 'test.yaml').write_text('user modified content')

        # Create manifest (representing old defaults)
        old_content = 'original default content'
        manifest = {'conf/test.yaml': _compute_hash_from_content(old_content)}
        (app_root / DEFAULTS_MANIFEST_FILE).write_text(json.dumps(manifest))

        # New defaults are same as old (update didn't change this file)
        (defaults_dir / 'conf' / 'test.yaml').write_text(old_content)

        # Current user file (from extraction) has new default content
        (app_root / 'conf' / 'test.yaml').write_text(old_content)

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(app_root)):
                with patch('vivarium.runtime.updater.get_defaults_dir', return_value=str(defaults_dir)):
                    result = perform_post_update_merge()

        # Should restore user's file
        assert result is not None
        assert 'test.yaml' in result['conf']['restored']
        assert len(result['conf']['conflicts']) == 0

        # Verify file was actually restored
        restored_content = (app_root / 'conf' / 'test.yaml').read_text()
        assert restored_content == 'user modified content'

    def test_conflict_when_both_user_and_update_modified(self, tmp_path):
        """
        When user modified a file AND update changed it,
        should keep new version and report conflict.
        """
        # Setup directory structure
        app_root = tmp_path / 'app'
        app_root.mkdir()
        defaults_dir = app_root / '_defaults'
        defaults_dir.mkdir()
        (defaults_dir / 'conf').mkdir(parents=True)
        (app_root / 'conf').mkdir()

        # Create backup with user's modified content
        backup_dir = app_root / 'update_backup_20250101_120000'
        backup_dir.mkdir()
        (backup_dir / 'conf').mkdir()
        (backup_dir / 'conf' / 'test.yaml').write_text('user modified content')

        # Create manifest (representing old defaults)
        old_content = 'original default content'
        manifest = {'conf/test.yaml': _compute_hash_from_content(old_content)}
        (app_root / DEFAULTS_MANIFEST_FILE).write_text(json.dumps(manifest))

        # New defaults are DIFFERENT (update changed this file)
        new_content = 'new updated default content'
        (defaults_dir / 'conf' / 'test.yaml').write_text(new_content)

        # Current user file (from extraction) has new default content
        (app_root / 'conf' / 'test.yaml').write_text(new_content)

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(app_root)):
                with patch('vivarium.runtime.updater.get_defaults_dir', return_value=str(defaults_dir)):
                    result = perform_post_update_merge()

        # Should report conflict, not restore
        assert result is not None
        assert 'test.yaml' in result['conf']['conflicts']
        assert len(result['conf']['restored']) == 0

        # Verify file still has new content (not restored)
        current_content = (app_root / 'conf' / 'test.yaml').read_text()
        assert current_content == new_content

    def test_no_action_when_user_didnt_modify(self, tmp_path):
        """
        When user didn't modify a file (backup matches old defaults),
        should keep new version without any action.
        """
        # Setup directory structure
        app_root = tmp_path / 'app'
        app_root.mkdir()
        defaults_dir = app_root / '_defaults'
        defaults_dir.mkdir()
        (defaults_dir / 'conf').mkdir(parents=True)
        (app_root / 'conf').mkdir()

        # Create backup with unmodified content (same as old defaults)
        old_content = 'original default content'
        backup_dir = app_root / 'update_backup_20250101_120000'
        backup_dir.mkdir()
        (backup_dir / 'conf').mkdir()
        (backup_dir / 'conf' / 'test.yaml').write_text(old_content)

        # Create manifest
        manifest = {'conf/test.yaml': _compute_hash_from_content(old_content)}
        (app_root / DEFAULTS_MANIFEST_FILE).write_text(json.dumps(manifest))

        # New defaults are different (update changed this file)
        new_content = 'new updated default content'
        (defaults_dir / 'conf' / 'test.yaml').write_text(new_content)
        (app_root / 'conf' / 'test.yaml').write_text(new_content)

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(app_root)):
                with patch('vivarium.runtime.updater.get_defaults_dir', return_value=str(defaults_dir)):
                    result = perform_post_update_merge()

        # Should return None (no action needed)
        assert result is None

        # Verify file still has new content
        current_content = (app_root / 'conf' / 'test.yaml').read_text()
        assert current_content == new_content

    def test_handles_multiple_files_mixed_scenarios(self, tmp_path):
        """Test with multiple files having different scenarios."""
        # Setup directory structure
        app_root = tmp_path / 'app'
        app_root.mkdir()
        defaults_dir = app_root / '_defaults'
        (defaults_dir / 'conf').mkdir(parents=True)
        (defaults_dir / 'notebooks').mkdir(parents=True)
        (app_root / 'conf').mkdir()
        (app_root / 'notebooks').mkdir()

        backup_dir = app_root / 'update_backup_20250101_120000'
        (backup_dir / 'conf').mkdir(parents=True)
        (backup_dir / 'notebooks').mkdir(parents=True)

        # File 1: User modified, update unchanged -> should restore
        old1 = 'old content 1'
        (backup_dir / 'conf' / 'restore_me.yaml').write_text('user version 1')
        (defaults_dir / 'conf' / 'restore_me.yaml').write_text(old1)
        (app_root / 'conf' / 'restore_me.yaml').write_text(old1)

        # File 2: User modified, update also modified -> conflict
        old2 = 'old content 2'
        (backup_dir / 'conf' / 'conflict.yaml').write_text('user version 2')
        (defaults_dir / 'conf' / 'conflict.yaml').write_text('new default 2')
        (app_root / 'conf' / 'conflict.yaml').write_text('new default 2')

        # File 3: User didn't modify -> no action
        old3 = 'old content 3'
        (backup_dir / 'notebooks' / 'unchanged.ipynb').write_text(old3)
        (defaults_dir / 'notebooks' / 'unchanged.ipynb').write_text('new content 3')
        (app_root / 'notebooks' / 'unchanged.ipynb').write_text('new content 3')

        # Create manifest
        manifest = {
            'conf/restore_me.yaml': _compute_hash_from_content(old1),
            'conf/conflict.yaml': _compute_hash_from_content(old2),
            'notebooks/unchanged.ipynb': _compute_hash_from_content(old3),
        }
        (app_root / DEFAULTS_MANIFEST_FILE).write_text(json.dumps(manifest))

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(app_root)):
                with patch('vivarium.runtime.updater.get_defaults_dir', return_value=str(defaults_dir)):
                    result = perform_post_update_merge()

        assert result is not None
        assert 'restore_me.yaml' in result['conf']['restored']
        assert 'conflict.yaml' in result['conf']['conflicts']
        # unchanged.ipynb should not appear in either list
        assert len(result['notebooks']['restored']) == 0
        assert len(result['notebooks']['conflicts']) == 0

        # Verify files
        assert (app_root / 'conf' / 'restore_me.yaml').read_text() == 'user version 1'
        assert (app_root / 'conf' / 'conflict.yaml').read_text() == 'new default 2'

    def test_clears_manifest_after_merge(self, tmp_path):
        """Manifest should be cleared after successful merge."""
        app_root = tmp_path / 'app'
        app_root.mkdir()
        defaults_dir = app_root / '_defaults'
        (defaults_dir / 'conf').mkdir(parents=True)
        (app_root / 'conf').mkdir()

        backup_dir = app_root / 'update_backup_20250101_120000'
        (backup_dir / 'conf').mkdir(parents=True)

        old_content = 'old content'
        (backup_dir / 'conf' / 'test.yaml').write_text('user modified')
        (defaults_dir / 'conf' / 'test.yaml').write_text(old_content)
        (app_root / 'conf' / 'test.yaml').write_text(old_content)

        manifest_path = app_root / DEFAULTS_MANIFEST_FILE
        manifest = {'conf/test.yaml': _compute_hash_from_content(old_content)}
        manifest_path.write_text(json.dumps(manifest))

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(app_root)):
                with patch('vivarium.runtime.updater.get_defaults_dir', return_value=str(defaults_dir)):
                    perform_post_update_merge()

        # Manifest should be cleared
        assert not manifest_path.exists()

    def test_includes_backup_dir_in_result(self, tmp_path):
        """Result should include the backup directory path."""
        app_root = tmp_path / 'app'
        app_root.mkdir()
        defaults_dir = app_root / '_defaults'
        (defaults_dir / 'conf').mkdir(parents=True)
        (app_root / 'conf').mkdir()

        backup_dir = app_root / 'update_backup_20250101_120000'
        (backup_dir / 'conf').mkdir(parents=True)

        old_content = 'old content'
        (backup_dir / 'conf' / 'test.yaml').write_text('user modified')
        (defaults_dir / 'conf' / 'test.yaml').write_text(old_content)
        (app_root / 'conf' / 'test.yaml').write_text(old_content)

        manifest = {'conf/test.yaml': _compute_hash_from_content(old_content)}
        (app_root / DEFAULTS_MANIFEST_FILE).write_text(json.dumps(manifest))

        with patch('vivarium.runtime.updater.is_frozen', return_value=True):
            with patch('vivarium.runtime.updater.get_app_root', return_value=str(app_root)):
                with patch('vivarium.runtime.updater.get_defaults_dir', return_value=str(defaults_dir)):
                    result = perform_post_update_merge()

        assert result is not None
        assert result['backup_dir'] == str(backup_dir)
