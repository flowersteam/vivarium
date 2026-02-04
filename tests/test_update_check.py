"""Tests for update checking functionality."""

import json
import urllib.error
import pytest
from unittest.mock import patch, MagicMock

from vivarium.utils.updater import check_for_updates, get_defaults_update_info


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

        with patch('vivarium.utils.updater.get_version', return_value='0.2.0'):
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

        with patch('vivarium.utils.updater.get_version', return_value='0.2.0'):
            with patch('urllib.request.urlopen', return_value=mock_response):
                with patch('sys.platform', 'darwin'):
                    result = check_for_updates()

        assert result is not None
        assert result['latest_version'] == '1.0.0'
        assert result['current_version'] == '0.2.0'
        assert 'release_url' in result

    def test_check_for_updates_handles_network_error(self):
        """Should return None on network errors (not raise)."""
        with patch('vivarium.utils.updater.get_version', return_value='0.2.0'):
            with patch('urllib.request.urlopen', side_effect=urllib.error.URLError('Network error')):
                result = check_for_updates()

        assert result is None

    def test_check_for_updates_handles_timeout(self):
        """Should return None on timeout (not raise)."""
        with patch('vivarium.utils.updater.get_version', return_value='0.2.0'):
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

        with patch('vivarium.utils.updater.is_frozen', return_value=True):
            with patch('vivarium.utils.updater.get_app_root', return_value=str(tmp_path)):
                with patch('vivarium.utils.updater.get_defaults_dir', return_value=str(defaults)):
                    with patch('vivarium.utils.updater.load_defaults_manifest', return_value=manifest):
                        with patch('vivarium.utils.updater.clear_defaults_manifest'):
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

        with patch('vivarium.utils.updater.is_frozen', return_value=True):
            with patch('vivarium.utils.updater.get_app_root', return_value=str(tmp_path)):
                with patch('vivarium.utils.updater.get_defaults_dir', return_value=str(defaults)):
                    with patch('vivarium.utils.updater.load_defaults_manifest', return_value=None):
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

        with patch('vivarium.utils.updater.is_frozen', return_value=True):
            with patch('vivarium.utils.updater.get_app_root', return_value=str(tmp_path)):
                with patch('vivarium.utils.updater.get_defaults_dir', return_value=str(defaults)):
                    with patch('vivarium.utils.updater.load_defaults_manifest', return_value=manifest):
                        with patch('vivarium.utils.updater.clear_defaults_manifest'):
                            result = get_defaults_update_info()

        assert result is not None
        assert 'config.yaml' in result['conf']['conflicts']

    def test_no_conflict_when_only_user_modified(self, tmp_path):
        """Should NOT report conflict when user modified but defaults unchanged."""
        from vivarium.utils.updater import _compute_file_hash
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

        with patch('vivarium.utils.updater.is_frozen', return_value=True):
            with patch('vivarium.utils.updater.get_app_root', return_value=str(tmp_path)):
                with patch('vivarium.utils.updater.get_defaults_dir', return_value=str(defaults)):
                    with patch('vivarium.utils.updater.load_defaults_manifest', return_value=manifest):
                        with patch('vivarium.utils.updater.clear_defaults_manifest'):
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

        with patch('vivarium.utils.updater.is_frozen', return_value=True):
            with patch('vivarium.utils.updater.get_app_root', return_value=str(tmp_path)):
                with patch('vivarium.utils.updater.get_defaults_dir', return_value=str(defaults)):
                    with patch('vivarium.utils.updater.load_defaults_manifest', return_value=manifest):
                        with patch('vivarium.utils.updater.clear_defaults_manifest'):
                            result = get_defaults_update_info()

        assert result is None
