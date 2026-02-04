"""Tests for runtime path utilities and frozen mode behavior."""

import os
import pytest
from unittest.mock import patch

from vivarium.utils.runtime import (
    is_frozen,
    get_bundle_root,
    get_app_root,
    get_config_dir,
    get_notebooks_dir,
    get_defaults_dir,
    initialize_user_data,
)
from vivarium.utils.updater import _version_is_newer


class TestPathResolution:
    """Tests for path resolution functions."""

    def test_is_frozen_returns_false_in_dev(self):
        """In development mode, is_frozen() should return False."""
        assert is_frozen() is False

    def test_get_bundle_root_in_dev(self):
        """get_bundle_root() should return project root in dev mode."""
        root = get_bundle_root()
        assert os.path.isdir(root)
        assert os.path.exists(os.path.join(root, 'vivarium'))
        assert os.path.exists(os.path.join(root, 'conf'))

    def test_get_app_root_in_dev(self):
        """get_app_root() should return project root in dev mode."""
        root = get_app_root()
        assert os.path.isdir(root)
        assert os.path.exists(os.path.join(root, 'vivarium'))

    def test_get_config_dir_exists(self):
        """get_config_dir() should return an existing directory."""
        config_dir = get_config_dir()
        assert os.path.isdir(config_dir)
        assert os.path.exists(os.path.join(config_dir, 'config.yaml'))

    def test_get_notebooks_dir_exists(self):
        """get_notebooks_dir() should return an existing directory."""
        notebooks_dir = get_notebooks_dir()
        assert os.path.isdir(notebooks_dir)

    def test_get_defaults_dir_in_dev(self):
        """In dev mode, get_defaults_dir() should equal get_bundle_root()."""
        assert get_defaults_dir() == get_bundle_root()


class TestFrozenModePaths:
    """Tests for path resolution in frozen mode (mocked)."""

    def test_get_app_root_frozen_uses_executable_dir(self, tmp_path):
        """In frozen mode, get_app_root() should use executable directory."""
        fake_exe = tmp_path / 'vivarium-interface'
        fake_exe.touch()

        with patch('vivarium.utils.runtime.is_frozen', return_value=True):
            with patch('sys.executable', str(fake_exe)):
                assert get_app_root() == str(tmp_path)

    def test_get_config_dir_frozen_uses_app_root(self, tmp_path):
        """In frozen mode, get_config_dir() should be under app root."""
        fake_exe = tmp_path / 'vivarium-interface'
        fake_exe.touch()

        with patch('vivarium.utils.runtime.is_frozen', return_value=True):
            with patch('sys.executable', str(fake_exe)):
                assert get_config_dir() == os.path.join(str(tmp_path), 'conf')

    def test_get_notebooks_dir_frozen_uses_app_root(self, tmp_path):
        """In frozen mode, get_notebooks_dir() should be under app root."""
        fake_exe = tmp_path / 'vivarium-interface'
        fake_exe.touch()

        with patch('vivarium.utils.runtime.is_frozen', return_value=True):
            with patch('sys.executable', str(fake_exe)):
                assert get_notebooks_dir() == os.path.join(str(tmp_path), 'notebooks')

    def test_get_defaults_dir_frozen(self, tmp_path):
        """In frozen mode, get_defaults_dir() should be _defaults under app root."""
        fake_exe = tmp_path / 'vivarium-interface'
        fake_exe.touch()

        with patch('vivarium.utils.runtime.is_frozen', return_value=True):
            with patch('sys.executable', str(fake_exe)):
                assert get_defaults_dir() == os.path.join(str(tmp_path), '_defaults')


class TestInitializeUserData:
    """Tests for first-run initialization."""

    def test_initialize_user_data_returns_false_in_dev(self):
        """initialize_user_data() should return False in dev mode."""
        assert initialize_user_data() is False

    def test_initialize_user_data_copies_from_defaults(self, tmp_path):
        """In frozen mode, should copy conf/ and notebooks/ from _defaults/."""
        # Setup directory structure
        fake_exe = tmp_path / 'vivarium-interface'
        fake_exe.touch()

        defaults = tmp_path / '_defaults'
        (defaults / 'conf').mkdir(parents=True)
        (defaults / 'conf' / 'config.yaml').write_text('test: true')
        (defaults / 'notebooks').mkdir()
        (defaults / 'notebooks' / 'test.ipynb').write_text('{}')

        with patch('vivarium.utils.runtime.is_frozen', return_value=True):
            with patch('sys.executable', str(fake_exe)):
                result = initialize_user_data()

        assert result is True
        assert (tmp_path / 'conf' / 'config.yaml').exists()
        assert (tmp_path / 'notebooks' / 'test.ipynb').exists()

    def test_initialize_user_data_skips_existing(self, tmp_path):
        """Should not overwrite existing user directories."""
        fake_exe = tmp_path / 'vivarium-interface'
        fake_exe.touch()

        # Create existing user conf
        (tmp_path / 'conf').mkdir()
        (tmp_path / 'conf' / 'config.yaml').write_text('user: config')

        # Create defaults
        defaults = tmp_path / '_defaults'
        (defaults / 'conf').mkdir(parents=True)
        (defaults / 'conf' / 'config.yaml').write_text('default: config')

        with patch('vivarium.utils.runtime.is_frozen', return_value=True):
            with patch('sys.executable', str(fake_exe)):
                result = initialize_user_data()

        # Should not have copied (user conf already exists)
        assert (tmp_path / 'conf' / 'config.yaml').read_text() == 'user: config'


class TestVersionComparison:
    """Tests for version comparison logic."""

    @pytest.mark.parametrize("latest,current,expected", [
        ("1.0.0", "0.9.0", True),
        ("1.0.1", "1.0.0", True),
        ("1.1.0", "1.0.9", True),
        ("2.0.0", "1.9.9", True),
        ("1.0.0", "1.0.0", False),
        ("0.9.0", "1.0.0", False),
        ("1.0.0", "1.0.1", False),
        ("1.2.3", "1.2", True),  # 1.2.3 vs 1.2.0
        ("1.2", "1.2.3", False),  # 1.2.0 vs 1.2.3
    ])
    def test_version_is_newer(self, latest, current, expected):
        """Test version comparison with various inputs."""
        assert _version_is_newer(latest, current) == expected

    @pytest.mark.parametrize("latest,current,expected", [
        ("1.0.0-beta", "1.0.0-alpha", False),  # No numeric suffix, both equal
        ("1.0.0", "1.0.0-beta", False),  # Both become (1.0.0, 0)
        ("1.0.1-beta", "1.0.0", True),  # 1.0.1 > 1.0.0
        ("1.0.0-test2", "1.0.0-test1", True),  # Same base, test2 > test1
        ("0.2.0-test2", "0.2.0-test1", True),  # Same base, test2 > test1
        ("0.2.0-test1", "0.2.0-test2", False),  # test1 < test2
        ("0.2.0-test10", "0.2.0-test9", True),  # test10 > test9
        ("1.0.0-rc2", "1.0.0-rc1", True),  # rc2 > rc1
    ])
    def test_version_is_newer_with_prerelease(self, latest, current, expected):
        """Test version comparison with pre-release suffixes."""
        assert _version_is_newer(latest, current) == expected
