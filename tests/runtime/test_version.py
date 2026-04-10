"""Tests for version management functionality."""

import os
import re
import pytest
from unittest.mock import patch

import vivarium
from vivarium.runtime.paths import get_version, get_app_root


def test_version_file_exists():
    """Verify VERSION file exists at expected location."""
    version_file = os.path.join(get_app_root(), 'VERSION')
    assert os.path.exists(version_file), f"VERSION file not found at {version_file}"


def test_version_file_valid_pep440():
    """Verify VERSION file contains a valid PEP 440 version string."""
    version = get_version()
    # Match PEP 440 patterns: 0.2.0, 1.0.0, 1.2.3rc1, 1.2.3a1, 1.2.3b1
    pep440_pattern = r'^\d+\.\d+\.\d+((a|b|rc)\d+)?$'
    assert re.match(pep440_pattern, version), f"Invalid version format: {version}. Expected PEP 440 format."


def test_get_version_returns_string():
    """Verify get_version() returns a non-empty string."""
    version = get_version()
    assert isinstance(version, str)
    assert len(version) > 0


def test_get_version_matches_file():
    """Verify get_version() returns the exact content of VERSION file."""
    version_file = os.path.join(get_app_root(), 'VERSION')
    with open(version_file, 'r') as f:
        expected = f.read().strip()
    assert get_version() == expected


def test_package_version_exposed():
    """Verify __version__ is exposed at top-level package."""
    assert hasattr(vivarium, '__version__')
    assert isinstance(vivarium.__version__, str)
    assert len(vivarium.__version__) > 0


def test_package_version_matches_get_version():
    """Verify vivarium.__version__ equals get_version()."""
    assert vivarium.__version__ == get_version()


def test_get_version_raises_when_file_missing(tmp_path):
    """Verify get_version() raises FileNotFoundError when VERSION missing."""
    with patch('vivarium.runtime.paths.get_app_root', return_value=str(tmp_path)):
        with pytest.raises(FileNotFoundError):
            get_version()


def test_get_version_frozen_mode_uses_app_root(tmp_path):
    """Verify get_version() reads from app root in frozen mode."""
    version_file = tmp_path / 'VERSION'
    version_file.write_text('1.2.3rc1')

    with patch('vivarium.runtime.paths.is_frozen', return_value=True):
        with patch('vivarium.runtime.paths.get_app_root', return_value=str(tmp_path)):
            version = get_version()
            assert version == '1.2.3rc1'
