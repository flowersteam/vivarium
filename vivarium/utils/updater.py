"""
Update utilities for Vivarium PyInstaller builds.

This module consolidates all update-related functionality:
- Checking for new versions on GitHub
- Downloading and applying updates
- Managing update state (pending markers, defaults comparison)
"""

import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from typing import Callable, Optional

from vivarium.utils.runtime import get_app_root, get_defaults_dir, get_version, is_frozen


lg = logging.getLogger(__name__)

# GitHub repository for update checks
GITHUB_REPO = "flowersteam/vivarium"

# Update marker file (created after download, cleared after restart)
UPDATE_PENDING_FILE = "UPDATE_PENDING"

# Manifest file storing hashes of _defaults before update
DEFAULTS_MANIFEST_FILE = ".defaults_manifest.json"

# Inner archive names by platform
INNER_ARCHIVE_UNIX = "vivarium-build.tar.gz"
INNER_ARCHIVE_WIN = "vivarium-build.zip"


class UpdateDownloadError(Exception):
    """Raised when an update download fails."""
    pass


def get_inner_archive_name() -> str:
    """Get the platform-appropriate inner archive name."""
    if sys.platform == "win32":
        return INNER_ARCHIVE_WIN
    return INNER_ARCHIVE_UNIX


def get_update_pending_path() -> str:
    """Get the path to the UPDATE_PENDING marker file."""
    return os.path.join(get_app_root(), UPDATE_PENDING_FILE)


def is_update_pending() -> bool:
    """Check if an update was recently applied (marker file exists)."""
    return os.path.exists(get_update_pending_path())


def get_update_pending_info() -> Optional[dict]:
    """
    Read the UPDATE_PENDING marker file if it exists.

    Returns:
        Dict with 'new_version', 'previous_version', 'updated_at', or None.
    """
    marker_path = get_update_pending_path()
    if not os.path.exists(marker_path):
        return None

    try:
        with open(marker_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        lg.warning(f"Failed to read update pending marker: {e}")
        return None


def mark_update_pending(new_version: str, previous_version: str) -> None:
    """
    Create the UPDATE_PENDING marker file.

    Args:
        new_version: The version being installed
        previous_version: The version being replaced
    """
    marker_path = get_update_pending_path()
    marker_data = {
        "new_version": new_version,
        "previous_version": previous_version,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }

    try:
        with open(marker_path, 'w') as f:
            json.dump(marker_data, f, indent=2)
        lg.info(f"Created update pending marker: {new_version}")
    except OSError as e:
        lg.warning(f"Failed to create update pending marker: {e}")


def clear_update_pending() -> None:
    """Remove the UPDATE_PENDING marker file."""
    marker_path = get_update_pending_path()
    if os.path.exists(marker_path):
        try:
            os.remove(marker_path)
            lg.info("Cleared update pending marker")
        except OSError as e:
            lg.warning(f"Failed to clear update pending marker: {e}")


def _get_defaults_manifest_path() -> str:
    """Get the path to the defaults manifest file."""
    return os.path.join(get_app_root(), DEFAULTS_MANIFEST_FILE)


def _compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def _build_defaults_manifest() -> dict:
    """
    Build a manifest of all files in _defaults/ with their hashes.

    Returns:
        Dict mapping relative paths to their MD5 hashes.
    """
    defaults_dir = get_defaults_dir()
    manifest = {}

    for folder in ['conf', 'notebooks']:
        folder_path = os.path.join(defaults_dir, folder)
        if not os.path.exists(folder_path):
            continue

        for root, _, files in os.walk(folder_path):
            for filename in files:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, defaults_dir)
                manifest[rel_path] = _compute_file_hash(filepath)

    return manifest


def save_defaults_manifest() -> None:
    """
    Save a manifest of current _defaults/ before applying an update.

    This allows us to detect what actually changed in the new version
    after the update is applied.
    """
    if not is_frozen():
        return

    manifest = _build_defaults_manifest()
    manifest_path = _get_defaults_manifest_path()

    try:
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        lg.info(f"Saved defaults manifest with {len(manifest)} files")
    except OSError as e:
        lg.warning(f"Failed to save defaults manifest: {e}")


def load_defaults_manifest() -> Optional[dict]:
    """
    Load the saved defaults manifest from before the update.

    Returns:
        Dict mapping relative paths to their old MD5 hashes, or None.
    """
    manifest_path = _get_defaults_manifest_path()
    if not os.path.exists(manifest_path):
        return None

    try:
        with open(manifest_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        lg.warning(f"Failed to load defaults manifest: {e}")
        return None


def clear_defaults_manifest() -> None:
    """Remove the defaults manifest file after processing."""
    manifest_path = _get_defaults_manifest_path()
    if os.path.exists(manifest_path):
        try:
            os.remove(manifest_path)
            lg.info("Cleared defaults manifest")
        except OSError as e:
            lg.warning(f"Failed to clear defaults manifest: {e}")


def _version_is_newer(latest: str, current: str) -> bool:
    """
    Compare two version strings.

    Returns True if latest is newer than current.
    Handles versions like "1.2.3", "1.2", "1.2.3-beta", "1.2.3rc1", "1.2.3-test1".

    Pre-release ordering:
    - 0.2.2 (release) > 0.2.2rc1 (pre-release of same base)
    - 0.2.2rc2 > 0.2.2rc1 (later pre-release)
    - 0.2.2rc1 > 0.2.1 (pre-release of 0.2.2 is newer than 0.2.1 release)
    """
    import re

    def parse_version(v: str) -> tuple:
        # Remove build metadata (everything after +)
        v = v.split('+')[0]

        # Handle both dash-style (1.2.3-rc1) and PEP 440-style (1.2.3rc1)
        # Pattern: base version, optional pre-release identifier, optional number
        match = re.match(r'^(\d+(?:\.\d+)*)(?:[-.]?(a|alpha|b|beta|rc|test|dev)(\d*))?$', v, re.IGNORECASE)

        if match:
            base = match.group(1)
            prerelease_type = match.group(2)  # e.g., "rc", "beta", "test"
            prerelease_num = match.group(3)   # e.g., "1", "2"
        else:
            # Fallback: try to extract base version from each part
            base = v
            prerelease_type = None
            prerelease_num = None

        # Parse base version parts
        parts = []
        for part in base.split('.'):
            # Extract leading digits from each part
            num_match = re.match(r'^(\d+)', part)
            if num_match:
                parts.append(int(num_match.group(1)))
            else:
                parts.append(0)

        # Pad to at least 3 parts
        while len(parts) < 3:
            parts.append(0)

        # Pre-release ordering: rc > beta > alpha > test > dev
        # We use a tuple: (base_version, is_release, prerelease_order, prerelease_num)
        # is_release: 1 for release, 0 for pre-release (so release sorts higher)
        prerelease_order = {
            'rc': 50,
            'beta': 40, 'b': 40,
            'alpha': 30, 'a': 30,
            'test': 20,
            'dev': 10,
        }

        if prerelease_type is None:
            # This is a release version
            is_release = 1
            pr_order = 0
            pr_num = 0
        else:
            # This is a pre-release
            is_release = 0
            pr_order = prerelease_order.get(prerelease_type.lower(), 0)
            pr_num = int(prerelease_num) if prerelease_num else 0

        return (tuple(parts), is_release, pr_order, pr_num)

    try:
        latest_parsed = parse_version(latest)
        current_parsed = parse_version(current)
        return latest_parsed > current_parsed
    except Exception:
        return False


def check_for_updates(timeout: float = 5.0, include_prereleases: bool = False) -> Optional[dict]:
    """
    Check GitHub releases for a newer version.

    Args:
        timeout: Request timeout in seconds
        include_prereleases: If True, also check pre-release versions

    Returns:
        Dict with update info if available, None otherwise.
        Dict contains: 'latest_version', 'current_version', 'download_url', 'release_url'
    """
    current = get_version()

    try:
        if include_prereleases:
            # Fetch all releases and find the newest one
            url = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
        else:
            # Only fetch the latest stable release
            url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

        request = urllib.request.Request(
            url,
            headers={'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'Vivarium'}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode('utf-8'))

        # Handle list vs single object response
        if include_prereleases:
            if not response_data:
                return None
            data = response_data[0]  # First release is the newest
        else:
            data = response_data

        latest = data.get("tag_name", "").lstrip("v")
        if not latest:
            return None

        # Simple version comparison (works for semver-like versions)
        if _version_is_newer(latest, current):
            # Find the appropriate asset for the current platform
            download_url = None
            for asset in data.get("assets", []):
                name = asset.get("name", "").lower()
                if sys.platform == "darwin" and "macos" in name:
                    download_url = asset.get("browser_download_url")
                    break
                elif sys.platform == "win32" and "windows" in name:
                    download_url = asset.get("browser_download_url")
                    break
                elif sys.platform.startswith("linux") and "linux" in name:
                    download_url = asset.get("browser_download_url")
                    break

            return {
                'latest_version': latest,
                'current_version': current,
                'download_url': download_url,
                'release_url': data.get("html_url"),
                'release_notes': data.get("body", ""),
            }

    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
        lg.debug(f"Update check failed: {e}")

    return None


def get_defaults_update_info() -> Optional[dict]:
    """
    Check for potential conflicts between updated defaults and user modifications.

    This function detects files where:
    1. The file changed in the new version (compared to the pre-update manifest)
    2. AND the user has local modifications to that file

    Also reports new files in _defaults/ that the user doesn't have.

    Returns:
        Dict with lists of 'new_files' and 'conflicts' for each folder,
        or None if not in frozen mode or no actionable changes found.

        'new_files': Files added in new version that user doesn't have
        'conflicts': Files that changed in new version AND user has modified
    """
    if not is_frozen():
        return None

    dist_dir = get_app_root()
    defaults_dir = get_defaults_dir()

    # Load the manifest from before the update
    old_manifest = load_defaults_manifest()

    result = {
        'conf': {'new_files': [], 'conflicts': []},
        'notebooks': {'new_files': [], 'conflicts': []},
    }

    for folder in ['conf', 'notebooks']:
        user_folder = os.path.join(dist_dir, folder)
        default_folder = os.path.join(defaults_dir, folder)

        if not os.path.exists(default_folder):
            continue

        for root, _, files in os.walk(default_folder):
            for filename in files:
                default_file = os.path.join(root, filename)
                rel_path_from_defaults = os.path.relpath(default_file, defaults_dir)
                rel_path_from_folder = os.path.relpath(default_file, default_folder)
                user_file = os.path.join(user_folder, rel_path_from_folder)

                # Compute current default hash
                new_default_hash = _compute_file_hash(default_file)

                if not os.path.exists(user_file):
                    # User doesn't have this file
                    if old_manifest is None:
                        # No manifest = first run or legacy, report as new
                        result[folder]['new_files'].append(rel_path_from_folder)
                    elif rel_path_from_defaults not in old_manifest:
                        # File didn't exist in old defaults = truly new in this version
                        result[folder]['new_files'].append(rel_path_from_folder)
                    # else: file existed before but user deleted it - don't nag
                else:
                    # User has this file - check for conflicts
                    user_hash = _compute_file_hash(user_file)

                    if old_manifest is not None:
                        old_default_hash = old_manifest.get(rel_path_from_defaults, "")

                        # Conflict: file changed in new version AND user has modifications
                        file_changed_in_update = (old_default_hash != new_default_hash)
                        user_has_modifications = (user_hash != old_default_hash)

                        if file_changed_in_update and user_has_modifications:
                            result[folder]['conflicts'].append(rel_path_from_folder)
                    # If no manifest, we can't determine conflicts accurately

    # Clean up the manifest after processing
    if old_manifest is not None:
        clear_defaults_manifest()

    # Return None if no actionable changes
    has_changes = any(
        result[folder]['new_files'] or result[folder]['conflicts']
        for folder in ['conf', 'notebooks']
    )

    return result if has_changes else None


def check_disk_space(required_bytes: int) -> bool:
    """
    Check if there's enough disk space for the update.

    Args:
        required_bytes: Estimated space needed in bytes

    Returns:
        True if sufficient space, False otherwise
    """
    try:
        usage = shutil.disk_usage(get_app_root())
        # Require at least 2x the download size for safety (download + extraction)
        return usage.free >= required_bytes * 2
    except Exception as e:
        lg.warning(f"Could not check disk space: {e}")
        return True  # Proceed anyway if check fails


def download_update(
    url: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    timeout: float = 300.0,
    cancel_flag: Optional[Callable[[], bool]] = None,
) -> str:
    """
    Download the outer archive from the given URL.

    Args:
        url: The download URL for the outer archive
        progress_callback: Optional callback(downloaded_bytes, total_bytes) for progress
        timeout: Download timeout in seconds
        cancel_flag: Optional callable that returns True to cancel the download

    Returns:
        Path to the downloaded archive file

    Raises:
        UpdateDownloadError: If download fails or is cancelled
    """
    if not is_frozen():
        raise UpdateDownloadError("Updates are only supported in frozen (PyInstaller) builds")

    app_root = get_app_root()

    # Determine archive extension based on platform
    if sys.platform == "win32":
        archive_name = "vivarium-update.zip"
    else:
        archive_name = "vivarium-update.tar.gz"

    download_path = os.path.join(app_root, archive_name)
    partial_path = download_path + ".partial"

    try:
        lg.info(f"Downloading update from: {url}")

        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Vivarium'}
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            total_size = int(response.headers.get('Content-Length', 0))

            # Check disk space if we know the size
            if total_size > 0 and not check_disk_space(total_size):
                raise UpdateDownloadError(
                    f"Insufficient disk space. Need ~{total_size * 2 // (1024*1024)} MB free."
                )

            downloaded = 0
            chunk_size = 64 * 1024  # 64KB chunks

            with open(partial_path, 'wb') as f:
                while True:
                    # Check for cancellation
                    if cancel_flag and cancel_flag():
                        raise UpdateDownloadError("Download cancelled")

                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback:
                        progress_callback(downloaded, total_size)

            # Verify download completeness
            if total_size > 0 and downloaded != total_size:
                raise UpdateDownloadError(
                    f"Download incomplete: got {downloaded} bytes, expected {total_size}"
                )

        # Move partial file to final location
        if os.path.exists(download_path):
            os.remove(download_path)
        os.rename(partial_path, download_path)

        lg.info(f"Download complete: {download_path}")
        return download_path

    except UpdateDownloadError:
        raise
    except urllib.error.URLError as e:
        raise UpdateDownloadError(f"Network error: {e.reason}")
    except urllib.error.HTTPError as e:
        raise UpdateDownloadError(f"HTTP error {e.code}: {e.reason}")
    except TimeoutError:
        raise UpdateDownloadError("Download timed out")
    except Exception as e:
        raise UpdateDownloadError(f"Download failed: {e}")
    finally:
        # Clean up partial download on error
        if os.path.exists(partial_path):
            try:
                os.remove(partial_path)
            except OSError:
                pass


def apply_downloaded_update(archive_path: str, new_version: str) -> None:
    """
    Extract the outer archive to the app root.

    This overwrites the launcher script and places the inner archive.
    The launcher will handle the actual extraction on next restart.

    Args:
        archive_path: Path to the downloaded outer archive
        new_version: Version being installed (for marker file)

    Raises:
        UpdateDownloadError: If extraction fails
    """
    if not is_frozen():
        raise UpdateDownloadError("Updates are only supported in frozen (PyInstaller) builds")

    app_root = get_app_root()
    current_version = get_version()

    # Save manifest of current _defaults before overwriting
    # This allows detecting what actually changed after restart
    save_defaults_manifest()

    try:
        lg.info(f"Extracting update to: {app_root}")

        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(app_root)
        else:
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(app_root)

        # Clear quarantine on macOS for the inner archive
        if sys.platform == "darwin":
            inner_archive = os.path.join(app_root, get_inner_archive_name())
            if os.path.exists(inner_archive):
                try:
                    import subprocess
                    subprocess.run(['xattr', '-cr', inner_archive], check=False, capture_output=True)
                    lg.info("Cleared quarantine on inner archive")
                except Exception as e:
                    lg.debug(f"Could not clear quarantine: {e}")

        # Create update pending marker
        mark_update_pending(new_version, current_version)

        # Remove the outer archive
        if os.path.exists(archive_path):
            os.remove(archive_path)
            lg.info("Removed outer archive")

        lg.info("Update applied successfully. Restart to complete installation.")

    except Exception as e:
        # Clean up on failure
        if os.path.exists(archive_path):
            try:
                os.remove(archive_path)
            except OSError:
                pass
        raise UpdateDownloadError(f"Failed to extract update: {e}")
