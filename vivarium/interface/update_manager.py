import os
import logging
import threading

import panel as pn

from vivarium.runtime.paths import is_frozen
from vivarium.runtime.updater import (
    check_for_updates,
    download_update,
    apply_downloaded_update,
    is_update_pending,
    get_update_pending_info,
    clear_update_pending,
    perform_post_update_merge,
    UpdateDownloadError,
)

lg = logging.getLogger(__name__)


class UpdateManager:
    """Manages update checking, downloading, and defaults merge notifications.

    Self-contained: owns its widgets, background thread, and notification panels.
    The host app places the panels in its layout via ``insert_into()``.
    """

    def __init__(self):
        # Track download state
        self._download_thread = None
        self._download_cancelled = False
        self._update_info = None
        self._defaults_info = None

        # --- Update notification widgets ---

        self.update_banner = pn.pane.Markdown(
            "",
            styles={
                'background-color': '#1a73e8',
                'color': 'white',
                'padding': '10px 20px',
                'border-radius': '5px',
                'text-align': 'center',
            },
            sizing_mode="stretch_width",
        )

        self.update_download_btn = pn.widgets.Button(
            name="Download Update",
            button_type="success",
            width=150,
        )
        self.update_download_btn.on_click(self._download_update)

        self.update_dismiss_btn = pn.widgets.Button(
            name="Dismiss",
            button_type="default",
            width=100,
        )
        self.update_dismiss_btn.on_click(self._dismiss_update_notification)

        self.update_progress = pn.widgets.Progress(
            name="Downloading...",
            value=0,
            max=100,
            sizing_mode="stretch_width",
            visible=False,
        )

        self.update_status_text = pn.pane.Markdown(
            "",
            styles={'text-align': 'center'},
            visible=False,
        )

        self.update_cancel_btn = pn.widgets.Button(
            name="Cancel",
            button_type="warning",
            width=100,
            visible=False,
        )
        self.update_cancel_btn.on_click(self._cancel_download)

        self.update_restart_btn = pn.widgets.Button(
            name="Restart to Apply",
            button_type="success",
            width=150,
            visible=False,
        )
        self.update_restart_btn.on_click(self._request_restart)

        self.update_notification_panel = pn.Column(
            self.update_banner,
            pn.Row(
                self.update_download_btn,
                self.update_cancel_btn,
                self.update_restart_btn,
                self.update_dismiss_btn,
                align="center",
            ),
            self.update_progress,
            self.update_status_text,
            visible=False,
            sizing_mode="stretch_width",
        )

        # --- Defaults change notification widgets ---

        self.defaults_banner = pn.pane.Markdown(
            "",
            styles={
                'background-color': '#2e7d32',
                'color': 'white',
                'padding': '10px 20px',
                'border-radius': '5px',
                'text-align': 'center',
            },
            sizing_mode="stretch_width",
        )

        self.defaults_dismiss_btn = pn.widgets.Button(
            name="Got it",
            button_type="default",
            width=100,
        )
        self.defaults_dismiss_btn.on_click(self._dismiss_defaults_notification)

        self.defaults_notification_panel = pn.Column(
            self.defaults_banner,
            pn.Row(self.defaults_dismiss_btn, align="center"),
            visible=False,
            sizing_mode="stretch_width",
        )

    def start_update_check(self, include_prereleases, scene_selection_panel):
        """Start checking for updates and defaults changes in a background thread.

        Args:
            include_prereleases: Whether to check pre-release versions.
            scene_selection_panel: The Panel layout to insert notifications into
                when the background check completes.
        """
        self._scene_selection_panel = scene_selection_panel

        def check_updates():
            try:
                # First, check if an update was recently applied
                if is_update_pending():
                    pending_info = get_update_pending_info()
                    if pending_info:
                        lg.info(f"Update pending from {pending_info.get('previous_version')} to {pending_info.get('new_version')}")

                    # Perform smart merge: restore user modifications for files not changed by update
                    # Files changed by both user and update will use new version (conflicts)
                    merge_info = perform_post_update_merge()
                    if merge_info:
                        self._merge_info = merge_info
                        if pn.state.curdoc:
                            pn.state.curdoc.add_next_tick_callback(self._show_defaults_notification)
                        else:
                            self._show_defaults_notification()

                    # Clear the pending marker
                    clear_update_pending()

                # Check for new updates
                update_info = check_for_updates(timeout=5.0, include_prereleases=include_prereleases)
                if update_info:
                    self._update_info = update_info
                    # Schedule UI update on main thread
                    if pn.state.curdoc:
                        pn.state.curdoc.add_next_tick_callback(self._show_update_notification)
                    else:
                        self._show_update_notification()
            except Exception as e:
                lg.debug(f"Update check failed: {e}")

        thread = threading.Thread(target=check_updates, daemon=True)
        thread.start()

    # --- Update notification callbacks ---

    def _show_update_notification(self):
        if not self._update_info:
            return

        current = self._update_info['current_version']
        latest = self._update_info['latest_version']

        self.update_banner.object = (
            f"**Update Available:** A new version of Vivarium ({latest}) is available. "
            f"You are running version {current}."
        )
        self.update_notification_panel.visible = True

        # Insert at the top of scene selection panel if not already there
        panel = self._scene_selection_panel
        if panel is not None and self.update_notification_panel not in panel:
            panel.insert(0, self.update_notification_panel)

    def _download_update(self, _event):
        if not self._update_info or not self._update_info.get('download_url'):
            lg.warning("No download URL available")
            return

        self._download_cancelled = False

        self.update_download_btn.disabled = True
        self.update_download_btn.visible = False
        self.update_cancel_btn.visible = True
        self.update_dismiss_btn.visible = False
        self.update_progress.visible = True
        self.update_progress.value = 0
        self.update_status_text.object = "Starting download..."
        self.update_status_text.visible = True

        def do_download():
            try:
                url = self._update_info['download_url']
                new_version = self._update_info['latest_version']

                def progress_callback(downloaded, total):
                    if total > 0:
                        percent = int(downloaded * 100 / total)
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)

                        def update_ui():
                            self.update_progress.value = percent
                            self.update_status_text.object = f"Downloading: {mb_downloaded:.1f} / {mb_total:.1f} MB ({percent}%)"

                        if pn.state.curdoc:
                            pn.state.curdoc.add_next_tick_callback(update_ui)
                        else:
                            update_ui()

                def cancel_check():
                    return self._download_cancelled

                archive_path = download_update(
                    url,
                    progress_callback=progress_callback,
                    cancel_flag=cancel_check,
                )

                apply_downloaded_update(archive_path, new_version)

                if pn.state.curdoc:
                    pn.state.curdoc.add_next_tick_callback(self._show_restart_prompt)
                else:
                    self._show_restart_prompt()

            except UpdateDownloadError as e:
                lg.error(f"Update download failed: {e}")
                if pn.state.curdoc:
                    pn.state.curdoc.add_next_tick_callback(lambda: self._show_download_error(str(e)))
                else:
                    self._show_download_error(str(e))
            except Exception as e:
                lg.exception(f"Unexpected error during update: {e}")
                if pn.state.curdoc:
                    pn.state.curdoc.add_next_tick_callback(lambda: self._show_download_error(f"Unexpected error: {e}"))
                else:
                    self._show_download_error(f"Unexpected error: {e}")

        self._download_thread = threading.Thread(target=do_download, daemon=True)
        self._download_thread.start()

    def _show_restart_prompt(self):
        self.update_progress.visible = False
        self.update_cancel_btn.visible = False
        self.update_restart_btn.visible = True
        self.update_dismiss_btn.visible = True
        self.update_status_text.object = "**Update downloaded!** Restart Vivarium to complete the installation."
        self.update_banner.object = (
            f"**Update Ready:** Version {self._update_info['latest_version']} is ready to install."
        )

    def _show_download_error(self, error_msg: str):
        self.update_progress.visible = False
        self.update_cancel_btn.visible = False
        self.update_download_btn.disabled = False
        self.update_download_btn.visible = True
        self.update_download_btn.name = "Retry Download"
        self.update_dismiss_btn.visible = True
        self.update_status_text.object = f"**Error:** {error_msg}"
        self._download_thread = None

    def _cancel_download(self, _event):
        self._download_cancelled = True
        self.update_status_text.object = "Cancelling download..."

    def _request_restart(self, _event):
        self.update_status_text.object = (
            "**Please close Vivarium from the terminal (Ctrl-C) and restart it using the launcher script.**"
        )
        self.update_restart_btn.visible = False

    # --- Defaults merge notification callbacks ---

    def _show_defaults_notification(self):
        if not hasattr(self, '_merge_info') or not self._merge_info:
            return

        backup_dir = self._merge_info.get('backup_dir', '')
        backup_name = os.path.basename(backup_dir) if backup_dir else 'backup folder'

        all_conflicts = []
        restored_count = 0

        for folder in ['conf', 'notebooks']:
            info = self._merge_info.get(folder, {})
            conflicts = info.get('conflicts', [])
            restored = info.get('restored', [])

            for filename in conflicts:
                parts = filename.replace("\\", "/").split("/")
                if any(part.startswith(".") for part in parts):
                    continue
                all_conflicts.append(f"{folder}/{filename}")
            restored_count += len(restored)

        if not all_conflicts:
            if restored_count > 0:
                lg.info(f"Update complete: {restored_count} user modification(s) preserved")
            return

        file_list = ", ".join(f"`{f}`" for f in all_conflicts)
        message = (
            f"**Update applied:** The following files you modified were also updated: {file_list}. "
            f"The new versions are now in use. "
            f"Your previous versions are saved in the **{backup_name}** folder "
            f"(located next to the `Start-Vivarium` script)."
        )

        self.defaults_banner.object = message
        self.defaults_notification_panel.visible = True

        # Insert at the top of scene selection panel if not already there
        panel = self._scene_selection_panel
        if panel is not None and self.defaults_notification_panel not in panel:
            panel.insert(0, self.defaults_notification_panel)

    def _dismiss_defaults_notification(self, _event):
        self.defaults_notification_panel.visible = False

    def _dismiss_update_notification(self, _event):
        self.update_notification_panel.visible = False
        self.update_download_btn.disabled = False
        self.update_download_btn.visible = True
        self.update_download_btn.name = "Download Update"
        self.update_progress.visible = False
        self.update_cancel_btn.visible = False
        self.update_restart_btn.visible = False
        self.update_status_text.visible = False
