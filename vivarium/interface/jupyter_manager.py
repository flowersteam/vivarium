import os
import time
import logging

import panel as pn

from vivarium.runtime.paths import get_app_root, DEFAULT_JUPYTER_PORT
from vivarium.runtime import (
    check_jupyter_running,
    register_jupyter_port,
    unregister_jupyter_port,
    find_next_available_port,
    start_jupyter_server,
    stop_jupyter_server,
    kill_port_processes,
)


lg = logging.getLogger(__name__)


class JupyterManager:
    """Manages Jupyter server lifecycle and notebook UI widgets.

    Self-contained: owns its widgets, server process handle, and all callbacks.
    The host app embeds the notebook tab via ``create_notebook_tab()`` and wires
    callbacks via ``set_callbacks()``.
    """

    def __init__(self, notebook_path=None, jupyter_port=DEFAULT_JUPYTER_PORT):
        self.notebook_path = notebook_path
        self.jupyter_port = jupyter_port

        # Jupyter server management
        self.jupyter_process = None
        self._jupyter_started_by_us = False

        # --- Widgets ---

        self.jupyter_status = pn.pane.Markdown(
            "**Jupyter Status:** Checking...",
            sizing_mode="stretch_width",
        )

        self.jupyter_port_input = pn.widgets.IntInput(
            name="Jupyter Port:",
            value=self.jupyter_port,
            start=8888,
            end=9999,
            step=1,
            width=120,
        )

        self.check_jupyter_btn = pn.widgets.Button(
            name="Check Server",
            button_type="default",
            width=120,
        )

        self.start_jupyter_btn = pn.widgets.Button(
            name="Start Jupyter Server",
            button_type="success",
            width=200,
        )

        self.stop_jupyter_btn = pn.widgets.Button(
            name="Stop Jupyter Server",
            button_type="danger",
            width=200,
            visible=False,
        )

        self.open_configured_notebook_btn = pn.widgets.Button(
            name=f"Open {os.path.basename(self.notebook_path) if self.notebook_path else 'Configured Notebook'}",
            button_type="primary",
            width=250,
            visible=False,
        )

        self.open_new_notebook_btn = pn.widgets.Button(
            name="Open New Notebook",
            button_type="primary",
            width=200,
            visible=False,
        )

        self.notebook_url = pn.widgets.TextInput(
            name="Or enter notebook URL:",
            placeholder=f"http://localhost:{self.jupyter_port}/notebooks/path/to/notebook.ipynb",
            value="",
            width=400,
            visible=False,
        )

        # Notebook iframe (initially empty)
        self.notebook_iframe = pn.pane.HTML(
            "",
            sizing_mode="stretch_both",
        )

        # --- Conflict resolution panel ---

        self._conflict_port = None
        self._conflict_message = pn.pane.Markdown("")
        self._suggested_port_msg = pn.pane.Markdown("")

        self._jupyter_use_existing_btn = pn.widgets.Button(
            name="Use Existing", button_type="primary", width=120
        )
        self._jupyter_kill_only_btn = pn.widgets.Button(
            name="Kill Server", button_type="danger", width=100
        )
        self._jupyter_kill_restart_btn = pn.widgets.Button(
            name="Kill & Restart", button_type="warning", width=120
        )
        self._jupyter_use_different_port_btn = pn.widgets.Button(
            name="Use Different Port", button_type="success", width=150
        )
        self._jupyter_cancel_conflict_btn = pn.widgets.Button(
            name="Cancel", button_type="default", width=80
        )

        self.jupyter_conflict_panel = pn.Column(
            pn.pane.Markdown("### Jupyter Server Already Running", styles={'color': 'orange'}),
            self._conflict_message,
            pn.Row(
                self._jupyter_use_existing_btn,
                self._jupyter_kill_only_btn,
                self._jupyter_kill_restart_btn,
                self._jupyter_use_different_port_btn,
            ),
            pn.Row(
                self._jupyter_cancel_conflict_btn,
            ),
            self._suggested_port_msg,
            visible=False,
        )

    def set_callbacks(self):
        """Wire up all Jupyter-related button callbacks."""
        self.check_jupyter_btn.on_click(self.check_jupyter_cb)
        self.start_jupyter_btn.on_click(self.start_jupyter_cb)
        self.stop_jupyter_btn.on_click(self.stop_jupyter_cb)
        self.open_configured_notebook_btn.on_click(self.open_configured_notebook_cb)
        self.open_new_notebook_btn.on_click(self.open_new_notebook_cb)
        self.notebook_url.param.watch(self.notebook_url_cb, "value")
        # Conflict resolution callbacks
        self._jupyter_use_existing_btn.on_click(self._jupyter_use_existing_cb)
        self._jupyter_kill_only_btn.on_click(self._jupyter_kill_only_cb)
        self._jupyter_kill_restart_btn.on_click(self._jupyter_kill_restart_cb)
        self._jupyter_use_different_port_btn.on_click(self._jupyter_use_different_port_cb)
        self._jupyter_cancel_conflict_btn.on_click(self._jupyter_cancel_conflict_cb)

    def create_notebook_tab(self):
        """Return the Panel column for the Notebook tab."""
        return pn.Column(
            self.jupyter_status,
            pn.Row(
                self.jupyter_port_input,
                self.check_jupyter_btn,
                self.start_jupyter_btn,
                self.stop_jupyter_btn,
            ),
            self.jupyter_conflict_panel,
            pn.Row(
                self.open_configured_notebook_btn,
                self.open_new_notebook_btn,
            ),
            self.notebook_url,
            self.notebook_iframe,
            sizing_mode="stretch_both",
        )

    def stop(self):
        """Stop the Jupyter server if running (cleanup helper)."""
        if self.jupyter_process is not None or self._jupyter_started_by_us:
            self.stop_jupyter_cb(None)

    # --- Callbacks ---

    def start_jupyter_cb(self, event):
        port = self.jupyter_port_input.value

        if check_jupyter_running(port):
            self._show_jupyter_conflict_panel(port)
            return

        self._do_start_jupyter(port)

    def _do_start_jupyter(self, port):
        project_root = get_app_root()

        try:
            lg.info(f"Starting Jupyter server on port {port}...")
            self.jupyter_status.object = f"**Jupyter Status:** 🔄 Starting on port {port}..."

            self.jupyter_process = start_jupyter_server(
                port=port,
                notebook_dir=project_root,
                show_output=False,
                return_process_object=True
            )

            self.jupyter_port = port
            self.jupyter_port_input.value = port

            register_jupyter_port(port)
            self._jupyter_started_by_us = True

            for _ in range(20):
                time.sleep(0.5)
                if check_jupyter_running(port):
                    lg.info(f"Jupyter server confirmed running on port {port}")
                    break

            self._check_jupyter_status()
        except RuntimeError as e:
            if "already in use" in str(e).lower():
                self._show_jupyter_conflict_panel(port)
            else:
                error_msg = str(e)
                lg.error(error_msg)
                self.jupyter_status.object = f"**Jupyter Status:** ❌ {error_msg}"

    def _show_jupyter_conflict_panel(self, port):
        self._conflict_port = port
        self._conflict_message.object = f"A Jupyter server is already running on port **{port}**. What would you like to do?"

        try:
            suggested = find_next_available_port(port + 1)
            self._suggested_port_msg.object = f"*Suggested available port: **{suggested}***"
        except RuntimeError:
            self._suggested_port_msg.object = "*No available ports found nearby*"

        self.jupyter_conflict_panel.visible = True

    def _jupyter_use_existing_cb(self, event):
        port = self._conflict_port
        self.jupyter_conflict_panel.visible = False

        self.jupyter_port = port
        self.jupyter_port_input.value = port
        self._jupyter_started_by_us = False
        self.jupyter_process = None

        lg.info(f"Using existing Jupyter server on port {port}")
        self._check_jupyter_status()

    def _jupyter_kill_only_cb(self, event):
        port = self._conflict_port
        self.jupyter_conflict_panel.visible = False

        lg.info(f"Killing Jupyter server on port {port}...")
        self.jupyter_status.object = f"**Jupyter Status:** 🔄 Stopping server on port {port}..."

        killed = kill_port_processes(port, servers_only=True)
        if killed:
            lg.info(f"Killed Jupyter processes: {killed}")

        unregister_jupyter_port(port)

        time.sleep(0.5)

        self._check_jupyter_status()

    def _jupyter_kill_restart_cb(self, event):
        port = self._conflict_port
        self.jupyter_conflict_panel.visible = False

        lg.info(f"Killing existing Jupyter on port {port} and restarting...")
        self.jupyter_status.object = f"**Jupyter Status:** 🔄 Restarting on port {port}..."

        killed = kill_port_processes(port, servers_only=True)
        if killed:
            lg.info(f"Killed existing Jupyter processes: {killed}")

        unregister_jupyter_port(port)

        time.sleep(0.5)

        self._do_start_jupyter(port)

    def _jupyter_use_different_port_cb(self, event):
        self.jupyter_conflict_panel.visible = False

        try:
            port = self._conflict_port
            suggested = find_next_available_port(port + 1)
            lg.info(f"Using alternative port {suggested}")
            self._do_start_jupyter(suggested)
        except RuntimeError as e:
            lg.error(f"Could not find available port: {e}")
            self.jupyter_status.object = f"**Jupyter Status:** ❌ {e}"

    def _jupyter_cancel_conflict_cb(self, event):
        self.jupyter_conflict_panel.visible = False
        self._check_jupyter_status()

    def check_jupyter_cb(self, event):
        port = self.jupyter_port_input.value
        self.jupyter_port = port
        self._check_jupyter_status()

    def stop_jupyter_cb(self, event):
        lg.info("Stopping Jupyter server...")
        self.jupyter_status.object = f"**Jupyter Status:** 🔄 Stopping..."

        stop_jupyter_server(self.jupyter_process, port=self.jupyter_port)

        unregister_jupyter_port(self.jupyter_port)
        self.jupyter_process = None
        self._jupyter_started_by_us = False

        self._check_jupyter_status()

    def open_configured_notebook_cb(self, event):
        if self.notebook_path:
            project_root = get_app_root()
            if not os.path.isabs(self.notebook_path):
                notebook_path = os.path.join(project_root, self.notebook_path)
            else:
                notebook_path = self.notebook_path

            lg.info(f"Opening notebook: {notebook_path}")
            notebook_rel_path = os.path.relpath(notebook_path, project_root)
            url = f"http://localhost:{self.jupyter_port}/notebooks/{notebook_rel_path}"
            self.notebook_url.value = url
            self._update_notebook()

    def open_new_notebook_cb(self, event):
        url = f"http://localhost:{self.jupyter_port}/tree"
        self.notebook_url.value = url
        self._update_notebook()

    def notebook_url_cb(self, event):
        if event.new:
            self._update_notebook()

    def _check_jupyter_status(self):
        is_running = check_jupyter_running(self.jupyter_port)

        if is_running:
            self.jupyter_status.object = f"**Jupyter Status:** ✓ Running on port {self.jupyter_port}"
            self.jupyter_port_input.visible = False
            self.start_jupyter_btn.visible = False
            self.stop_jupyter_btn.visible = True
            self.open_configured_notebook_btn.visible = bool(self.notebook_path)
            self.open_new_notebook_btn.visible = True
            self.notebook_url.visible = True
        else:
            self.jupyter_status.object = f"**Jupyter Status:** ✗ Not running"
            self.jupyter_port_input.visible = True
            self.start_jupyter_btn.visible = True
            self.stop_jupyter_btn.visible = False
            self.open_configured_notebook_btn.visible = False
            self.open_new_notebook_btn.visible = False
            self.notebook_url.visible = False
            self.notebook_iframe.object = ""

    def _update_notebook(self):
        url = self.notebook_url.value
        if url:
            iframe_html = f'''<iframe
                src="{url}"
                width="100%"
                height="100%"
                frameborder="0"
                style="border: 1px solid #ddd;"
                sandbox="allow-same-origin allow-scripts allow-forms allow-modals allow-popups allow-downloads"
                allow="clipboard-read; clipboard-write"
            ></iframe>'''
            self.notebook_iframe.object = iframe_html
        else:
            self.notebook_iframe.object = ""
