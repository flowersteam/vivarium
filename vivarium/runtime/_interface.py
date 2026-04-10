"""
Panel web interface lifecycle management.

Start and stop the Panel web interface process.
"""

import re
import subprocess
import time
import logging

from vivarium.runtime.paths import get_interface_command


lg = logging.getLogger(__name__)


def wait_for_http(url, retries=1, delay=5.0, timeout=10.0):
    """Wait for an HTTP endpoint to respond with status 200.

    Useful for checking if a web server (e.g., Panel interface) is ready.
    Can be used programmatically or from command line for CI health checks.

    Args:
        url: The URL to check (e.g., 'http://localhost:5006/run_interface')
        retries: Maximum number of attempts (default: 1, no retries)
        delay: Seconds to wait between retries (default: 5.0)
        timeout: Timeout in seconds for each HTTP request (default: 10.0)

    Returns:
        True if URL responded with 200, False otherwise

    Example:
        # Programmatic use
        if wait_for_http('http://localhost:5006/run_interface', retries=12, delay=5):
            print("Panel is ready!")

        # CI use (will exit with code 1 on failure)
        python -c "from vivarium.runtime import wait_for_http; \\
                   assert wait_for_http('http://localhost:5006/run_interface', retries=12)"
    """
    import urllib.request
    import urllib.error

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    lg.info(f"HTTP check OK: {url} (attempt {attempt}/{retries})")
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            lg.debug(f"HTTP check attempt {attempt}/{retries} for {url}: {e}")

        if attempt < retries:
            time.sleep(delay)

    lg.warning(f"HTTP check failed after {retries} attempts: {url}")
    return False


def start_process_and_parse_url(process_command, show_output=True, timeout=10):
    """Start a process and parse URL from its output

    :param process_command: command to start the process
    :param show_output: whether to echo subprocess stdout/stderr
    :param timeout: seconds to wait for URL to appear in output
    :return: tuple of (Popen process object, URL string or None)
    """
    import threading

    process = subprocess.Popen(
        process_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    url_found = [None]  # Use list to allow modification in thread

    def read_output():
        url_pattern = re.compile(r'(http://[^\s]+)')
        try:
            for line in process.stdout:
                if show_output:
                    print(line, end='')
                if url_found[0] is None:
                    match = url_pattern.search(line)
                    if match:
                        url_found[0] = match.group(1)
        except Exception:
            pass

    # Start thread to read output
    output_thread = threading.Thread(target=read_output, daemon=True)
    output_thread.start()

    # Wait for URL with timeout
    start_time = time.time()
    while url_found[0] is None and time.time() - start_time < timeout:
        if process.poll() is not None:  # Process terminated
            break
        time.sleep(0.1)

    return process, url_found[0]


def start_panel_interface(allow_external_origins=False, show_output=True, timeout=10):
    """Start only the Panel web interface (assumes server is running).

    Args:
        allow_external_origins: Whether to allow websocket connections from
                               external origins (e.g., ngrok).
        show_output: Whether to show interface output in console.
        timeout: Seconds to wait for URL to appear in output.

    Returns:
        tuple: (Popen process, interface_url or None)
    """
    interface_command = get_interface_command(allow_external_origins)
    lg.info("Starting web interface...")

    interface_process, interface_url = start_process_and_parse_url(
        interface_command,
        show_output=show_output,
        timeout=timeout
    )

    if interface_url:
        lg.info(f"Interface available at: {interface_url}")
    else:
        # If we can't parse the URL from output, construct it
        interface_url = "http://localhost:5006/run_interface"
        lg.info(f"Interface should be available at: {interface_url}")

    return interface_process, interface_url


def stop_panel_interface(interface_process):
    """Stop a Panel interface process.

    Args:
        interface_process: Popen process object to terminate
    """
    if interface_process is None:
        return

    lg.info(f"Stopping Panel interface (PID: {interface_process.pid})...")

    try:
        interface_process.terminate()
        try:
            interface_process.wait(timeout=5)
            lg.info("Interface terminated gracefully")
        except subprocess.TimeoutExpired:
            lg.warning("Interface did not terminate gracefully, forcing kill...")
            interface_process.kill()
            interface_process.wait(timeout=3)
            lg.info("Interface killed")
    except Exception as e:
        lg.warning(f"Error stopping interface: {e}")
