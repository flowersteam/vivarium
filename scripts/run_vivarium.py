import logging

import webbrowser
import sys

from vivarium.utils.handle_server_interface import (
    get_server_interface_pids,
    start_server_and_interface,
    stop_server_and_interface,
)


lg = logging.getLogger(__name__)


def main(cmd_args):
    interface_pids, server_pids = get_server_interface_pids()
    print(f"Interface PIDs: {interface_pids}")
    print(f"Server PIDs: {server_pids}")
    stop_server_and_interface(safe_mode=True)
    start_server_and_interface(cmd_args, wait_time=6)


if __name__ == "__main__":
    cmd_args = sys.argv[1:]
    main(cmd_args)
    webbrowser.open("http://localhost:5006/run_interface")
    # try:
    #     # Keep the script running to allow the server to run
    #     while True:
    #         pass
    # except KeyboardInterrupt:
    #     lg.info("Script stopped by user. Cleaning up...")
    #     stop_server_and_interface(safe_mode=True)
