import time

import pytest

from vivarium.utils.handle_server_interface import (
    start_server_and_interface,
    stop_server_and_interface,
)

WAIT_TIME = 0


@pytest.fixture
def start_and_stop_server():
    def _start_and_stop(scene_name, notebook_mode):
        start_server_and_interface(
            scene_name=scene_name, notebook_mode=notebook_mode, wait_time=WAIT_TIME
        )
        time.sleep(1)
        yield
        stop_server_and_interface()

    return _start_and_stop


def test_start_stop(start_and_stop_server):
    scene_name = "quickstart"
    start_and_stop_server(scene_name, False)
    assert True


def test_start_stop_notebook_mode(start_and_stop_server):
    scene_name = "session_3"
    start_and_stop_server(scene_name, True)
    assert True
