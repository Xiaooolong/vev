import pytest

from evals.tests.fake_server import FakeServer


@pytest.fixture(scope="session")
def _server():
    srv = FakeServer().start()
    yield srv
    srv.stop()


@pytest.fixture
def fake(_server):
    _server.reset()
    return _server
