"""
Shared pytest configuration for the backend test suite.

* Makes both `import job_manager` (backend/ on sys.path) and `import backend.x` (repo root on sys.path) work.
* Runs `async def` tests without requiring the pytest-asyncio plugin.
* Skips live-API tests when no SatQuery server is listening on localhost:8000.
"""
import asyncio
import inspect
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BACKEND_DIR)
for path in (BACKEND_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# Scratch files kept out of version control (see .gitignore)
collect_ignore_glob = ["tests/test_temp*.py"]

LIVE_API_MODULES = ("test_regression_api.py", "test_s1aad_e2e.py")


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: run the coroutine test in a fresh event loop")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**kwargs))
        return True
    return None


def _server_up() -> bool:
    try:
        import requests
        return requests.get("http://localhost:8000/api/system", timeout=3).ok
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    live_items = [i for i in items if os.path.basename(str(i.fspath)) in LIVE_API_MODULES]
    if live_items and not _server_up():
        skip = pytest.mark.skip(reason="SatQuery API server is not running on localhost:8000")
        for item in live_items:
            item.add_marker(skip)
