"""#0020 — ``hypha.mcp`` must honor ``HYPHA_LOGLEVEL`` like every other module.

``hypha/mcp.py`` hardcoded ``logger.setLevel(logging.DEBUG)`` at import time,
diverging from the whole rest of the codebase (which does
``logger.setLevel(LOGLEVEL)`` where ``LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL",
"WARNING")``). Two harms in a prod deployment that sets ``HYPHA_LOGLEVEL=INFO``:

1. **Volume:** a single per-request ``logger.debug("MCP Middleware: Processing
   HTTP request to ...")`` line (``hypha/mcp.py``) then fires for *every* HTTP
   request — measured at ~39% of all prod log lines in 24h.
2. **Observability trap (worse):** ``hypha.mcp`` DEBUG stays *enabled* in prod
   while every other module's DEBUG is suppressed at INFO. That makes DEBUG look
   "on" and nearly produced a false "verified — event eliminated" conclusion
   during a log review (the module whose DEBUG survives is not representative).

These are hermetic subprocess tests: each spawns a fresh interpreter with a
controlled ``HYPHA_LOGLEVEL``, imports ``hypha.mcp`` (running its module-level
``setLevel``), and reports the resulting logger level — a real import, no mocks,
no shared global-state bleed into the rest of the suite.
"""
import logging
import os
import subprocess
import sys

import pytest


def _imported_logger_level(module: str, logger_name: str, loglevel_env: str) -> int:
    """Spawn a fresh interpreter with HYPHA_LOGLEVEL set, import ``module``, and
    return the numeric level of ``logger_name`` after the module's import-time
    ``setLevel`` runs.

    ``logger_name`` is the *actual* name the module configures — which is NOT the
    dotted module path for most of hypha: websocket.py names its logger
    ``"websocket-server"``, http.py ``"http"``, core ``"core"``. mcp.py alone uses
    ``getLogger(__name__)`` → ``"hypha.mcp"`` (the very ``DEBUG:hypha.mcp`` prefix
    seen in prod).
    """
    code = (
        "import logging, importlib;"
        f"importlib.import_module('{module}');"
        f"print(logging.getLogger('{logger_name}').level)"
    )
    env = {**os.environ, "HYPHA_LOGLEVEL": loglevel_env}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"importing {module} under HYPHA_LOGLEVEL={loglevel_env} failed:\n"
        f"{proc.stderr}"
    )
    return int(proc.stdout.strip().splitlines()[-1])


# The bare logger names each module actually configures (see docstring above).
_MCP = ("hypha.mcp", "hypha.mcp")
_WEBSOCKET = ("hypha.websocket", "websocket-server")


def test_mcp_logger_respects_info_loglevel():
    """With HYPHA_LOGLEVEL=INFO, hypha.mcp must be INFO — NOT forced to DEBUG.

    Reproduces the bug: the hardcoded ``setLevel(logging.DEBUG)`` yields level
    10 (DEBUG) regardless of the env, so this asserts 20 (INFO) and fails until
    mcp.py reads LOGLEVEL like every other module.
    """
    level = _imported_logger_level(*_MCP, "INFO")
    assert level == logging.INFO, (
        f"hypha.mcp forced level {logging.getLevelName(level)} under "
        f"HYPHA_LOGLEVEL=INFO; expected INFO (it must not hardcode DEBUG)"
    )


def test_mcp_logger_matches_other_modules():
    """hypha.mcp must not diverge from the rest of the codebase.

    Under the same HYPHA_LOGLEVEL, hypha.mcp and hypha.websocket must resolve to
    the same level — this is the invariant the hardcoded DEBUG broke (the
    'observability trap': one module's DEBUG surviving while others are muted).
    """
    mcp_level = _imported_logger_level(*_MCP, "INFO")
    ws_level = _imported_logger_level(*_WEBSOCKET, "INFO")
    assert ws_level == logging.INFO, (
        "reference module hypha.websocket did not resolve to INFO under "
        f"HYPHA_LOGLEVEL=INFO (got {logging.getLevelName(ws_level)}); the "
        "convention assumption is wrong — investigate before trusting this test"
    )
    assert mcp_level == ws_level, (
        f"hypha.mcp level ({logging.getLevelName(mcp_level)}) diverges from "
        f"the shared convention that hypha.websocket follows "
        f"({logging.getLevelName(ws_level)}) under the same HYPHA_LOGLEVEL — "
        f"mcp.py must read LOGLEVEL from the env like every other module"
    )


def test_mcp_logger_still_tunable_to_debug():
    """The fix must read the env, not hardcode INFO the other way.

    With HYPHA_LOGLEVEL=DEBUG, hypha.mcp must be DEBUG — proving DEBUG is still
    reachable when an operator actually asks for it.
    """
    level = _imported_logger_level(*_MCP, "DEBUG")
    assert level == logging.DEBUG, (
        f"hypha.mcp is {logging.getLevelName(level)} under HYPHA_LOGLEVEL=DEBUG; "
        f"expected DEBUG (the fix must honor the env in both directions)"
    )
