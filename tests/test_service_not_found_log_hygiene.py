"""#0002 — an unregistered-service HTTP request must be a clean 404 + single
WARNING, NOT an ERROR + nested tracebacks.

Production symptom (early-osprey 2026-07-31, graceful-salmon 2026-08-11, prod
0.21.107): a client requesting a service that does not exist on the ASGI apps
path (``GET /{workspace}/apps/{service_id}/...``) produced a full ERROR:http log
line plus FOUR nested tracebacks (~120 log lines) for what is semantically a
404. The chain: ``get_service_info`` (``core/workspace.py``) raises
``KeyError("Service not found: reef-imaging/*:agent-lens@*")``; that crosses the
RPC boundary to the ASGI middleware as a ``hypha_rpc.rpc.RemoteException`` (the
original ``KeyError`` type is lost — only the message survives); on 0.21.107 the
middleware had no ``except RemoteException`` arm, so it fell into the generic
``except Exception`` -> HTTP 500 + ``logger.exception`` (traceback at ERROR).

This is the SAME surface as #0017 — the ASGI ``__call__`` missing-service path.
#0017 (0.21.127) added the ``except RemoteException`` arm that maps the exception
to an HTTP status via ``_get_status_for_remote_exception`` and gates the ERROR
traceback behind ``status_code >= 500`` (``hypha/http.py`` ~647-652):

    status_code = _get_status_for_remote_exception(exp)
    if status_code >= 500:
        logger.exception(...)        # genuine fault: keep the traceback
    else:
        logger.warning(...)          # client-side miss: one line, no traceback

Prod 0.21.107 predates 0.21.127, which is why the noise was still observed there;
on main it is already fixed. #0017's full-stack test
(``tests/test_asgi_missing_service.py``) proves the *response* is a clean 404.
What it does NOT assert — and what #0002 is specifically about — is that the
missing-service signature routes to the ``logger.warning`` branch (status < 500)
rather than ``logger.exception``. This module adds that guard as real unit tests
on the pure classifier, mirroring the two-layer approach of #0019
(``test_service_timeout_log_hygiene.py``): a pure-classifier layer here + the
existing full-stack layer in ``test_asgi_missing_service.py``.

These are real tests: they construct genuine ``RemoteException`` instances with
the byte-for-byte shape of a production line and call the real
``_get_status_for_remote_exception`` — no mocks. The non-masking cases prove the
downgrade did NOT widen: a genuine 500-class server fault keeps its 500 (and thus
its ERROR + traceback), and a multiple-match ambiguity surfaces as 400 rather
than being hidden.
"""

from hypha_rpc.rpc import RemoteException

from hypha.http import _get_status_for_remote_exception


# ---------------------------------------------------------------------------
# The exact production signature -> 404 (the logger.warning branch).
# ---------------------------------------------------------------------------


def test_service_not_found_keyerror_maps_to_404():
    """The byte-for-byte prod line: a ``KeyError('Service not found: ...')`` that
    crossed the RPC boundary -> 404, so the middleware logs WARNING, not ERROR."""
    exc = RemoteException(
        'KeyError: "Service not found: reef-imaging/*:agent-lens@*"'
    )
    status = _get_status_for_remote_exception(exc)
    assert status == 404, f"expected 404 for a missing service, got {status}"
    assert status < 500, (
        "status must be < 500 so the ASGI arm takes the logger.warning branch "
        "(no ERROR + traceback) — that is the whole point of #0002."
    )


def test_service_not_found_message_without_keyerror_marker_maps_to_404():
    """A 'not found' message that lost its ``KeyError:`` prefix on the wire still
    maps to 404 via the case-insensitive 'not found' substring — the classifier
    does not depend on the exact exception-type prefix surviving RPC."""
    exc = RemoteException("Service not found: public/*:no-such-service-xyz@*")
    assert _get_status_for_remote_exception(exc) == 404


# ---------------------------------------------------------------------------
# Non-masking guards: the downgrade must NOT swallow genuine faults or
# ambiguities. (Same discipline as #0019's classifier-rejection tests.)
# ---------------------------------------------------------------------------


def test_genuine_server_fault_keeps_500():
    """A real server-side fault (no 'not found'/'KeyError:'/client-error marker)
    stays 500 -> keeps ERROR + traceback. The missing-service downgrade must not
    widen into a blanket 'everything is a 404'."""
    exc = RemoteException("RuntimeError: database connection pool exhausted")
    status = _get_status_for_remote_exception(exc)
    assert status == 500, f"genuine fault must stay 500, got {status}"
    assert status >= 500, (
        "a genuine fault must keep status >= 500 so its traceback is still logged "
        "at ERROR — otherwise the fix would mask real server errors."
    )


def test_multiple_match_ambiguity_surfaces_as_400_not_masked():
    """A multiple-service-match raises AssertionError on the manager side; it must
    surface as 400 (a client-fixable ambiguity), NOT be hidden. This guards the
    CLAUDE.md rule against masking multi-service issues with a default."""
    exc = RemoteException(
        "AssertionError: Multiple services found for public/*:agent-lens@*"
    )
    assert _get_status_for_remote_exception(exc) == 400


def test_permission_denied_maps_to_403():
    """A permission failure crossing the boundary maps to 403 (not folded into the
    404 miss), so an authorization problem is reported honestly."""
    exc = RemoteException(
        "PermissionError: Permission denied for reef-imaging/*:agent-lens@*"
    )
    assert _get_status_for_remote_exception(exc) == 403
