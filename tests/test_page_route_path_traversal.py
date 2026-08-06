"""#0018 — path-traversal / illegal-path probes on the ``/{page:path}`` fallback
route must return a clean 4xx, NOT a 500 + full ERROR:http traceback.

PHP-scanner bots sweep hypha with wordlist paths (``/wp-ws68.php``,
``//aa.php``, ``/../../etc/passwd`` …). The catch-all ``get_pages`` handler
(``hypha/http.py``) resolves the page under the templates dir via
``safe_join``. The traversal *defence* is correct — ``safe_join`` blocks the
escape — but it did so by raising a **bare ``Exception``**
(``hypha/utils/__init__.py``) that no FastAPI handler caught, so an
unauthenticated remote client received an HTTP **500** and the server emitted a
full multi-frame traceback per probe. That is an attacker-controlled
log-amplification / CPU-burn primitive (a bot walking a wordlist makes hypha
write unbounded tracebacks) plus minor work-directory info-disclosure. A bad
path is a **client** error (404), not a server fault.

Fix: ``safe_join`` raises a typed ``UnsafePathError`` (a ``ValueError``
subclass, so any existing ``except Exception`` still catches it — backward
compatible); ``get_pages`` catches it and returns a clean 404 JSON with a
single ``logger.warning`` line (no traceback). The traversal is still blocked —
nothing escapes the templates dir — this only fixes the *error handling*.
"""
import httpx
import pytest

from . import SERVER_URL_SQLITE

pytestmark = pytest.mark.asyncio


# Paths that make ``safe_join`` reject (absolute / dot-segment escape). Sent
# percent-encoded so the client does not normalise the dot-segments away before
# they reach the server. Each must resolve to a clean 4xx, never a 500.
_ILLEGAL_PAGES = [
    "%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # ../../etc/passwd
    "%2e%2e%2fsecret",  # ../secret
    "%2fetc%2fpasswd",  # /etc/passwd (absolute)
]

# Ordinary non-existent probe paths — already returned a clean 404, must stay so.
# (Dash-free: a ``-`` in the first path segment takes the workspace-routing
# branch and 307-redirects, which is pre-existing behavior unrelated to #0018.)
_MISSING_PAGES = [
    "aa.php",
    "xpohywjo.php",
    "config.php",
]


@pytest.mark.parametrize("page", _ILLEGAL_PAGES)
async def test_illegal_page_path_returns_4xx_not_500(fastapi_server_sqlite, page):
    """An illegal/traversal page path returns a clean 4xx, never a 500."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{SERVER_URL_SQLITE}/{page}")

    assert resp.status_code in (400, 403, 404), (
        f"illegal path {page!r} should be a clean client error, "
        f"got {resp.status_code}: {resp.text[:200]}"
    )
    # A server fault (500) means the bare Exception escaped -> traceback spam.
    assert resp.status_code != 500, (
        f"path {page!r} escaped to a 500 (bare Exception not handled): "
        f"{resp.text[:200]}"
    )


@pytest.mark.parametrize("page", _MISSING_PAGES)
async def test_missing_page_still_returns_404(fastapi_server_sqlite, page):
    """A plain non-existent page (no traversal) still returns a clean 404."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{SERVER_URL_SQLITE}/{page}")

    assert resp.status_code == 404, (
        f"missing page {page!r} should be 404, got {resp.status_code}: "
        f"{resp.text[:200]}"
    )


async def test_legitimate_index_page_still_serves(fastapi_server_sqlite):
    """A legitimate page (root -> index.html) still serves 200 after the
    get_pages restructure — the try/except only affects the reject path."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{SERVER_URL_SQLITE}/")

    assert resp.status_code == 200, (
        f"root index page should serve 200, got {resp.status_code}: "
        f"{resp.text[:200]}"
    )


def test_safe_join_raises_typed_unsafe_path_error():
    """``safe_join`` rejects an escape with the typed ``UnsafePathError``.

    A direct ``Exception`` subclass (NOT ``ValueError``) so the ~90 existing
    ``safe_join`` callers that only ``except Exception`` are unaffected, while a
    path-aware caller can catch it *specifically* and map it to a 404.
    """
    from hypha.utils import safe_join, UnsafePathError

    assert issubclass(UnsafePathError, Exception)
    # NOT a ValueError subclass -> does not get newly swallowed by any existing
    # ``except ValueError`` sitting near another safe_join caller.
    assert not issubclass(UnsafePathError, ValueError)
    with pytest.raises(UnsafePathError):
        safe_join("/tmp/base", "../../etc/passwd")
    with pytest.raises(UnsafePathError):
        safe_join("/tmp/base", "/etc/passwd")
    # A legitimate relative path still joins cleanly.
    assert safe_join("/tmp/base", "index.html") == "/tmp/base/index.html"
