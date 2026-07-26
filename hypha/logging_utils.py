"""Logging helpers for the Hypha server.

Currently this hosts a single, narrowly-scoped log-hygiene filter for a benign
asyncio transport warning that is chatty under MCP/SSE client churn.
"""

import logging

# The exact messages asyncio's stream/datagram transports emit when application
# code keeps calling ``transport.write()`` on a socket whose connection is
# ALREADY lost. See CPython ``asyncio/selector_events.py`` and
# ``asyncio/proactor_events.py``: the warning is logged only from the
# ``if self._conn_lost:`` branch, and only once the futile-write counter passes
# ``constants.LOG_THRESHOLD_FOR_CONNLOST_WRITES`` (5). It carries NO exception
# and NO traceback — the real connection-lost error was already surfaced
# separately via the transport's ``connection_lost(exc)`` path. So the message
# is purely "you wrote to a dead socket 5+ times"; suppressing it hides nothing
# actionable.
#
# In Hypha this fires in bursts around MCP ``/rpc`` (streamable-HTTP / SSE)
# request churn: the vendored ``mcp`` library's transport pushes several events
# to a client that has already disconnected. The write loop lives inside that
# third-party transport, not in Hypha's code, so we cannot cleanly guard each
# write — instead we drop the confirmed-benign warning at the ``asyncio`` logger.
_CONNLOST_WRITE_MESSAGES = frozenset(
    {
        "socket.send() raised exception.",
        "socket.sendto() raised exception.",
    }
)


class ConnLostWriteFilter(logging.Filter):
    """Drop asyncio's benign "socket.send()/sendto() raised exception." warnings.

    These are emitted only for writes to an already-lost connection (no
    traceback, real error handled elsewhere), so they are noise. Every other
    record — including any other asyncio warning — is passed through unchanged.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Match on the raw, un-formatted message. asyncio logs these with no
        # args, so getMessage() equals the literal string; exact-match keeps the
        # filter as narrow as possible (no substring false-positives).
        try:
            message = record.getMessage()
        except Exception:
            # If a record cannot be formatted, it is definitely not one of ours;
            # never swallow it.
            return True
        return message not in _CONNLOST_WRITE_MESSAGES


def install_asyncio_connlost_filter() -> ConnLostWriteFilter:
    """Attach :class:`ConnLostWriteFilter` to the ``asyncio`` logger (idempotent).

    Returns the installed filter instance. Safe to call multiple times: a second
    call does not add a duplicate filter.
    """
    asyncio_logger = logging.getLogger("asyncio")
    for existing in asyncio_logger.filters:
        if isinstance(existing, ConnLostWriteFilter):
            return existing
    connlost_filter = ConnLostWriteFilter()
    asyncio_logger.addFilter(connlost_filter)
    return connlost_filter
