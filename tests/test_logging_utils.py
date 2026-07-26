"""Tests for hypha.logging_utils — the asyncio conn-lost-write log filter.

Background: under MCP/SSE client churn the vendored `mcp` transport keeps
writing to already-disconnected clients, so asyncio logs bursts of the benign
`socket.send()/sendto() raised exception.` warning (no traceback; the real
connection-lost error is surfaced separately). We filter ONLY those two exact
messages, only on the `asyncio` logger, and must never swallow anything else.
"""

import logging

import pytest

from hypha.logging_utils import (
    ConnLostWriteFilter,
    install_asyncio_connlost_filter,
)


def _record(logger_name, msg, args=()):
    return logging.LogRecord(
        name=logger_name,
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_filter_drops_the_two_exact_connlost_messages():
    """The two exact asyncio futile-write warnings are dropped (filter -> False)."""
    f = ConnLostWriteFilter()
    assert f.filter(_record("asyncio", "socket.send() raised exception.")) is False
    assert f.filter(_record("asyncio", "socket.sendto() raised exception.")) is False


def test_filter_passes_other_asyncio_warnings():
    """Any OTHER asyncio message must pass through untouched (filter -> True).

    This guards against over-suppression: a real asyncio problem (e.g. a fatal
    transport error, an unretrieved exception) must still be logged.
    """
    f = ConnLostWriteFilter()
    assert f.filter(_record("asyncio", "Fatal write error on socket transport")) is True
    assert f.filter(_record("asyncio", "Task was destroyed but it is pending!")) is True
    assert (
        f.filter(_record("asyncio", "Future exception was never retrieved")) is True
    )
    # A message that merely CONTAINS the phrase but isn't the exact literal is
    # not one of ours — exact match keeps the filter maximally narrow.
    assert (
        f.filter(
            _record("asyncio", "custom: socket.send() raised exception. extra detail")
        )
        is True
    )


def test_filter_does_not_break_on_unformattable_record():
    """A record whose args don't format must NEVER be swallowed (fail-open)."""
    f = ConnLostWriteFilter()
    # %d with a non-numeric arg raises inside getMessage(); filter must return
    # True (keep) rather than crash or drop.
    bad = _record("asyncio", "value=%d", args=("not-an-int",))
    assert f.filter(bad) is True


def test_install_is_idempotent_and_attached_to_asyncio_logger():
    """install_asyncio_connlost_filter attaches exactly one filter, idempotently."""
    asyncio_logger = logging.getLogger("asyncio")
    # Clean slate for any previously-installed instance (e.g. via importing
    # server in the same test session).
    asyncio_logger.filters = [
        flt
        for flt in asyncio_logger.filters
        if not isinstance(flt, ConnLostWriteFilter)
    ]

    first = install_asyncio_connlost_filter()
    second = install_asyncio_connlost_filter()
    assert first is second, "second install must not create a duplicate filter"

    installed = [
        flt
        for flt in asyncio_logger.filters
        if isinstance(flt, ConnLostWriteFilter)
    ]
    assert len(installed) == 1

    asyncio_logger.removeFilter(first)


def test_installed_filter_suppresses_real_asyncio_log_call(caplog):
    """End-to-end: after install, an asyncio.warning of the exact message is not
    emitted, while a different asyncio warning still is."""
    asyncio_logger = logging.getLogger("asyncio")
    install_asyncio_connlost_filter()
    try:
        with caplog.at_level(logging.WARNING, logger="asyncio"):
            asyncio_logger.warning("socket.send() raised exception.")
            asyncio_logger.warning("Fatal write error on socket transport")
        messages = [r.getMessage() for r in caplog.records]
        assert "socket.send() raised exception." not in messages
        assert "Fatal write error on socket transport" in messages
    finally:
        for flt in list(asyncio_logger.filters):
            if isinstance(flt, ConnLostWriteFilter):
                asyncio_logger.removeFilter(flt)
