"""Unit test for the start-up "ready" log line.

When the Visualization_Server has bound its loopback sockets and is
ready to accept HTTP connections,
:func:`project_knowledge_mcp.visualization_server.emit_ready_log` SHALL
emit exactly one log record whose message is
``"Visualization_Server ready at http://127.0.0.1:{port}"`` at
``logging.WARNING`` level. The wording is fixed by Requirement 12.7
and reproduced verbatim by :data:`READY_LOG_TEMPLATE`; this test pins
both the wording and the level so a regression is caught at unit-test
time.

The level deviates from Requirement 12.7's literal "INFO" because
operators want the line visible at Python's default root threshold
(``WARNING``) without having to bump every third-party library's
verbosity too. Operationally the "ready" banner is in the same class
as the per-project refresh progress lines: an important signal that
should always surface.

The test attaches a list-collecting :class:`logging.Handler` to a
dedicated logger that is then handed to
:func:`emit_ready_log` via its ``logger`` keyword argument, so the
captured records come from a single, isolated source. A second test
exercises the default code path (no ``logger=`` argument) by capturing
records from the module's own logger via the pytest ``caplog`` fixture,
confirming that the module-level logger (``_LOG``) is the default
emitter.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from project_knowledge_mcp.visualization_server import (
    READY_LOG_TEMPLATE,
    emit_ready_log,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit


#: Fully-qualified name of the visualization_server module logger
#: (``logging.getLogger(__name__)`` inside the module). Pinned here so
#: the test will fail loudly if the module is renamed without the
#: corresponding test update.
_VIS_LOGGER_NAME = "project_knowledge_mcp.visualization_server"


class _ListHandler(logging.Handler):
    """Minimal :class:`logging.Handler` that records every emitted record.

    Used in preference to :class:`logging.handlers.MemoryHandler` because
    we want to inspect the raw :class:`logging.LogRecord` objects (level,
    name, message) rather than buffer them for forwarding to another
    handler.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def isolated_logger() -> Iterator[tuple[logging.Logger, _ListHandler]]:
    """Yield a fresh logger plus an attached list-collecting handler.

    The logger has propagation disabled so records do not bubble to the
    root logger (and therefore do not pollute pytest's own log capture
    or any other handlers configured for the session). The handler is
    detached on teardown so successive tests start clean.
    """
    # Use a unique logger name so parallel/iterated test runs cannot
    # accidentally inherit handlers from an earlier invocation.
    logger = logging.getLogger("test_visualization_server_ready_log.isolated")
    handler = _ListHandler()
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        yield logger, handler
    finally:
        logger.removeHandler(handler)
        logger.handlers.clear()


def test_emit_ready_log_emits_documented_line_exactly_once(
    isolated_logger: tuple[logging.Logger, _ListHandler],
) -> None:
    """One call to :func:`emit_ready_log` produces one WARNING record.

    The single emitted record must have the exact documented wording
    (Requirement 12.7) with the bound port interpolated, and it must be
    logged at ``logging.WARNING`` level so operators see the line
    under Python's default root threshold without bumping third-party
    library verbosity.
    """
    logger, handler = isolated_logger
    port = 7345  # The documented default visualization port.

    emit_ready_log(port=port, logger=logger)

    # Exactly one record is the central guarantee tested here: callers
    # invoke ``emit_ready_log`` once after a successful bind and the
    # documented line must appear exactly once in the operator's logs.
    assert len(handler.records) == 1

    record = handler.records[0]
    assert record.levelno == logging.WARNING

    # The wording is fixed verbatim by the design; reproduce it
    # literally so a regression in :data:`READY_LOG_TEMPLATE` is caught
    # here as well as a regression in the call site.
    expected_line = READY_LOG_TEMPLATE.format(port=port)
    assert expected_line == f"Visualization_Server ready at http://127.0.0.1:{port}"
    assert record.getMessage() == expected_line


def test_emit_ready_log_uses_module_logger_by_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without an explicit ``logger`` argument the module logger is used.

    The default code path (``emit_ready_log(port=...)``) emits the
    documented line through ``logging.getLogger(__name__)`` inside
    :mod:`project_knowledge_mcp.visualization_server`. Capturing from
    that logger with :func:`caplog.at_level` pins both the emitter
    identity and the wording.
    """
    port = 8080  # Arbitrary in-range port; wording is the invariant.

    with caplog.at_level(logging.WARNING, logger=_VIS_LOGGER_NAME):
        emit_ready_log(port=port)

    # Filter strictly to records from the visualization_server module
    # logger so unrelated WARNING-level chatter from other modules
    # cannot mask a missing or duplicated emission of the documented
    # line.
    records = [r for r in caplog.records if r.name == _VIS_LOGGER_NAME]
    assert len(records) == 1

    record = records[0]
    assert record.levelno == logging.WARNING
    expected_line = READY_LOG_TEMPLATE.format(port=port)
    assert record.getMessage() == expected_line
    assert expected_line == f"Visualization_Server ready at http://127.0.0.1:{port}"
