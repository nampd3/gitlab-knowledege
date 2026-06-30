# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 12: For all sequences of refresh requests (full or single-project, from MCP tools or the scheduler), at most one Ingestion_Job is in the running state at any moment; every refresh request issued while another job is running SHALL be rejected with the documented "Ingestion_Job already in progress" message and SHALL leave the coordinator state and the Knowledge_Store unchanged; every refresh request issued while the coordinator is idle SHALL be accepted.
"""Property test for the at-most-one ``Ingestion_Job`` single-flight contract.

**Validates Requirement 8.6** (Property 12 in the design).

This test exercises the ``IngestionCoordinator``'s ``idle → running``
CAS at the unit level — i.e. directly against
:meth:`IngestionCoordinator.try_start`, without driving the full
:meth:`IngestionCoordinator.start_full_refresh` /
:meth:`IngestionCoordinator.start_single_project_refresh` plumbing.
Tasks 8.2 and 8.3 (and the property tests for Properties 9, 10, 11,
29, 30) cover the higher-level procedures that wrap the CAS. What
Property 12 asserts is the underlying single-flight invariant that
those procedures rely on:

1. **At most one job is in the ``running`` state at any moment.** The
   coordinator's ``current_state().state`` is the system-of-record;
   the test maintains a parallel expected-state model that mirrors
   the operation just performed and asserts agreement with the
   coordinator after every step.
2. **Every refresh request issued while another job is running is
   rejected** with :class:`IngestionInProgressError` carrying the
   canonical message ``"Ingestion_Job is already in progress"``
   (Requirement 8.6) and leaves the coordinator state unchanged. The
   test snapshots the coordinator's ``CoordinatorStatus`` immediately
   before each rejected start and asserts equality immediately after,
   so any state mutation through the rejection path would surface as
   an inequality on the in-flight ``snapshot_id`` / ``trigger`` /
   ``started_at`` triple.
3. **Every refresh request issued while the coordinator is idle is
   accepted.** When the model says ``idle``, the test calls
   ``try_start`` and asserts that no exception is raised, that the
   handle's metadata reflects the request, and that
   ``current_state().state`` is now ``RUNNING``.

The property's "Knowledge_Store unchanged" clause is satisfied by
construction at this unit level: ``try_start`` does not hold a
reference to a ``KnowledgeStore`` and cannot, therefore, modify it.
The store-level proof for the higher-level procedures lives in
property tests 9/10/11, which exercise the writer/reader interfaces
directly.

The state-space is explored with Hypothesis's
:class:`RuleBasedStateMachine`, which generates randomly interleaved
traces of ``try_start_full``, ``try_start_single_project``,
``complete``, and ``abort`` rules. The ``@invariant`` decorators run
after every rule, so the at-most-one-running guarantee and the
model-agreement guarantee are checked at every machine position
Hypothesis visits, not only at the end of each trace.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)

from project_knowledge_mcp.errors import IngestionInProgressError
from project_knowledge_mcp.ingestion_coordinator import (
    CoordinatorState,
    CoordinatorStatus,
    IngestionCoordinator,
    JobHandle,
)
from project_knowledge_mcp.models import SnapshotTrigger

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

# Hypothesis-driven rules cover both refresh "sources" enumerated in the
# property text by exercising both triggers the coordinator's MCP-tool
# path and scheduler path can pass into ``try_start``: ``FULL`` for the
# scheduler tick and the ``refresh_all`` MCP tool, and
# ``SINGLE_PROJECT`` for the ``refresh_project`` MCP tool. The
# coordinator's CAS is trigger-agnostic (it serializes regardless of
# which trigger is presented), so the property must hold uniformly for
# either choice.
_TRIGGERS_UNDER_TEST: tuple[SnapshotTrigger, ...] = (
    SnapshotTrigger.FULL,
    SnapshotTrigger.SINGLE_PROJECT,
)

# A small pool of distinct ``snapshot_id``s keeps the model interesting
# (so the rejection-path's "state unchanged" check is meaningful) while
# bounding the search space for fast example generation. The values
# themselves are arbitrary; they only need to be distinct integers.
_SNAPSHOT_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

# Canonical message wording fixed by Requirement 8.6 / Property 12.
_CANONICAL_REJECTION_MESSAGE: str = "Ingestion_Job is already in progress"


# ---------------------------------------------------------------------------
# Stateful machine
# ---------------------------------------------------------------------------


class IngestionSingleFlightMachine(RuleBasedStateMachine):
    """RuleBasedStateMachine modelling the single-flight invariant.

    The machine maintains a tiny parallel model:

    * ``self.expected_state`` — the model's view of the coordinator
      state (``IDLE`` or ``RUNNING``). The model transitions on each
      successfully-applied rule and is never mutated on rejection
      (because the property requires the real coordinator to behave
      that way too).
    * ``self.handle`` — the currently-active :class:`JobHandle` when
      the model says ``RUNNING``, or ``None`` when ``IDLE``. The
      ``complete`` and ``abort`` rules have a ``RUNNING`` precondition
      and consume this handle.

    Each rule applies one operation to ``self.coordinator`` and to
    ``self.expected_state`` together, so the post-rule invariants can
    cross-check that the two evolved in step.
    """

    def __init__(self) -> None:
        super().__init__()
        # A fresh coordinator per example. The state machine is fully
        # in-memory; no Knowledge_Store is wired in because
        # ``try_start`` does not interact with one (the property's
        # "Knowledge_Store unchanged" clause therefore holds
        # vacuously at this layer, see module docstring).
        self.coordinator: IngestionCoordinator = IngestionCoordinator()
        # The model's view of the coordinator state.
        self.expected_state: CoordinatorState = CoordinatorState.IDLE
        # Handle on the running job when ``expected_state == RUNNING``.
        self.handle: JobHandle | None = None
        # A monotonically-increasing fallback id; the rules combine it
        # with a Hypothesis-drawn pool id so the test exercises both
        # repeated and novel snapshot ids without ever colliding on a
        # value that is currently in flight.
        self._next_snapshot_id: int = max(_SNAPSHOT_IDS) + 1

    # -- Helpers ---------------------------------------------------------

    def _attempt_try_start(
        self,
        trigger: SnapshotTrigger,
        snapshot_id: int,
        started_at: datetime,
    ) -> None:
        """Drive a single ``try_start`` and reconcile model + assertions.

        When the model says ``IDLE`` the call must succeed: the
        returned :class:`JobHandle` is captured into ``self.handle``,
        the model transitions to ``RUNNING``, and the coordinator's
        observable state is asserted to match. When the model says
        ``RUNNING`` the call must raise :class:`IngestionInProgressError`
        with the canonical message; the coordinator's state immediately
        before and after the rejected call must be byte-for-byte
        identical so the "leave the coordinator state ... unchanged"
        clause is satisfied.
        """

        if self.expected_state is CoordinatorState.IDLE:
            # Idle-state requests SHALL be accepted (Property 12).
            handle = self.coordinator.try_start(
                trigger=trigger,
                snapshot_id=snapshot_id,
                started_at=started_at,
            )
            assert handle.is_active is True
            assert handle.snapshot_id == snapshot_id
            assert handle.trigger is trigger
            assert handle.started_at == started_at
            self.handle = handle
            self.expected_state = CoordinatorState.RUNNING

            # Coordinator must now report ``RUNNING`` with the
            # supplied metadata.
            status = self.coordinator.current_state()
            assert status.state is CoordinatorState.RUNNING
            assert status.snapshot_id == snapshot_id
            assert status.trigger is trigger
            assert status.started_at == started_at
            return

        # ``expected_state is RUNNING``: the request must be rejected
        # with the canonical wording, and the coordinator's
        # CoordinatorStatus must be unchanged across the call.
        before: CoordinatorStatus = self.coordinator.current_state()
        with pytest.raises(IngestionInProgressError) as excinfo:
            self.coordinator.try_start(
                trigger=trigger,
                snapshot_id=snapshot_id,
                started_at=started_at,
            )
        # Canonical message is fixed by Requirement 8.6 so the MCP
        # tool result and the scheduler log line read identically.
        assert excinfo.value.message == _CANONICAL_REJECTION_MESSAGE
        assert str(excinfo.value) == _CANONICAL_REJECTION_MESSAGE

        # State is unchanged: same triple (snapshot_id, trigger,
        # started_at), same overall ``state`` field. ``CoordinatorStatus``
        # is a frozen dataclass, so equality compares all fields.
        after: CoordinatorStatus = self.coordinator.current_state()
        assert after == before, (
            "rejected start mutated coordinator state: "
            f"before={before}, after={after}"
        )

    # -- Rules -----------------------------------------------------------

    @rule(
        snapshot_pool_id=st.sampled_from(_SNAPSHOT_IDS),
        started_at=st.datetimes(
            min_value=datetime(2024, 1, 1, 0, 0, 0),
            max_value=datetime(2030, 12, 31, 23, 59, 59),
            timezones=st.just(UTC),
        ),
    )
    def try_start_full(
        self,
        snapshot_pool_id: int,
        started_at: datetime,
    ) -> None:
        """Issue a refresh request with ``trigger == FULL``.

        Models the scheduler tick / ``refresh_all`` MCP tool path. The
        coordinator's CAS is trigger-agnostic, so the same
        accept-when-idle / reject-when-running invariant must hold.
        """

        self._attempt_try_start(
            trigger=SnapshotTrigger.FULL,
            snapshot_id=snapshot_pool_id,
            started_at=started_at,
        )

    @rule(
        snapshot_pool_id=st.sampled_from(_SNAPSHOT_IDS),
        started_at=st.datetimes(
            min_value=datetime(2024, 1, 1, 0, 0, 0),
            max_value=datetime(2030, 12, 31, 23, 59, 59),
            timezones=st.just(UTC),
        ),
    )
    def try_start_single_project(
        self,
        snapshot_pool_id: int,
        started_at: datetime,
    ) -> None:
        """Issue a refresh request with ``trigger == SINGLE_PROJECT``.

        Models the ``refresh_project`` MCP tool path. Together with
        :meth:`try_start_full` this rule covers both refresh "sources"
        (MCP tools and the scheduler) named in the property text.
        """

        self._attempt_try_start(
            trigger=SnapshotTrigger.SINGLE_PROJECT,
            snapshot_id=snapshot_pool_id,
            started_at=started_at,
        )

    @precondition(lambda self: self.expected_state is CoordinatorState.RUNNING)
    @rule()
    def complete_running_job(self) -> None:
        """End the in-flight job via ``JobHandle.complete``.

        The handle's ``complete`` releases the running slot under the
        coordinator's lock, so the very next ``try_start`` rule must
        succeed. The model transitions back to ``IDLE``.
        """

        assert self.handle is not None
        self.handle.complete()
        assert self.handle.is_active is False
        self.handle = None
        self.expected_state = CoordinatorState.IDLE

        # Coordinator must now report ``IDLE`` with all in-flight
        # fields cleared.
        status = self.coordinator.current_state()
        assert status.state is CoordinatorState.IDLE
        assert status.snapshot_id is None
        assert status.trigger is None
        assert status.started_at is None

    @precondition(lambda self: self.expected_state is CoordinatorState.RUNNING)
    @rule()
    def abort_running_job(self) -> None:
        """End the in-flight job via ``JobHandle.abort``.

        Symmetric to :meth:`complete_running_job`. The coordinator's
        single-flight contract treats ``complete`` and ``abort``
        identically with respect to slot release, so the same
        post-conditions apply.
        """

        assert self.handle is not None
        self.handle.abort()
        assert self.handle.is_active is False
        self.handle = None
        self.expected_state = CoordinatorState.IDLE

        status = self.coordinator.current_state()
        assert status.state is CoordinatorState.IDLE
        assert status.snapshot_id is None
        assert status.trigger is None
        assert status.started_at is None

    # -- Invariants ------------------------------------------------------

    @invariant()
    def at_most_one_running_job(self) -> None:
        """The headline Property 12 guarantee.

        Hypothesis runs every ``@invariant`` after every applied rule
        (and once before the first rule), so this check is performed
        at every machine position visited during the trace — including
        immediately after a rejected ``try_start`` — not only at the
        end. Because the coordinator exposes a single
        ``CoordinatorStatus`` whose ``state`` field is one of two
        values, "at most one running job" is equivalent to "the
        observable state is ``IDLE`` or ``RUNNING``" combined with
        agreement between the model and the coordinator.
        """

        status = self.coordinator.current_state()
        # The state field is a closed set of size 2; agreement with
        # the model is what proves the headline invariant.
        assert status.state in (CoordinatorState.IDLE, CoordinatorState.RUNNING)
        assert status.state is self.expected_state

    @invariant()
    def is_idle_agrees_with_state(self) -> None:
        """``is_idle()`` must agree with ``current_state().state``.

        The scheduler's tick logic and the MCP tool dispatcher both
        read ``is_idle()`` to decide whether to even attempt
        ``try_start`` (or, for the scheduler, whether to log the
        rejection line). They MUST therefore see the same answer the
        CAS itself would produce; otherwise the property's "rejected
        request is unchanged" clause could be silently violated by an
        is_idle/try_start race that returns inconsistent answers.
        """

        is_idle = self.coordinator.is_idle()
        status_state = self.coordinator.current_state().state
        assert is_idle is (status_state is CoordinatorState.IDLE)

    @invariant()
    def running_status_carries_in_flight_metadata(self) -> None:
        """When running, ``snapshot_id`` / ``trigger`` / ``started_at`` are present.

        The property text talks about the running state being a
        well-defined state — Requirement 8.6 specifies the retained
        ``snapshot_id``, ``trigger``, and ``started_at`` triple while
        running. Verifying it here as an invariant guards against a
        partial CAS that flips ``state`` to ``RUNNING`` without
        populating the triple (in which case "rejected request leaves
        coordinator state unchanged" would be observable but
        meaningless).
        """

        status = self.coordinator.current_state()
        if status.state is CoordinatorState.RUNNING:
            assert status.snapshot_id is not None
            assert status.trigger is not None
            assert status.started_at is not None
            # Cross-check the model's recorded handle.
            assert self.handle is not None
            assert status.snapshot_id == self.handle.snapshot_id
            assert status.trigger is self.handle.trigger
            assert status.started_at == self.handle.started_at
        else:
            # IDLE: the triple is fully cleared and the model has no
            # active handle.
            assert status.snapshot_id is None
            assert status.trigger is None
            assert status.started_at is None
            assert self.handle is None


# Bind the Hypothesis settings (max_examples=100 per the design's
# property-test convention) onto the auto-generated TestCase. The
# stateful runner uses ``TestCase.settings``, not the ``@settings``
# decorator on the class itself; ``stateful_step_count`` is set wide
# enough that each example exercises a meaningful sequence of
# operations and explores both legs of the state machine multiple
# times within a single trace.
IngestionSingleFlightMachine.TestCase.settings = settings(
    max_examples=100,
    deadline=None,
    stateful_step_count=50,
)


# Pytest collects the class produced by the state machine as the test
# entry point. The ``property`` marker matches the convention shared
# by every other property test in this suite and is registered in
# ``pyproject.toml``.
TestPropertyTwelveSingleFlight = IngestionSingleFlightMachine.TestCase
TestPropertyTwelveSingleFlight.pytestmark = [pytest.mark.property]
