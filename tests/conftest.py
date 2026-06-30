"""Pytest configuration and shared fixtures.

This module registers the Hypothesis ``ci`` profile that all property-based
tests in this suite run under. The profile sets ``max_examples=100`` (the
minimum iteration count required by the design's property-based test
specification) and disables Hypothesis' per-test deadline so that tests which
set up small in-memory fixtures (SQLite stores, fake GitLab trees, etc.) are
not flaked by GC pauses or cold-start overhead.

The profile is both *registered* and *loaded* here so that simply running
``pytest`` from the repo root automatically picks it up; individual tests do
not need to opt in via ``HYPOTHESIS_PROFILE``.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

# Register the "ci" profile used by every property-based test in this suite.
# - max_examples=100 is the minimum iteration count mandated by the design
#   for every Hypothesis-driven property test (see tasks.md, "Conventions").
# - deadline=None disables per-example deadlines; many of the property tests
#   exercise SQLite write/commit cycles whose timing varies.
# - suppress_health_check relaxes the data-generation health checks that fire
#   when stateful tests build moderately complex fixtures.
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
    ],
)

# Activate the profile for the entire test session.
settings.load_profile("ci")
