"""Skeleton sanity check: the package imports and exposes a version."""

from __future__ import annotations

import vidaudit


def test_version_is_exposed() -> None:
    assert vidaudit.__version__ == "0.1.0"
