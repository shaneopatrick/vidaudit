"""Shared pytest fixtures.

Per CLAUDE.md §6, tests must never hit real VLM APIs or require real video
files — mock subprocess/SDK calls. Fixtures land here as components are built.
"""

from __future__ import annotations
