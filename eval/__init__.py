"""Evaluation harness for vidaudit.

Builds a labeled clean-vs-mutated dataset from FineVideo chapters and runs
vidaudit against it to produce precision / recall / F1 versus the
text-comparison baseline. The eval is the project's primary deliverable and
runs in Colab on the canonical open-weight Qwen backend.
"""

from __future__ import annotations
