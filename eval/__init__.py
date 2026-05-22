"""Evaluation harness for vidaudit (PLAN.md component 9, DD-13).

Builds a labeled clean-vs-mutated dataset from FineVideo chapters and runs
vidaudit against it to produce precision / recall / F1 versus the
text-comparison baseline. The eval is the project's primary deliverable
(DD-15) and runs in Colab on the canonical Qwen backend (DD-16).
"""

from __future__ import annotations
