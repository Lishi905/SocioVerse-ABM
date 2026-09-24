> **Legacy snapshot, not maintained.** This folder keeps the original script verbatim for provenance and as the reference behaviour for the refactored task in [`../`](../). It is not imported by the task at runtime and is not part of the test suite. The script needs its own environment (`ndlib`, `networkx`, `numpy`, `matplotlib`).

# legacy — Hegselmann-Krause parity baseline

`HK_simple.py` is the original rule-based script built on ndlib's `HKModel`
(bounded-confidence opinion dynamics). The refactored `../rule_f.py` reimplements the
same mean-of-confidants update natively, per agent and per step, so the LLM behavior
function can be compared against it; this script is kept as the ground truth.
