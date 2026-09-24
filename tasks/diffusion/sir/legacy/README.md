> **Legacy snapshot, not maintained.** This folder keeps the original script verbatim for provenance and as the reference behaviour for the refactored task in [`../`](../). It is not imported by the task at runtime and is not part of the test suite. The script needs its own environment (`ndlib`, `networkx`, `matplotlib`).

# legacy — parity baseline (do not edit)

`SIR_simple.py` is the **original, unmodified** rule-based script (ndlib `SIRModel`
+ matplotlib) as it was before the refactor. It is kept verbatim as the parity ground
truth: the refactored `../rule_f.py` reimplements the same contact process per-agent,
and the LLM path is ultimately checked against this behavior.

- Needs `ndlib`, `networkx` and `matplotlib` (`pip install ndlib networkx matplotlib`).
- Writes result figures to `res_figs_simple/`; those PNG artifacts are not tracked
  (regenerate them by running the script).
