# Vendored: Chicago Segregation ABM (legacy `SegregationModel`)

This directory is a copy of the validated Chicago residential-segregation model
(vendored Chicago segregation model @ a1964b8), included in SocioVerse-ABM so the
`chicago_segregation` task is self-contained.

## Source
- Upstream: the authors' Chicago segregation model (not published separately)
- Commit: `a1964b8160726fa8f335dcc6ed94ce84e58429bd`
- Date:   2026-04-23
- Subject: *Race-aware mobility controls to curb D_asian_white overshoot*

## What was copied (and what was NOT)
Copied verbatim, except for the release changes listed below:
- `src/` — `model.py` (`SegregationModel`), `agents.py`, `environment.py`, `llm_client.py`, `metrics.py`
- `config/` — `archetypes.csv` (240 archetypes), `archetype_tract_mapping.csv` (11,454 rows), `persona_parameters.json`
- `processed_data/` — `chicago_tracts.geojson` (2010), `chicago_tracts_2000.geojson` (for `init_mode="census_2000"`), `DATA_README.md`
- `run_prototype.py` — kept for `select_prototype_tracts` (small/middle subsets)

NOT copied (out of scope for the benchmark task): the policy sandbox, the data-processing
scripts, raw inputs, run outputs, tests and internal docs.

## Changes made for the public release
The snapshot is otherwise unmodified. For the public release:
- `src/environment.py` (the only change to the model's behaviour, kept on purpose):
  `TractEnvironment.get_neighbors` returns the neighbouring tract ids sorted
  (`sorted(visited)`) instead of in Python set order (`list(visited)`). The ids are
  strings, so set order changed with `PYTHONHASHSEED` from one process to the next, and
  the seeded `random.sample` / `shuffle` calls that consume the list (in `src/model.py`
  and in the task's rule functions) then produced different runs for the same seed. The
  set of neighbours and the dynamics are unchanged; runs with the same seed are now
  identical across processes. Trade-off: a rerun does not reproduce earlier runs of the
  unmodified snapshot step by step. Those runs depended on the process's hash seed, so
  they could not be reproduced exactly either.
- `run_prototype.py`: the default `SV_LLM_BASE_URL` now points at the official OpenAI
  endpoint (`https://api.openai.com/v1`) instead of a third-party relay.
- `processed_data/DATA_README.md`: trimmed to the files actually vendored here, and
  `chicago_tracts_2000.geojson` documented.
- `DATA_LICENSE.md`: added (data terms; the data are not covered by the Apache-2.0 code license).

## Why paths resolve with zero patching
`src/model.py` computes `PROJECT_ROOT = Path(__file__).resolve().parent.parent`, then
`CONFIG_DIR = PROJECT_ROOT/config` and `DATA_DIR = PROJECT_ROOT/processed_data`. Because
`src/`, `config/`, and `processed_data/` are vendored preserving that sibling layout, the
model reads *this* directory's data automatically — no monkey-patch needed for the base run.

## Rules
- **Treat as a pinned snapshot.** Do not hand-edit the dynamics; to pull a new upstream
  version, re-copy it, bump the commit above and re-apply the release changes listed above.
- The `chicago_segregation` task wraps this code (does not fork its logic); the wrapper lives one
  level up (`../model.py`, `../env.py`, `../llm_f.py`, `../rule_f.py`, `../evaluate.py`).
- The same import surface (`src.model`, module-global `CONFIG_DIR`/`DATA_DIR`, the private
  `_assess_… / _evaluate_… / _apply_move_budget / _compute_metrics` phases) is what
  SocioVerse2's `studies/chicago_schelling` adapter also drives, so SocioVerse2 uses this
  vendored copy directly when SocioVerse-ABM is checked out next to it.
