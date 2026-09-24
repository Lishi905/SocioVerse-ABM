# chicago_segregation — Schelling on real Chicago 2010 geography

**Family:** organization **SocioVerse2 study:** `chicago_schelling` **Source ABM:** Schelling
segregation on real census geography.

The real-world segregation case (Case Study 3 in the SocioVerse2 technical report):
multi-group residential segregation in the City of Chicago (781 census tracts, Queen
contiguity), grounded in `B = f(P, E)`. It is the real-geography member of the Schelling /
*organization* family: a classical Schelling threshold runs as a baseline on this very
geography, and the LLM behaviour function is compared against it in a multi-group,
preference-asymmetric setting that single-statistic rules cannot express.

## Wrap, not reimplement

Unlike the 11 from-scratch tasks, this task **wraps** the validated Chicago model
(`SegregationModel`, ~1.5k lines of calibrated dynamics) vendored as a pinned snapshot in
[`legacy/`](legacy/) (see [`legacy/PROVENANCE.md`](legacy/PROVENANCE.md), which lists the few
changes made for the public release). Re-deriving its
published numbers from scratch would be a research-grade validation task; wrapping the
original **preserves the effect by construction** — a parity test proves the wrapped path
reproduces legacy `model.step()` bit-for-bit.

| B=f(P,E) | here |
|---|---|
| **E** environment | `env.py` → `ChicagoTractEnv` wraps `legacy` `TractEnvironment` (tract graph, dynamic per-tract race counts) |
| **P** population | `agents.py` projects the model's archetype `HouseholdAgent`s (240 archetypes on real tracts) |
| **f** rule | `rule_f.py` — classical similarity threshold τ on the Queen graph (the rule-based baseline) |
| **f** llm | `llm_f.py` — the validated two-phase LLM (`_assess_archetype_satisfaction` + `_evaluate_move_candidates`) |
| **B** metrics | `evaluate.py` — Dissimilarity / Isolation / Exposure / R²_recovery over the trajectory |

Both `rule` and `llm` decide *intent* differently but execute through the **same** validated
machinery (`_apply_move_budget` + `shuffle_do`) in `env.apply` — i.e. the same per-step move
budget, a like-for-like comparison. `_engine.py` owns the one shared model instance and the
import seam; it mirrors SocioVerse2's `chicago_schelling` adapter, which drives this same
vendored copy when SocioVerse-ABM is checked out next to SocioVerse2.

## Run

```bash
pip install -e ".[chicago]"                          # geopandas / libpysal / mesa / pandas
sv-abm run chicago_segregation --mode rule            # classical Schelling baseline (offline, no key)
sv-abm run chicago_segregation --mode llm             # validated LLM f (needs SV_LLM_API_KEY or OPENAI_API_KEY)

# Full-city recovery run: edit config.yaml → scale: full, init_mode: perturbed_n, steps: 15
```

Runs are reproducible: the same `seed` gives the same trajectory in every process. This
relies on the one behavioural change made to the vendored model for the public release:
`TractEnvironment.get_neighbors` (`legacy/src/environment.py`) returns neighbour ids sorted
instead of in Python set order, which varied with `PYTHONHASHSEED`. The neighbours and the
dynamics are unchanged, but a rerun does not match earlier runs of the unmodified snapshot
step by step, because those depended on the process's hash seed. See
[`legacy/PROVENANCE.md`](legacy/PROVENANCE.md).

LLM mode reads `SV_LLM_API_KEY` (or `OPENAI_API_KEY`) and `SV_LLM_BASE_URL` (or
`OPENAI_BASE_URL`; default: the official OpenAI endpoint) from the environment or from a
gitignored `.env` at the repository root. The model is `behavior.llm_model` in
[`config.yaml`](config.yaml) (default `gpt-4o`); edit it or pass `--config` with your own
file. `SV_LLM_MODEL` is used only when `llm_model` is left empty.

Config knobs live in [`config.yaml`](config.yaml): `scale` (small≈25 / middle≈218 / full=781
tracts), `init_mode` (`perturbed_n` = the 2010-perturbation recovery test; `census_2000`
starts from the 2000 census), `steps`, `homophily` (τ, rule mode), `move_budget_pct`. Any
extra `model:` keys pass straight through to `SegregationModel` (defaults match the
published config).

## Supplementary ablations (E0-E4)

`ablations.py` and `rule_variants.py` implement the reference conditions and ablations,
all switched from `config.yaml` without editing the vendored code (E1 needs no switch):

| id | switch | effect |
|---|---|---|
| E0 | `behavior.rule_variant: classical \| classical_random \| calibrated_multigroup \| logit` | rule baselines (rule mode): the classical threshold with best-improvement (default) or random destination choice, and the calibrated multi-group and logit rules |
| E1 | no switch: rerun the unablated `llm` condition with several `seed` values | seed-to-seed confidence intervals for the main LLM result |
| E2 | `ablations.race_blind: true` | strip racial information from all LLM-visible text |
| E3 | `model:` constraints-off preset (see `config.yaml`) | remove the calibrated mobility constraints |
| E4 | `ablations.shuffle_tract_attributes: <seed>` | permute non-demographic tract profiles |

## Tests / parity

```bash
pytest tasks/organization/chicago_segregation/ -q
```

`test_chicago.py` proves: (1) the task is discoverable without the geo stack; (2) **parity** —
the env-driven wrapper matches legacy `model.step()` on every segregation index under the
`DeterministicLLMClient`; (3) `rule` mode runs offline and moves the indices. The deterministic
client (`_fake_llm.py`) also powers no-token `llm`-path runs. `test_ablations.py` covers the
E0-E4 switches.

If a SOCKS proxy is set in your environment (`ALL_PROXY=socks5://...`), either unset it or
`pip install "httpx[socks]"`: the vendored model constructs an OpenAI client even in rule mode.

## Data

The vendored census geography and configuration are documented in
[`legacy/processed_data/DATA_README.md`](legacy/processed_data/DATA_README.md). The data are
**not** covered by the repository's Apache-2.0 code license; see
[`legacy/DATA_LICENSE.md`](legacy/DATA_LICENSE.md) (the HOLC-derived fields are
non-commercial only).
