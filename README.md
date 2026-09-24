<!-- ───────────────────────── Challenge banner ─────────────────────────
     Files: assets/challenge-banner.svg (dark card) + assets/challenge-banner-light.svg.
     Both are self-contained (text outlined to paths, no fonts/scripts/external refs),
     so GitHub renders them through <img>. <picture> follows the viewer's GitHub theme.
     Remove this block (and the News line) after the awards on 2026-11-08.            -->
<p align="center">
  <a href="https://socioverse.fudan-disc.com/challenge/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/challenge-banner.svg">
      <source media="(prefers-color-scheme: light)" srcset="assets/challenge-banner-light.svg">
      <img alt="SocioVerse Challenge 2026: AI4SS Challenge for Human-AI Collaboration and Social Governance. Three research tracks, $21,050 in prizes and API support, final submission October 31, 2026." src="assets/challenge-banner.svg" width="100%">
    </picture>
  </a>
</p>

<h1 align="center">SocioVerse-ABM</h1>

<p align="center">
  <b>Can LLM agents stand in for the hand-coded rules of classic agent-based models?</b><br>
  Classic ABMs with their rule-based agents replaced by LLM agents: a benchmark of whether
  LLM-driven agents reproduce the collective behavior the rules produce.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.24911"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.24911-b31b1b.svg"></a>
  <a href="https://github.com/sii-research/SocioVerse2"><img alt="SocioVerse2" src="https://img.shields.io/badge/SocioVerse2-runtime-6f42c1.svg"></a>
  <a href="https://socioverse.fudan-disc.com/"><img alt="Homepage" src="https://img.shields.io/badge/Homepage-socioverse.fudan--disc.com-e67e22.svg"></a>
  <a href="https://socioverse.fudan-disc.com/docs/"><img alt="Docs" src="https://img.shields.io/badge/Docs-user%20manual-2563eb.svg"></a>
  <a href="https://socioverse.fudan-disc.com/challenge/"><img alt="Challenge 2026" src="https://img.shields.io/badge/Challenge%202026-open-f97316.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white">
</p>

<p align="center">
  <a href="https://github.com/sii-research/SocioVerse2">SocioVerse2</a> ·
  <a href="https://socioverse.fudan-disc.com/">Homepage</a> ·
  <a href="https://socioverse.fudan-disc.com/docs/">User manual</a> ·
  <a href="https://arxiv.org/abs/2609.24911">Technical report</a> ·
  <a href="https://socioverse.fudan-disc.com/challenge/">Challenge 2026</a> ·
  <a href="README_zh.md">简体中文</a>
</p>

---

## News

- **2026-09** SocioVerse-ABM v0.2.0 is open-sourced as the ABM benchmark companion of [SocioVerse2](https://github.com/sii-research/SocioVerse2).
- **2026-09-21** The SocioVerse2 technical report is on arXiv: [SocioVerse2: A Longitudinal Dynamic Social Simulation Framework under a Human-AI Co-evolutionary Paradigm](https://arxiv.org/abs/2609.24911). This repository backs its Case Study 1 (*Reproducing Canonical ABMs with LLM Agents*) and Case Study 3 (*Chicago Segregation with Real Census Data*).
- **2026-09-15** [SocioVerse Challenge 2026](https://socioverse.fudan-disc.com/challenge/) (AI4SS Challenge for Human–AI Collaboration and Social Governance) opens registration: three tracks, $21,050 in prizes and API support, proposals due 2026-10-09, final submissions 2026-10-31.

## Overview

Every model here is written as Lewin's **B = f(P, E)**: behavior `B` is a function `f` of
the population `P` and its environment `E`, the same anchor as the SocioVerse2 runtime.
Each task implements `f` twice over **identical observations and typed actions**:

- **rule `f`**: the classic hand-coded rule of the source model (the reference);
- **LLM `f`**: an LLM that receives the same observation and must return the same kind of
  action (move or stay, cooperate or defect, spread or ignore, a heading, a price stance).

Because both behavior functions act in the same action space on the same `(P, E)`, their
runs can be compared step by step and scored against the rule-based reference
(`socioverse_abm.eval` provides the agreement score; each task's `evaluate.py` provides
its outcome metrics).

A `hybrid` mode splits each step's agents between the two with a routing policy, the
`route_to_llm` argument of `hybrid_decide` in
[`socioverse_abm/behavior_engine/hybrid.py`](socioverse_abm/behavior_engine/hybrid.py).
Its default policy is `always_rule`, and the shipped tasks keep that default, so
`sv-abm run <task> --mode hybrid` sends every agent to the rule, makes no LLM call, needs
no API key and gives the same result as `--mode rule`. To mix rule and LLM agents, pass a
policy where the task's `model.py` (`_behavior_fn`) calls `hybrid_decide`: `always_llm`,
or any function that takes an `Observation` and returns `True` for the agents that should
ask the LLM (for example by `agent_id`). Such a run then needs the same API key as
`--mode llm`.

## Tasks

Twelve tasks in four families. Ten form the benchmark suite of the SocioVerse2 technical
report (Case Study 1); `lux_marchesi` is an extra task outside that suite; and
`chicago_segregation` is the real-geography case (Case Study 3), which SocioVerse2 runs as
its `chicago_schelling` study.

| family | task (`sv-abm` name) | source model | SocioVerse2 study | in the report |
|---|---|---|---|---|
| flow | [`nasch`](tasks/flow/nasch/) | Nagel & Schreckenberg (1992) traffic cellular automaton | `abm_nasch` | Case Study 1 |
| flow | [`boids`](tasks/flow/boids/) | Reynolds (1987) flocking | `abm_boids` | Case Study 1 |
| flow | [`social_force`](tasks/flow/social_force/) | Helbing social force pedestrian evacuation | `abm_social_force` | Case Study 1 |
| market | [`sugarscape`](tasks/market/sugarscape/) | Epstein & Axtell (1996) Sugarscape | `abm_sugarscape` | Case Study 1 |
| market | [`minority_game`](tasks/market/minority_game/) | Challet & Zhang (1997) Minority Game | `abm_minority_game` | Case Study 1 |
| market | [`axelrod`](tasks/market/axelrod/) | Axelrod (1984) iterated Prisoner's Dilemma tournament | `abm_axelrod` | Case Study 1 |
| market | [`lux_marchesi`](tasks/market/lux_marchesi/) | Lux & Marchesi (1999) interacting-agent market | `abm_lux_marchesi` | extra, not in the suite |
| organization | [`schelling`](tasks/organization/schelling/) | Schelling (1971) residential segregation | `abm_schelling` | Case Study 1 |
| organization | [`civil_violence`](tasks/organization/civil_violence/) | Epstein (2002) civil violence | `abm_civil_violence` | Case Study 1 |
| organization | [`chicago_segregation`](tasks/organization/chicago_segregation/) | Schelling on 2010 Chicago census tracts (vendored Chicago segregation model @ a1964b8) | `chicago_schelling` | Case Study 3 |
| diffusion | [`sir`](tasks/diffusion/sir/) | SIR process read as rumor spreading | `abm_sir` | Case Study 1 |
| diffusion | [`hegselmann_krause`](tasks/diffusion/hegselmann_krause/) | Hegselmann & Krause (2002) bounded confidence | `abm_hegselmann_krause` | Case Study 1 |

> [!NOTE]
> The default `config.yaml` of each task is a small, fast **demo**, not the setting used in
> the technical report's tables (for example NaSch runs 50 vehicles instead of 200, the
> Minority Game 101 agents with memory 5 instead of 301 with memory 3, and Axelrod 20
> players instead of 64). To reproduce a reported setting, edit the task's `config.yaml`
> or pass your own file with `--config`. The report queries GPT-4o, DeepSeek-V3 and
> Qwen3-235B at temperature 0.7 with at most 256 output tokens, with 10 rule-based control
> runs and 3 runs per LLM per task.

## Quickstart

```bash
git clone https://github.com/Lishi905/SocioVerse-ABM.git
cd SocioVerse-ABM

conda create -n sv-abm python=3.11 -y && conda activate sv-abm   # or: python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

sv-abm list                        # 12 tasks, each with its SocioVerse2 study id
sv-abm run sir --mode rule         # rule-based run, no API key needed
pytest tasks/ socioverse_abm/ -q   # the chicago tests skip without the [chicago] extra
```

SocioVerse-ABM is used from a git checkout with an editable install, as above; it is not
published on PyPI. A regular install (`pip install .` or `pip install git+https://...`)
runs the eleven classic tasks but leaves out the Chicago data files, so
`chicago_segregation` then needs `SV_CHICAGO_LEGACY` pointing at a checkout's
`tasks/organization/chicago_segregation/legacy/` folder.

Every task runs in rule mode offline: `sv-abm run <task> --mode rule`. The Chicago task
needs the geo stack:

```bash
pip install -e ".[chicago]"        # geopandas, libpysal, mesa 3, pandas
sv-abm run chicago_segregation --mode rule
```

### LLM mode

The classic tasks call an OpenAI-compatible endpoint through `socioverse_abm.behavior_engine.llm_f`:

```bash
export OPENAI_API_KEY=...                       # required
export OPENAI_BASE_URL=https://api.openai.com/v1  # optional; any OpenAI-compatible endpoint
sv-abm run schelling --mode llm
```

The model is `behavior.llm_model` in the task's `config.yaml` (`gpt-4o` in the shipped
files). To query another model, edit that value or pass your own config file with
`--config`. `sv-abm run` always applies that value, so `SV_LLM_MODEL` has no effect on it;
`SV_LLM_MODEL` (default `gpt-4o`) only sets the starting model of a task's `llm_f` when
you call it from your own code without `llm_f.configure(model)`.

`chicago_segregation` reads `SV_LLM_API_KEY` and `SV_LLM_BASE_URL` (falling back to
`OPENAI_API_KEY` / `OPENAI_BASE_URL`) from the environment or from a gitignored `.env` at
the repository root, and also takes its model from `behavior.llm_model` (it falls back
to `SV_LLM_MODEL` only when that value is empty). LLM runs cost money; start with a small
`config.yaml`.

## Use with SocioVerse2

SocioVerse2's 11 `abm_*` studies and its `chicago_schelling` study load this repository
as a sibling checkout; it does not need to be pip-installed there.

```bash
git clone https://github.com/sii-research/SocioVerse2.git
git clone https://github.com/Lishi905/SocioVerse-ABM.git   # the folder name must be exactly SocioVerse-ABM
cd SocioVerse2
pip install -e ".[dev,chicago,workbench]"
pytest -q                                                # the ABM and Chicago tests now run instead of skipping
```

This is the install command of the SocioVerse2 README. `chicago` runs the Chicago tests;
`workbench` also runs the `hisim_roe` and `germany_auto_market` tests that skip without it.

- Keep the two folders side by side and named exactly `SocioVerse2` and `SocioVerse-ABM`.
- Leave `SV_ABM_ROOT` unset; the sibling layout is found automatically.
- If your shell sets a SOCKS proxy (`ALL_PROXY=socks5://...`), either unset it or
  `pip install "httpx[socks]"`: the vendored Chicago model constructs an OpenAI client even
  for offline runs, and httpx refuses SOCKS proxies without that extra.

## Repository layout

```
socioverse_abm/            the shared kernel (see socioverse_abm/README.md)
  behavior_engine/         typed Observation / Action, hybrid routing, LLM helpers
  social_env_engine/       ContinuousField2D for the continuous-space tasks
  scenario_engine/         the task registry used by sv-abm and by SocioVerse2
  eval.py                  rule-vs-LLM agreement score
  runner.py                the sv-abm CLI
tasks/<family>/<task>/     one folder per task:
  config.yaml              every hyperparameter
  agents.py                P: the population
  env.py                   E: observe / apply / advance / snapshot
  rule_f.py, llm_f.py      f: the rule and LLM behavior functions
  evaluate.py              B: outcome metrics
  model.py                 assembly, run loop and registry.register(...)
  test_*.py                offline tests (rule mode and a deterministic fake LLM)
  legacy/                  the original contributors' scripts, kept verbatim for provenance
```

The `legacy/` folders are historical snapshots: they are not maintained and are not
imported at runtime. Their scripts need their own environments, which differ by folder
(for example Mesa 2.x or 3.x, ndlib, the Axelrod library or the pre-1.0 `openai` API);
each legacy README lists them. The one exception is
`tasks/organization/chicago_segregation/legacy/`, which is imported at runtime: it is the
vendored Chicago model that the Chicago task and SocioVerse2's `chicago_schelling` run on.

## Data

The Chicago task ships 2010 and 2000 census-tract data for Chicago under
[`tasks/organization/chicago_segregation/legacy/`](tasks/organization/chicago_segregation/legacy/).
These data are **not** covered by the Apache-2.0 code license. They combine public-domain
U.S. Census Bureau data, City of Chicago Data Portal data, OpenStreetMap-derived counts
(ODbL) and HOLC redlining fields from the Mapping Inequality census crosswalk, which are
licensed CC BY-NC and therefore **for non-commercial use only**. See
[`DATA_LICENSE.md`](tasks/organization/chicago_segregation/legacy/DATA_LICENSE.md) and
[`DATA_README.md`](tasks/organization/chicago_segregation/legacy/processed_data/DATA_README.md).
The other tasks generate their populations synthetically and ship no data.

## Citation

If you use SocioVerse-ABM, please cite the SocioVerse2 technical report:

```bibtex
@misc{zhang2026socioverse2,
  title         = {SocioVerse2: A Longitudinal Dynamic Social Simulation Framework under a Human-AI Co-evolutionary Paradigm},
  author        = {Xinnong Zhang and Jiayu Lin and Jia Wang and Yixu Huang and Xinyi Mou and Yingqian Wu and Jingcong Liang and Shijun Lei and Jianing Shi and Guanying Li and Siyuan Wang and Hanjia Lyu and Zhenfei Yin and Yunlu Yin and Siming Chen and Yulan He and Jiebo Luo and Xuanjing Huang and Liyin Jin and Baohua Zhou and Hanqi Yan and Zhongyu Wei},
  year          = {2026},
  eprint        = {2609.24911},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.24911}
}
```

## License

The code is released under the [Apache License 2.0](LICENSE); see [`NOTICE`](NOTICE).
Three legacy files adapted from the Mesa examples keep their Apache-2.0 attribution to the
Core Mesa Team and contributors. The vendored Chicago data carry their own terms (see
[Data](#data)).

## Acknowledgements

The models and the shared kernel in this repository were originally implemented by:

- **Xinnong Zhang** ([@Lishi905](https://github.com/Lishi905)): SIR, Hegselmann-Krause, Sugarscape, the shared kernel and the Chicago segregation model.
- **Shijun Lei** ([@ShijunLei-cn](https://github.com/ShijunLei-cn)): Schelling and Civil Violence.
- **Jianing Shi** ([@Brishian427](https://github.com/Brishian427)): Boids, NaSch and Social Force.
- **Chenyu Li** ([@if111111111111111111111](https://github.com/if111111111111111111111)): Axelrod and the Minority Game.
- **Zijian Ling** ([@Georgelingzj](https://github.com/Georgelingzj)): Lux-Marchesi.
- **Jia Wang** ([@JiaWANG-TJ](https://github.com/JiaWANG-TJ)): the consolidated Minority Game simulation.

The original scripts are preserved in the `legacy/` folders. The Schelling and Civil Violence
legacy models build on the examples of [Mesa](https://github.com/projectmesa/mesa)
(Apache-2.0, Core Mesa Team and contributors); the Axelrod legacy code uses the
[Axelrod library](https://github.com/Axelrod-Python/Axelrod).

Questions and collaboration: contact@socioverse.fudan-disc.com.
