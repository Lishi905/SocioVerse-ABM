# `socioverse_abm/` — the shared kernel

One kernel, 12 tasks depend on it. Every task expresses **B = f(P, E)** through the same
contracts instead of each re-writing its own f / E / P. The package mirrors the provider
interfaces of [SocioVerse2](https://github.com/sii-research/SocioVerse2) (`socioverse.abc`),
so a task written here plugs into SocioVerse2 without extra glue.

```
socioverse_abm/
  behavior_engine/        # f  +  B
    action.py             # typed Observation / Action / Decision (the f contract)   [dep-free]
    hybrid.py             # rule+LLM routing for --mode hybrid                         [dep-free]
    llm_f.py              # LLM helpers: generate / parse / embedding (lazy client)    [openai]
    prompt.py             # shared prompt templates
  social_env_engine/      # E
    field.py              # ContinuousField2D (toroidal or bounded 2-D space)          [numpy]
  scenario_engine/        # task registry
    registry.py           # name -> TaskSpec; how the runner and SocioVerse2 discover tasks
  eval.py                 # consistency score (LLM-f vs rule-f agreement)
  runner.py               # CLI: `sv-abm list` / `sv-abm run <task> --mode rule|llm|hybrid`
```

Task-specific pieces (populations, environments, rule and LLM behavior functions,
metrics) live under `tasks/<family>/<task>/`.

## B = f(P, E) → SocioVerse2 interface mapping

| Lewin | where it lives | SocioVerse2 interface | key call |
|-------|----------------|-----------------------|----------|
| **E** | task `env.py` (+ `social_env_engine.ContinuousField2D`) | `EnvironmentProvider` | `reset / observe_batch / apply / advance / snapshot` |
| **P** | task `agents.py` | `PopulationProvider` | `build(cfg, seed) -> (env, population)` |
| **f** | task `rule_f.py` / `llm_f.py` + `behavior_engine.hybrid` | `DecisionModel` | `f(list[Observation]) -> list[Action]` |
| **B** | `behavior_engine.action` + task `evaluate.py` | `MetricCollector` | per-task metrics over the trajectory |

## Import contract

`import socioverse_abm` pulls in only the dependency-free core (`Action`/`Observation`/
`Decision`, `registry`, `eval`). `behavior_engine.llm_f` needs `openai` and is imported
lazily by the tasks' LLM behavior functions, so rule-mode runs and `sv-abm list` work in a
bare environment. The OpenAI client is created on first use from `OPENAI_API_KEY` and
`OPENAI_BASE_URL` (default: the official OpenAI endpoint). The model comes from the task's
`config.yaml` (`behavior.llm_model`), which each task's `run()` passes to its
`llm_f.configure()`; `SV_LLM_MODEL` only sets the fallback used when a task's `llm_f` is
called without `configure()`.

## Registering a task

```python
from socioverse_abm.scenario_engine import registry

registry.register(
    name="schelling",
    family="organization",              # flow | market | organization | diffusion
    source_abm="Schelling (1971)",
    socioverse2_study="abm_schelling",  # the matching SocioVerse2 study id
    build=build,                        # (cfg, seed) -> (env, population)
    rule_f=schelling_rule_f,            # list[Observation] -> list[Action]
    llm_f=schelling_llm_f,
    evaluate=segregation_metrics,
    run=run,                            # (mode, config) -> results (CLI entry)
)
```

`sv-abm list` prints every registered task with its SocioVerse2 study id.

## Install

```bash
pip install -e .                 # kernel + core deps
pip install -e ".[chicago]"      # + geopandas / libpysal / mesa / pandas for chicago_segregation
pip install -e ".[dev]"          # + pytest / pre-commit
```
