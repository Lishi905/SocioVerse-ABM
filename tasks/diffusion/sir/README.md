# `sir` — rumor spreading (SIR analogy)

**Family:** diffusion **SocioVerse2 study:** `abm_sir` **Source ABM:** SIR epidemic process
read as rumor diffusion (Ignorant→Spreader→Stifler ≡ Susceptible→Infected→Removed).

This is the **template** task: it shows the standard layout every task follows and how a
monolithic legacy script becomes the kernel's per-step `B = f(P, E)` contract.

## Layout (the standard contract)

| file | role | maps to SocioVerse2 interface |
|------|------|------------------|
| `config.yaml` | every hyperparameter (N, topology, seed, steps, p/q/γ, mode) | — |
| `agents.py` | **P**: network nodes + interaction structure (graph) | `PopulationProvider` |
| `env.py` | **E**: per-step observe / apply / snapshot over the graph | `EnvironmentProvider` |
| `rule_f.py` | **f (rule)**: native per-agent SIR contact process — the parity baseline | `DecisionModel` |
| `llm_f.py` | **f (llm)**: same contract, decisions from an LLM (lazy `openai`) | `DecisionModel` |
| `evaluate.py` | **B**: final_reach / peak / time-to-peak | `MetricCollector` |
| `model.py` | assembles P+E+f, runs the loop, **registers the task** | engine loop |
| `legacy/SIR_simple.py` | the original ndlib+matplotlib script, verbatim, as parity ground truth | — |

## Why rule_f is reimplemented (not ndlib)

The legacy script runs ndlib's `iteration_bunch` — a whole trajectory in one call.
The project's question ("does an LLM agent reproduce the rule agent?") needs decisions
compared **agent-by-agent, step-by-step**, so `rule_f.py` reimplements the same contact
process natively: an Ignorant with `k` spreading neighbours starts spreading with prob
`1 - (1 - β_eff)^k` (β_eff = p·(1−q)); a Spreader stops with prob γ. The ndlib original
stays in `legacy/` and is the baseline the consistency score checks against.

## Run

```bash
sv-abm run sir --mode rule          # native rule baseline (no LLM needed)
sv-abm run sir --mode llm           # LLM behavior function (needs openai + key)
sv-abm run sir --mode hybrid        # per-agent routing; the default policy sends all to the rule
# or: python -m tasks.diffusion.sir.model
```

Edit `config.yaml` for N / topology (ER|BA|WS) / seed / β,γ. Consistency between the
rule and LLM paths is exercised in the test suite with a deterministic fake LLM.
