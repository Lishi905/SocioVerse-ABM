# `sugarscape` — Epstein-Axtell wealth dynamics

**Family:** market **SocioVerse2 study:** `abm_sugarscape` **Source ABM:** Epstein & Axtell
(1996) Sugarscape (G1).

First market-family task. Citizens forage a toroidal sugar grid; heterogeneous vision
and metabolism produce an emergent, highly unequal wealth distribution (the Gini
coefficient is the headline outcome).

## Layout

| file | role |
|------|------|
| `config.yaml` | grid size, N, capacity, regrowth, metabolism/vision/sugar ranges |
| `agents.py` | **P**: citizens with vision, metabolism, sugar |
| `env.py` | **E**: two-hill sugar grid + regrowth + harvest/metabolism/death (native, no mesa) |
| `rule_f.py` | **f (rule)**: move to richest visible empty cell (nearest tie) — parity baseline |
| `llm_f.py` | **f (llm)**: LLM picks from a numbered menu of reachable cells; lazy openai |
| `evaluate.py` | **B**: Gini / survivors / mean wealth |
| `model.py` | assemble + run loop + `registry.register` |
| `legacy/` | original Mesa `sugarscape_simple.py` + `sugarscape_LLM.py` — parity ground truth |

## Behaviour

Citizens that run out of sugar starve (no respawn), so population falls and wealth
concentrates — Gini rises over the run. Moves are discrete cell choices, so the
rule↔LLM consistency score uses strict action equality.

## Run

```bash
sv-abm run sugarscape --mode rule     # native foraging (no LLM needed)
sv-abm run sugarscape --mode llm      # LLM forager (needs openai + key)
```
