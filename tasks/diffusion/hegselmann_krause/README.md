# `hegselmann_krause` — bounded-confidence opinion dynamics

**Family:** diffusion **SocioVerse2 study:** `abm_hegselmann_krause` **Source ABM:** Hegselmann
& Krause (2002).

Its environment is unusual: not a network or space but **opinion proximity** — each
agent listens only to others within its confidence bound `epsilon`.

## Layout

| file | role |
|------|------|
| `config.yaml` | N, epsilon (confidence bound), steps, seed |
| `agents.py` | **P**: N agents with an opinion in [0,1] (fully mixed) |
| `env.py` | **E**: shows each agent all opinions; it averages those within epsilon |
| `rule_f.py` | **f (rule)**: HK mean-of-confidants update — parity baseline (native, no ndlib) |
| `llm_f.py` | **f (llm)**: LLM updates its opinion from its confidants; lazy openai |
| `evaluate.py` | **B**: cluster count (consensus/polarization/fragmentation) + `opinion_equals(tol)` |
| `model.py` | assemble + synchronous loop (stops on convergence) + `registry.register` |
| `legacy/HK_simple.py` | original ndlib `HKModel` script — parity ground truth |

## Behaviour

Small `epsilon` → fragmentation (many clusters); large `epsilon` → consensus (1
cluster). Sweep `epsilon` in `config.yaml` to reproduce the classic transition. Rule
and LLM opinions are continuous, so consistency uses `evaluate.opinion_equals(tol)`.

## Run

```bash
sv-abm run hegselmann_krause --mode rule    # native HK (no LLM needed)
sv-abm run hegselmann_krause --mode llm     # LLM opinion update (needs openai + key)
```
