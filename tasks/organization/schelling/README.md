# `schelling` — residential segregation

**Family:** organization **SocioVerse2 study:** `abm_schelling` **Source ABM:** Schelling (1971).

Two groups on a grid; a resident relocates when too few neighbours share its group.
Mild individual preference produces strong global segregation.

## The four variants, as parameters

The ABM study's LCM/LBF/OCM/TBF matrix is exposed as two config keys (they only affect the
LLM path; rule mode is the classic deterministic Schelling):

| key | values | meaning |
|-----|--------|---------|
| `context` | `ocm` / `lcm` | how **E** is described: raw neighbour counts vs a short narrative |
| `llm_behavior` | `tbf` / `lbf` | how **f** decides: apply the stated rule vs free reasoning |

So `ocm+tbf`, `ocm+lbf`, `lcm+tbf`, `lcm+lbf` are the four special cases of one task.

## Layout

| file | role |
|------|------|
| `config.yaml` | grid, density, homophily threshold, + `context`/`llm_behavior` |
| `agents.py` | **P**: two groups on a grid with empty cells |
| `env.py` | **E**: Moore-neighbourhood grid; renders ocm vs lcm context |
| `rule_f.py` | **f (rule)**: move when same-group fraction < threshold — parity baseline |
| `llm_f.py` | **f (llm)**: tbf/lbf prompt over the ocm/lcm context; lazy openai |
| `evaluate.py` | **B**: segregation index + happy fraction |
| `model.py` | assemble + run loop + `registry.register` |
| `legacy/` | original Mesa Schelling + the `extends/` variant scripts (ocm/lcm × tbf/lbf) |

## Run

```bash
sv-abm run schelling --mode rule     # classic segregation (no LLM needed)
sv-abm run schelling --mode llm      # set context/llm_behavior in config (needs openai + key)
```
