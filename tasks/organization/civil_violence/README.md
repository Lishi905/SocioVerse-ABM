# `civil_violence` — Epstein civil violence

**Family:** organization **SocioVerse2 study:** `abm_civil_violence` **Source ABM:** Epstein (2002).

Citizens (with private hardship and risk aversion) and a small police force on a grid.
A citizen rebels when its grievance outweighs the risk of arrest — producing the model's
signature **punctuated outbursts** of unrest.

## The four variants, as parameters

Same matrix as `schelling` (they only affect the LLM path; rule mode is the classic
deterministic Epstein rule):

| key | values | meaning |
|-----|--------|---------|
| `context` | `ocm` / `lcm` | **E**: numeric (grievance/arrest_prob) vs narrative |
| `llm_behavior` | `tbf` / `lbf` | **f**: apply Epstein's rule vs free reasoning |

## Layout

| file | role |
|------|------|
| `config.yaml` | grid, densities, legitimacy, vision, arrest k, threshold, jail + variant keys |
| `agents.py` | **P**: citizens (hardship, risk aversion) + cops |
| `env.py` | **E**: grid, vision scan, cop arrests, jail, movement; ocm/lcm rendering |
| `rule_f.py` | **f (rule)**: rebel iff grievance − net_risk > threshold — parity baseline |
| `llm_f.py` | **f (llm)**: tbf/lbf prompt over the ocm/lcm context; lazy openai |
| `evaluate.py` | **B**: peak / mean unrest, burstiness, jailed |
| `model.py` | assemble + run loop + `registry.register` |
| `legacy/` | original Mesa model + the `Civil_Violence_{LCM_LBF,OCM_*}` variant scripts |

## Run

```bash
sv-abm run civil_violence --mode rule    # classic Epstein dynamics (no LLM needed)
sv-abm run civil_violence --mode llm     # set context/llm_behavior in config (needs openai + key)
```
