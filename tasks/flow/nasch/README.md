# `nasch` — Nagel-Schreckenberg single-lane traffic

**Family:** flow **SocioVerse2 study:** `abm_nasch` **Source ABM:** Nagel & Schreckenberg (1992)
cellular-automaton traffic model.

A layout template (like `sir`). Where SIR's environment is a **network**,
NaSch's is a **1-D cellular lattice with motion** — chosen deliberately to prove the
standard task layout and the kernel `f(P, E)` contract survive a very different E.

## Layout

| file | role |
|------|------|
| `config.yaml` | L (lane), density ρ, vmax, p, steps, seed, mode |
| `agents.py` | **P**: N vehicles on a ring, kept sorted by position |
| `env.py` | **E**: gaps + synchronous move (`observe → set speed → move all`) |
| `rule_f.py` | **f (rule)**: the 4-rule NaSch speed update — parity baseline |
| `llm_f.py` | **f (llm)**: original prompt + integer parsing, via kernel LLM helper (lazy openai) |
| `evaluate.py` | **B**: mean speed / flow (fundamental diagram), post-warmup |
| `model.py` | assemble + run loop + `registry.register` |
| `legacy/` | the original NaSch implementation (`nasch_mesa.py`, `behavior_engine/`, `test_consistency.py`) as parity ground truth |

## Mechanics (one synchronous step)

Each vehicle, from the *current* gaps: `v=min(v+1, vmax)` → `v=min(v, gap)` →
with prob `p`, `v=max(v-1,0)`; then **all** vehicles move `pos=(pos+v) mod L`. The
behavior function returns the new speed (rules 1-3); the env does the move (rule 4).
A vehicle never overtakes the one ahead (the gap caps its speed), so lane order is
preserved.

## Run

```bash
sv-abm run nasch --mode rule        # native CA baseline (no LLM needed)
sv-abm run nasch --mode llm         # LLM driver (needs openai + key)
# or: python -m tasks.flow.nasch.model
```

Edit `config.yaml` for density (sweep ρ for the q-vs-ρ fundamental diagram). Rule↔LLM
consistency is exercised in the test suite with a deterministic fake LLM driver.
