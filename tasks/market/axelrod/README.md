# `axelrod` — iterated Prisoner's Dilemma tournament

**Family:** market **SocioVerse2 study:** `abm_axelrod` **Source ABM:** Axelrod (1984)
evolution of cooperation. (Classified as **market**: it is a strategic-interaction game.)

A round-robin tournament: every pair of agents plays an iterated Prisoner's Dilemma,
all matches advancing one round per kernel step. The headline question is Axelrod's —
does cooperation pay against a field of strategies?

## Layout

| file | role |
|------|------|
| `config.yaml` | strategy mix, copies, rounds, payoff matrix (T>R>P>S) |
| `agents.py` | **P**: agents, each running a fixed strategy |
| `env.py` | **E**: round-robin arena, pairwise scoring, grudge/last-move tracking |
| `rule_f.py` | **f (rule)**: tit_for_tat / grudger / all-C / all-D / random — parity baseline |
| `llm_f.py` | **f (llm)**: LLM plays C/D per opponent given history; lazy openai |
| `evaluate.py` | **B**: cooperation rate + strategy ranking by total payoff |
| `model.py` | assemble + run loop + `registry.register` |
| `legacy/` | the original `axelrod`-library tournament code — parity ground truth |

## Behaviour

Each agent's action is its move (C/D) toward **every** opponent this round, decided
from that opponent's history (so tit_for_tat cooperates with cooperators and retaliates
against defectors). Moves are discrete, so the rule↔LLM consistency uses strict
equality over the per-opponent move set.

## Run

```bash
sv-abm run axelrod --mode rule     # native strategies (no LLM needed)
sv-abm run axelrod --mode llm      # LLM plays each strategy (needs openai + key; costly)
```
