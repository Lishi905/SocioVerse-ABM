# `minority_game` — Challet-Zhang Minority Game

**Family:** market **SocioVerse2 study:** `abm_minority_game` **Source ABM:** Challet & Zhang (1997).
(Classified as **market**: it is a resource-competition
game, per the ABM study's taxonomy.)

Each round N agents pick a side; whoever is in the **minority** wins. Agents hold S
strategies (tables of size 2^M keyed on the recent winning-side history) and use the
one with the best virtual score — a clean test of inductive, bounded-rational play.

## Layout

| file | role |
|------|------|
| `config.yaml` | N (odd), memory M, strategies S, steps |
| `agents.py` | **P**: agents with S strategy tables + virtual scores |
| `env.py` | **E**: the game — minority wins, counterfactual score & history update |
| `rule_f.py` | **f (rule)**: play the best-scoring strategy's action — parity baseline |
| `llm_f.py` | **f (llm)**: LLM predicts the minority side from history; lazy openai |
| `evaluate.py` | **B**: volatility sigma^2/N (coordination efficiency) |
| `model.py` | assemble + run loop + `registry.register` |
| `legacy/` | original Mesa model (`minority_game_*.py`) + a consolidated simulation script — parity ground truth |

## Behaviour

The volatility `sigma^2/N` of attendance measures how well the population coordinates;
it depends on `alpha = 2^M / N` (the classic MG phase transition). Sides are discrete,
so the rule↔LLM consistency uses strict action equality.

## Run

```bash
sv-abm run minority_game --mode rule    # native inductive play (no LLM needed)
sv-abm run minority_game --mode llm     # LLM player (needs openai + key)
```
