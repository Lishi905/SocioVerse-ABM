# `lux_marchesi` — interacting-agents financial market

**Family:** market **SocioVerse2 study:** `abm_lux_marchesi` **Source ABM:** Lux & Marchesi (1999).

An extra task, not part of the ten-model suite in the SocioVerse2 technical report.
Traders are fundamentalists
(price anchor) or noise traders that **herd** between optimist and pessimist; the
herding feedback generates the stylized facts of real markets — volatility clustering
and fat-tailed returns.

## Layout

| file | role |
|------|------|
| `config.yaml` | trader counts, price/value, demand & herding constants (mirror legacy paper config) |
| `agents.py` | **P**: traders with a stance (fundamentalist / optimist / pessimist) |
| `env.py` | **E**: excess-demand price formation + fundamental-value random walk |
| `rule_f.py` | **f (rule)**: Lux herding `P(optimist)=sigmoid(a1·x + a2·tanh(trend))` — parity baseline |
| `llm_f.py` | **f (llm)**: LLM picks the next stance; lazy openai |
| `evaluate.py` | **B**: volatility, excess kurtosis (fat tails), price range |
| `model.py` | assemble + run loop + `registry.register` |
| `legacy/` | the full multi-mode original model (`lux_marchesi_model.py`, evaluation/run/viz) — parity ground truth |

## Faithfulness note

This is the **core herding model**: a fixed fundamentalist anchor plus optimist⇄pessimist
herding among noise traders, which already produces volatility clustering / fat tails.
The full version's chartist⇄fundamentalist strategy switching, burst substeps and
multiple price-update modes live in `legacy/lux_marchesi_model.py`.

## Run

```bash
sv-abm run lux_marchesi --mode rule    # native herding market (no LLM needed)
sv-abm run lux_marchesi --mode llm     # LLM traders (needs openai + key)
```
