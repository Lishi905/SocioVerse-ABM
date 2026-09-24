# `social_force` — Helbing pedestrian evacuation (continuous 2-D)

**Family:** flow **SocioVerse2 study:** `abm_social_force` **Source ABM:** Helbing & Molnár
Social Force Model.

Completes the flow family. Reuses the kernel's `ContinuousField2D` with **wrap=False**
(a bounded room — agents clamp at walls instead of wrapping), complementing Boids'
toroidal use. Continuous vector actions, so consistency uses a heading tolerance.

## Layout

| file | role |
|------|------|
| `config.yaml` | room size, exit, N, tau, desired speed (Gaussian), v_max, A/B repulsion, exit radius |
| `agents.py` | **P**: crowd with heterogeneous desired speeds, shared exit goal |
| `env.py` | **E**: bounded field; evacuated agents drop out |
| `rule_f.py` | **f (rule)**: driving force to exit + exponential inter-agent repulsion — parity baseline |
| `llm_f.py` | **f (llm)**: LLM picks a heading toward the exit avoiding the crowd; lazy openai |
| `evaluate.py` | **B**: evacuation fraction / time + `heading_equals(tol)` |
| `model.py` | assemble + run loop (until evacuated) + `registry.register` |
| `legacy/` | full SI-unit Helbing engine (walls, time-to-collision, LLM policy), kept verbatim — parity ground truth |

## Faithfulness note

`rule_f` keeps Helbing's **structure** — `driving = (v_des·e_exit − v)/τ` plus
`Σ A·exp((2r − d)/B)·n_away` repulsion — but uses tamed, dimensionless force constants
so the template is stable at `dt=0.1` and verifiable (most agents evacuate). The full
SI-unit version with walls lives in `legacy/`.

## Run

```bash
sv-abm run social_force --mode rule     # native social force (no LLM needed)
sv-abm run social_force --mode llm      # LLM pedestrian (needs openai + key)
# or: python -m tasks.flow.social_force.model
```
