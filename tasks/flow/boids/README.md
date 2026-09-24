# `boids` — Reynolds flocking (continuous 2-D)

**Family:** flow **SocioVerse2 study:** `abm_boids` **Source ABM:** Reynolds (1987) Boids.

The **continuous-field** template: where SIR's E is a network and NaSch's is a lattice, Boids' E is **continuous 2-D
space with vector state**, built on the kernel's reusable `ContinuousField2D`. It
also introduces **tolerance-based consistency** — continuous vector actions can't use
strict equality, so the rule↔LLM score compares headings within a tolerance.

## Layout

| file | role |
|------|------|
| `config.yaml` | N, world size, max_speed/force, perception/separation radius, weights, steps, seed |
| `agents.py` | **P**: N boids, positions/velocities as NumPy `(N,2)` arrays |
| `env.py` | **E**: `ContinuousField2D` (toroidal) + synchronous integrate |
| `rule_f.py` | **f (rule)**: Reynolds steering (separation/alignment/cohesion) — parity baseline |
| `llm_f.py` | **f (llm)**: LLM picks a heading; lazy openai |
| `evaluate.py` | **B**: polarization (order parameter) + `heading_equals(tol)` for consistency |
| `model.py` | assemble + run loop + `registry.register` |
| `legacy/` | original `boids_core.py` / `boids_llm.py` / `boids_traditional.py` / `evaluate_*` (original scripts, kept verbatim) — parity ground truth |

## Mechanics (one synchronous step)

Each boid sees flockmates within `perception_radius`. Three steering forces
(`steer = limit(desired - velocity, max_force)`): **cohesion** toward the centroid,
**alignment** to mean neighbour velocity, **separation** away from neighbours within
`separation_radius`. The weighted sum (clamped to `max_force`) updates the velocity
(clamped to `max_speed`); the env integrates position on a toroidal field. Order
emerges: polarization rises from ~random toward ~1.

## Run

```bash
sv-abm run boids --mode rule        # native flocking (no LLM needed)
sv-abm run boids --mode llm         # LLM bird (needs openai + key)
# or: python -m tasks.flow.boids.model
```

Rule↔LLM consistency uses `evaluate.heading_equals(tol_deg)`; the test suite exercises it
with a deterministic fake LLM.
