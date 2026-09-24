> **Legacy snapshot, not maintained.** This folder keeps the original scripts verbatim for provenance and as the reference behaviour for the refactored task in [`../`](../). It is not imported by the task at runtime and is not part of the test suite. The scripts need their own environment: Mesa 2.x (`mesa.time`), and for `sugarscape_LLM.py` the pre-1.0 `openai` API (`openai<1`). They do not run against the current kernel.
>
> Edits for the public release: `sugarscape_LLM.py` reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the environment and defaults to the official OpenAI endpoint instead of a third-party relay.

# legacy — Sugarscape parity baseline

- `sugarscape_simple.py`: the original Mesa Sugarscape (G1): citizens forage a sugar grid
  by vision and metabolism; the rule baseline for `../rule_f.py`.
- `sugarscape_LLM.py`: the same model with an LLM choosing each citizen's move
  (legacy `openai.ChatCompletion` API, i.e. `openai<1`).

Both scripts serve a Mesa visualization (default http://127.0.0.1:8521).
