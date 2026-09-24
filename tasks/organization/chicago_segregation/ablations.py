"""Switches for the supplementary ablations (E0-E4): race-blind (E2) and attribute shuffle (E4).

Applied AFTER ``ChicagoEngine.ensure_built()`` via instance-level wraps and
module-global patches only — zero edits to the vendored ``legacy/`` code (the same
seam philosophy as the CONFIG_DIR monkey-patch and SocioVerse2's information-injection wrap).

- ``race_blind`` (E2): strip racial information from everything the LLM sees — the two
  system prompts (module globals in ``src.llm_client``), every archetype's persona
  description, and the environment's tract descriptions. Mechanics (moves, budgets,
  capacities, metrics) are untouched: the ablation asks whether the *recovery* uses
  racial semantics, not whether segregation remains measurable.
- ``shuffle_tract_attributes`` (E4): one seeded permutation relocates each tract's
  non-demographic attribute PROFILE (income, crime, schools, transit, amenities, HOLC)
  onto another tract, breaking the geography-context correspondence while leaving
  racial composition, geometry, adjacency and capacities intact. If the LLM's recovery
  uses real geographic signal, it should degrade or collapse under this shuffle.

Config (config.yaml)::

    ablations:
      race_blind: true
      shuffle_tract_attributes: 42   # int = its own seed; true = reuse the study seed
"""
from __future__ import annotations

import random
import re
from typing import Optional

# ── race-blind replacement system prompts ───────────────────────────────────
# Same stay-bias / moving-cost logic and the EXACT same JSON output contract as the
# originals in legacy/src/llm_client.py; all demographic guidance removed.

_RB_SATISFACTION_PROMPT = """\
You are simulating a household's residential satisfaction in Chicago (year 2010).
You must evaluate how satisfied this household is with their current neighborhood.

Consider ALL of these factors — use the importance weights from the household profile:
- School quality and educational access
- Safety and crime levels
- Housing affordability relative to income
- Commute and transit access
- Local amenities (parks, grocery, religious places)
- Proximity to social network (family, friends, community)

IMPORTANT — STAY BIAS: Moving is costly in real life: packing, transaction
costs, broken school continuity, lost neighborhood ties, and uncertainty
about the new area.  The default for an established household is to STAY.
Only set would_move=true when a TOP-weighted priority (weight >= 20%) is
badly unmet — typically scoring 4/10 or worse — AND it is plausible that
other reachable neighborhoods would do noticeably better on that priority
without sacrificing the household's other top priorities.  A household
that is mildly dissatisfied (satisfaction 5-6) on several factors but has
no single severely unmet top priority should keep would_move=false.

Respond ONLY with valid JSON in this exact format:
{"satisfaction": <float 0-10>, "would_move": <bool>, "key_reasons": ["reason1", "reason2"]}

Where satisfaction 0 = extremely dissatisfied, 10 = perfectly satisfied.
would_move = true if this household would actively seek to relocate."""

_RB_MOVE_PROMPT = """\
You are simulating a household's neighborhood choice decision in Chicago (year 2010).
Evaluate candidate neighborhoods and pick the best one for relocation.

Moving imposes real costs (packing, transaction fees, broken school
continuity, lost neighborhood ties). The stay_score should already
reflect a modest "inertia premium" for the current neighborhood — a
candidate must be meaningfully better on a top priority to justify the
move, not merely marginally better.

Weigh ALL factors from the household profile — schools, safety, affordability,
commute, amenities, and social networks — according to the stated importance weights.

Score the current neighborhood honestly based on how well it meets the household's
needs right now. Score candidate neighborhoods the same way.

Recommend a move (best_score > stay_score) whenever a candidate is clearly better
on the household's top-weighted priorities. Even a meaningful improvement in ONE top
priority is a valid reason to move if other factors are roughly equal.

Respond ONLY with valid JSON in this exact format:
{"ranked_candidates": [{"tract_id": "<tract_id>", "score": <float 0-10>}, ...], "stay_score": <float 0-10>, "reason": "<one sentence>"}

ranked_candidates = a list of UP TO 3 candidate tracts sorted by your preference
(best first). Include fewer than 3 only if fewer candidates are genuinely
plausible. Each entry gives your attractiveness score (0-10) for that tract.
stay_score = how attractive staying in the current neighborhood is (0-10).
If stay_score >= the top candidate's score, the household should stay — in
that case return an empty ranked_candidates list."""

_ORIGINAL_PROMPTS: Optional[tuple] = None  # saved once so tests can restore


def _apply_race_blind(engine) -> None:
    global _ORIGINAL_PROMPTS
    import src.llm_client as _lc                      # the vendored module the engine uses
    from src.agents import RACE_LABELS, RACIAL_COMFORT_DESC  # noqa: F401 (labels drive the scrubber)

    if _ORIGINAL_PROMPTS is None:
        _ORIGINAL_PROMPTS = (_lc.SATISFACTION_SYSTEM_PROMPT, _lc.MOVE_EVAL_SYSTEM_PROMPT)
    _lc.SATISFACTION_SYSTEM_PROMPT = _RB_SATISFACTION_PROMPT
    _lc.MOVE_EVAL_SYSTEM_PROMPT = _RB_MOVE_PROMPT

    # persona scrubber: drop the demographic-comfort line, the race label, and the
    # "demographic familiarity (NN%)" priority item from the archetype description.
    labels = sorted(RACE_LABELS.values(), key=len, reverse=True)
    label_pat = re.compile("|".join(re.escape(v) for v in labels))

    def scrub_persona(text: str) -> str:
        lines = [l for l in text.splitlines()
                 if not l.strip().startswith("Regarding neighborhood demographics:")]
        t = "\n".join(lines)
        t = label_pat.sub("", t)
        t = re.sub(r"demographic familiarity \(\d+%\), ", "", t)
        t = re.sub(r", demographic familiarity \(\d+%\)", "", t)
        return re.sub(r"[ ]{2,}", " ", t)

    for arch in engine.model.active_archetypes.values():
        orig = arch.describe_for_llm
        arch.describe_for_llm = (lambda _o=orig: scrub_persona(_o()))

    # tract scrubber: drop the "Racial composition:" line and the race list on the
    # "Surrounding area (...)" line (neighbour count and population are kept).
    def scrub_tract(text: str) -> str:
        out = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("Racial composition:"):
                continue
            if s.startswith("Surrounding area"):
                line = re.sub(r"(Surrounding area \([^)]*\)):.*$", r"\1.", line)
            out.append(line)
        return "\n".join(out)

    env = engine.model.environment
    for name in ("describe_tract_for_llm", "describe_tract_with_context_for_llm"):
        orig = getattr(env, name)
        setattr(env, name, (lambda geoid, _o=orig: scrub_tract(_o(geoid))))


def restore_prompts() -> None:
    """Undo the system-prompt patch (used by tests; one experiment per process otherwise)."""
    global _ORIGINAL_PROMPTS
    if _ORIGINAL_PROMPTS is not None:
        import src.llm_client as _lc

        _lc.SATISFACTION_SYSTEM_PROMPT, _lc.MOVE_EVAL_SYSTEM_PROMPT = _ORIGINAL_PROMPTS
        _ORIGINAL_PROMPTS = None


# ── shuffled tract attributes (E4) ───────────────────────────────────────────

_SHUFFLE_COLS = [
    "per_capita_income", "hardship_index", "crime_count_2010",
    "pct_below_poverty", "pct_unemployed",
    "cta_stations", "cta_bus_stops", "schools", "religious_places",
    "police_stations", "fire_stations", "parks",
    "holc_grade", "holc_score", "holc_coverage",
]


def _apply_shuffle(engine, seed: int) -> None:
    gdf = engine.model.environment.gdf
    cols = [c for c in _SHUFFLE_COLS if c in gdf.columns]
    rng = random.Random(seed)
    perm = list(range(len(gdf)))
    rng.shuffle(perm)
    for c in cols:      # ONE permutation for all columns: attribute profiles move as blocks
        gdf[c] = gdf[c].iloc[perm].to_numpy()


# ── entry point ──────────────────────────────────────────────────────────────

def apply(engine, ab_cfg: dict, seed: Optional[int] = None) -> list:
    """Apply the configured ablations to a built engine. Returns the applied list."""
    engine.ensure_built()
    applied = []
    if ab_cfg.get("race_blind"):
        _apply_race_blind(engine)
        applied.append("race_blind")
    sh = ab_cfg.get("shuffle_tract_attributes")
    if sh:
        sh_seed = sh if (isinstance(sh, int) and not isinstance(sh, bool)) else (
            seed if seed is not None else 42)
        _apply_shuffle(engine, sh_seed)
        applied.append("shuffle_tract_attributes(seed=%s)" % sh_seed)
    engine._sv_ablations = applied
    return applied
