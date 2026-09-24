"""
Agent module for Chicago Segregation ABM.

Implements the AgentTorch-style Archetype strategy:
- Each HouseholdAgent represents a group of households sharing the same archetype
- Archetypes are defined by: race x income_bracket x family_type x neighborhood_type
- LLM calls are made at the archetype level, not individual agent level
- Individual agents sample from the archetype's decision distribution
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import mesa

logger = logging.getLogger(__name__)

RACE_LABELS = {
    "nh_white": "Non-Hispanic White",
    "nh_black": "Non-Hispanic Black",
    "nh_asian": "Non-Hispanic Asian",
    "hispanic": "Hispanic/Latino",
    "nh_other": "Other",
}

INCOME_LABELS = {
    "low": "low-income (below $15,500/year per capita)",
    "lower_middle": "lower-middle income ($15,500-$23,000/year per capita)",
    "upper_middle": "upper-middle income ($23,000-$35,800/year per capita)",
    "high": "high-income (above $35,800/year per capita)",
}

FAMILY_LABELS = {
    "single_no_children": "single adult or couple without children",
    "family_with_children": "family with children",
    "elderly": "elderly",
}

# Qualitative descriptions of racial comfort preferences.
# Avoids giving the LLM exact numeric thresholds that it would mechanically optimize.
# Instead, the LLM must weigh racial familiarity against all other factors holistically.
RACIAL_COMFORT_DESC = {
    # White: strong racial preference (D_bw=0.825 target). Being in a tract with
    # very low White presence is a genuine source of dissatisfaction, but does not
    # override good schools, safety, or affordability on its own.
    "nh_white": (
        "they prefer neighborhoods where White residents are a meaningful share of the "
        "population. Being in a neighborhood where White residents make up less than 15% "
        "of the population creates real discomfort, particularly when the surrounding "
        "area is also low in White residents. That said, strong schools, safety, and "
        "affordability can meaningfully offset demographic discomfort."
    ),
    # Black: strong racial preference (D_bw=0.825 from the other side). Social networks
    # and community ties are closely tied to neighborhood racial composition for Black
    # households, due to historic patterns of discrimination and community formation.
    "nh_black": (
        "they strongly value living near a Black community for social support, cultural "
        "familiarity, and network access. A neighborhood with very little Black presence "
        "(under 10%) is a notable source of dissatisfaction. However, affordability, "
        "safety, and proximity to family/friends also carry major weight — they would "
        "not choose a high-Black-presence neighborhood that is unaffordable or unsafe."
    ),
    # Asian: low-moderate racial preference (D_aw=0.449 city-wide).
    # SES-driven choice, BUT with an asymmetric weak ethnic anchor.
    # Earlier versions with zero Asian-presence pull caused D_aw to drift
    # off GT in both directions depending on scale.  A symmetric weak
    # anchor then over-corrected: enclave tracts started hoarding Asians
    # (strong stay-pull in already-Asian tracts), while non-enclave Asians
    # kept leaving for amenity — net over-clustering, D_aw overshooting
    # GT.  The anchor is now asymmetric: it only matters when the
    # household currently has essentially no Asian neighbors.  Being in a
    # tract that is already Asian-heavy provides no additional retention
    # pull beyond the normal amenity calculus.
    "nh_asian": (
        "their residential choice is driven primarily by school quality, safety, "
        "and commute convenience. They feel comfortable in diverse, mixed, or "
        "White-majority neighborhoods and do not need a large Asian population "
        "to be at home. Having a complete absence of Asian neighbors is a mild "
        "negative — enough that, if the current tract has essentially zero "
        "Asian presence, moving to a candidate with at least some visible "
        "Asian community (grocery stores, places of worship, a few familiar "
        "faces) becomes a small but real tiebreaker. However, when the "
        "household already lives in an Asian-heavy or Asian-enclave tract, the "
        "ethnic dimension is already satisfied and should not create "
        "additional stickiness — amenities should drive the stay/move "
        "decision. Do not recommend a move primarily to chase Asian "
        "clustering, and do not treat an already-strong Asian presence as "
        "an override reason to stay when schools, safety, or transit clearly "
        "point elsewhere."
    ),
    # Hispanic: moderate racial preference (D_hw=0.563 city-wide; 0.707
    # in the middle South+West Side subset).  The earlier "do not seek
    # out Hispanic-majority areas" framing undershot GT; a stronger
    # Latino anchor then over-corrected — move rate collapsed to ~5% and
    # D_hw still climbed far too slowly.  The current framing keeps the
    # Latino anchor real but loosens the threshold: *comparable* (not
    # strictly higher) Latino presence is sufficient to preserve
    # community ties, and the household will accept a moderate reduction
    # in Latino neighbors when a real amenity upgrade (safety, schools,
    # transit, cost) is on the table.
    "hispanic": (
        "they value proximity to family, affordable housing, and convenient "
        "transit above all else. Their social networks are often rooted in "
        "Latino community institutions — Spanish-speaking churches, "
        "family-run businesses, extended kin on nearby blocks — so a tract "
        "with visible Hispanic presence feels more connected and livable, "
        "while a tract with very few Latino neighbors feels isolating.  They "
        "are comfortable alongside non-Latino residents and will not move "
        "purely to reach a Hispanic-majority area, but they treat losing "
        "access to Latino community ties as a meaningful negative in any "
        "move evaluation.  A candidate tract does not need *more* Hispanic "
        "presence than the current one — a comparable or only modestly "
        "lower Latino share still preserves enough community connection to "
        "be acceptable when other top priorities (cost, transit, family "
        "proximity) are improved.  Ethnic familiarity is a real and "
        "meaningful factor in the decision, on par with the other top "
        "priorities rather than overriding them."
    ),
    "nh_other": (
        "they are comfortable in diverse neighborhoods and prioritize practical "
        "factors like affordability, safety, and transit access over ethnic composition."
    ),
}


@dataclass
class ArchetypeParams:
    """Parameters for a single archetype, loaded from archetypes.csv."""
    archetype_key: str
    race: str
    income_bracket: str
    family_type: str
    neighborhood_type: str
    population: int
    mobility_rate: float
    context_satisfaction_bonus: float
    ideal_own_group_pct: float
    min_own_group_pct: float
    search_n_mean: int
    search_radius_hops: int
    search_bias_own: float
    search_bias_mixed: float
    social_anchor_weight: float
    satisfaction_weights: dict = field(default_factory=dict)

    # Runtime state - updated each step by LLM
    current_satisfaction: float = 5.0
    would_move: bool = False
    key_reasons: list = field(default_factory=list)
    move_rankings: list = field(default_factory=list)

    def describe_for_llm(self) -> str:
        """Generate natural language description for LLM prompt."""
        race_label = RACE_LABELS.get(self.race, self.race)
        income_label = INCOME_LABELS.get(self.income_bracket, self.income_bracket)
        family_label = FAMILY_LABELS.get(self.family_type, self.family_type)

        # Build priority list from satisfaction weights
        # Rename "racial_composition" to softer label to avoid LLM over-indexing
        WEIGHT_DISPLAY = {"racial_composition": "demographic familiarity"}
        priorities = sorted(self.satisfaction_weights.items(), key=lambda x: -x[1])
        top_priorities = [
            f"{WEIGHT_DISPLAY.get(k, k).replace('_', ' ')} ({v:.0%})"
            for k, v in priorities[:4]
        ]

        # Qualitative racial-comfort description instead of explicit numeric thresholds
        # to prevent the LLM from mechanically optimizing toward a target percentage.
        comfort_desc = RACIAL_COMFORT_DESC.get(self.race, "some presence of their own community nearby")

        return (
            f"A {race_label} {family_label} household, {income_label}.\n"
            f"They currently live in a {self.neighborhood_type.replace('_', ' ')} neighborhood.\n"
            f"Their top priorities for where to live: {', '.join(top_priorities)}.\n"
            f"Regarding neighborhood demographics: {comfort_desc}"
        )


class HouseholdAgent(mesa.Agent):
    """
    A household agent in the segregation model.

    Each agent belongs to an archetype and resides in a Census Tract.
    Decisions are driven by LLM calls at the archetype level.
    """

    def __init__(self, model, archetype: ArchetypeParams, tract_id: str, pop_count: int):
        super().__init__(model)
        self.archetype = archetype
        self.tract_id = tract_id
        self.race = archetype.race
        self.pop_count = pop_count  # How many actual households this agent represents
        self.satisfaction = 5.0
        self.years_in_tract = np.random.randint(1, 15)
        self.moved_this_step = False
        # Tract the agent was in at the START of the current step.
        # Used by direction metrics to measure per-mover alignment with GT.
        self._prev_tract = tract_id
        # Set by the model's budget gate each step.  False blocks a move
        # even if the cell has a target.
        self._allowed_to_move = False

    def step(self):
        """
        Agent step: use archetype-level LLM decision to decide move/stay.
        Actual LLM calls happen at the model level (archetype batching).
        self.satisfaction was already set by the model's per-tract LLM assessment.
        """
        self.moved_this_step = False
        # Record pre-move tract for the direction indicator.
        self._prev_tract = self.tract_id

        # Apply context bonus and clamp to [0, 10]
        effective_sat = max(0.0, min(10.0,
            self.satisfaction + self.archetype.context_satisfaction_bonus))
        self.satisfaction = effective_sat

        # Determine if this agent moves.
        # The LLM has already decided (a) would_move and (b) best target.
        # If BOTH are true, the agent executes the move directly — no additional
        # stochastic gate. The mobility realism is captured by:
        #   1. The LLM's would_move=False for satisfied agents (the main filter)
        #   2. The LLM's best_score <= stay_score for agents with no good alternative
        # Adding a further move_probability gate would make recovery take 5-10x more
        # steps without adding behavioral realism — the LLM already embeds the
        # "moving is costly" logic in its stay_score vs best_score comparison.
        group_wants_move = self.model.group_would_move.get(
            (self.archetype_key, self.tract_id), self.archetype.would_move
        )
        if group_wants_move:
            has_target = bool(self.model.move_targets.get(
                (self.archetype_key, self.tract_id)
            ))
            if has_target and self._allowed_to_move:
                # LLM confirmed a better destination AND budget gate admitted us
                self._attempt_move()
            # Else: stay (no target, or budget exhausted this step)

        self.years_in_tract += 1

    def _attempt_move(self):
        """Try to move to the best available tract from per-(archetype, tract) targets.

        Iterates through the ranked candidate list (top-k from LLM).  For
        each candidate, re-checks capacity AT EXECUTION TIME because
        concurrent movers in the same step may have filled a destination
        that was feasible when the LLM saw it.  If every candidate is full,
        the agent stays this step — natural-queuing semantics: wait for a
        vacancy next step rather than forcing an over-cap move.
        """
        # Look up move target for this agent's specific (archetype, tract) group
        rankings = self.model.move_targets.get(
            (self.archetype_key, self.tract_id), []
        )
        if not rankings:
            return

        env = self.model.environment
        for candidate in rankings:
            target_id = candidate.get("tract_id")
            if not target_id or target_id not in env.dynamic_pop or target_id == self.tract_id:
                continue
            # Capacity re-check at execution time (handles within-step races
            # where earlier agents have filled this destination).
            if not env.has_capacity(target_id, self.pop_count):
                continue
            # Per-step (tract, race) inflow cap — rate limit on single-step
            # demographic shock. Skips to the next ranked candidate if this
            # tract has already absorbed its quota of this race this step.
            inflow_cap = self.model.get_step_inflow_cap(target_id)
            already = self.model._step_inflow[(target_id, self.race)]
            if already + self.pop_count > inflow_cap:
                continue
            # Attempt the move; move_population returns False if capacity
            # was invalidated between check and write (belt-and-braces).
            ok = env.move_population(self.tract_id, target_id, self.race, self.pop_count)
            if not ok:
                continue
            self.model._step_inflow[(target_id, self.race)] += self.pop_count
            old_tract = self.tract_id
            self.tract_id = target_id
            self.years_in_tract = 0
            self.moved_this_step = True
            logger.debug(
                f"Agent {self.unique_id} ({self.archetype.archetype_key}) "
                f"moved {old_tract} -> {target_id}"
            )
            return
        # All ranked candidates were infeasible (full or invalid) — stay.

    @property
    def archetype_key(self) -> str:
        return self.archetype.archetype_key
