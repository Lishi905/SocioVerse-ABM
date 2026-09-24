"""P — the archetype household population.

Unlike grid Schelling (two synthetic groups), the Chicago population is materialised
by the vendored SegregationModel from real data: 240 archetypes
(race × income × family × neighbourhood_type) placed onto census tracts via
``archetype_tract_mapping.csv`` at true 2010 scale. We do not re-synthesise it here;
we project the model's HouseholdAgents into a light, inspectable view for the kernel's
``build() -> (env, population)`` contract (and for the SocioVerse2 integration).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Household:
    """A projected view of one HouseholdAgent (a group of ``weight`` real households)."""
    agent_id: int
    archetype_key: str
    race: str
    income_bracket: str
    family_type: str
    neighborhood_type: str
    weight: int                      # pop_count — real households this agent represents
    init_tract: str


@dataclass
class Population:
    households: list[Household] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.households)

    @property
    def total_weight(self) -> int:
        return sum(h.weight for h in self.households)


def build_population(engine: Any) -> Population:
    """Project the (already built) model's agents into a Population view."""
    engine.ensure_built()
    hh = []
    for agent in engine.agent_order:
        a = agent.archetype
        hh.append(Household(
            agent_id=agent._sv_id,
            archetype_key=agent.archetype_key,
            race=agent.race,
            income_bracket=a.income_bracket,
            family_type=a.family_type,
            neighborhood_type=a.neighborhood_type,
            weight=int(agent.pop_count),
            init_tract=agent.tract_id,
        ))
    return Population(hh)
