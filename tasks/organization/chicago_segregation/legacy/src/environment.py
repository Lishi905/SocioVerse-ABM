"""
Environment module for Chicago Segregation ABM.

Manages the spatial environment:
- Load Census Tract data from processed GeoJSON
- Build spatial adjacency graph (Queen contiguity)
- Provide neighborhood descriptions for LLM prompts
- Track dynamic population changes as agents move
"""

import json
import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import Queen

logger = logging.getLogger(__name__)

RACE_COLS = ["nh_white", "nh_black", "nh_asian", "hispanic", "nh_other"]
PCT_COLS = ["pct_nh_white", "pct_nh_black", "pct_nh_asian", "pct_hispanic", "pct_nh_other"]
RACE_LABELS = {
    "nh_white": "Non-Hispanic White",
    "nh_black": "Non-Hispanic Black",
    "nh_asian": "Non-Hispanic Asian",
    "hispanic": "Hispanic/Latino",
    "nh_other": "Other",
}


class TractEnvironment:
    """
    Manages the spatial environment of Census Tracts.

    Each tract is a spatial cell with demographic, economic, and amenity attributes.
    Agents reside in tracts and can move between adjacent tracts.
    """

    def __init__(self, geojson_path: str, persona_config: dict | None = None):
        self.gdf = gpd.read_file(geojson_path)
        self.gdf = self.gdf[self.gdf["total_pop"] > 0].copy()
        self.gdf = self.gdf.set_index("GEOID10", drop=False)

        # Fill NaN values
        for c in PCT_COLS:
            self.gdf[c] = self.gdf[c].fillna(0)
        for c in ["per_capita_income", "hardship_index", "crime_count_2010"]:
            if c in self.gdf.columns:
                self.gdf[c] = self.gdf[c].fillna(self.gdf[c].median())

        # Build adjacency graph
        logger.info("Building Queen contiguity adjacency matrix...")
        self.weights = Queen.from_dataframe(self.gdf, use_index=True)
        logger.info(f"Loaded {len(self.gdf)} tracts with {self.weights.n} adjacency entries")

        # Dynamic population tracking (updated as agents move)
        self.dynamic_pop = {}
        for geoid in self.gdf.index:
            row = self.gdf.loc[geoid]
            self.dynamic_pop[geoid] = {
                race: int(row[race]) for race in RACE_COLS
            }

        # Tract capacities — populated later via init_capacities() once the
        # subset and Census baseline are locked.  If unset, has_capacity()
        # returns True (no capacity constraint).
        self.tract_capacity: dict[str, int] = {}

        # Normalization ranges for environment quality scores
        self.norm_ranges = {}
        if persona_config and "environmental_quality_normalization" in persona_config:
            self.norm_ranges = persona_config["environmental_quality_normalization"]

        # HOLC desirability scores
        self.holc_scores = {}
        if persona_config and "holc_legacy_effect" in persona_config:
            self.holc_scores = persona_config["holc_legacy_effect"]["grade_desirability_score"]

    @property
    def tract_ids(self) -> list[str]:
        return list(self.gdf.index)

    def get_neighbors(self, geoid: str, hops: int = 1) -> list[str]:
        """Get neighboring tract IDs within `hops` adjacency steps."""
        if geoid not in self.weights.neighbors:
            return []

        visited = {geoid}
        frontier = {geoid}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for nbr in self.weights.neighbors.get(node, []):
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.add(nbr)
            frontier = next_frontier

        visited.discard(geoid)
        return sorted(visited)

    def get_tract_info(self, geoid: str) -> dict:
        """Get static + dynamic info for a tract."""
        row = self.gdf.loc[geoid]
        pop = self.dynamic_pop[geoid]
        total = sum(pop.values())

        info = {
            "geoid": geoid,
            "total_pop": total,
            "per_capita_income": row.get("per_capita_income", 0),
            "hardship_index": row.get("hardship_index", 50),
            "crime_count_2010": row.get("crime_count_2010", 5000),
            "cta_stations": int(row.get("cta_stations", 0)),
            "cta_bus_stops": int(row.get("cta_bus_stops", 0)),
            "schools": int(row.get("schools", 0)),
            "parks": int(row.get("parks", 0)),
            "religious_places": int(row.get("religious_places", 0)),
            "holc_grade": row.get("holc_grade", "C"),
            "holc_score": row.get("holc_score", 2.5),
        }

        # Add dynamic racial composition
        for race in RACE_COLS:
            info[race] = pop[race]
            info[f"pct_{race}"] = (pop[race] / total * 100) if total > 0 else 0

        return info

    def _compute_safety_score(self, crime_count: float) -> tuple[float, str]:
        """Convert raw crime count to a 1-10 safety score with label."""
        # Based on environmental_quality_normalization ranges
        crime_min = self.norm_ranges.get("crime_count_2010", {}).get("min", 778)
        crime_max = self.norm_ranges.get("crime_count_2010", {}).get("max", 15801)
        # Normalize to 0-1 (inverted: low crime = high safety)
        normalized = 1.0 - max(0, min(1, (crime_count - crime_min) / (crime_max - crime_min)))
        score = round(1 + normalized * 9, 1)  # Scale to 1-10
        if score >= 8:
            label = "very safe"
        elif score >= 6:
            label = "moderately safe"
        elif score >= 4:
            label = "below average safety"
        elif score >= 2:
            label = "high crime area"
        else:
            label = "very high crime area"
        return score, label

    def _compute_school_score(self, n_schools: int) -> tuple[float, str]:
        """Convert school count to a quality score with label."""
        school_max = self.norm_ranges.get("schools", {}).get("max", 6)
        normalized = min(1, n_schools / max(1, school_max))
        score = round(1 + normalized * 9, 1)
        if score >= 7:
            label = "good school access"
        elif score >= 4:
            label = "moderate school access"
        elif score >= 2:
            label = "limited school access"
        else:
            label = "no nearby schools"
        return score, label

    def _compute_transit_score(self, stations: int, bus_stops: int) -> tuple[float, str]:
        """Convert transit counts to an accessibility score with label."""
        station_max = self.norm_ranges.get("cta_stations", {}).get("max", 4)
        bus_max = self.norm_ranges.get("cta_bus_stops", {}).get("max", 50)
        station_norm = min(1, stations / max(1, station_max))
        bus_norm = min(1, bus_stops / max(1, bus_max))
        combined = station_norm * 0.6 + bus_norm * 0.4  # L stations weighted more
        score = round(1 + combined * 9, 1)
        if score >= 7:
            label = "excellent transit"
        elif score >= 4:
            label = "moderate transit"
        else:
            label = "limited transit"
        return score, label

    def get_surrounding_area_summary(self, geoid: str) -> dict:
        """
        Compute aggregate racial composition of the surrounding area
        (immediate 1-hop neighbors) using dynamic population data.

        Returns dict with keys: n_neighbors, total_pop, and pct_{race} for each race,
        plus a 'dominant_race' label.
        """
        neighbors = self.get_neighbors(geoid, hops=1)
        if not neighbors:
            return {"n_neighbors": 0, "total_pop": 0, "dominant_race": "unknown"}

        totals = {race: 0 for race in RACE_COLS}
        grand_total = 0
        for nbr in neighbors:
            pop = self.dynamic_pop.get(nbr, {})
            for race in RACE_COLS:
                totals[race] += pop.get(race, 0)
            grand_total += sum(pop.get(r, 0) for r in RACE_COLS)

        result = {"n_neighbors": len(neighbors), "total_pop": grand_total}
        for race in RACE_COLS:
            result[f"pct_{race}"] = (totals[race] / grand_total * 100) if grand_total > 0 else 0

        # Determine dominant race (plurality)
        if grand_total > 0:
            dominant = max(RACE_COLS, key=lambda r: totals[r])
            result["dominant_race"] = RACE_LABELS[dominant]
            result["dominant_pct"] = result[f"pct_{dominant}"]
        else:
            result["dominant_race"] = "unknown"
            result["dominant_pct"] = 0

        return result

    def _format_surrounding_context(self, geoid: str) -> str:
        """Format surrounding area summary as a text block for LLM prompts."""
        ctx = self.get_surrounding_area_summary(geoid)
        if ctx["n_neighbors"] == 0:
            return "  Surrounding area: no adjacent tracts"

        parts = []
        for race in RACE_COLS:
            pct = ctx[f"pct_{race}"]
            if pct >= 1:
                parts.append(f"{RACE_LABELS[race]}: {pct:.1f}%")

        return (
            f"  Surrounding area ({ctx['n_neighbors']} adjacent tracts, "
            f"combined pop {ctx['total_pop']:,}): "
            f"{', '.join(parts)}"
        )

    def describe_tract_for_llm(self, geoid: str) -> str:
        """
        Generate a natural language description of a tract for LLM prompts.
        Includes ONLY the tract itself (no surrounding context).

        Uses normalized scores (1-10) instead of raw counts to prevent
        large numbers (e.g. "9,810 crime incidents") from dominating
        the LLM's assessment over more nuanced factors like racial composition.
        """
        info = self.get_tract_info(geoid)
        total = info["total_pop"]

        composition_parts = []
        for race in RACE_COLS:
            pct = info[f"pct_{race}"]
            if pct >= 1:
                composition_parts.append(f"{RACE_LABELS[race]}: {pct:.1f}%")

        safety_score, safety_label = self._compute_safety_score(info["crime_count_2010"])
        school_score, school_label = self._compute_school_score(info["schools"])
        transit_score, transit_label = self._compute_transit_score(
            info["cta_stations"], info["cta_bus_stops"])

        holc_desc = {"A": "historically well-invested (Grade A)",
                     "B": "historically moderate (Grade B)",
                     "C": "historically declining (Grade C)",
                     "D": "historically redlined/disinvested (Grade D)"}

        desc = (
            f"Tract {geoid} (population {total:,}):\n"
            f"  Racial composition: {', '.join(composition_parts)}\n"
            f"  Per capita income: ${info['per_capita_income']:,.0f}\n"
            f"  Safety: {safety_score}/10 ({safety_label})\n"
            f"  School quality: {school_score}/10 ({school_label})\n"
            f"  Transit access: {transit_score}/10 ({transit_label})\n"
            f"  Amenities: {info['parks']} parks, {info['religious_places']} religious places\n"
            f"  Historical context: {holc_desc.get(info['holc_grade'], 'unknown')}"
        )
        return desc

    def describe_tract_with_context_for_llm(self, geoid: str) -> str:
        """
        Generate a tract description WITH surrounding area racial context.

        This is the enriched version that includes the aggregate racial
        composition of adjacent tracts, enabling the LLM to reason about
        spatial clustering (e.g. "this tract is in a predominantly Black area"
        vs "this tract is an isolated pocket").
        """
        base_desc = self.describe_tract_for_llm(geoid)
        surrounding = self._format_surrounding_context(geoid)
        return f"{base_desc}\n{surrounding}"

    def init_capacities(self, tolerance_ratio: float, max_pop_per_agent: int) -> None:
        """
        Compute per-tract capacity caps based on the CURRENT dynamic_pop
        (which should be the pre-perturbation Census baseline at the
        moment this is called).

        Formula (agreed design):
            base      = sum(dynamic_pop[t])                  # Census total
            headroom_n = max(1, ceil(tolerance_ratio * base / max_pop_per_agent))
            capacity[t] = base + headroom_n * max_pop_per_agent

        Guarantees:
          * Every tract has headroom that is an exact multiple of
            max_pop_per_agent (so at least one max-sized agent can always
            enter a fresh tract).
          * Headroom scales roughly linearly with tract size while staying
            aligned to the agent granularity.
          * No capacity drops below base + max_pop_per_agent regardless of
            how small the tract is.

        Call this BEFORE any perturbation / state restore so capacities
        reflect the physical housing stock, not a transient shuffled state.
        """
        if max_pop_per_agent <= 0:
            raise ValueError(f"max_pop_per_agent must be > 0, got {max_pop_per_agent}")
        self.tract_capacity = {}
        for geoid in self.gdf.index:
            base = int(sum(self.dynamic_pop[geoid].values()))
            if base <= 0:
                # Empty tract: still give it one headroom unit so agents
                # could in principle move in if they wanted to.
                headroom_n = 1
            else:
                headroom_n = max(1, math.ceil(tolerance_ratio * base / max_pop_per_agent))
            self.tract_capacity[geoid] = base + headroom_n * max_pop_per_agent
        logger.info(
            f"Initialized tract capacities for {len(self.tract_capacity)} tracts "
            f"(tolerance_ratio={tolerance_ratio}, quantum={max_pop_per_agent})"
        )

    def tract_total(self, geoid: str) -> int:
        """Return current total population in a tract (sum across races)."""
        return int(sum(self.dynamic_pop[geoid].values()))

    def has_capacity(self, geoid: str, count: int) -> bool:
        """
        Check whether `count` more people can fit in tract `geoid`.

        If capacities have not been initialized, this returns True (no
        capacity constraint — preserves legacy behavior).
        """
        if not self.tract_capacity:
            return True
        cap = self.tract_capacity.get(geoid)
        if cap is None:
            return True
        return self.tract_total(geoid) + int(count) <= cap

    def move_population(self, from_geoid: str, to_geoid: str, race: str, count: int) -> bool:
        """
        Move `count` people of `race` from one tract to another.

        Returns True on success, False if the destination has no capacity
        for the requested count (caller should fall back to another target
        or stay put).
        """
        # Capacity guard: if the destination cannot absorb `count` more
        # people, reject the move entirely (partial moves would split the
        # agent conceptually and break pop_count accounting).
        if not self.has_capacity(to_geoid, count):
            return False
        actual = min(count, self.dynamic_pop[from_geoid][race])
        if actual <= 0:
            return False
        self.dynamic_pop[from_geoid][race] -= actual
        self.dynamic_pop[to_geoid][race] += actual
        return True

    def get_racial_composition_array(self) -> pd.DataFrame:
        """Return current dynamic racial composition as DataFrame."""
        rows = []
        for geoid in self.gdf.index:
            pop = self.dynamic_pop[geoid]
            total = sum(pop.values())
            row = {"GEOID10": geoid, "total_pop": total}
            for race in RACE_COLS:
                row[race] = pop[race]
                row[f"pct_{race}"] = (pop[race] / total * 100) if total > 0 else 0
            rows.append(row)
        return pd.DataFrame(rows)

    def rebuild_dynamic_pop_from_agents(self, agents):
        """
        Rebuild dynamic_pop entirely from current agent positions.

        Called after agent positions are shuffled (non-census init modes).
        Replaces the Census-based population counts with agent-based counts.

        Args:
            agents: iterable of HouseholdAgent with .tract_id, .race, .pop_count
        """
        # Reset all tracts to zero
        for geoid in self.gdf.index:
            self.dynamic_pop[geoid] = {race: 0 for race in RACE_COLS}

        # Accumulate from agents
        for agent in agents:
            tid = agent.tract_id
            race = agent.race
            pop = agent.pop_count
            if tid in self.dynamic_pop:
                self.dynamic_pop[tid][race] += pop

        # Log summary
        total = sum(sum(v.values()) for v in self.dynamic_pop.values())
        logger.info(f"Rebuilt dynamic_pop from agents: {total:,} total population")

    def subset(self, tract_ids: list[str]) -> "TractEnvironment":
        """Create a subset environment with only specified tracts (for prototyping)."""
        # This modifies self in place for simplicity in prototype
        mask = self.gdf.index.isin(tract_ids)
        self.gdf = self.gdf[mask].copy()
        self.dynamic_pop = {k: v for k, v in self.dynamic_pop.items() if k in tract_ids}
        # Also filter capacities if they were already set (defensive; the
        # normal flow computes capacities AFTER subset)
        if self.tract_capacity:
            self.tract_capacity = {
                k: v for k, v in self.tract_capacity.items() if k in tract_ids
            }
        self.weights = Queen.from_dataframe(self.gdf, use_index=True)
        logger.info(f"Subset to {len(self.gdf)} tracts")
        return self
