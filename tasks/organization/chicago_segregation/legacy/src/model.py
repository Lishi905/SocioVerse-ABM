"""
Main model orchestrator for Chicago Segregation ABM.

Coordinates:
1. Environment setup from Census Tract data
2. Agent creation from archetype population mapping
3. Per-step LLM calls at archetype level (AgentTorch strategy)
4. Agent decision execution (move/stay)
5. Metrics collection each step
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import mesa
import numpy as np
import pandas as pd

from .agents import ArchetypeParams, HouseholdAgent
from .environment import TractEnvironment
from .llm_client import LLMCallRecord, LLMClient, LLMConfig
from .metrics import compute_all_metrics, recovery_r_squared

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "processed_data"

# Income bracket → priority score on a 0–10 scale, commensurate with
# (10 - satisfaction).  Wealthier households have a higher score so that,
# all else equal, they outcompete poorer households for scarce slots in
# desirable destinations.  This encodes the empirical fact that
# dissatisfaction alone does not cause relocation in the real world —
# economic capacity is the binding constraint that prevents 2010 Census
# from being a satisfaction-equilibrium.
INCOME_PRIORITY_SCORE = {
    "low": 0.0,
    "lower_middle": 10.0 / 3.0,       # ~3.33
    "upper_middle": 20.0 / 3.0,       # ~6.67
    "high": 10.0,
}


def load_archetypes(csv_path: str | Path) -> dict[str, ArchetypeParams]:
    """Load archetype parameters from CSV into ArchetypeParams objects."""
    df = pd.read_csv(csv_path)
    archetypes = {}
    weight_cols = [c for c in df.columns if c.startswith("w_")]

    for _, row in df.iterrows():
        weights = {c.replace("w_", ""): row[c] for c in weight_cols}
        params = ArchetypeParams(
            archetype_key=row["archetype_key"],
            race=row["race"],
            income_bracket=row["income_bracket"],
            family_type=row["family_type"],
            neighborhood_type=row["neighborhood_type"],
            population=int(row["population"]),
            mobility_rate=row["mobility_rate"],
            context_satisfaction_bonus=row["context_satisfaction_bonus"],
            ideal_own_group_pct=row["ideal_own_group_pct"],
            min_own_group_pct=row["min_own_group_pct"],
            search_n_mean=int(row["search_n_mean"]),
            search_radius_hops=int(row["search_radius_hops"]),
            search_bias_own=row["search_bias_own"],
            search_bias_mixed=row["search_bias_mixed"],
            social_anchor_weight=row["social_anchor_weight"],
            satisfaction_weights=weights,
        )
        archetypes[params.archetype_key] = params

    return archetypes


class SegregationModel(mesa.Model):
    """
    LLM-Agent driven Schelling model variant for Chicago racial segregation.

    Key design:
    - AgentTorch Archetype strategy: LLM calls per archetype, not per agent
    - Each step: assess satisfaction (LLM) -> decide movers -> evaluate destinations (LLM) -> execute moves
    - Metrics tracked every step for validation against 2010 Census data

    Initialization modes:
    - "census" (default): Agents placed at real 2010 Census locations.
      Purpose: validate equilibrium stability (does D stay ~0.825?)
    - "census_2000": Agents placed at real 2000 Census locations (on 2010 boundaries).
      Purpose: test whether LLM agents can evolve from 2000→2010 distribution.
      Uses processed_data/chicago_tracts_2000.geojson for demographics and
      socioeconomic data, while environment infrastructure (schools, transit, etc.)
      comes from the 2010 dataset (reasonable proxy).
    - "random": Agents shuffled uniformly across all tracts.
      Purpose: test emergence (can segregation arise from D≈0?)
    - "semi_random": Agents shuffled within same Community Area.
      Purpose: partial mixing, test local re-segregation dynamics.
    - "inverted": Minorities placed in White tracts and vice versa.
      Purpose: extreme perturbation, test convergence speed.
    - "perturbed_n": Like "perturbed" but destinations are restricted to
      foreign-race tracts within `perturb_radius_hops` adjacency hops of
      the agent's original tract (slightly wider than the per-step search
      radius used during simulation, so recovery is reachable).
    """

    INIT_MODES = (
        "census", "census_2000", "random", "semi_random",
        "inverted", "perturbed", "perturbed_n",
    )

    def __init__(
        self,
        llm_config: LLMConfig,
        n_agents: int = 100,
        tract_ids: list[str] | None = None,
        max_archetypes: int | None = None,
        seed: int = 42,
        init_mode: str = "census",
        perturb_pct: float = 0.15,
        perturb_radius_hops: int = 7,
        move_budget_pct: float = 0.05,
        own_race_ceiling_ratio: float = 1.3,
        max_pop_per_agent: int = 250,
        capacity_tolerance: float = 0.10,
        move_priority_income_weight: float = 0.0,
        race_size_budget_scaling: bool = True,
        per_step_tract_inflow_cap_ratio: float = 0.03,
        enclave_candidate_bias: float = 0.1,
    ):
        super().__init__(seed=seed)

        if init_mode not in self.INIT_MODES:
            raise ValueError(f"init_mode must be one of {self.INIT_MODES}, got '{init_mode}'")

        self.perturb_pct = perturb_pct
        # Neighborhood radius (in adjacency hops) used by "perturbed_n" init
        # mode when choosing a foreign-race destination.  Should be slightly
        # larger than the largest archetype.search_radius_hops used during
        # simulation so that perturbed agents can realistically search their
        # way back during recovery.
        self.perturb_radius_hops = perturb_radius_hops
        # Per-step mover budget: cap the total pop_count that is allowed to
        # relocate in one step at this fraction of the total represented
        # population.  Movers are picked by descending _move_priority
        # (weighted blend of dissatisfaction and income — see
        # move_priority_income_weight) so that both the desire to move
        # AND the economic capacity to do so gate admission.  ~5% matches
        # the roughly 5–7% annual residential mobility rate in US metros.
        self.move_budget_pct = move_budget_pct
        # Own-race ceiling for candidate filtering: a candidate tract is
        # dropped from the LLM's consideration set if its own-race share is
        # more than this ratio times the household's current tract's own-race
        # share.  Set > 1 to soften; set to <= 0 to disable.
        self.own_race_ceiling_ratio = own_race_ceiling_ratio
        # Maximum population (people) represented by a single HouseholdAgent.
        # Mapping cells with pop > this value are split into several clones
        # sharing the same (archetype, tract) but each carrying a smaller
        # pop_count.  Smaller clones prevent overshoot (a single move
        # relocating 6,000+ people in one step) and let the per-step move
        # budget admit a partial fraction of any cell.  Set to a very large
        # value (e.g. 10**9) to disable splitting.
        self.max_pop_per_agent = max_pop_per_agent
        # Per-tract capacity tolerance (fraction of initial Census pop),
        # quantized to multiples of max_pop_per_agent.  See
        # TractEnvironment.init_capacities for the exact formula.
        self.capacity_tolerance = capacity_tolerance
        # Weight on income (0..1) vs. dissatisfaction (1 - w) when ranking
        # agents for the per-race move budget.  Higher w means richer
        # households grab scarce slots more often; lower w means the
        # most-dissatisfied move first regardless of income.
        # DEFAULT 0.0 = legacy pure-dissatisfaction ordering (exactly
        # reproduces the old `sort(key=satisfaction, asc)` behavior, since
        # priority = 10 - sat sorted desc is monotonically equivalent).
        # Set to a positive value (e.g. 0.4) to let economic capacity
        # contest scarce slots in desirable tracts.
        if not 0.0 <= move_priority_income_weight <= 1.0:
            raise ValueError(
                f"move_priority_income_weight must be in [0, 1], "
                f"got {move_priority_income_weight}"
            )
        self.move_priority_income_weight = move_priority_income_weight
        # Race-aware mobility controls (anti-overshoot).  None of these
        # reference the Census 2010 target values; they only use the city-wide
        # share of each race (public 2010 demographic common knowledge) plus
        # currently-observed simulation state.
        # ─ race_size_budget_scaling:
        #     Cut the move budget for small races so that a 5% mover fraction
        #     does not dominate the D metric purely because the race is
        #     numerically small.  Scale ∝ sqrt(city_share / median_share),
        #     capped at 1.0.
        # ─ per_step_tract_inflow_cap_ratio:
        #     Per-step (tract, race) net inflow cap as a fraction of that
        #     tract's capacity.  Prevents runaway single-step jumps like a
        #     tract's own-race share doubling in one step.  0 disables.
        # ─ enclave_candidate_bias:
        #     ε in the inverse-frequency candidate weight w = 1 / (ε + share).
        #     Smaller ε → stronger downweighting of already-concentrated
        #     candidates.  0 disables (uniform sampling).
        self.race_size_budget_scaling = race_size_budget_scaling
        self.per_step_tract_inflow_cap_ratio = per_step_tract_inflow_cap_ratio
        self.enclave_candidate_bias = enclave_candidate_bias
        # Per-step inflow tracker: (tract_id, race) -> pop moved in this step.
        # Reset at the start of every step() call.
        self._step_inflow: dict[tuple[str, str], int] = defaultdict(int)

        self.llm = LLMClient(llm_config)
        self.n_agents = n_agents
        self.init_mode = init_mode

        # Load persona config
        with open(CONFIG_DIR / "persona_parameters.json") as f:
            self.persona_config = json.load(f)

        # Setup environment (always uses 2010 boundaries + infrastructure)
        logger.info("Loading tract environment...")
        self.environment = TractEnvironment(
            str(DATA_DIR / "chicago_tracts.geojson"),
            self.persona_config,
        )

        # Optionally subset to specific tracts (for prototyping)
        if tract_ids:
            self.environment.subset(tract_ids)
            logger.info(f"Subset environment to {len(tract_ids)} tracts")

        self.tract_ids = self.environment.tract_ids

        # Lock per-tract capacities to the Census 2010 baseline NOW, before
        # any agent creation / perturbation / restore overwrites
        # dynamic_pop.  Capacity represents physical housing stock and must
        # not drift with the population shuffle.
        self.environment.init_capacities(
            tolerance_ratio=self.capacity_tolerance,
            max_pop_per_agent=self.max_pop_per_agent,
        )

        # For census_2000 mode, load the 2000 demographic data to build
        # archetype-tract mapping from 2000 populations
        if init_mode == "census_2000":
            logger.info("Loading Census 2000 data for initial population distribution...")
            tract_mapping_2000 = self._build_2000_tract_mapping()
            tract_mapping = tract_mapping_2000
        else:
            tract_mapping = None

        # Load archetypes and (2010-based) tract mapping
        logger.info("Loading archetypes...")
        all_archetypes = load_archetypes(CONFIG_DIR / "archetypes.csv")

        if tract_mapping is None:
            # Default: use 2010 archetype-tract mapping
            tract_mapping = pd.read_csv(CONFIG_DIR / "archetype_tract_mapping.csv")
            tract_mapping["GEOID10"] = tract_mapping["GEOID10"].astype(str)

        # Filter mapping to our tracts
        tract_mapping = tract_mapping[tract_mapping["GEOID10"].isin(self.tract_ids)]

        # Create agents from tract-archetype mapping
        self._create_agents(all_archetypes, tract_mapping, max_archetypes)

        # Per-(archetype, tract) state for move evaluation
        # Populated each step by _assess_archetype_satisfaction
        self.group_would_move: dict[tuple[str, str], bool] = {}
        # Populated each step by _evaluate_move_candidates
        self.move_targets: dict[tuple[str, str], list] = {}

        # ── Reference baselines ──────────────────────────────────────
        # 1) Census 2010 ground-truth (tract-scale): the REAL 2010 demographic
        #    counts for the selected tracts.  This is what the simulation is
        #    ultimately trying to reproduce.
        self.census_2010_baseline = self.environment.get_racial_composition_array()
        self.census_ground_truth_metrics = compute_all_metrics(self.census_2010_baseline)

        # City-wide race share on the selected tracts.  NOT GT leakage:
        # these are publicly reported aggregate demographic statistics for
        # Chicago 2010 (roughly 32% White / 32% Black / 29% Hispanic / 5%
        # Asian / 2% Other), which any 2010 resident would reasonably
        # know.  Used only as a structural prior for the ceiling floor
        # and budget scaling — NOT to steer agents toward specific tracts.
        race_cols = ["nh_white", "nh_black", "nh_asian", "hispanic", "nh_other"]
        total_city_pop = float(
            self.census_2010_baseline[race_cols].values.sum()
        )
        if total_city_pop > 0:
            self.city_race_share = {
                r: float(self.census_2010_baseline[r].sum()) / total_city_pop
                for r in race_cols
            }
        else:
            # Fallback: uniform.  Should never hit this on real data.
            self.city_race_share = {r: 1.0 / len(race_cols) for r in race_cols}
        _nz_shares = [v for v in self.city_race_share.values() if v > 0]
        self.city_median_race_share = (
            sorted(_nz_shares)[len(_nz_shares) // 2] if _nz_shares else 0.2
        )
        logger.info(
            "[baseline] City-wide race shares: "
            + ", ".join(
                f"{r}={v*100:.1f}%" for r, v in self.city_race_share.items()
            )
            + f" (median={self.city_median_race_share*100:.1f}%)"
        )
        logger.info(
            f"[baseline] Census 2010 ground truth on {len(self.tract_ids)} tracts: "
            f"D_bw={self.census_ground_truth_metrics['D_black_white']:.4f}, "
            f"D_hw={self.census_ground_truth_metrics['D_hispanic_white']:.4f}, "
            f"D_aw={self.census_ground_truth_metrics['D_asian_white']:.4f}, "
            f"Iso_B={self.census_ground_truth_metrics['Isolation_black']:.4f}"
        )

        # 2) Agent-scale pre-perturbation baseline: what the model looks like
        #    when agents are at their Census 2010 positions, aggregated at
        #    agent scale (n_agents, with each agent = pop_count households).
        #    This is the fair comparison point for in-simulation metrics
        #    because the metric scale matches the simulation scale.
        self.environment.rebuild_dynamic_pop_from_agents(list(self.agents))
        pre_perturb_df = self.environment.get_racial_composition_array()
        self.agent_scale_baseline_df = pre_perturb_df.copy()
        self.agent_scale_baseline_metrics = compute_all_metrics(pre_perturb_df)
        self.agent_scale_baseline_metrics["R2_recovery"] = recovery_r_squared(
            pre_perturb_df, self.census_2010_baseline
        )
        logger.info(
            f"[baseline] Agent-scale pre-perturbation ({len(list(self.agents))} agents): "
            f"D_bw={self.agent_scale_baseline_metrics['D_black_white']:.4f}, "
            f"D_hw={self.agent_scale_baseline_metrics['D_hispanic_white']:.4f}, "
            f"D_aw={self.agent_scale_baseline_metrics['D_asian_white']:.4f}, "
            f"Iso_B={self.agent_scale_baseline_metrics['Isolation_black']:.4f}, "
            f"R²={self.agent_scale_baseline_metrics['R2_recovery']:.4f}"
        )

        # Apply initialization mode (shuffle agent positions if not census-like)
        if init_mode not in ("census", "census_2000"):
            self._apply_init_mode(init_mode, tract_mapping)
        elif init_mode == "census_2000":
            # Already using 2000 data; rebuild from agents on top of that
            self.environment.rebuild_dynamic_pop_from_agents(list(self.agents))

        # Metrics history
        self.metrics_history = []
        self.step_log = []

        # Compute initial metrics (step 0 = post-perturbation state for perturbed mode,
        # or identical to agent-scale baseline for census mode)
        initial_metrics = self._compute_metrics()
        initial_metrics["step"] = 0
        self.metrics_history.append(initial_metrics)
        logger.info(
            f"[step 0] Init mode '{init_mode}': "
            f"D_bw={initial_metrics['D_black_white']:.4f}, "
            f"D_hw={initial_metrics['D_hispanic_white']:.4f}, "
            f"D_aw={initial_metrics['D_asian_white']:.4f}, "
            f"Iso_B={initial_metrics['Isolation_black']:.4f}, "
            f"R²={initial_metrics.get('R2_recovery', 'n/a')}"
        )

    def _build_2000_tract_mapping(self) -> pd.DataFrame:
        """
        Build an archetype-tract mapping from Census 2000 data.

        Uses the same archetype key structure as the 2010 mapping, but derives
        population counts from the 2000 census distribution on 2010 tract boundaries.

        The mapping logic mirrors generate_archetypes.py:
        1. Classify each tract's neighborhood_type by 2000 racial composition
        2. Estimate income bracket distribution from 2000 per_capita_income
        3. Apply standard family_type splits (same across years)
        4. Build (archetype_key, GEOID10, population) tuples
        """
        import geopandas as gpd

        geojson_2000 = DATA_DIR / "chicago_tracts_2000.geojson"
        if not geojson_2000.exists():
            raise FileNotFoundError(
                f"Census 2000 processed data not found: {geojson_2000}\n"
                "Run: python scripts/process_data_2000.py"
            )

        gdf_2000 = gpd.read_file(str(geojson_2000))
        gdf_2000["GEOID10"] = gdf_2000["GEOID10"].astype(str)
        logger.info(f"Loaded 2000 census data: {len(gdf_2000)} tracts, "
                     f"total pop: {gdf_2000['total_pop'].sum():,}")

        # --- Classify neighborhood_type based on 2000 racial composition ---
        def classify_neighborhood(row):
            pct_w = row.get("pct_nh_white", 0) or 0
            pct_b = row.get("pct_nh_black", 0) or 0
            pct_h = row.get("pct_hispanic", 0) or 0
            pct_a = row.get("pct_nh_asian", 0) or 0
            if pct_b >= 50:
                return "black_dominant"
            elif pct_h >= 50:
                return "hispanic_dominant"
            elif pct_w >= 50:
                return "white_dominant"
            elif pct_a >= 25:
                return "asian_enclave"
            else:
                return "mixed"

        gdf_2000["neighborhood_type"] = gdf_2000.apply(classify_neighborhood, axis=1)

        # --- Classify income brackets based on 2000 per_capita_income ---
        # Use quartiles of the 2000 income distribution
        income = gdf_2000["per_capita_income"].dropna()
        q25, q50, q75 = income.quantile([0.25, 0.5, 0.75])

        def classify_income(pci):
            if pd.isna(pci):
                return "lower_middle"
            if pci <= q25:
                return "low"
            elif pci <= q50:
                return "lower_middle"
            elif pci <= q75:
                return "upper_middle"
            else:
                return "high"

        gdf_2000["income_bracket"] = gdf_2000["per_capita_income"].apply(classify_income)

        # --- Family type splits (consistent across years) ---
        # Standard CPS-era household composition: ~35% families w/children,
        # ~40% singles/couples no children, ~25% elderly
        family_splits = {
            "family_with_children": 0.35,
            "single_no_children": 0.40,
            "elderly": 0.25,
        }

        # --- Build mapping rows ---
        RACE_COLS = ["nh_white", "nh_black", "nh_asian", "hispanic", "nh_other"]
        rows = []

        for _, tract in gdf_2000.iterrows():
            geoid = str(tract["GEOID10"])
            if geoid not in self.tract_ids:
                continue

            nbhd_type = tract["neighborhood_type"]
            income_br = tract["income_bracket"]

            for race in RACE_COLS:
                race_pop = int(tract.get(race, 0) or 0)
                if race_pop <= 0:
                    continue

                for fam_type, fam_frac in family_splits.items():
                    pop = max(1, round(race_pop * fam_frac))
                    archetype_key = f"{race}__{income_br}__{fam_type}__{nbhd_type}"
                    rows.append({
                        "GEOID10": geoid,
                        "archetype_key": archetype_key,
                        "population": pop,
                    })

        mapping = pd.DataFrame(rows)
        logger.info(f"Built 2000 tract mapping: {len(mapping)} entries, "
                     f"{mapping['archetype_key'].nunique()} unique archetypes, "
                     f"total pop: {mapping['population'].sum():,}")

        return mapping

    @staticmethod
    def _find_closest_archetype(key: str, archetypes: dict) -> str | None:
        """
        Find the closest matching archetype for a key not in the archetypes dict.

        Matching priority: same (race, income, family), any neighborhood_type.
        Falls back to same (race, family), any income/neighborhood.
        """
        parts = key.split("__")
        if len(parts) != 4:
            return None
        race, income, family, nbhd = parts

        # Priority 1: same race + income + family, different neighborhood
        for akey in archetypes:
            ap = akey.split("__")
            if len(ap) == 4 and ap[0] == race and ap[1] == income and ap[2] == family:
                return akey

        # Priority 2: same race + family, different income/neighborhood
        for akey in archetypes:
            ap = akey.split("__")
            if len(ap) == 4 and ap[0] == race and ap[2] == family:
                return akey

        # Priority 3: same race, any match
        for akey in archetypes:
            ap = akey.split("__")
            if len(ap) == 4 and ap[0] == race:
                return akey

        return None

    def _create_agents(self, archetypes: dict, tract_mapping: pd.DataFrame,
                       max_archetypes: int | None):
        """
        Create one HouseholdAgent per (archetype, tract) row in the mapping.

        Each agent's `pop_count` is the ORIGINAL Census population for that
        (archetype, tract) cell — no downsampling, no `max(1, round(...))`
        flooring. This guarantees that the agent-scale pre-perturbation
        demographic distribution exactly matches the Census 2010 ground truth
        (modulo the archetype × tract discretization already baked into the
        mapping).

        `self.n_agents` is overwritten after creation with the actual number
        of agents produced (= number of active mapping rows, post filters).
        The original `--agents` CLI value is ignored for calibration; it no
        longer drives a separate downsample step.

        `max_archetypes`, if set, still caps the number of distinct archetypes
        by global population (useful for full-city runs where the tail of
        rare archetypes can be dropped). It does NOT alter per-cell pop_count.
        """
        # Group by archetype to find active archetypes in our tracts
        active_keys = tract_mapping["archetype_key"].unique()

        # If max_archetypes specified, keep the most populous ones only
        if max_archetypes and len(active_keys) > max_archetypes:
            arch_pops = tract_mapping.groupby("archetype_key")["population"].sum()
            arch_pops = arch_pops.sort_values(ascending=False)
            active_keys = arch_pops.head(max_archetypes).index.tolist()
            dropped = len(tract_mapping) - tract_mapping["archetype_key"].isin(active_keys).sum()
            tract_mapping = tract_mapping[tract_mapping["archetype_key"].isin(active_keys)]
            logger.info(f"max_archetypes={max_archetypes} filter dropped "
                         f"{dropped} mapping rows (tail archetypes)")

        # Build fallback mapping for keys not in archetypes dict (e.g. census_2000 keys)
        fallback_map = {}
        missing_keys = set()

        self.active_archetypes = {}
        agent_count = 0
        total_pop_realized = 0

        for _, row in tract_mapping.iterrows():
            key = row["archetype_key"]
            geoid = str(row["GEOID10"])
            pop = int(row["population"])

            if geoid not in self.tract_ids:
                continue
            if pop <= 0:
                continue

            # Resolve archetype: direct match or fallback
            resolved_key = key
            if key not in archetypes:
                if key not in fallback_map:
                    closest = self._find_closest_archetype(key, archetypes)
                    fallback_map[key] = closest
                    if closest is None:
                        missing_keys.add(key)
                resolved_key = fallback_map.get(key)
                if resolved_key is None:
                    continue

            # Get or create archetype params
            # Use the original key as the agent's archetype key (preserves 2000 identity)
            # but clone parameters from the resolved archetype
            if key not in self.active_archetypes:
                source = archetypes[resolved_key]
                if resolved_key == key:
                    self.active_archetypes[key] = source
                else:
                    # Clone with the original key but source's behavioral parameters
                    self.active_archetypes[key] = ArchetypeParams(
                        archetype_key=key,
                        race=source.race,
                        income_bracket=source.income_bracket,
                        family_type=source.family_type,
                        neighborhood_type=key.split("__")[3] if len(key.split("__")) == 4 else source.neighborhood_type,
                        population=source.population,
                        mobility_rate=source.mobility_rate,
                        context_satisfaction_bonus=source.context_satisfaction_bonus,
                        ideal_own_group_pct=source.ideal_own_group_pct,
                        min_own_group_pct=source.min_own_group_pct,
                        search_n_mean=source.search_n_mean,
                        search_radius_hops=source.search_radius_hops,
                        search_bias_own=source.search_bias_own,
                        search_bias_mixed=source.search_bias_mixed,
                        social_anchor_weight=source.social_anchor_weight,
                        satisfaction_weights=dict(source.satisfaction_weights),
                    )

            # Split this (archetype, tract) cell into one or more clones,
            # each carrying at most `max_pop_per_agent` people.
            # Largest-remainder allocation ensures the sum of clone
            # pop_counts equals `pop` exactly (no rounding drift).
            cap = max(1, int(self.max_pop_per_agent))
            n_clones = max(1, (pop + cap - 1) // cap)
            base = pop // n_clones
            remainder = pop - base * n_clones
            for i in range(n_clones):
                clone_pop = base + (1 if i < remainder else 0)
                if clone_pop <= 0:
                    continue
                HouseholdAgent(
                    model=self,
                    archetype=self.active_archetypes[key],
                    tract_id=geoid,
                    pop_count=clone_pop,
                )
                agent_count += 1
            total_pop_realized += pop

        if fallback_map:
            n_fallback = sum(1 for v in fallback_map.values() if v is not None)
            logger.info(f"Archetype fallback: {n_fallback} keys mapped to closest match, "
                         f"{len(missing_keys)} keys unresolvable (skipped)")

        # Overwrite n_agents with the actual count so downstream code
        # (logging, metrics, etc.) reflects reality.
        if self.n_agents != agent_count:
            logger.info(f"n_agents overridden from CLI value {self.n_agents} "
                         f"to actual mapping-derived count {agent_count}")
            self.n_agents = agent_count

        logger.info(
            f"Created {agent_count} agents (max {self.max_pop_per_agent} pop/agent, "
            f"avg {total_pop_realized/max(agent_count,1):.1f}) "
            f"across {len(self.active_archetypes)} archetypes, "
            f"{len(self.tract_ids)} tracts; total pop = {total_pop_realized:,}"
        )

    def _apply_init_mode(self, mode: str, tract_mapping: pd.DataFrame):
        """
        Redistribute agents across tracts according to the initialization mode.

        This is called AFTER agents are created at their Census positions,
        then shuffles them to test whether segregation can emerge or re-emerge.

        Modes:
        - "random": Uniformly random placement across all tracts.
          Each agent is assigned to a random tract. Environment dynamic_pop
          is rebuilt from scratch based on new agent positions.
          Expected initial D ≈ 0 (perfect integration).

        - "semi_random": Shuffle within Community Area clusters.
          Agents are only moved to other tracts within the same Community Area.
          Preserves broad spatial patterns but removes micro-level segregation.
          Expected initial D: somewhere between 0 and census value.

        - "inverted": Swap racial distributions - minorities go to historically
          White tracts and vice versa. Tests how quickly the system re-segregates.
          Expected initial D: high but inverted spatial pattern.
        """
        rng = self.random
        all_tracts = list(self.tract_ids)
        agent_list = list(self.agents)

        logger.info(f"Applying init_mode='{mode}' to {len(agent_list)} agents "
                     f"across {len(all_tracts)} tracts...")

        if mode == "random":
            # Uniformly random: each agent goes to a random tract
            for agent in agent_list:
                new_tract = rng.choice(all_tracts)
                agent.tract_id = new_tract

        elif mode == "semi_random":
            # Shuffle within Community Area
            # Build CA -> tract mapping
            gdf = self.environment.gdf
            ca_tracts = defaultdict(list)
            for geoid in all_tracts:
                ca = gdf.loc[geoid].get("community_area_num", 0)
                ca_tracts[ca].append(geoid)

            # For each agent, find its current CA and shuffle within it
            tract_to_ca = {}
            for ca, tracts in ca_tracts.items():
                for t in tracts:
                    tract_to_ca[t] = ca

            for agent in agent_list:
                ca = tract_to_ca.get(agent.tract_id)
                if ca is not None and len(ca_tracts[ca]) > 1:
                    new_tract = rng.choice(ca_tracts[ca])
                    agent.tract_id = new_tract

        elif mode == "inverted":
            # Classify tracts by dominant race
            white_tracts = []
            minority_tracts = []
            for geoid in all_tracts:
                info = self.environment.get_tract_info(geoid)
                if info.get("pct_nh_white", 0) >= 50:
                    white_tracts.append(geoid)
                else:
                    minority_tracts.append(geoid)

            # Swap: White agents -> minority tracts, minority agents -> White tracts
            for agent in agent_list:
                if agent.race == "nh_white":
                    if minority_tracts:
                        agent.tract_id = rng.choice(minority_tracts)
                else:
                    if white_tracts:
                        agent.tract_id = rng.choice(white_tracts)

        elif mode == "perturbed":
            # Perturbation recovery: move a fraction of agents to tracts where
            # their race is NOT dominant, reducing segregation in a controlled way.
            # The model should then "recover" back toward the real 2010 pattern.
            pct = self.perturb_pct

            # Classify tracts by dominant race
            tract_by_dominant = defaultdict(list)  # dominant_race -> [tract_ids]
            for geoid in all_tracts:
                info = self.environment.get_tract_info(geoid)
                best_race = max(
                    ("nh_white", "nh_black", "nh_asian", "hispanic"),
                    key=lambda r: info.get(f"pct_{r}", 0),
                )
                tract_by_dominant[best_race].append(geoid)

            # Build "foreign tracts" for each race: tracts NOT dominated by that race
            foreign_tracts = {}
            for race in ("nh_white", "nh_black", "nh_asian", "hispanic", "nh_other"):
                foreign_tracts[race] = [
                    t for t in all_tracts if t not in tract_by_dominant.get(race, [])
                ]
                # Fallback: if all tracts are same-race dominant, use all tracts
                if not foreign_tracts[race]:
                    foreign_tracts[race] = all_tracts

            # Select agents to perturb
            n_perturb = max(1, round(len(agent_list) * pct))
            perturb_agents = rng.sample(agent_list, min(n_perturb, len(agent_list)))

            for agent in perturb_agents:
                targets = foreign_tracts[agent.race]
                agent.tract_id = rng.choice(targets)

            logger.info(f"Perturbed {len(perturb_agents)}/{len(agent_list)} agents "
                         f"({pct*100:.0f}%) to foreign-race tracts")

        elif mode == "perturbed_n":
            # Like "perturbed" but destinations are restricted to foreign-race
            # tracts within perturb_radius_hops of each agent's ORIGINAL tract.
            # Keeps recovery reachable under the per-step hop-limited search.
            pct = self.perturb_pct
            hops = self.perturb_radius_hops

            # Classify tracts by dominant race
            dominant_race_by_tract: dict[str, str] = {}
            for geoid in all_tracts:
                info = self.environment.get_tract_info(geoid)
                dominant_race_by_tract[geoid] = max(
                    ("nh_white", "nh_black", "nh_asian", "hispanic"),
                    key=lambda r: info.get(f"pct_{r}", 0),
                )

            # Global foreign-tracts fallback (used if local neighborhood has
            # no foreign-race tract within the given radius)
            global_foreign: dict[str, list[str]] = {}
            for race in ("nh_white", "nh_black", "nh_asian", "hispanic", "nh_other"):
                global_foreign[race] = [
                    t for t in all_tracts
                    if dominant_race_by_tract.get(t) != race
                ] or all_tracts

            # Select agents to perturb
            n_perturb = max(1, round(len(agent_list) * pct))
            perturb_agents = rng.sample(agent_list, min(n_perturb, len(agent_list)))

            n_local = 0
            n_fallback = 0
            for agent in perturb_agents:
                neighbors = self.environment.get_neighbors(agent.tract_id, hops)
                local_foreign = [
                    t for t in neighbors
                    if dominant_race_by_tract.get(t) != agent.race
                ]
                if local_foreign:
                    agent.tract_id = rng.choice(local_foreign)
                    n_local += 1
                else:
                    # No foreign-race tract within radius — fall back to
                    # the global foreign pool (keeps the perturbation target
                    # rate honest even for isolated agents).
                    agent.tract_id = rng.choice(global_foreign[agent.race])
                    n_fallback += 1

            logger.info(
                f"Perturbed {len(perturb_agents)}/{len(agent_list)} agents "
                f"({pct*100:.0f}%) within {hops}-hop neighborhoods "
                f"[local={n_local}, global_fallback={n_fallback}]"
            )

        # Rebuild environment dynamic_pop from new agent positions
        self.environment.rebuild_dynamic_pop_from_agents(agent_list)

        logger.info(f"Init mode '{mode}' applied. Agent positions redistributed.")

    def step(self):
        """
        Execute one simulation step:
        1. LLM satisfaction assessment for each active archetype
        2. For dissatisfied archetypes: LLM evaluates candidate neighborhoods
        2.5  Budget gate: cap total pop_count that may move this step
        3. Agents execute move/stay decisions
        4. Compute metrics + direction indicator
        """
        current_step = len(self.metrics_history)
        logger.info(f"=== Step {current_step} ===")

        # Reset per-step inflow tracker used by the (tract, race) inflow cap.
        # Any mover admitted this step increments _step_inflow[(dst, race)];
        # further movers targeting the same (dst, race) are rejected once
        # the tract-specific cap is reached.  This is a pure rate limit on
        # single-step demographic shock — no GT is consulted.
        self._step_inflow = defaultdict(int)

        # Track LLM calls for this step
        llm_call_id_before = self.llm.get_current_call_id()

        # Snapshot the racial distribution BEFORE this step's moves, used
        # by the direction indicator to measure gap reduction toward GT.
        prev_race_df = self.environment.get_racial_composition_array().copy()

        # Phase 1: Archetype-level satisfaction assessment via LLM
        self._assess_archetype_satisfaction()

        # Phase 2: For dissatisfied archetypes, evaluate candidate destinations
        self._evaluate_move_candidates()

        # Phase 2.5: Budget gate — cap per-step relocation volume
        self._apply_move_budget()

        # Phase 3: Agents execute decisions
        self.agents.shuffle_do("step")

        # Phase 4: Compute metrics + direction indicator
        metrics = self._compute_metrics()
        metrics["step"] = current_step

        # Count movers
        movers = [a for a in self.agents if a.moved_this_step]
        metrics["n_movers"] = len(movers)
        metrics["pct_movers"] = len(movers) / len(self.agents) if self.agents else 0
        metrics["pop_moved"] = sum(a.pop_count for a in movers)
        metrics["pct_pop_moved"] = (
            metrics["pop_moved"] / sum(a.pop_count for a in self.agents)
            if self.agents else 0
        )

        # Direction indicator: gap reduction vs Census 2010 ground truth.
        # See _compute_direction_metrics for definitions.
        direction = self._compute_direction_metrics(prev_race_df, movers)
        metrics.update(direction)

        self.metrics_history.append(metrics)

        # Collect per-step LLM call records
        step_llm_records = self.llm.get_call_log_since(llm_call_id_before)

        # Log LLM stats
        llm_stats = self.llm.get_stats()
        step_info = {
            "step": current_step,
            "movers": len(movers),
            "llm_calls_this_step": len(step_llm_records),
            "llm_calls_total": llm_stats["total_calls"],
            "D_black_white": metrics["D_black_white"],
            "agent_snapshots": self.get_agent_snapshot(),
            "llm_calls": self.get_llm_log_dicts(step_llm_records),
        }
        self.step_log.append(step_info)
        logger.info(
            f"Step {current_step}: {len(movers)} movers "
            f"({metrics['pct_pop_moved']*100:.1f}% of pop), "
            f"D_bw={metrics['D_black_white']:.4f}, "
            f"gap_reduction={metrics.get('gap_reduction_pct', 0.0):+.2f}% "
            f"(aligned_movers={metrics.get('mover_alignment_rate', 0.0)*100:.1f}%), "
            f"LLM calls total={llm_stats['total_calls']}"
        )

    def _assess_archetype_satisfaction(self):
        """
        Assess satisfaction per (archetype, tract) group via LLM.

        Uses batch async calls: all (archetype, tract) groups that need
        LLM evaluation are collected first, then sent concurrently.

        For cost control, we deduplicate: same archetype in same tract = one call.
        Model Cascading: re-evaluate every 3 steps even if previously satisfied.
        """
        current_step = len(self.metrics_history)

        # Group agents by (archetype_key, tract_id)
        groups = defaultdict(list)
        for agent in self.agents:
            groups[(agent.archetype_key, agent.tract_id)].append(agent)

        # Separate into skipped (cascading) and needs-LLM groups
        llm_tasks = []  # List of {archetype_desc, neighborhood_desc, archetype_key, tract_id}
        llm_task_keys = []  # Parallel list of (key, tract_id) for mapping results back

        for (key, tract_id), agents_in_group in groups.items():
            archetype = self.active_archetypes[key]

            # Model Cascading: skip if satisfied, but re-evaluate every 3 steps
            if (archetype.current_satisfaction >= 7.0
                    and not archetype.would_move
                    and current_step % 3 != 0):
                for agent in agents_in_group:
                    agent.satisfaction = archetype.current_satisfaction
                # Carry forward: this group doesn't want to move
                self.group_would_move[(key, tract_id)] = False
                continue

            llm_tasks.append({
                "archetype_desc": archetype.describe_for_llm(),
                "neighborhood_desc": self.environment.describe_tract_with_context_for_llm(tract_id),
                "archetype_key": key,
                "tract_id": tract_id,
            })
            llm_task_keys.append((key, tract_id))

        if not llm_tasks:
            return

        # Batch async LLM calls
        logger.info(f"  Satisfaction: {len(llm_tasks)} LLM calls (async batch)...")
        results = self.llm.batch_assess_satisfaction(llm_tasks)

        # Apply results back to agents and store per-group would_move
        archetype_results = defaultdict(list)

        for (key, tract_id), result in zip(llm_task_keys, results):
            agents_in_group = groups[(key, tract_id)]

            sat = max(0.0, min(10.0, float(result.get("satisfaction", 5.0))))
            would_move = bool(result.get("would_move", False))
            reasons = result.get("key_reasons", [])

            for agent in agents_in_group:
                agent.satisfaction = sat

            # Store per-(archetype, tract) move intent
            self.group_would_move[(key, tract_id)] = would_move

            archetype_results[key].append({
                "satisfaction": sat,
                "would_move": would_move,
                "reasons": reasons,
                "n_agents": len(agents_in_group),
            })

            logger.debug(
                f"Archetype {key} @ {tract_id}: satisfaction={sat:.1f}, "
                f"would_move={would_move}, reasons={reasons}"
            )

        # Consolidate to archetype level: use population-weighted average
        for key, results in archetype_results.items():
            archetype = self.active_archetypes[key]
            total_agents = sum(r["n_agents"] for r in results)
            avg_sat = sum(r["satisfaction"] * r["n_agents"] for r in results) / total_agents
            any_would_move = any(r["would_move"] for r in results)

            archetype.current_satisfaction = avg_sat
            archetype.would_move = any_would_move
            archetype.key_reasons = results[0]["reasons"]

    def _evaluate_move_candidates(self):
        """
        For dissatisfied (archetype, tract) groups, evaluate candidate
        neighborhoods via LLM.

        Evaluates per (archetype_key, tract_id) — each agent group searches
        neighbors from its OWN tract rather than sharing one representative.
        Results are stored in self.move_targets[(key, tract_id)].
        """
        # Reset move targets
        self.move_targets.clear()

        # Group agents by (archetype_key, tract_id)
        groups = defaultdict(list)
        for agent in self.agents:
            groups[(agent.archetype_key, agent.tract_id)].append(agent)

        # Phase 1: Collect all move evaluation tasks
        move_tasks = []
        move_task_keys = []  # (archetype_key, tract_id)

        for (key, tract_id), agents_in_group in groups.items():
            archetype = self.active_archetypes[key]

            # Skip if this specific group doesn't want to move
            if not self.group_would_move.get((key, tract_id), archetype.would_move):
                continue

            # Get candidate tracts from THIS agent group's actual tract
            candidates = self.environment.get_neighbors(
                tract_id, archetype.search_radius_hops
            )

            if not candidates:
                continue

            # Limit candidates based on archetype search parameters
            n_consider = min(archetype.search_n_mean, len(candidates))
            candidates = self._bias_candidate_selection(
                candidates, archetype, n_consider, current_tract_id=tract_id
            )
            if not candidates:
                continue

            # Capacity pre-filter: drop destinations that cannot physically
            # absorb the largest agent in this (archetype, tract) group.
            # This prevents the LLM from picking a full tract only for the
            # agent to fail to move at execution time.  We use the MAX
            # pop_count so the filter reflects the most demanding clone;
            # smaller clones of the same group may still find other means
            # of moving in future steps once capacity opens up.
            max_pop_in_group = max(
                (a.pop_count for a in agents_in_group), default=1
            )
            feasible = [
                c for c in candidates
                if self.environment.has_capacity(c, max_pop_in_group)
            ]
            if not feasible:
                # All neighbors are full — no feasible move this step.
                continue
            candidates = feasible

            # Build descriptions from the actual tract
            current_desc = self.environment.describe_tract_with_context_for_llm(tract_id)
            candidate_descs = [self.environment.describe_tract_with_context_for_llm(cid) for cid in candidates]
            candidates_text = "\n\n".join(candidate_descs)

            move_tasks.append({
                "archetype_desc": archetype.describe_for_llm(),
                "current_desc": current_desc,
                "candidates_desc": candidates_text,
                "archetype_key": key,
                "tract_id": tract_id,
            })
            move_task_keys.append((key, tract_id))

        if not move_tasks:
            return

        # Phase 2: Batch async LLM calls
        logger.info(f"  Move eval: {len(move_tasks)} LLM calls (async batch)...")
        results = self.llm.batch_evaluate_move(move_tasks)

        # Phase 3: Store results per (archetype, tract)
        # Response format (new): ranked_candidates = [{tract_id, score}, ...]
        # Fallback (legacy): best_tract + best_score
        for (key, tract_id), result in zip(move_task_keys, results):
            stay_score = float(result.get("stay_score", 5))
            ranked_list = result.get("ranked_candidates")

            parsed_rankings: list[dict] = []
            if isinstance(ranked_list, list):
                # New top-k format
                for entry in ranked_list[:3]:
                    tid = entry.get("tract_id")
                    try:
                        sc = float(entry.get("score", 0))
                    except (TypeError, ValueError):
                        sc = 0.0
                    if tid and sc > stay_score:
                        parsed_rankings.append({"tract_id": tid, "score": sc})
            else:
                # Legacy single-best format (backward compat for older responses)
                best_tract = result.get("best_tract")
                try:
                    best_score = float(result.get("best_score", 0))
                except (TypeError, ValueError):
                    best_score = 0.0
                if best_tract and best_score > stay_score:
                    parsed_rankings.append({"tract_id": best_tract, "score": best_score})

            self.move_targets[(key, tract_id)] = parsed_rankings

        # Update archetype-level move_rankings summary (for output/logging)
        for key, archetype in self.active_archetypes.items():
            group_rankings = [
                self.move_targets[k]
                for k in self.move_targets
                if k[0] == key and self.move_targets[k]
            ]
            if group_rankings:
                # Pick the most common target as archetype summary
                archetype.move_rankings = group_rankings[0]
            else:
                archetype.move_rankings = []

    def _bias_candidate_selection(self, candidates: list[str],
                                  archetype: ArchetypeParams,
                                  n: int,
                                  current_tract_id: str | None = None) -> list[str]:
        """
        Select n candidates from geographic neighbors with an own-race ceiling.

        Strategy:
        1. Filter out tracts whose own-race share exceeds
           ``current_tract.pct_own_race * own_race_ceiling_ratio``.
           This prevents feeding the LLM a candidate set dominated by
           more-segregated same-race enclaves and is a DIRECTIONAL fix for
           the "LLM drifts toward runaway segregation" problem.
        2. If the ceiling filter wipes the candidate set (no safer options
           nearby), fall back to the unfiltered set so we never strand the
           household — the LLM's own prompt constraints will handle the
           final call.
        3. Uniformly random-sample ``n`` from whatever survives.

        Args:
            candidates: tract ids returned by the environment's neighbor query.
            archetype: the household archetype (we use its `race`).
            n: number to return.
            current_tract_id: the mover's current tract, used to compute the
                ceiling.  If None, the ceiling is skipped.
        """
        if not candidates:
            return []

        ratio = self.own_race_ceiling_ratio
        own_race = archetype.race

        # City-wide share of this race (percentage points).  Used to scale
        # the ceiling floor so that minority races cannot legally step
        # into a +40pp concentrated tract in a single move (which was the
        # original runaway channel for D_asian_white).
        city_share_pct = 100.0 * self.city_race_share.get(own_race, 0.05)

        if current_tract_id is not None and ratio is not None and ratio > 0:
            cur_info = self.environment.get_tract_info(current_tract_id)
            cur_share = cur_info.get(f"pct_{own_race}", 0.0)
            # Race-aware ceiling floor:
            #   floor_pp ≈ 1.5 × city_share_pct, clamped to [3, 40].
            # So for NH Asian (city share ≈ 5.4%) the floor is ≈ 8pp —
            # an Asian agent in a 2% Asian tract may move to at most a
            # ≈10% Asian tract.  For NH White (≈31.7%) the floor stays
            # at 40pp — majority recovery is unaffected.
            # If the agent is CURRENTLY at less than half the city share
            # of its own race, relax the floor (up to the 40pp cap) so
            # perturbed minority agents can still find recovery paths.
            floor_pp = max(3.0, min(40.0, 1.5 * city_share_pct))
            if cur_share < 0.5 * city_share_pct:
                floor_pp = min(40.0, floor_pp * 2.0)
            ceiling = max(cur_share * ratio, cur_share + floor_pp)

            filtered = []
            for cid in candidates:
                cand_info = self.environment.get_tract_info(cid)
                cand_share = cand_info.get(f"pct_{own_race}", 0.0)
                if cand_share <= ceiling:
                    filtered.append(cid)

            # Fall back to unfiltered if ceiling eliminated everything.
            if filtered:
                candidates = filtered

        if len(candidates) <= n:
            return candidates

        # Inverse-frequency weighted sample (Efraimidis–Spirakis) to
        # down-weight candidate tracts whose current own-race share is
        # already high.  This breaks the "always sample toward the
        # nearest enclave" bias of uniform random sampling WITHOUT
        # looking at any GT target.  Uses live tract state (pct_own_race)
        # only.
        bias = self.enclave_candidate_bias
        if bias is None or bias <= 0:
            return self.random.sample(candidates, n)

        import math
        weighted = []
        for cid in candidates:
            cand_share_frac = (
                self.environment.get_tract_info(cid).get(f"pct_{own_race}", 0.0)
                / 100.0
            )
            # w = 1 / (bias + share); smaller bias → stronger downweight.
            w = 1.0 / (bias + max(0.0, cand_share_frac))
            u = self.random.random()
            if u <= 0.0:
                u = 1e-12
            key = math.log(u) / w
            weighted.append((key, cid))
        # Highest key wins (Efraimidis–Spirakis).
        weighted.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in weighted[:n]]

    def get_step_inflow_cap(self, tract_id: str) -> int:
        """
        Per-step single-(tract, race) inflow cap, in pop_count units.

        Formula: ``max(max_pop_per_agent, ratio × tract_capacity)``.  The
        lower bound guarantees at least one max-sized clone can enter a
        tract per step per race (otherwise the cap would deadlock
        recovery when tract capacity is small).  The upper bound scales
        with tract size so large tracts don't become artificially
        bottlenecked.

        Rate limit only — uses tract_capacity (a physical housing-stock
        proxy set at init time) and is race-blind per tract; it does not
        consult Census GT.
        """
        ratio = self.per_step_tract_inflow_cap_ratio
        if ratio is None or ratio <= 0:
            return 10**9  # effectively unlimited
        cap = self.environment.tract_capacity.get(tract_id)
        if cap is None or cap <= 0:
            return 10**9
        return max(int(self.max_pop_per_agent), int(ratio * cap))

    def _move_priority(self, agent) -> float:
        """
        Rank-score for admitting movers under the per-race budget.

        Blends two signals on a common 0–10 scale:
          * Desire to move:      ``10 - satisfaction``
          * Economic capacity:   ``INCOME_PRIORITY_SCORE[income_bracket]``

        ``priority = (1 - w) * dissat + w * income_score``

        Higher priority → admitted first.  Models the real-world dynamic
        that many 2010 Census residents were dissatisfied but lacked the
        economic means to actually relocate; among competitors for a
        single scarce slot in a desirable tract, the wealthier (and still
        dissatisfied enough) household wins.  When w=0 we recover the
        legacy "pure dissatisfaction" ordering; at w=1 we rank purely by
        income.
        """
        w = self.move_priority_income_weight
        dissat = 10.0 - float(agent.satisfaction)
        income_score = INCOME_PRIORITY_SCORE.get(
            agent.archetype.income_bracket, 5.0
        )
        return (1.0 - w) * dissat + w * income_score

    def _apply_move_budget(self):
        """
        Cap the per-race pop_count that may relocate this step (agent-level).

        Uses a **per-race quota**: each race gets its own budget of
        ``move_budget_pct × race_pop``.  This prevents a single race whose
        LLM satisfaction happens to be low (e.g. Asian agents with
        avg_sat=3.4) from monopolizing the global budget and shutting out
        other races whose recovery is equally important (e.g. Black
        agents with avg_sat=4.8).

        Within each race we sort candidate agents by descending
        ``_move_priority`` — a weighted blend of dissatisfaction and
        income — and admit until that race's budget is exhausted.
        Admitted agents have ``_allowed_to_move = True``; everyone else
        stays this step.  Clones of the same cell compete individually
        against the race quota (see `max_pop_per_agent`).
        """
        # Reset flags every step
        for agent in self.agents:
            agent._allowed_to_move = False

        if self.move_budget_pct <= 0 or self.move_budget_pct >= 1.0:
            for agent in self.agents:
                agent._allowed_to_move = True
            return

        # Bucket agents by race and compute per-race populations
        by_race: dict[str, list] = defaultdict(list)
        pop_by_race: dict[str, int] = defaultdict(int)
        for agent in self.agents:
            pop_by_race[agent.race] += agent.pop_count
            targets = self.move_targets.get((agent.archetype_key, agent.tract_id), [])
            if targets:
                by_race[agent.race].append(agent)

        if not by_race:
            return

        total_admitted_pop = 0
        total_candidates = 0
        per_race_stats: list[str] = []

        for race, candidates in by_race.items():
            total_candidates += len(candidates)
            race_pop = pop_by_race.get(race, 0)
            if race_pop <= 0 or not candidates:
                continue

            # Race-size-aware budget.  Rationale: the raw rule
            # (budget = move_budget_pct × race_pop) is fair in per-race
            # mobility rate, but it is *unfair* in per-race D impact —
            # a single Asian mover nudges D_aw roughly 6× more than a
            # single Black mover because Asian is ~6× smaller in Chicago.
            # We scale the budget by min(1, sqrt(city_share /
            # median_city_share)) so majority races (Hispanic/Black/White
            # at or above the median) keep the full 5% budget, while
            # smaller races (Asian, Other) get a reduced budget that
            # matches their D-sensitivity.  Uses city-wide shares only
            # (public 2010 demographic knowledge) — NOT GT targets.
            if self.race_size_budget_scaling:
                import math
                share = self.city_race_share.get(race, 0.0)
                if share > 0 and self.city_median_race_share > 0:
                    scale = min(
                        1.0,
                        math.sqrt(share / self.city_median_race_share),
                    )
                else:
                    scale = 1.0
            else:
                scale = 1.0
            race_budget = self.move_budget_pct * race_pop * scale

            # Shuffle to randomize ties, then stable-sort by move-priority
            # (higher priority first = lower satisfaction and/or higher
            # income, per self.move_priority_income_weight).
            self.random.shuffle(candidates)
            candidates.sort(key=self._move_priority, reverse=True)

            cum = 0.0
            admitted = 0
            for agent in candidates:
                # Always admit the first (most dissatisfied) to avoid a
                # single large agent deadlocking this race's recovery.
                if admitted == 0 or cum + agent.pop_count <= race_budget:
                    agent._allowed_to_move = True
                    cum += agent.pop_count
                    admitted += 1
                # Keep iterating — small agents later may still fit.

            total_admitted_pop += cum
            per_race_stats.append(
                f"{race}={admitted}/{len(candidates)}"
                f"({cum/race_pop*100:.1f}%,×{scale:.2f})"
            )

        total_pop = sum(pop_by_race.values())
        logger.info(
            f"  Budget gate (per-race): admitted {total_admitted_pop:,.0f}/{total_pop:,} "
            f"({total_admitted_pop/max(total_pop,1)*100:.1f}% of pop); "
            + " ".join(per_race_stats)
        )

    def _compute_direction_metrics(self, prev_race_df, movers) -> dict:
        """
        Measure whether this step's moves push the system TOWARD Census GT.

        We compare the racial composition before and after this step against
        the Census 2010 baseline (tract-scale ground truth).  Two indicators:

        gap_reduction_pct:
            ``(gap_before - gap_after) / gap_before * 100``
            where gap is ``Σ |race_count_tract - census_target_tract|``
            summed across every (tract, race) cell.
            Positive → the step recovered toward GT.
            Negative → the step drifted further away.
            0       → no net change (e.g. in census mode with no movers).

        mover_alignment_rate:
            Fraction of individual movers whose specific move reduced the
            2-tract (source, destination) × own-race absolute deviation.
            High rate → most movers are "filling gaps" in a recovery sense.
            Low rate → most movers are "over-shooting" / racially sorting.

        Both rely on ``self.census_2010_baseline`` being set (it always is
        after __init__).
        """
        out = {
            "gap_before": 0.0,
            "gap_after": 0.0,
            "gap_reduction_pct": 0.0,
            "mover_alignment_rate": 0.0,
        }
        if self.census_2010_baseline is None:
            return out

        race_cols = ["nh_white", "nh_black", "nh_asian", "hispanic", "nh_other"]
        target = self.census_2010_baseline.set_index("GEOID10")[race_cols]
        pre = prev_race_df.set_index("GEOID10")[race_cols]
        post = self.environment.get_racial_composition_array().set_index("GEOID10")[race_cols]

        # Align on shared index (should already match)
        idx = target.index.intersection(pre.index).intersection(post.index)
        target = target.loc[idx]
        pre = pre.loc[idx]
        post = post.loc[idx]

        gap_before = float((pre - target).abs().values.sum())
        gap_after = float((post - target).abs().values.sum())
        out["gap_before"] = round(gap_before, 1)
        out["gap_after"] = round(gap_after, 1)
        out["gap_reduction_pct"] = (
            round((gap_before - gap_after) / gap_before * 100, 3)
            if gap_before > 0 else 0.0
        )

        # Per-mover alignment: did each mover's own race move "closer to GT"
        # at both source (by leaving) and destination (by arriving)?
        if movers:
            aligned = 0
            for a in movers:
                race = a.race
                src = getattr(a, "_prev_tract", None)
                dst = a.tract_id
                if src is None or src == dst or src not in pre.index or dst not in pre.index:
                    continue
                # Deviation at source for this race BEFORE step: excess if > 0
                dev_src_before = pre.loc[src, race] - target.loc[src, race]
                dev_dst_before = pre.loc[dst, race] - target.loc[dst, race]
                # Leaving a tract where race was OVER-represented (dev>0) helps;
                # arriving where race was UNDER-represented (dev<0) also helps.
                # Either direction individually counts as recovery-aligned.
                score = (1 if dev_src_before > 0 else 0) + (1 if dev_dst_before < 0 else 0)
                if score >= 1:
                    aligned += 1
            out["mover_alignment_rate"] = round(aligned / len(movers), 3)

        return out

    def _compute_metrics(self) -> dict:
        """Compute current segregation metrics from dynamic population."""
        race_df = self.environment.get_racial_composition_array()
        metrics = compute_all_metrics(race_df)

        # Add R² recovery metric (how close current distribution is to 2010 baseline)
        if self.census_2010_baseline is not None:
            r2 = recovery_r_squared(race_df, self.census_2010_baseline)
            metrics["R2_recovery"] = r2

        return metrics

    def get_results_summary(self) -> str:
        """Generate a human-readable summary of the simulation results."""
        if not self.metrics_history:
            return "No simulation steps completed."

        initial = self.metrics_history[0]
        final = self.metrics_history[-1]
        llm_stats = self.llm.get_stats()

        # Four reference columns:
        #   Census GT = actual Census 2010 on these tracts (tract-scale, ground truth)
        #   PrePerturb = agent-scale representation of Census 2010 (what recovery should target)
        #   Initial = post-perturbation state (step 0)
        #   Final = after simulation
        ct = getattr(self, "census_ground_truth_metrics", None) or {}
        pp = getattr(self, "agent_scale_baseline_metrics", None) or {}

        def col(metric_key, d, fmt="{:.4f}"):
            v = d.get(metric_key)
            return fmt.format(v) if v is not None else "   —   "

        lines = [
            "=" * 78,
            "SIMULATION RESULTS SUMMARY",
            "=" * 78,
            f"Steps completed: {len(self.metrics_history) - 1}",
            f"Total agents: {len(self.agents)}",
            f"Active archetypes: {len(self.active_archetypes)}",
            f"Tracts: {len(self.tract_ids)}",
            "",
            "--- Segregation Metrics ---",
            f"{'':24}{'CensusGT':>10}  {'PrePerturb':>10}  {'Initial':>10}  {'Final':>10}  {'City Tgt':>10}",
            f"{'D (Black-White):':24}{col('D_black_white', ct):>10}  {col('D_black_white', pp):>10}  {col('D_black_white', initial):>10}  {col('D_black_white', final):>10}  {'0.8250':>10}",
            f"{'D (Hispanic-White):':24}{col('D_hispanic_white', ct):>10}  {col('D_hispanic_white', pp):>10}  {col('D_hispanic_white', initial):>10}  {col('D_hispanic_white', final):>10}  {'0.5630':>10}",
            f"{'D (Asian-White):':24}{col('D_asian_white', ct):>10}  {col('D_asian_white', pp):>10}  {col('D_asian_white', initial):>10}  {col('D_asian_white', final):>10}  {'0.4490':>10}",
            f"{'Isolation (Black):':24}{col('Isolation_black', ct):>10}  {col('Isolation_black', pp):>10}  {col('Isolation_black', initial):>10}  {col('Isolation_black', final):>10}  {'0.8000':>10}",
            f"{'Isolation (White):':24}{col('Isolation_white', ct):>10}  {col('Isolation_white', pp):>10}  {col('Isolation_white', initial):>10}  {col('Isolation_white', final):>10}  {'':>10}",
            f"{'Isolation (Hispanic):':24}{col('Isolation_hispanic', ct):>10}  {col('Isolation_hispanic', pp):>10}  {col('Isolation_hispanic', initial):>10}  {col('Isolation_hispanic', final):>10}  {'':>10}",
        ]

        # R² recovery metric (available for all init modes)
        if "R2_recovery" in initial:
            lines.append(
                f"{'R² vs Census 2010:':24}{'1.0000':>10}  {col('R2_recovery', pp):>10}  {col('R2_recovery', initial):>10}  {col('R2_recovery', final):>10}  {'1.0000':>10}"
            )

        lines.append("")
        lines.append("  Legend:")
        lines.append("    CensusGT   = Real 2010 Census counts on selected tracts (population-scale ground truth)")
        lines.append("    PrePerturb = Agent-scale Census 2010 representation (what recovery should aim for)")
        lines.append("    Initial    = Post-initialization step 0 (after perturbation for 'perturbed' mode)")
        lines.append("    Final      = After N simulation steps")
        lines.append("    City Tgt   = Whole-Chicago Census 2010 targets (for reference; subset may differ)")

        lines.extend([
            "",
            "--- LLM Usage ---",
            f"Total API calls: {llm_stats['total_calls']}",
            f"Total input tokens: {llm_stats['total_input_tokens']:,}",
            f"Total output tokens: {llm_stats['total_output_tokens']:,}",
        ])

        if len(self.metrics_history) > 1:
            movers_per_step = [m.get("n_movers", 0) for m in self.metrics_history[1:]]
            lines.extend([
                "",
                "--- Movement ---",
                f"Total moves: {sum(movers_per_step)}",
                f"Avg movers/step: {np.mean(movers_per_step):.1f}",
            ])

        return "\n".join(lines)

    def get_agent_snapshot(self) -> list[dict]:
        """Return per-agent state for the current step."""
        current_step = len(self.metrics_history) - 1
        snapshots = []
        for agent in self.agents:
            snapshots.append({
                "step": current_step,
                "agent_id": agent.unique_id,
                "archetype_key": agent.archetype_key,
                "race": agent.race,
                "tract_id": agent.tract_id,
                "pop_count": agent.pop_count,
                "satisfaction": round(agent.satisfaction, 3),
                "would_move": self.group_would_move.get(
                    (agent.archetype_key, agent.tract_id), agent.archetype.would_move
                ),
                "moved_this_step": agent.moved_this_step,
                "years_in_tract": agent.years_in_tract,
                "archetype_satisfaction": round(agent.archetype.current_satisfaction, 3),
            })
        return snapshots

    def get_llm_log_dicts(self, records: list["LLMCallRecord"]) -> list[dict]:
        """Convert LLMCallRecord list to serializable dicts (without full prompts for compactness)."""
        out = []
        for r in records:
            out.append({
                "call_id": r.call_id,
                "call_type": r.call_type,
                "archetype_key": r.archetype_key,
                "tract_id": r.tract_id,
                "raw_response": r.raw_response,
                "parsed_result": r.parsed_result,
                "is_fallback": r.is_fallback,
                "fallback_reason": r.fallback_reason,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": round(r.latency_ms, 1),
                "error": r.error,
            })
        return out

    def get_llm_log_dicts_full(self, records: list["LLMCallRecord"]) -> list[dict]:
        """Convert LLMCallRecord list to serializable dicts (including full prompts)."""
        out = []
        for r in records:
            out.append({
                "call_id": r.call_id,
                "call_type": r.call_type,
                "archetype_key": r.archetype_key,
                "tract_id": r.tract_id,
                "system_prompt": r.system_prompt,
                "user_prompt": r.user_prompt,
                "raw_response": r.raw_response,
                "parsed_result": r.parsed_result,
                "is_fallback": r.is_fallback,
                "fallback_reason": r.fallback_reason,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": round(r.latency_ms, 1),
                "error": r.error,
            })
        return out
