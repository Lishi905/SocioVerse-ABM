"""
Phase 1 Prototype Runner for Chicago Segregation ABM.

Small-scale test: ~50-100 agents, a cluster of adjacent tracts, 3-5 steps.
Purpose: Validate LLM integration, agent behavior, and metric computation.

Usage:
    python run_prototype.py                    # Default: census init
    python run_prototype.py --init random      # Random placement (test emergence)
    python run_prototype.py --init semi_random # Shuffle within Community Areas
    python run_prototype.py --init inverted    # Swap racial distributions
    python run_prototype.py --steps 10         # Custom step count
    python run_prototype.py --agents 100       # Custom agent count

Resume from a previous run:
    python run_prototype.py --resume output/20260320_211933 --steps 5
    # Continues from the final state of that run, adding 5 more steps.
    # Creates a NEW output folder with full provenance linking back.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.llm_client import LLMConfig
from src.model import SegregationModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# LLM Configuration — credentials come from the environment (see .env.example).
# Never hard-code API keys in source; the key is read at call time, not baked in.
LLM_CONFIG = LLMConfig(
    api_key=os.environ.get("SV_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
    base_url=os.environ.get("SV_LLM_BASE_URL", "https://api.openai.com/v1"),
    model=os.environ.get("SV_LLM_MODEL", "gpt-4o-2024-08-06"),
    temperature=0.7,
    max_tokens=512,
)

# Prototype parameters
N_AGENTS = 50          # Small agent count for testing
N_STEPS = 3            # Few steps to validate flow
MAX_ARCHETYPES = 241    # Limit archetype diversity for cost control
SEED = 42
INIT_MODE = "census"   # "census" | "random" | "semi_random" | "inverted"


def select_prototype_tracts(scale: str = "small", n_tracts: int = 25) -> list[str]:
    """
    Select a study area of Chicago tracts at one of three scales.

    scale = "small"  → 25-tract cluster, BFS from CA 61 anchor
    scale = "middle" → ~218-tract bounding-box on South+West Side (Option 2+)

    (scale = "full" is handled by the caller as tract_ids = None.)

    ── "small" (legacy prototype subset) ────────────────────────────
    Study area: **South-Southwest Side Back-of-the-Yards corridor**,
    anchored at New City (CA 61, tract 17031842600) and expanded by
    Queen-contiguity BFS.  This anchor was selected by grid-searching
    every Chicago tract as a BFS seed and minimizing the summed gap
    between subset indices and city-wide values:

      Subset  vs  City-wide:
        D_bw  = 0.816  vs  0.835   (Δ -0.019)   ← matches canonical 0.825
        D_hw  = 0.558  vs  0.608   (Δ -0.050)
        D_aw  = 0.328  vs  0.434
        Iso_B = 0.614  vs  0.797
        Iso_H = 0.732  vs  0.612
        Total gap = 0.707 (best of all tested anchors)

    Race composition: W=17%, B=15%, H=60%, A=7% (city-wide 32/32/29/5).

    ── "middle" (Option 2+ bounding box) ────────────────────────────
    A rectangular slice of the South + West Side plus two extra tract
    rows north to capture more White-dominant tracts.  Selected by
    filtering tract centroids to:
        lon ∈ [-87.755, -87.58]
        lat ∈ [ 41.81 ,  41.91]
    This yields ~218 contiguous tracts spanning West Side, Near
    West/South, Loop fringe, and the Black Belt + Back of the Yards.
    Race composition is closer to city-wide (W≈27%, vs city 31.7%)
    while keeping the run tractable.
    """
    import geopandas as gpd

    data_dir = Path(__file__).parent / "processed_data"
    gdf = gpd.read_file(data_dir / "chicago_tracts.geojson")
    gdf["GEOID10"] = gdf["GEOID10"].astype(str)
    gdf = gdf.reset_index(drop=True).set_index("GEOID10")

    if scale == "small":
        selected = _select_small_tracts(gdf, n_tracts=n_tracts)
    elif scale == "middle":
        selected = _select_middle_tracts(gdf)
    else:
        raise ValueError(
            f"select_prototype_tracts: unknown scale '{scale}' "
            f"(expected 'small' or 'middle')"
        )

    # Summary log: race shares + CA breakdown
    RACE_COLS = ["nh_white", "nh_black", "nh_asian", "hispanic", "nh_other"]
    sub = gdf.loc[selected]
    t = sub[RACE_COLS].sum()
    shares = {r: t[r] / max(t.sum(), 1) for r in RACE_COLS}
    logger.info(f"Selected {len(selected)} tracts (scale={scale})")
    logger.info(
        f"  Race shares: W={shares['nh_white']*100:.1f}% "
        f"B={shares['nh_black']*100:.1f}% "
        f"H={shares['hispanic']*100:.1f}% "
        f"A={shares['nh_asian']*100:.1f}%  "
        f"(city: W=31.7% B=32.1% H=29.3% A=5.3%)"
    )
    logger.info(f"  Total pop: {int(t.sum()):,}")

    ca_names = {
        28: "Near West Side", 31: "Lower West Side (Pilsen)",
        32: "Loop", 33: "Near South Side",
        34: "Armour Square (Chinatown)", 35: "Douglas (Bronzeville)",
        37: "Fuller Park", 38: "Grand Boulevard (Black Belt)",
        40: "Washington Park", 41: "Hyde Park",
        58: "Brighton Park", 59: "McKinley Park",
        60: "Bridgeport", 61: "New City (Back of the Yards)",
        63: "Gage Park", 66: "Chicago Lawn (Marquette Park)",
        67: "West Englewood", 68: "Englewood",
    }
    ca_counts = gdf.loc[selected, "community_area_num"].value_counts().to_dict()
    # Only spell out CAs for small scale; middle is too many to enumerate
    if scale == "small":
        for ca, n in sorted(ca_counts.items(), key=lambda x: -x[1]):
            name = ca_names.get(int(ca), f"CA {int(ca)}")
            logger.info(f"  CA {int(ca):3d} ({name}): {n} tracts")
    else:
        logger.info(f"  Spans {len(ca_counts)} Community Areas")

    return list(selected)


def _select_small_tracts(gdf, n_tracts: int = 25) -> list[str]:
    """BFS from CA 61 anchor; 25-tract prototype subset."""
    from collections import deque
    from libpysal.weights import Queen

    # Build Queen contiguity on the whole city, indexed by GEOID10
    W = Queen.from_dataframe(gdf, use_index=True)

    # Anchor chosen by grid-search: among all 785 Chicago tracts as
    # potential BFS seeds, 17031842600 (CA 61 New City) minimizes
    # Σ |subset_index - city_index| across (D_bw, D_hw, D_aw, Iso_B,
    # Iso_W, Iso_H) with a gap score of 0.707.
    ANCHOR = "17031842600"
    if ANCHOR not in gdf.index:
        alt = gdf[gdf["community_area_num"] == 61]
        if len(alt) == 0:
            raise RuntimeError(
                f"Prototype anchor tract {ANCHOR} not found and no CA 61 fallback available"
            )
        anchor = alt.index[0]
        logger.warning(f"Anchor {ANCHOR} not in data; using CA 61 fallback {anchor}")
    else:
        anchor = ANCHOR

    # Pure BFS with sorted neighbor expansion for cross-process
    # determinism (libpysal's neighbor dicts are sets whose iteration
    # order depends on PYTHONHASHSEED).
    visited = {anchor}
    order = [anchor]
    queue = deque([anchor])
    while queue and len(order) < n_tracts:
        cur = queue.popleft()
        for nb in sorted(W.neighbors.get(cur, [])):
            if nb in gdf.index and nb not in visited:
                visited.add(nb)
                order.append(nb)
                queue.append(nb)
                if len(order) >= n_tracts:
                    break

    if len(order) < n_tracts:
        logger.warning(
            f"BFS from {anchor} exhausted at {len(order)} tracts "
            f"(< requested {n_tracts}); subset may be disconnected."
        )
    return order


# Option 2+ bounding box (South+West Side, extended 2 tract rows north).
# Chosen to yield ~218 tracts with race shares closer to city-wide.
_MIDDLE_BBOX = (-87.755, 41.81, -87.58, 41.91)  # (lon_min, lat_min, lon_max, lat_max)


def _select_middle_tracts(gdf) -> list[str]:
    """Centroid-in-bounding-box filter; ~218 tracts (Option 2+)."""
    lon_min, lat_min, lon_max, lat_max = _MIDDLE_BBOX

    # Centroids in lon/lat.  We assume geometries are in EPSG:4326 (the
    # stored GeoJSON is WGS84).  If not, reproject just for centroid.
    geom = gdf.geometry
    if gdf.crs is None or str(gdf.crs).lower().endswith("4326"):
        cent = geom.centroid
    else:
        cent = geom.to_crs("EPSG:4326").centroid

    mask = (
        (cent.x >= lon_min) & (cent.x <= lon_max) &
        (cent.y >= lat_min) & (cent.y <= lat_max)
    )
    selected = sorted(gdf.index[mask].tolist())
    if not selected:
        raise RuntimeError(
            f"Middle-scale bbox {_MIDDLE_BBOX} selected 0 tracts; check CRS"
        )
    return selected


def _legacy_select_prototype_tracts_unused():
    """Old implementation retained for reference only; not called."""
    import geopandas as gpd
    data_dir = Path(__file__).parent / "processed_data"
    gdf = gpd.read_file(data_dir / "chicago_tracts.geojson")
    target_cas = [31, 33, 34, 35, 41, 60]
    subset = gdf[gdf["community_area_num"].isin(target_cas)]
    tract_ids = subset["GEOID10"].tolist()[:25]
    ca_names = {31: "Lower West Side (Hispanic)", 33: "Near South Side (mixed)",
                34: "Armour Square (Asian)", 35: "Douglas (Black)",
                41: "Hyde Park (diverse)", 60: "Bridgeport (White)"}
    for ca in target_cas:
        ca_tracts = subset[subset["community_area_num"] == ca]
        if len(ca_tracts) == 0:
            continue
    return tract_ids


def create_run_dir() -> Path:
    """Create timestamped output directory for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent / "output" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(run_dir: Path, tract_ids: list[str],
                    n_agents: int = N_AGENTS, n_steps: int = N_STEPS,
                    max_archetypes: int = MAX_ARCHETYPES, seed: int = SEED,
                    init_mode: str = INIT_MODE, perturb_pct: float = 0.15,
                    perturb_radius_hops: int = 7,
                    move_budget_pct: float = 0.05,
                    own_race_ceiling: float = 1.3,
                    max_pop_per_agent: int = 250,
                    capacity_tolerance: float = 0.10,
                    move_priority_income_weight: float = 0.0,
                    race_size_budget_scaling: bool = True,
                    per_step_tract_inflow_cap_ratio: float = 0.03,
                    enclave_candidate_bias: float = 0.1,
                    resume_from: str = None,
                    start_step: int = 0, llm_config: LLMConfig = None,
                    scale: str = "small",
                    tract_capacities: dict | None = None):
    """Save run configuration for reproducibility."""
    _llm = llm_config or LLM_CONFIG
    config = {
        "timestamp": datetime.now().isoformat(),
        "llm_model": _llm.model,
        "llm_base_url": _llm.base_url,
        "llm_temperature": _llm.temperature,
        "llm_max_tokens": _llm.max_tokens,
        "n_agents": n_agents,
        "n_steps": n_steps,
        "max_archetypes": max_archetypes,
        "seed": seed,
        "init_mode": init_mode,
        "perturb_pct": perturb_pct,
        "perturb_radius_hops": perturb_radius_hops,
        "move_budget_pct": move_budget_pct,
        "own_race_ceiling_ratio": own_race_ceiling,
        "max_pop_per_agent": max_pop_per_agent,
        "capacity_tolerance": capacity_tolerance,
        "move_priority_income_weight": move_priority_income_weight,
        "race_size_budget_scaling": race_size_budget_scaling,
        "per_step_tract_inflow_cap_ratio": per_step_tract_inflow_cap_ratio,
        "enclave_candidate_bias": enclave_candidate_bias,
        "scale": scale,
        "n_tracts": len(tract_ids),
        "tract_ids": tract_ids,
        "resume_from": resume_from,
        "start_step": start_step,
    }
    if tract_capacities is not None:
        config["tract_capacities"] = tract_capacities
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    return config


def save_step_log(run_dir: Path, step_info: dict, step_num: int):
    """Save per-step detailed log as individual JSON files."""
    step_dir = run_dir / "steps"
    step_dir.mkdir(exist_ok=True)

    # Save agent snapshots for this step
    with open(step_dir / f"step_{step_num:03d}_agents.json", "w") as f:
        json.dump(step_info["agent_snapshots"], f, indent=2)

    # Save LLM call details for this step
    with open(step_dir / f"step_{step_num:03d}_llm_calls.json", "w") as f:
        json.dump(step_info["llm_calls"], f, indent=2)

    # Save step summary (without bulky nested data)
    summary = {k: v for k, v in step_info.items() if k not in ("agent_snapshots", "llm_calls")}
    with open(step_dir / f"step_{step_num:03d}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def save_full_llm_log(run_dir: Path, model: SegregationModel):
    """Save complete LLM call log with full prompts for deep analysis."""
    all_records = model.get_llm_log_dicts_full(model.llm.call_log)
    with open(run_dir / "llm_calls_full.json", "w") as f:
        json.dump(all_records, f, indent=2)

    # Also save a compact summary
    fallbacks = [r for r in all_records if r["is_fallback"]]
    summary = {
        "total_calls": len(all_records),
        "satisfaction_calls": sum(1 for r in all_records if r["call_type"] == "satisfaction"),
        "move_eval_calls": sum(1 for r in all_records if r["call_type"] == "move_eval"),
        "fallback_count": len(fallbacks),
        "fallback_details": [
            {"call_id": r["call_id"], "type": r["call_type"],
             "archetype": r["archetype_key"], "reason": r["fallback_reason"]}
            for r in fallbacks
        ],
        "total_input_tokens": sum(r["input_tokens"] for r in all_records),
        "total_output_tokens": sum(r["output_tokens"] for r in all_records),
        "avg_latency_ms": round(
            sum(r["latency_ms"] for r in all_records) / len(all_records), 1
        ) if all_records else 0,
    }
    with open(run_dir / "llm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ── Per-step visualization (pic_log) ────────────────────────────

def save_step_pic(run_dir: Path, model: SegregationModel, step_num: int,
                  gdf=None, gdf_sub=None):
    """Save a racial distribution map for this step into pic_log/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba

    RACE_COLORS = {
        "nh_white": "#4575b4", "nh_black": "#1a9850",
        "hispanic": "#e66101", "nh_asian": "#d73027", "nh_other": "#998ec3",
    }
    RACE_LABELS = {
        "nh_white": "NH White", "nh_black": "NH Black",
        "hispanic": "Hispanic", "nh_asian": "NH Asian", "nh_other": "Other",
    }
    RACE_COLS = ["nh_white", "nh_black", "nh_asian", "hispanic", "nh_other"]

    pic_dir = run_dir / "pic_log"
    pic_dir.mkdir(exist_ok=True)

    # Build per-tract racial percentages from current agent positions
    from collections import defaultdict
    tract_pops = defaultdict(lambda: {r: 0 for r in RACE_COLS})
    for agent in model.agents:
        tract_pops[agent.tract_id][agent.race] += agent.pop_count
    tract_pcts = {}
    for tid, pops in tract_pops.items():
        total = sum(pops.values())
        tract_pcts[tid] = {r: (pops[r] / total * 100) if total > 0 else 0
                           for r in RACE_COLS}

    # Get current metrics
    metrics = model.metrics_history[-1] if model.metrics_history else {}
    d_bw = metrics.get("D_black_white", 0)
    iso_b = metrics.get("Isolation_black", 0)
    movers = metrics.get("n_movers", 0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor="white")

    # Background city
    gdf.boundary.plot(ax=ax, color="#eeeeee", linewidth=0.06)

    # Color prototype tracts
    for idx, row in gdf_sub.iterrows():
        rp = tract_pcts.get(idx, {})
        if rp:
            dom = max(RACE_COLS, key=lambda r: rp.get(r, 0))
            pct = rp.get(dom, 0)
            alpha = max(0.2, min(1.0, pct / 100))
            color = to_rgba(RACE_COLORS[dom], alpha=alpha)
        else:
            color = "#f0f0f0"
        geom = row["geometry"]
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, color=color, edgecolor="#aaaaaa", linewidth=0.3)

    # Zoom to active area (full city if gdf_sub == gdf, else prototype subset)
    b = gdf_sub.total_bounds
    margin = 0.005 if len(gdf_sub) < 100 else 0.01
    ax.set_xlim(b[0] - margin, b[2] + margin)
    ax.set_ylim(b[1] - margin, b[3] + margin)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Legend
    patches = [mpatches.Patch(color=RACE_COLORS[r], label=RACE_LABELS[r])
               for r in RACE_COLS]
    ax.legend(handles=patches, loc="lower left", fontsize=8, framealpha=0.9,
              title="Dominant Group", title_fontsize=9)

    label = "Initial" if step_num == 0 else f"Step {step_num}"
    movers_text = f"  |  Movers: {movers}" if step_num > 0 else ""
    ax.set_title(
        f"{label}  |  D_bw={d_bw:.4f}  |  Iso_B={iso_b:.4f}{movers_text}",
        fontsize=12, fontweight="bold")

    out = pic_dir / f"step_{step_num:03d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Resume helpers ──────────────────────────────────────────────

def load_previous_run(resume_dir: Path) -> dict:
    """Load all necessary data from a previous run to resume simulation."""
    resume_dir = Path(resume_dir)
    if not resume_dir.is_absolute():
        resume_dir = Path(__file__).parent / resume_dir

    if not resume_dir.exists():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

    data = {}
    for name in ["run_config", "metrics_history", "final_agents", "archetype_decisions"]:
        p = resume_dir / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing required file for resume: {p}")
        with open(p) as f:
            data[name] = json.load(f)

    logger.info(f"Loaded previous run from {resume_dir.name}")
    prev_config = data["run_config"]
    last_step = data["metrics_history"][-1]["step"]
    logger.info(f"  Previous run: {prev_config.get('init_mode', 'census')} init, "
                f"{last_step} steps completed, "
                f"D_bw={data['metrics_history'][-1]['D_black_white']:.4f}")

    return data


def restore_model_state(model: SegregationModel, prev_data: dict, start_step: int):
    """
    Restore a model to the final state of a previous run.

    This sets:
    - Agent positions (tract_id) and satisfaction from final_agents
    - Archetype decisions from archetype_decisions
    - metrics_history from previous run
    - Environment dynamic_pop rebuilt from agent positions
    - Step counter offset so new steps continue from start_step
    """
    final_agents = prev_data["final_agents"]
    prev_metrics = prev_data["metrics_history"]
    prev_archetypes = prev_data["archetype_decisions"]

    # Build lookup: agent_id -> final state
    agent_lookup = {a["agent_id"]: a for a in final_agents}

    # Restore agent positions and satisfaction
    restored = 0
    for agent in model.agents:
        state = agent_lookup.get(agent.unique_id)
        if state:
            agent.tract_id = state["tract_id"]
            agent.satisfaction = state.get("satisfaction", 5.0)
            agent.years_in_tract = state.get("years_in_tract", 0)
            agent.moved_this_step = False
            restored += 1

    # Rebuild environment dynamic_pop from restored agent positions
    model.environment.rebuild_dynamic_pop_from_agents(list(model.agents))

    # Restore archetype decisions
    for key, arch_state in prev_archetypes.items():
        if key in model.active_archetypes:
            arch = model.active_archetypes[key]
            arch.current_satisfaction = arch_state.get("current_satisfaction", 5.0)
            arch.would_move = arch_state.get("would_move", False)
            arch.key_reasons = arch_state.get("key_reasons", [])
            arch.move_rankings = arch_state.get("move_rankings", [])

    # Restore metrics history (all previous steps)
    model.metrics_history = list(prev_metrics)

    # Clear step_log (fresh for this continuation run)
    model.step_log = []

    logger.info(f"Restored {restored}/{len(list(model.agents))} agents from previous run")
    logger.info(f"Metrics history: {len(prev_metrics)} entries, "
                f"continuing from step {start_step}")


# ── Args ────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments, falling back to module-level defaults."""
    parser = argparse.ArgumentParser(description="Chicago Segregation ABM Prototype")
    parser.add_argument("--init", type=str, default=INIT_MODE,
                        choices=["census", "census_2000", "random", "semi_random",
                                 "inverted", "perturbed", "perturbed_n"],
                        help="Initialization mode (default: census)")
    parser.add_argument("--perturb-pct", type=float, default=0.15,
                        help="Fraction of agents to perturb in 'perturbed' / 'perturbed_n' "
                             "modes (default: 0.15)")
    parser.add_argument("--perturb-radius-hops", type=int, default=7,
                        help="Neighborhood radius in adjacency hops for 'perturbed_n' "
                             "init mode. Should be slightly larger than the simulation's "
                             "search_radius_hops (max 5 in current archetypes) so that "
                             "perturbed agents can search their way back during recovery "
                             "(default: 7)")
    parser.add_argument("--move-budget-pct", type=float, default=0.05,
                        help="Cap on total pop_count allowed to move per step, as fraction "
                             "of total population (default: 0.05, i.e. 5%% per step)")
    parser.add_argument("--own-race-ceiling", type=float, default=1.3,
                        help="Own-race ceiling ratio for candidate filtering. A candidate "
                             "tract is dropped if its own-race share > current_share * ratio. "
                             "Set <=0 to disable (default: 1.3)")
    parser.add_argument("--max-pop-per-agent", type=int, default=250,
                        help="Maximum people represented by a single HouseholdAgent. "
                             "Large mapping cells are split into multiple clones so per-step "
                             "moves are fine-grained. Set very large to disable (default: 250)")
    parser.add_argument("--steps", type=int, default=N_STEPS,
                        help=f"Number of simulation steps (default: {N_STEPS})")
    parser.add_argument("--agents", type=int, default=N_AGENTS,
                        help=f"Number of agents (default: {N_AGENTS})")
    parser.add_argument("--archetypes", type=int, default=MAX_ARCHETYPES,
                        help=f"Max archetypes (default: {MAX_ARCHETYPES})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a previous run directory to continue from")
    parser.add_argument("--scale", type=str, default="small",
                        choices=["small", "middle", "full"],
                        help="Study-area scale: "
                            "'small' = 25-tract CA-61 BFS prototype (default), "
                            "'middle' = ~218-tract South+West Side bbox (Option 2+), "
                            "'full' = all 785 Chicago tracts")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model name (e.g. gpt-5.4-nano)")
    parser.add_argument("--move-priority-income-weight", type=float, default=0.0,
                        help="Weight on income vs. dissatisfaction when ranking movers for "
                            "the per-race budget (0.0 = pure dissatisfaction, 1.0 = pure "
                            "income, 0.4 = suggested blend). Must be in [0, 1]. "
                            )
    parser.add_argument("--capacity-tolerance", type=float, default=0.10,
                        help="Per-tract capacity headroom as a fraction of initial Census "
                             "population (quantized to multiples of --max-pop-per-agent). "
                             "0.10 gives ~10%% breathing room; 0.15 recommended for full-scale "
                             "runs to reduce destination saturation on Hispanic/Black clustering "
                             "(default: 0.10)")
    parser.add_argument("--race-size-budget-scaling",
                        dest="race_size_budget_scaling",
                        action="store_true", default=True,
                        help="Scale the per-race move budget by "
                             "sqrt(city_share/median_share) so small races (Asian, Other) "
                             "do not dominate D via outsized mobility (default: on)")
    parser.add_argument("--no-race-size-budget-scaling",
                        dest="race_size_budget_scaling",
                        action="store_false",
                        help="Disable race-size-aware budget scaling "
                             "(fallback to flat move_budget_pct × race_pop)")
    parser.add_argument("--per-step-tract-inflow-cap-ratio", type=float, default=0.03,
                        help="Per-step (tract, race) net inflow cap as a fraction of that "
                             "tract's capacity.  Lower-bounded at max_pop_per_agent so at "
                             "least one max-sized clone can enter.  Set to 0 to disable "
                             "(default: 0.03)")
    parser.add_argument("--enclave-candidate-bias", type=float, default=0.1,
                        help="ε in the inverse-frequency candidate weight w = 1/(ε + share). "
                             "Smaller ε → stronger downweighting of already-concentrated "
                             "destinations when sampling the move-eval candidate set. "
                             "Set to 0 to disable and fall back to uniform sampling "
                             "(default: 0.1)")
    return parser.parse_args()


# ── Main ────────────────────────────────────────────────────────

def main():
    args = parse_args()

    n_agents = args.agents
    n_steps = args.steps
    max_archetypes = args.archetypes
    init_mode = args.init
    perturb_pct = args.perturb_pct
    perturb_radius_hops = args.perturb_radius_hops
    move_budget_pct = args.move_budget_pct
    own_race_ceiling = args.own_race_ceiling
    max_pop_per_agent = args.max_pop_per_agent
    seed = args.seed
    resume_dir = args.resume
    scale = args.scale
    use_full_city = (scale == "full")

    # Override LLM model if specified
    llm_config = LLM_CONFIG
    if args.model:
        llm_config = LLMConfig(
            api_key=LLM_CONFIG.api_key,
            base_url=LLM_CONFIG.base_url,
            model=args.model,
            temperature=LLM_CONFIG.temperature,
            max_tokens=LLM_CONFIG.max_tokens,
            min_call_interval=0.5,  # Rate limit for non-default models
        )

    # ── Resume mode ──
    is_resume = resume_dir is not None
    prev_data = None
    start_step = 0

    if is_resume:
        prev_data = load_previous_run(resume_dir)
        prev_config = prev_data["run_config"]
        # Inherit settings from previous run unless explicitly overridden
        init_mode = prev_config.get("init_mode", "census")
        n_agents = prev_config.get("n_agents", n_agents)
        max_archetypes = prev_config.get("max_archetypes", max_archetypes)
        seed = prev_config.get("seed", seed)
        # Inherit scale if present; older runs predate this field
        scale = prev_config.get("scale", scale)
        start_step = prev_data["metrics_history"][-1]["step"]
        logger.info("=" * 60)
        logger.info("Chicago Segregation ABM - RESUME RUN")
        logger.info(f"  Resuming from: {resume_dir}")
        logger.info(f"  Previous steps: {start_step}, adding {n_steps} more")
        logger.info(f"  Init mode (inherited): {init_mode}")
        logger.info("=" * 60)
    else:
        scale_label = {
            "small": "Prototype (25 tracts)",
            "middle": "Middle (~218 tracts, South+West Side bbox)",
            "full": "FULL CITY (785 tracts)",
        }[scale]
        logger.info("=" * 60)
        logger.info(f"Chicago Segregation ABM - {scale_label}")
        logger.info(f"  Init mode: {init_mode}")
        logger.info(f"  Agents: {n_agents}, Archetypes: {max_archetypes}, Model: {llm_config.model}")
        logger.info("=" * 60)

    # Create timestamped run directory
    run_dir = create_run_dir()
    logger.info(f"Run output directory: {run_dir}")

    # Get tract_ids
    if is_resume:
        tract_ids = prev_data["run_config"]["tract_ids"]
        logger.info(f"Using {len(tract_ids)} tracts from previous run")
    elif scale == "full":
        # Use all tracts (pass None to model, which skips subset)
        tract_ids = None
        logger.info("Using ALL tracts (full Chicago)")
    else:
        tract_ids = select_prototype_tracts(scale=scale)

    # Save run config
    resume_ref = str(Path(resume_dir).resolve()) if resume_dir else None
    # For full-city mode, save tract_ids as empty list (will be populated after model init)
    config_tract_ids = tract_ids if tract_ids is not None else []
    # Per-tract capacity headroom (fraction of initial Census pop,
    # quantized to multiples of max_pop_per_agent).  Exposed as --capacity-tolerance.
    # 0.10 is a good default for small/middle scale; full scale benefits from 0.15
    # because more tracts = more concurrent Schelling flows = more risk of hitting
    # destination caps before the cluster can re-form.
    capacity_tolerance = args.capacity_tolerance
    move_priority_income_weight = args.move_priority_income_weight
    race_size_budget_scaling = args.race_size_budget_scaling
    per_step_tract_inflow_cap_ratio = args.per_step_tract_inflow_cap_ratio
    enclave_candidate_bias = args.enclave_candidate_bias
    config = save_run_config(
        run_dir, config_tract_ids, n_agents=n_agents, n_steps=n_steps,
        max_archetypes=max_archetypes, seed=seed, init_mode=init_mode,
        perturb_pct=perturb_pct,
        perturb_radius_hops=perturb_radius_hops,
        move_budget_pct=move_budget_pct, own_race_ceiling=own_race_ceiling,
        max_pop_per_agent=max_pop_per_agent,
        capacity_tolerance=capacity_tolerance,
        move_priority_income_weight=move_priority_income_weight,
        race_size_budget_scaling=race_size_budget_scaling,
        per_step_tract_inflow_cap_ratio=per_step_tract_inflow_cap_ratio,
        enclave_candidate_bias=enclave_candidate_bias,
        resume_from=resume_ref, start_step=start_step, llm_config=llm_config,
        scale=scale,
    )

    # Create model (always from census first, then apply init_mode or restore)
    if is_resume:
        # For resume: create model in census mode (base positions don't matter,
        # we'll overwrite from final_agents)
        logger.info(f"\nCreating model on {len(tract_ids)} tracts "
                    f"(will restore from previous run; actual agent count "
                    f"comes from mapping)...")
        model = SegregationModel(
            llm_config=llm_config,
            n_agents=n_agents,
            tract_ids=tract_ids,
            max_archetypes=max_archetypes,
            seed=seed,
            init_mode="census",  # Base creation, will be overwritten
            move_budget_pct=move_budget_pct,
            own_race_ceiling_ratio=own_race_ceiling,
            max_pop_per_agent=max_pop_per_agent,
            capacity_tolerance=capacity_tolerance,
            move_priority_income_weight=move_priority_income_weight,
            race_size_budget_scaling=race_size_budget_scaling,
            per_step_tract_inflow_cap_ratio=per_step_tract_inflow_cap_ratio,
            enclave_candidate_bias=enclave_candidate_bias,
        )
        # Restore to previous final state
        restore_model_state(model, prev_data, start_step)
    else:
        n_tracts_label = "ALL" if tract_ids is None else str(len(tract_ids))
        logger.info(f"\nInitializing model on {n_tracts_label} tracts, "
                    f"{max_archetypes} max archetypes, init_mode={init_mode}, "
                    f"model={llm_config.model}...")
        logger.info(f"  (--agents {n_agents} is advisory; actual agent count "
                    f"= number of archetype×tract rows in the mapping, "
                    f"each agent's pop_count = its Census population)")
        logger.info(f"  move_budget_pct={move_budget_pct}, "
                    f"own_race_ceiling_ratio={own_race_ceiling}, "
                    f"capacity_tolerance={capacity_tolerance}, "
                    f"move_priority_income_weight={move_priority_income_weight}")
        logger.info(f"  race_size_budget_scaling={race_size_budget_scaling}, "
                    f"per_step_tract_inflow_cap_ratio={per_step_tract_inflow_cap_ratio}, "
                    f"enclave_candidate_bias={enclave_candidate_bias}")
        model = SegregationModel(
            llm_config=llm_config,
            n_agents=n_agents,
            tract_ids=tract_ids,
            max_archetypes=max_archetypes,
            seed=seed,
            init_mode=init_mode,
            perturb_pct=perturb_pct,
            perturb_radius_hops=perturb_radius_hops,
            move_budget_pct=move_budget_pct,
            own_race_ceiling_ratio=own_race_ceiling,
            max_pop_per_agent=max_pop_per_agent,
            capacity_tolerance=capacity_tolerance,
            move_priority_income_weight=move_priority_income_weight,
            race_size_budget_scaling=race_size_budget_scaling,
            per_step_tract_inflow_cap_ratio=per_step_tract_inflow_cap_ratio,
            enclave_candidate_bias=enclave_candidate_bias,
        )

    # Now that model is initialized, get actual tract_ids for full-city mode
    # and update n_agents to reflect the actual mapping-derived count.
    config_needs_update = False
    if tract_ids is None:
        tract_ids = model.tract_ids
        config["tract_ids"] = list(tract_ids)
        config["n_tracts"] = len(tract_ids)
        config_needs_update = True
    actual_n_agents = len(list(model.agents))
    if config.get("n_agents") != actual_n_agents:
        config["n_agents_cli"] = config.get("n_agents")
        config["n_agents"] = actual_n_agents
        config_needs_update = True
    # Record per-tract capacities (base_pop + tolerance headroom rounded to
    # max_pop_per_agent units) so the run config documents the exact ceilings
    # enforced by environment.move_population during this run.
    tract_caps = getattr(model.environment, "tract_capacity", None)
    if tract_caps:
        serialised_caps = {str(k): int(v) for k, v in tract_caps.items()}
        if config.get("tract_capacities") != serialised_caps:
            config["tract_capacities"] = serialised_caps
            config_needs_update = True
    if config_needs_update:
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f, indent=2)

    # Print agent distribution
    logger.info("\nAgent distribution by race:")
    race_counts = {}
    for agent in model.agents:
        race_counts[agent.race] = race_counts.get(agent.race, 0) + 1
    for race, count in sorted(race_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {race}: {count} agents")

    logger.info(f"\nAgent distribution by tract:")
    tract_counts = {}
    for agent in model.agents:
        tract_counts[agent.tract_id] = tract_counts.get(agent.tract_id, 0) + 1
    for tid, count in sorted(tract_counts.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  Tract {tid}: {count} agents")

    # Save initial agent state (= start state for this segment)
    initial_agents = []
    for agent in model.agents:
        initial_agents.append({
            "agent_id": agent.unique_id,
            "archetype_key": agent.archetype_key,
            "race": agent.race,
            "tract_id": agent.tract_id,
            "pop_count": agent.pop_count,
            "satisfaction": round(agent.satisfaction, 3),
            "years_in_tract": agent.years_in_tract,
        })
    with open(run_dir / "initial_agents.json", "w") as f:
        json.dump(initial_agents, f, indent=2)

    # ── Load GeoDataFrames for pic_log ──
    import geopandas as gpd
    data_dir = Path(__file__).parent / "processed_data"
    gdf = gpd.read_file(data_dir / "chicago_tracts.geojson")
    gdf = gdf[gdf["total_pop"] > 0].copy()
    gdf = gdf.set_index("GEOID10", drop=False)
    gdf_sub = gdf[gdf.index.isin(set(tract_ids))].copy()

    # Save initial state pic (step 0 or start_step)
    save_step_pic(run_dir, model, start_step, gdf=gdf, gdf_sub=gdf_sub)
    logger.info(f"Saved pic_log/step_{start_step:03d}.png (initial state)")

    # Run simulation
    logger.info(f"\nRunning {n_steps} simulation steps (step {start_step+1} to {start_step+n_steps})...")
    for i in range(n_steps):
        model.step()

        # The model's internal step counter
        step_info = model.step_log[-1]
        actual_step = step_info["step"]

        # Save per-step logs
        save_step_log(run_dir, step_info, actual_step)

        # Save per-step distribution map
        save_step_pic(run_dir, model, actual_step, gdf=gdf, gdf_sub=gdf_sub)

    # ── Print results ──
    # Build summary covering only this segment's steps
    print("\n" + model.get_results_summary())

    # Print step-by-step metrics (full history including resumed steps)
    has_r2 = "R2_recovery" in model.metrics_history[0]
    print("\n--- Step-by-Step Metrics ---")
    if has_r2:
        print(f"{'Step':>10}  {'D_bw':>8}  {'D_hw':>8}  {'D_aw':>8}  {'Iso_B':>8}  {'R²':>8}  {'Movers':>7}")
        print("-" * 72)
    else:
        print(f"{'Step':>10}  {'D_bw':>8}  {'D_hw':>8}  {'D_aw':>8}  {'Iso_B':>8}  {'Movers':>7}")
        print("-" * 62)

    # Print the two baselines first as reference rows
    ct = getattr(model, "census_ground_truth_metrics", None)
    pp = getattr(model, "agent_scale_baseline_metrics", None)
    if ct:
        row = (f"{'CensusGT':>10}  {ct['D_black_white']:>8.4f}  {ct['D_hispanic_white']:>8.4f}  "
               f"{ct['D_asian_white']:>8.4f}  {ct['Isolation_black']:>8.4f}")
        if has_r2:
            row += f"  {'1.0000':>8}"
        row += f"  {'—':>7}"
        print(row)
    if pp:
        row = (f"{'PrePerturb':>10}  {pp['D_black_white']:>8.4f}  {pp['D_hispanic_white']:>8.4f}  "
               f"{pp['D_asian_white']:>8.4f}  {pp['Isolation_black']:>8.4f}")
        if has_r2 and "R2_recovery" in pp:
            row += f"  {pp['R2_recovery']:>8.4f}"
        elif has_r2:
            row += f"  {'—':>8}"
        row += f"  {'—':>7}"
        print(row)
    if ct or pp:
        print("-" * (72 if has_r2 else 62))

    for m in model.metrics_history:
        movers = m.get("n_movers", "-")
        base = (f"{m['step']:>10}  {m['D_black_white']:>8.4f}  {m['D_hispanic_white']:>8.4f}  "
                f"{m['D_asian_white']:>8.4f}  {m['Isolation_black']:>8.4f}")
        if has_r2:
            base += f"  {m.get('R2_recovery', 0):>8.4f}"
        base += f"  {str(movers):>7}"
        print(base)

    # Print archetype decision details
    print("\n--- Archetype Decisions (Final Step) ---")
    for key, arch in sorted(model.active_archetypes.items(),
                             key=lambda x: x[1].current_satisfaction):
        status = "MOVE" if arch.would_move else "STAY"
        reasons = ", ".join(arch.key_reasons[:2]) if arch.key_reasons else "-"
        print(f"  [{status}] {key}: sat={arch.current_satisfaction:.1f}, reasons: {reasons}")

    # ── Save outputs ──
    # Save full metrics history (including previous run if resumed)
    with open(run_dir / "metrics_history.json", "w") as f:
        json.dump(model.metrics_history, f, indent=2)

    # Save reference baselines alongside the history so downstream
    # visualization / analysis can plot them as horizontal reference lines.
    baselines = {
        "census_ground_truth": getattr(model, "census_ground_truth_metrics", {}),
        "agent_scale_pre_perturbation": getattr(model, "agent_scale_baseline_metrics", {}),
        "city_wide_targets_2010": {
            "D_black_white": 0.825,
            "D_hispanic_white": 0.563,
            "D_asian_white": 0.449,
            "Isolation_black": 0.80,
        },
        "notes": (
            "census_ground_truth = real 2010 Census population counts on the selected tracts. "
            "agent_scale_pre_perturbation = metrics computed from agent positions BEFORE "
            "the perturbation step (i.e., the recovery target for 'perturbed' init mode). "
            "city_wide_targets_2010 = whole-Chicago Census 2010 values for reference; a small "
            "tract subset will generally differ from city-wide values."
        ),
    }
    with open(run_dir / "metrics_baselines.json", "w") as f:
        json.dump(baselines, f, indent=2, default=lambda v: float(v) if hasattr(v, "item") else v)

    # Save final agent state
    final_agents = []
    for agent in model.agents:
        final_agents.append({
            "agent_id": agent.unique_id,
            "archetype_key": agent.archetype_key,
            "race": agent.race,
            "tract_id": agent.tract_id,
            "pop_count": agent.pop_count,
            "satisfaction": round(agent.satisfaction, 3),
            "moved_this_step": agent.moved_this_step,
            "years_in_tract": agent.years_in_tract,
        })
    with open(run_dir / "final_agents.json", "w") as f:
        json.dump(final_agents, f, indent=2)

    # Save archetype decisions (with per-tract move targets)
    archetype_decisions = {}
    for key, arch in model.active_archetypes.items():
        # Collect per-tract move targets for this archetype
        per_tract_targets = {
            tract_id: targets
            for (ak, tract_id), targets in model.move_targets.items()
            if ak == key and targets
        }
        archetype_decisions[key] = {
            "race": arch.race,
            "income_bracket": arch.income_bracket,
            "family_type": arch.family_type,
            "neighborhood_type": arch.neighborhood_type,
            "current_satisfaction": round(arch.current_satisfaction, 3),
            "would_move": arch.would_move,
            "key_reasons": arch.key_reasons,
            "move_rankings": arch.move_rankings,
            "per_tract_targets": per_tract_targets,
        }
    with open(run_dir / "archetype_decisions.json", "w") as f:
        json.dump(archetype_decisions, f, indent=2)

    # Save full LLM call log
    save_full_llm_log(run_dir, model)

    # Print LLM usage summary
    stats = model.llm.get_stats()
    print(f"\nLLM API Usage: {stats['total_calls']} calls, "
          f"{stats['total_input_tokens']:,} input tokens, "
          f"{stats['total_output_tokens']:,} output tokens")

    logger.info(f"\nAll output saved to {run_dir}/")
    print(f"\nOutput directory: {run_dir}")

    # ── Run post-run visualizations ──
    logger.info("Generating post-run visualizations...")
    try:
        from scripts.generate_run_visuals import (
            load_run, plot_comparison, plot_metrics_timeline,
            plot_satisfaction_map, plot_satisfaction_by_race,
            plot_movement_flows, plot_city_context, plot_summary_dashboard,
        )
        data = load_run(run_dir)
        plot_comparison(gdf, data, run_dir)
        plot_metrics_timeline(data, run_dir)
        plot_satisfaction_map(gdf, data, run_dir)
        plot_satisfaction_by_race(data, run_dir)
        plot_movement_flows(gdf, data, run_dir)
        plot_city_context(gdf, data, run_dir)
        plot_summary_dashboard(gdf, data, run_dir)
    except Exception as e:
        logger.warning(f"Post-run visualization failed: {e}")
        logger.info("You can generate visualizations manually with: "
                     f"python scripts/generate_run_visuals.py {run_dir}")


if __name__ == "__main__":
    main()
