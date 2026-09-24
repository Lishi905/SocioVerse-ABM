#!/usr/bin/env python3
"""
NaSch traffic model simulation consistency evaluation script
"""
import os
import sys
import time
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
try:
    import pandas as pd
except ImportError:
    pd = None

sys.path.insert(0, str(Path(__file__).parent))

from nasch_mesa import NaSchModel

def consistency(g, r):
    """
    Calculate consistency between ground-truth and predicted values.
    
    Args:
        g: array-like, ground-truth values g_k
        r: array-like, predicted/returned values r_k
    
    Returns:
        Consistency value between 0 and 1 (higher is better)
    """
    g = np.array(g, dtype=float)
    r = np.array(r, dtype=float)
    if np.any(g == 0):
        raise ValueError("Consistency denominator g_k contains zero; Eq. 1 is undefined.")

    term = ((np.abs(g - r) / g) ** 2)
    mean_term = np.mean(term)
    
    value = 1 - np.sqrt(mean_term)
    return max(0.0, value)

# Global figure storage for cumulative snapshots (key: (run_type, run_idx))
_cumulative_figures = {}

def init_cumulative_snapshot(model: NaSchModel, log_dir: Path, run_type: str, run_idx: int, num_steps: int):
    """
    Initialize a cumulative snapshot figure for a simulation run.
    This figure will be reused across all steps to accumulate vehicle positions.
    
    Args:
        model: NaSchModel instance
        log_dir: Directory to save snapshots
        run_type: "ABM" or "LLM"
        run_idx: Run index
        num_steps: Total number of steps (for Y-axis range)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure and axes (only once per run)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up plot with fixed axes (Y-axis fixed to max depth = num_steps)
    ax.set_xlim(-0.5, model.L - 0.5)
    ax.set_ylim(-0.5, num_steps - 0.5)  # Fixed Y-axis range: 0 to num_steps
    ax.set_xlabel("Position (Road Cell)")
    ax.set_ylabel("Time Step")
    
    # Lock axes limits to prevent auto-scaling during plotting
    ax.set_autoscale_on(False)  # Disable auto-scaling
    
    # Grid background - major grid lines at integer positions (every position)
    # But labels only every 10 steps for X-axis, every 5 steps for Y-axis
    ax.set_xticks(np.arange(0, model.L + 1, 1), minor=False)  # Grid lines at all integers
    ax.set_xticklabels([str(i) if i % 10 == 0 else '' for i in range(model.L + 1)], minor=False)  # Labels every 10
    
    ax.set_yticks(np.arange(0, num_steps + 1, 1), minor=False)  # Grid lines at all integers
    ax.set_yticklabels([str(i) if i % 5 == 0 else '' for i in range(num_steps + 1)], minor=False)  # Labels every 5
    
    ax.grid(which='major', color='lightgray', linewidth=0.5, alpha=0.5, linestyle='-')
    
    # Minor grid lines for finer grid (at half-integer positions)
    ax.set_xticks(np.arange(-0.5, model.L + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_steps + 0.5, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linewidth=0.2, alpha=0.3, linestyle='--')
    
    # Build title
    L = model.L
    rho = model.rho
    N = len(model.schedule.agents)
    vmax = model.vmax
    p = model.p
    
    title = f"NaSch {run_type} Mode - Cumulative Space-Time Diagram (Run {run_idx})\n"
    title += f"L={L}, ρ={rho:.2f}, N={N}, vmax={vmax}, p={p:.2f}"
    ax.set_title(title)
    
    # Store figure and axes
    key = (run_type, run_idx)
    _cumulative_figures[key] = {
        'fig': fig,
        'ax': ax,
        'log_dir': log_dir,
        'num_steps': num_steps,
        'L': model.L
    }
    
    return fig, ax

def save_cumulative_snapshot(model: NaSchModel, step: int, run_type: str, run_idx: int):
    """
    Save a cumulative snapshot: overlay current step's vehicle positions on top of all previous steps.
    Each snapshot shows the accumulation of all steps up to the current one.
    
    Args:
        model: NaSchModel instance
        step: Current step number
        run_type: "ABM" or "LLM"
        run_idx: Run index
    """
    key = (run_type, run_idx)
    if key not in _cumulative_figures:
        print(f"Warning: Cumulative figure not initialized for {run_type} run {run_idx}", file=sys.stderr)
        return
    
    fig = _cumulative_figures[key]['fig']
    ax = _cumulative_figures[key]['ax']
    log_dir = _cumulative_figures[key]['log_dir']
    num_steps = _cumulative_figures[key]['num_steps']
    L = _cumulative_figures[key]['L']
    
    # Ensure axes limits remain fixed (prevent auto-scaling)
    # Get the original limits from initialization
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Get current step's vehicle positions
    positions = [agent.position for agent in model.schedule.agents]
    
    # Overlay current step's positions using cross markers ('+')
    # Cross size should match grid cell size
    if positions:
        # Calculate appropriate marker size to match grid cell
        # For a figure of width 12 inches and L cells, each cell is approximately 12/L inches
        # Marker size 's' in scatter is in points^2, where 1 inch = 72 points
        # We want the cross to approximately fill the grid cell
        fig_width_inches = fig.get_figwidth()
        fig_height_inches = fig.get_figheight()
        dpi = fig.dpi
        
        # Approximate cell size in points
        x_cell_size_points = (fig_width_inches * dpi) / L
        y_cell_size_points = (fig_height_inches * dpi) / num_steps
        cell_size_points = min(x_cell_size_points, y_cell_size_points)
        
        # Marker size for cross (s parameter is area in points^2)
        # Make cross slightly smaller than cell to ensure visibility of grid
        marker_size = (cell_size_points * 0.8) ** 2
        
        # Use blue '+' markers, matching space-time diagram style
        # Line width for cross lines
        ax.scatter(positions, [step] * len(positions), 
                  s=marker_size, marker='+', alpha=0.8, color='blue', 
                  linewidths=1.5, zorder=5)
    
    # Re-apply fixed limits to ensure they don't change
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Save current cumulative state
    filename = f"step_{step:03d}.png"
    filepath = log_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    # Note: Do NOT close the figure - keep it open for next step

def close_cumulative_snapshot(run_type: str, run_idx: int):
    """
    Close the cumulative snapshot figure for a run.
    
    Args:
        run_type: "ABM" or "LLM"
        run_idx: Run index
    """
    key = (run_type, run_idx)
    if key in _cumulative_figures:
        fig = _cumulative_figures[key]['fig']
        plt.close(fig)
        del _cumulative_figures[key]

def save_snapshot(model: NaSchModel, step: int, log_dir: Path, run_type: str, run_idx: int):
    """
    Save cumulative snapshot (overlay current step on previous steps).
    This function is called each step to accumulate vehicle positions.
    """
    save_cumulative_snapshot(model, step, run_type, run_idx)

def save_spacetime_diagram(model: NaSchModel, log_dir: Path, run_type: str, run_idx: int):
    """
    Save a space-time diagram showing vehicle trajectories over all simulation steps.
    Based on original nasch_abm.py create_abm_visualization() logic.
    
    This generates a proper cellular automaton visualization where:
    - X-axis: Position on road
    - Y-axis: Time step
    - Each point represents a vehicle at a specific position and time
    - Diagonal lines show vehicle movement trajectories
    
    Args:
        model: NaSchModel instance (should have datacollector with all steps)
        log_dir: Directory to save diagram
        run_type: "ABM" or "LLM"
        run_idx: Run index
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if pd is None:
        print("Warning: pandas not available, cannot generate space-time diagram", file=sys.stderr)
        return
    
    # Get data from datacollector (contains all steps)
    try:
        df = model.datacollector.get_agent_vars_dataframe()
    except Exception as e:
        print(f"Warning: Could not get datacollector data: {e}", file=sys.stderr)
        return
    
    if df.empty:
        print("Warning: No data available for space-time diagram", file=sys.stderr)
        return
    
    # Extract positions and times for all steps
    positions = []
    times = []
    
    for step in df.index.get_level_values('Step').unique():
        step_data = df.loc[step]
        if not step_data.empty:
            if isinstance(step_data, pd.Series):
                # Only one vehicle recorded for this step
                agent_positions = [float(step_data.get('Position', 0.0))]
            else:
                agent_positions = step_data['Position'].values
            positions.extend(agent_positions)
            times.extend([step] * len(agent_positions))
    
    if not positions:
        print("Warning: No position data available", file=sys.stderr)
        return
    
    # Create space-time diagram (matching original demo)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot as scatter (matching original demo style: blue '+' markers)
    ax.scatter(positions, times, s=4, marker='+', alpha=0.7, color='blue')
    ax.invert_yaxis()  # Time increases downward
    
    ax.set_xlabel("Position (Road Cell)")
    ax.set_ylabel("Time Step")
    
    # Build title with parameters
    L = model.L
    rho = model.rho
    N = len(model.schedule.agents)
    vmax = model.vmax
    p = model.p
    steps = model.step_count
    
    title = f"NaSch {run_type} Mode - Space-Time Diagram (Run {run_idx})\n"
    title += f"L={L}, ρ={rho:.2f}, N={N}, steps={steps}, vmax={vmax}, p={p:.2f}"
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save diagram
    filename = f"spacetime_diagram_run{run_idx:02d}.png"
    filepath = log_dir / filename
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)

def save_step_log(model: NaSchModel, step: int, log_dir: Path, run_type: str, run_idx: int, save_snapshots: bool = False):
    """
    Save step-by-step log in Sugarscape format.
    
    Args:
        model: NaSchModel instance
        step: Current step number
        log_dir: Directory to save logs
        run_type: "ABM" or "LLM"
        run_idx: Run index
        save_snapshots: Whether to save visual snapshots (default: False)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate metrics for this step
    metrics = compute_metrics(model)
    
    # Collect all vehicles data
    vehicles_data = []
    for agent in model.schedule.agents:
        vehicle_data = {
            "id": agent.unique_id,
            "position": int(agent.position),
            "speed": int(agent.speed),
            "vmax": int(model.vmax) if hasattr(model, 'vmax') else 5
        }
        vehicles_data.append(vehicle_data)
    
    # Create step log in Sugarscape format
    step_log = {
        "step": step,
        "average_speed": float(metrics["average_speed"]),
        "density": float(metrics["density"]),
        "flow_rate": float(metrics["flow_rate"]),
        "num_vehicles": len(model.schedule.agents),
        "L": int(model.L),
        "vehicles": vehicles_data,
        "llm_responses": getattr(model, "last_llm_step_debug", []) if run_type.upper() == "LLM" else []
    }
    
    # Save to JSON file
    filename = f"step_{step:03d}.json"
    filepath = log_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(step_log, f, indent=2, ensure_ascii=False)
    
    # Save cumulative snapshot if enabled
    if save_snapshots:
        try:
            save_snapshot(model, step, log_dir, run_type, run_idx)
        except Exception as e:
            # Log error but don't stop execution
            print(f"Warning: Failed to save cumulative snapshot for step {step}: {e}", file=sys.stderr)

def compute_metrics(model: NaSchModel) -> Dict[str, float]:
    """Compute key metrics for NaSch simulation."""
    if not model.schedule.agents:
        return {
            "average_speed": 0.0,
            "density": 0.0,
            "flow_rate": 0.0
        }
    
    speeds = [agent.speed for agent in model.schedule.agents]
    avg_speed = np.mean(speeds)
    density = len(model.schedule.agents) / model.L
    flow_rate = avg_speed * density
    
    return {
        "average_speed": avg_speed,
        "density": density,
        "flow_rate": flow_rate
    }

def test_consistency(num_abm_runs: int = 10, num_llm_runs: int = 3, num_steps: int = 500, num_vehicles: int = 50, save_raw_data: bool = True, save_step_logs: bool = False, save_snapshots: bool = False):
    """Run simulation consistency evaluation.
    
    Args:
        num_abm_runs: Number of ABM runs (default: 10)
        num_llm_runs: Number of LLM runs (default: 3)
        num_steps: Number of simulation steps per run (default: 500)
        num_vehicles: Number of vehicles in simulation (default: 50)
        save_raw_data: Whether to save raw data to JSON file (default: True)
        save_step_logs: Whether to save step-by-step logs (default: False)
        save_snapshots: Whether to save visual snapshots (default: False)
    """
    
    print("=" * 80)
    print("NaSch Traffic Model Simulation Consistency Evaluation")
    print(f"Configuration: ABM runs={num_abm_runs}, LLM runs={num_llm_runs}, Steps={num_steps}, Vehicles={num_vehicles}")
    print(f"Step-by-step logs: {'Enabled' if save_step_logs else 'Disabled'}")
    print(f"Snapshots: {'Enabled' if save_snapshots else 'Disabled'}")
    print("=" * 80)
    
    # Create base output directory for step logs and snapshots
    if save_step_logs or save_snapshots:
        base_output_dir = Path(__file__).parent / "step_logs"
        base_output_dir.mkdir(exist_ok=True)
    else:
        base_output_dir = None
    
    # Check API key only if LLM runs are requested
    if num_llm_runs > 0:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM runs")
    
    L = 250
    rho = 0.20
    vmax = 5
    p = 0.50
    
    print(f"\n1. Running ABM {num_abm_runs} times...")
    abm_runs_metrics = {
        "average_speed": [],
        "density": [],
        "flow_rate": []
    }
    
    for run_idx in tqdm(range(num_abm_runs), desc="ABM runs", ncols=80, leave=False):
        abm_model = NaSchModel(L=L, rho=rho, vmax=vmax, p=p, mode="abm", seed=42+run_idx)
        
        # Create log directory for this run if step logs or snapshots are enabled
        run_log_dir = None
        if (save_step_logs or save_snapshots) and base_output_dir is not None:
            run_log_dir = base_output_dir / f"run{run_idx}_ABM" / "logs"
            run_log_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize cumulative snapshot figure if snapshots enabled
            if save_snapshots:
                try:
                    init_cumulative_snapshot(abm_model, run_log_dir, "ABM", run_idx, num_steps)
                    # Save initial state (step 0)
                    save_snapshot(abm_model, 0, run_log_dir, "ABM", run_idx)
                except Exception as e:
                    print(f"Warning: Failed to initialize cumulative snapshot for ABM run {run_idx}: {e}", file=sys.stderr)
            
            # Save initial state (step 0) if step logs enabled
            if save_step_logs:
                save_step_log(abm_model, 0, run_log_dir, "ABM", run_idx, save_snapshots)
        
        for step in tqdm(range(num_steps), desc=f"ABM run {run_idx+1}", ncols=80, leave=False):
            abm_model.step()
            
            # Save step log and cumulative snapshot if enabled
            if save_step_logs:
                save_step_log(abm_model, step + 1, run_log_dir, "ABM", run_idx, save_snapshots)
            elif save_snapshots:
                # If only snapshots enabled (no step logs), still need to save snapshot
                save_snapshot(abm_model, step + 1, run_log_dir, "ABM", run_idx)
        
        # Close cumulative snapshot figure after simulation completes
        if save_snapshots:
            try:
                close_cumulative_snapshot("ABM", run_idx)
            except Exception as e:
                print(f"Warning: Failed to close cumulative snapshot for ABM run {run_idx}: {e}", file=sys.stderr)
        
        metrics = compute_metrics(abm_model)
        abm_runs_metrics["average_speed"].append(metrics["average_speed"])
        abm_runs_metrics["density"].append(metrics["density"])
        abm_runs_metrics["flow_rate"].append(metrics["flow_rate"])
    
    abm_average = {
        "average_speed": np.mean(abm_runs_metrics["average_speed"]),
        "density": np.mean(abm_runs_metrics["density"]),
        "flow_rate": np.mean(abm_runs_metrics["flow_rate"])
    }
    
    print(f"\n   ABM Average Metrics:")
    print(f"      Average Speed: {abm_average['average_speed']:.4f}")
    print(f"      Density: {abm_average['density']:.4f}")
    print(f"      Flow Rate: {abm_average['flow_rate']:.4f}")
    
    print("\n2. Calculating ABM self-consistency...")
    abm_self_consistencies = {}
    for metric_name in abm_runs_metrics.keys():
        metric_values = abm_runs_metrics[metric_name]
        consistencies = []
        for val in metric_values:
            cons = consistency([abm_average[metric_name]], [val])
            consistencies.append(cons)
        abm_self_consistencies[metric_name] = np.mean(consistencies)
        print(f"   {metric_name.replace('_', ' ').title()}: {abm_self_consistencies[metric_name]:.4f}")
    
    print(f"\n3. Running LLM simulation {num_llm_runs} time(s)...")
    llm_runs_metrics = {
        "average_speed": [],
        "density": [],
        "flow_rate": []
    }
    
    for llm_run_idx in range(num_llm_runs):
        if num_llm_runs > 1:
            print(f"\n   LLM Run {llm_run_idx + 1}/{num_llm_runs}...")
        
        llm_model = NaSchModel(L=L, rho=rho, vmax=vmax, p=p, mode="llm", seed=42+llm_run_idx)
        
        llm_model.llm_decision_tick_interval = 1
        llm_model.llm_cache_duration = 1
        llm_model.synchronous_mode = True
        
        # Create log directory for this run if step logs or snapshots are enabled
        run_log_dir = None
        if (save_step_logs or save_snapshots) and base_output_dir is not None:
            run_log_dir = base_output_dir / f"run{llm_run_idx}_LLM" / "logs"
            run_log_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize cumulative snapshot figure if snapshots enabled
            if save_snapshots:
                try:
                    init_cumulative_snapshot(llm_model, run_log_dir, "LLM", llm_run_idx, num_steps)
                    # Save initial state (step 0)
                    save_snapshot(llm_model, 0, run_log_dir, "LLM", llm_run_idx)
                except Exception as e:
                    print(f"Warning: Failed to initialize cumulative snapshot for LLM run {llm_run_idx}: {e}", file=sys.stderr)
            
            # Save initial state (step 0) if step logs enabled
            if save_step_logs:
                save_step_log(llm_model, 0, run_log_dir, "LLM", llm_run_idx, save_snapshots)
        
        start_time = time.time()
        
        run_completed_all_steps = True
        failed_step = None
        
        for step in tqdm(range(num_steps), desc=f"LLM run {llm_run_idx+1}", ncols=80, leave=False):
            try:
                llm_model.run_synchronous_llm_step()
                llm_model.step()
                
                # Save step log and cumulative snapshot if enabled
                if save_step_logs:
                    save_step_log(llm_model, step + 1, run_log_dir, "LLM", llm_run_idx, save_snapshots)
                elif save_snapshots:
                    # If only snapshots enabled (no step logs), still need to save snapshot
                    save_snapshot(llm_model, step + 1, run_log_dir, "LLM", llm_run_idx)
            except Exception as e:
                tqdm.write(f"✗ ERROR at step {step}: {e}")
                import traceback
                traceback.print_exc()
                run_completed_all_steps = False
                failed_step = step
                break
        
        # Close cumulative snapshot figure after simulation completes
        if save_snapshots:
            try:
                close_cumulative_snapshot("LLM", llm_run_idx)
            except Exception as e:
                print(f"Warning: Failed to close cumulative snapshot for LLM run {llm_run_idx}: {e}", file=sys.stderr)
        
        if run_completed_all_steps:
            metrics = compute_metrics(llm_model)
            llm_runs_metrics["average_speed"].append(metrics["average_speed"])
            llm_runs_metrics["density"].append(metrics["density"])
            llm_runs_metrics["flow_rate"].append(metrics["flow_rate"])
        else:
            print(f"Run {llm_run_idx} failed at step {failed_step}; no metrics computed")
        
        elapsed_time = time.time() - start_time
        if num_llm_runs == 1:
            print(f"\n   Total time: {elapsed_time:.2f} seconds")
        else:
            print(f"      Run {llm_run_idx + 1} completed in {elapsed_time:.2f} seconds")
    
    if not llm_runs_metrics["average_speed"]:
        raise RuntimeError("All NaSch LLM runs failed; no metrics computed")

    llm_average = {
        "average_speed": np.mean(llm_runs_metrics["average_speed"]),
        "density": np.mean(llm_runs_metrics["density"]),
        "flow_rate": np.mean(llm_runs_metrics["flow_rate"])
    }
    
    print(f"\n   LLM Average Metrics ({num_llm_runs} run(s)):")
    print(f"      Average Speed: {llm_average['average_speed']:.4f}")
    print(f"      Density: {llm_average['density']:.4f}")
    print(f"      Flow Rate: {llm_average['flow_rate']:.4f}")
    
    if num_llm_runs > 1:
        print("\n   Calculating LLM self-consistency...")
        llm_self_consistencies = {}
        for metric_name in llm_runs_metrics.keys():
            metric_values = llm_runs_metrics[metric_name]
            consistencies = []
            for val in metric_values:
                cons = consistency([llm_average[metric_name]], [val])
                consistencies.append(cons)
            llm_self_consistencies[metric_name] = np.mean(consistencies)
            print(f"      {metric_name.replace('_', ' ').title()}: {llm_self_consistencies[metric_name]:.4f}")
    
    print("\n4. Calculating ABM-LLM Consistency...")
    abm_llm_consistencies = {}
    for metric_name in ["average_speed", "density", "flow_rate"]:
        g = [abm_average[metric_name]]
        r = [llm_average[metric_name]]
        cons = consistency(g, r)
        abm_llm_consistencies[metric_name] = cons
        print(f"   {metric_name.replace('_', ' ').title()}: {cons:.4f}")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"ABM Average Metrics:")
    for metric_name, value in abm_average.items():
        print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")
    print(f"\nLLM Average Metrics:")
    for metric_name, value in llm_average.items():
        print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")
    print(f"\nABM-LLM Consistency:")
    for metric_name, value in abm_llm_consistencies.items():
        print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")
    
    # Prepare result dictionary with all data
    result = {
        "metadata": {
            "model_name": "NaSch",
            "mode": "abm+llm",
            "llm_provider": os.getenv("LLM_PROVIDER", "gpt-4o"),
            "llm_model_string": os.getenv("LLM_MODEL_NAME", "gpt-4o"),
            "llm_temperature": 0.1,
            "llm_max_tokens": 16,
            "random_seed": 42,
            "num_agents": num_vehicles,
            "num_steps": num_steps,
            "parameters": {"L": L, "rho": rho, "vmax": vmax, "p": p},
            "total_llm_calls": len(llm_runs_metrics["average_speed"]) * num_steps * num_vehicles,
            "total_input_tokens": None,
            "total_output_tokens": None,
            "failed_calls": max(0, num_llm_runs - len(llm_runs_metrics["average_speed"])) * num_steps * num_vehicles,
            "run_completed": len(llm_runs_metrics["average_speed"]) == num_llm_runs,
            "timestamp": datetime.now().isoformat()
        },
        "experiment_config": {
            "num_abm_runs": num_abm_runs,
            "num_llm_runs": num_llm_runs,
            "num_steps": num_steps,
            "num_vehicles": num_vehicles,
            "L": L,
            "rho": rho,
            "vmax": vmax,
            "p": p,
            "timestamp": datetime.now().isoformat()
        },
        "abm_average": abm_average,
        "llm_average": llm_average,
        "abm_llm_consistencies": abm_llm_consistencies,
        "abm_self_consistencies": abm_self_consistencies,
        # Include raw data
        "abm_raw_data": {
            "runs": []
        },
        "llm_raw_data": {
            "runs": []
        }
    }
    
    # Add raw ABM data (each run's metrics)
    for run_idx in range(num_abm_runs):
        result["abm_raw_data"]["runs"].append({
            "run_index": run_idx,
            "seed": 42 + run_idx,
            "average_speed": abm_runs_metrics["average_speed"][run_idx],
            "density": abm_runs_metrics["density"][run_idx],
            "flow_rate": abm_runs_metrics["flow_rate"][run_idx]
        })
    
    # Add raw LLM data (each run's metrics)
    for run_idx in range(len(llm_runs_metrics["average_speed"])):
        result["llm_raw_data"]["runs"].append({
            "run_index": run_idx,
            "seed": 42 + run_idx,
            "average_speed": llm_runs_metrics["average_speed"][run_idx],
            "density": llm_runs_metrics["density"][run_idx],
            "flow_rate": llm_runs_metrics["flow_rate"][run_idx]
        })
    
    # Save raw data to JSON file if requested
    if save_raw_data:
        output_dir = Path(__file__).parent / "raw_data"
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp and config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nasch_consistency_{num_abm_runs}abm_{num_llm_runs}llm_{num_steps}steps_{num_vehicles}vehicles_{timestamp}.json"
        output_path = output_dir / filename
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_result = convert_to_serializable(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Raw data saved to: {output_path}")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NaSch traffic model simulation consistency evaluation"
    )
    parser.add_argument(
        "--num-abm-runs",
        type=int,
        default=10,
        help="Number of ABM runs (default: 10)"
    )
    parser.add_argument(
        "--num-llm-runs",
        type=int,
        default=3,
        help="Number of LLM runs (default: 3)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500,
        help="Number of simulation steps per run (default: 500)"
    )
    parser.add_argument(
        "--num-vehicles",
        type=int,
        default=50,
        help="Number of vehicles in simulation (default: 50)"
    )
    parser.add_argument(
        "--no-save-raw-data",
        action="store_true",
        help="Do not save raw data to JSON file"
    )
    
    args = parser.parse_args()
    
    test_consistency(
        num_abm_runs=args.num_abm_runs,
        num_llm_runs=args.num_llm_runs,
        num_steps=args.num_steps,
        num_vehicles=args.num_vehicles,
        save_raw_data=not args.no_save_raw_data
    )
