"""
Run Schelling segregation simulation with LLM-based agent decision making.
Now with async/concurrent support for significantly faster performance.
Enhanced with step-by-step visualization and GIF generation for all engine types.
"""

import matplotlib
matplotlib.use("Agg")

import os
import logging
import asyncio
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from PIL import Image

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import LLMConfig, SimulationConfig, ExperimentConfig, print_config
from schelling_model import SchellingModel
from llm_integration import LLMDecisionMaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_step_image(model, step, step_dir, experiment_name=""):
    """
    Save a single step image showing agent distribution at current state.
    
    Args:
        model: SchellingModel instance
        step: Current step number
        step_dir: Directory to save the image
        experiment_name: Name of the experiment for the title
    """
    # Ensure directory exists
    os.makedirs(step_dir, exist_ok=True)
    
    # Get the agent data
    agent_counts = np.zeros((model.grid.width, model.grid.height))
    
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            cell_content = model.grid.get_cell_list_contents([(x, y)])
            if cell_content:
                agent_counts[x][y] = cell_content[0].type
            else:
                agent_counts[x][y] = -1  # Empty cell
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    # Plot the state
    colors = ['red', 'blue', 'white']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    ax.imshow(agent_counts, cmap=cmap, norm=norm)
    
    # Title with experiment name
    title = f'Schelling Segregation - {experiment_name} - Step {step}' if experiment_name else f'Step {step}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    
    # Create legend
    red_patch = mpatches.Patch(color='red', label='Type 0 (Majority)')
    blue_patch = mpatches.Patch(color='blue', label='Type 1 (Minority)')
    white_patch = mpatches.Patch(color='white', label='Empty')
    ax.legend(handles=[red_patch, blue_patch, white_patch], loc='upper right')
    
    # Add statistics text box
    total_agents = model.schedule.get_agent_count()
    happiness_ratio = model.happy / total_agents if total_agents > 0 else 0
    
    stats_text = f'Step: {step}\nHappy: {model.happy}/{total_agents}\nRatio: {happiness_ratio:.1%}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='lightgray', alpha=0.9, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(f'{step_dir}/step_{step:04d}.png', dpi=100, bbox_inches='tight')
    plt.close()


def create_simulation_gif(step_dir, total_steps, experiment_name=""):
    """
    Create a GIF animation from saved step images.
    
    Args:
        step_dir: Directory containing step images
        total_steps: Total number of steps in simulation
        experiment_name: Name of the experiment
    """
    logger.info(f"Creating GIF animation for {experiment_name}...")
    
    # Get all step images
    image_files = []
    for i in range(total_steps + 1):
        filename = f'{step_dir}/step_{i:04d}.png'
        if os.path.exists(filename):
            image_files.append(filename)
    
    if not image_files:
        logger.warning(f"No step images found in {step_dir}")
        return
    
    # Load images
    images = []
    for filename in image_files:
        try:
            img = Image.open(filename)
            images.append(img)
        except Exception as e:
            logger.error(f"Failed to open image {filename}: {e}")
    
    if not images:
        logger.warning("No images loaded for GIF creation")
        return
    
    # Create GIF
    gif_filename = f'{step_dir}/{experiment_name}_simulation.gif' if experiment_name else f'{step_dir}/simulation.gif'
    try:
        images[0].save(
            gif_filename,
            save_all=True,
            append_images=images[1:],
            duration=300,  # 300ms per frame
            loop=0  # Infinite loop
        )
        logger.info(f"GIF animation saved: {gif_filename} ({len(images)} frames)")
    except Exception as e:
        logger.error(f"Failed to create GIF: {e}")


def run_experiment(experiment_name: str, experiment_config: dict, steps: int = 100, use_async: bool = True):
    """
    Run a single experiment comparing LLM-based vs rule-based agents.
    
    Args:
        experiment_name: Name of the experiment
        experiment_config: Configuration dict for the experiment
        steps: Number of simulation steps
        use_async: Whether to use async LLM calls for better performance
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Mode: {'Async (Concurrent LLM)' if use_async and experiment_config.get('use_llm') else 'Synchronous'}")
    logger.info(f"{'='*70}")
    
    # Initialize LLM decision maker if needed
    llm_decision_maker = None
    if experiment_config.get("use_llm", False):
        try:
            logger.info(f"Initializing LLM decision maker with provider: {LLMConfig.PROVIDER}")
            llm_decision_maker = LLMDecisionMaker(
                provider_name=LLMConfig.PROVIDER,
                use_cache=LLMConfig.USE_CACHE,
                **LLMConfig.get_provider_config()
            )
            logger.info("LLM decision maker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            logger.warning("Falling back to rule-based decision making")
            experiment_config["use_llm"] = False
    
    # Create model
    model = SchellingModel(
        height=SimulationConfig.GRID_HEIGHT,
        width=SimulationConfig.GRID_WIDTH,
        density=SimulationConfig.DENSITY,
        minority_pc=SimulationConfig.MINORITY_PERCENTAGE,
        homophily=SimulationConfig.HOMOPHILY,
        use_llm=experiment_config.get("use_llm", False),
        llm_decision_maker=llm_decision_maker,
        llm_only_percentage=experiment_config.get("llm_only_percentage", 1.0)
    )
    
    logger.info(f"Model created with {model.schedule.get_agent_count()} agents")
    if llm_decision_maker:
        logger.info(f"LLM agents: {model.llm_agents_count} ({100*model.llm_agents_count/model.schedule.get_agent_count():.1f}%)")
    
    # Prepare visualization directory
    step_dir = None
    if SimulationConfig.SAVE_GIF:
        step_dir = os.path.join(SimulationConfig.OUTPUT_DIR, f"{experiment_name}_steps")
        os.makedirs(step_dir, exist_ok=True)
        logger.info(f"Will save step visualizations to: {step_dir}")
    
    # Run simulation
    converged_step = None
    
    if use_async and experiment_config.get("use_llm", False):
        logger.info("Using async concurrent LLM calls...")
        with tqdm(total=steps, desc=f"Running {experiment_name} (async)", unit="step") as pbar:
            for step in range(steps):
                # Use async step with concurrency
                model.step_with_concurrency()
                pbar.update(1)
                
                # Save step visualization
                if step_dir and step % SimulationConfig.SAVE_EVERY_N_STEPS == 0:
                    save_step_image(model, step, step_dir, experiment_name)
                
                if not model.running:
                    converged_step = step
                    pbar.set_description(f"{experiment_name} (async) - Converged at step {step}")
                    # Save final step
                    if step_dir:
                        save_step_image(model, step, step_dir, experiment_name)
                    break
    else:
        logger.info("Using standard synchronous mode...")
        with tqdm(total=steps, desc=f"Running {experiment_name}", unit="step") as pbar:
            for step in range(steps):
                model.step()
                pbar.update(1)
                
                # Save step visualization
                if step_dir and step % SimulationConfig.SAVE_EVERY_N_STEPS == 0:
                    save_step_image(model, step, step_dir, experiment_name)
                
                if not model.running:
                    converged_step = step
                    pbar.set_description(f"{experiment_name} - Converged at step {step}")
                    # Save final step
                    if step_dir:
                        save_step_image(model, step, step_dir, experiment_name)
                    break
    
    # Create GIF if visualization was enabled
    if step_dir:
        actual_steps = converged_step if converged_step is not None else steps - 1
        create_simulation_gif(step_dir, actual_steps, experiment_name)
    
    # Extract results
    total_agents = model.schedule.get_agent_count()
    final_happy = model.happy
    happiness_ratio = final_happy / total_agents if total_agents > 0 else 0
    
    results = {
        "experiment_name": experiment_name,
        "use_llm": experiment_config.get("use_llm", False),
        "llm_percentage": experiment_config.get("llm_only_percentage", 0),
        "total_agents": total_agents,
        "final_happy": final_happy,
        "happiness_ratio": happiness_ratio,
        "converged_step": converged_step if converged_step is not None else steps,
        "converged": model.running == False,
        "model_data": model.datacollector.get_model_vars_dataframe(),
        "used_async": use_async and experiment_config.get("use_llm", False)
    }
    
    logger.info(f"\nExperiment Results: {experiment_name}")
    logger.info(f"  Total Agents: {total_agents}")
    logger.info(f"  Final Happy Agents: {final_happy}")
    logger.info(f"  Happiness Ratio: {happiness_ratio:.2%}")
    logger.info(f"  Converged: {results['converged']}")
    logger.info(f"  Convergence Step: {results['converged_step']}")
    
    return results


def compare_experiments(experiments_to_run: list = None, use_async: bool = True):
    """
    Run multiple experiments and compare results.
    
    Args:
        experiments_to_run: List of experiment names to run
        use_async: Whether to use async LLM calls
    """
    if experiments_to_run is None:
        experiments_to_run = [
            "llm_all",
            "rule_based_only",
            "llm_hybrid_50"
        ]
    
    results_list = []
    
    for exp_name in experiments_to_run:
        try:
            exp_config = ExperimentConfig.get_experiment_config(exp_name)
            results = run_experiment(exp_name, exp_config, steps=SimulationConfig.STEPS, use_async=use_async)
            results_list.append(results)
        except Exception as e:
            logger.error(f"Failed to run experiment {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison visualization
    if results_list:
        create_comparison_plot(results_list)


def create_comparison_plot(results_list: list):
    """
    Create visualization comparing different experiments.
    
    Args:
        results_list: List of experiment results
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    exp_names = [r["experiment_name"] for r in results_list]
    happiness_ratios = [r["happiness_ratio"] for r in results_list]
    converged_steps = [r["converged_step"] for r in results_list]
    llm_percentages = [r["llm_percentage"] * 100 for r in results_list]
    
    # Plot 1: Happiness Ratio Comparison
    ax1 = axes[0, 0]
    colors = ['blue' if not r["use_llm"] else 'red' for r in results_list]
    bars1 = ax1.bar(range(len(exp_names)), happiness_ratios, color=colors, alpha=0.7)
    ax1.set_ylabel('Happiness Ratio')
    ax1.set_title('Final Happiness Ratio Comparison')
    ax1.set_xticks(range(len(exp_names)))
    ax1.set_xticklabels(exp_names, rotation=45, ha='right')
    ax1.set_ylim([0, 1])
    
    # Add value labels on bars
    for i, (bar, ratio) in enumerate(zip(bars1, happiness_ratios)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{ratio:.1%}', ha='center', va='bottom')
    
    # Plot 2: Convergence Steps
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(exp_names)), converged_steps, color=colors, alpha=0.7)
    ax2.set_ylabel('Steps to Convergence')
    ax2.set_title('Convergence Speed')
    ax2.set_xticks(range(len(exp_names)))
    ax2.set_xticklabels(exp_names, rotation=45, ha='right')
    
    # Plot 3: LLM Percentage
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(exp_names)), llm_percentages, color='green', alpha=0.7)
    ax3.set_ylabel('Percentage of LLM Agents (%)')
    ax3.set_title('LLM Agent Percentage')
    ax3.set_xticks(range(len(exp_names)))
    ax3.set_xticklabels(exp_names, rotation=45, ha='right')
    ax3.set_ylim([0, 100])
    
    # Plot 4: Happiness Over Time
    ax4 = axes[1, 1]
    for result in results_list:
        model_data = result["model_data"]
        ax4.plot(model_data.index, model_data['Happy'], label=result["experiment_name"], linewidth=2)
    
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Happy Agents')
    ax4.set_title('Happiness Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = SimulationConfig.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'llm_experiment_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\nComparison plot saved to: {output_path}")
    
    plt.close()


def run_single_llm_simulation(use_async: bool = True):
    """
    Run a single LLM-powered simulation with visualization.
    
    Args:
        use_async: Whether to use async concurrent LLM calls
    """
    logger.info("\n" + "="*70)
    logger.info(f"Running Single LLM-Powered Simulation ({'Async' if use_async else 'Sync'})")
    logger.info("="*70)
    
    # Initialize LLM decision maker
    try:
        logger.info(f"Initializing LLM with provider: {LLMConfig.PROVIDER}")
        llm_decision_maker = LLMDecisionMaker(
            provider_name=LLMConfig.PROVIDER,
            use_cache=LLMConfig.USE_CACHE,
            **LLMConfig.get_provider_config()
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        logger.warning("Falling back to rule-based decision making")
        llm_decision_maker = None
    
    # Create model with LLM
    model = SchellingModel(
        height=SimulationConfig.GRID_HEIGHT,
        width=SimulationConfig.GRID_WIDTH,
        density=SimulationConfig.DENSITY,
        minority_pc=SimulationConfig.MINORITY_PERCENTAGE,
        homophily=SimulationConfig.HOMOPHILY,
        use_llm=True if llm_decision_maker else False,
        llm_decision_maker=llm_decision_maker,
        llm_only_percentage=1.0
    )
    
    logger.info(f"Created model with {model.schedule.get_agent_count()} agents")
    logger.info(f"LLM agents: {model.llm_agents_count}")
    
    # Run simulation
    if use_async and llm_decision_maker:
        logger.info("Using async concurrent LLM calls for faster execution...")
        with tqdm(total=SimulationConfig.STEPS, desc="Running LLM Simulation (async)", unit="step") as pbar:
            for step in range(SimulationConfig.STEPS):
                model.step_with_concurrency()
                pbar.update(1)
                if not model.running:
                    pbar.set_description(f"Converged at step {step}")
                    break
    else:
        logger.info("Using standard synchronous mode...")
        with tqdm(total=SimulationConfig.STEPS, desc="Running LLM Simulation", unit="step") as pbar:
            for step in range(SimulationConfig.STEPS):
                model.step()
                pbar.update(1)
                if not model.running:
                    pbar.set_description(f"Converged at step {step}")
                    break
    
    # Create visualization
    model_data = model.datacollector.get_model_vars_dataframe()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(model_data.index, model_data['Happy'], linewidth=2, label='Happy Agents')
    ax.plot(model_data.index, model_data['LLMAgents'], linewidth=2, label='LLM Agents', linestyle='--')
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.set_title(f'LLM-Powered Schelling Segregation Simulation ({("Async" if use_async else "Sync")})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = SimulationConfig.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'llm_simulation_{"async" if use_async else "sync"}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Simulation plot saved to: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    # Print configuration
    print_config()
    
    # Run experiments
    logger.info("\nStarting LLM-based Schelling Segregation Experiments")
    logger.info(f"Simulator will use: {LLMConfig.PROVIDER}")
    logger.info("Using ASYNC/CONCURRENT mode for better performance!")
    
    # You can run different types of experiments:
    
    # 1. Compare different approaches (with async enabled)
    # Start with rule-based for quick test, then hybrid, then full LLM
    compare_experiments(experiments_to_run=[
        "llm_all",
        "rule_based_only",
        "llm_hybrid_50",
    ], use_async=True)
    
    # 2. Or run a single LLM simulation with async
    # run_single_llm_simulation(use_async=True)
    
    logger.info("\n" + "="*70)
    logger.info("All experiments completed!")
    logger.info("="*70)
