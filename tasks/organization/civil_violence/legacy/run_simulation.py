# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from civil_violence_model import CivilViolenceModel
from agents import Citizen, Cop


def run_civil_violence_simulation(steps=200, height=40, width=40, citizen_density=0.2,
                                 cop_density=0.001, legitimacy=0.5, save_gif=True):
    """
    Run a civil violence simulation and visualize the results.

    Args:
        steps: Number of steps to run
        height: Grid height
        width: Grid width
        citizen_density: Density of citizens
        cop_density: Density of cops
        legitimacy: Government legitimacy (0-1)
        save_gif: Whether to save individual step images and create GIF
    """
    # Create directories for output
    simulation_steps_dir = "./raw_simulations/organization_model/civil_violence/simulation_steps"
    output_figs_dir = "./raw_simulations/organization_model/civil_violence/output_figs"
    
    if save_gif:
        if not os.path.exists(simulation_steps_dir):
            os.makedirs(simulation_steps_dir)
        if not os.path.exists(output_figs_dir):
            os.makedirs(output_figs_dir)
    
    # Create and run the model
    model = CivilViolenceModel(
        height=height, width=width, citizen_density=citizen_density,
        cop_density=cop_density, legitimacy=legitimacy
    )

    print(f"Running Civil Violence Model for {steps} steps...")
    print(f"Initial setup: {model.count_type(model, 'Citizen')} citizens, "
          f"{model.count_type(model, 'Cop')} cops")

    # Save initial state
    if save_gif:
        # Create directory if it doesn't exist
        os.makedirs(simulation_steps_dir, exist_ok=True)
        save_step_image(model, 0, simulation_steps_dir)

    for i in range(steps):
        model.step()
        
        # Save image at key intervals
        if save_gif and (i % 1 == 0):  # More frequent saves in early steps
            save_step_image(model, i + 1, simulation_steps_dir)
        
        if i % 50 == 0:
            active = model.count_type(model, 'Active')
            jailed = model.count_type(model, 'Jailed')
            print(f"Step {i}: {active} active citizens, {jailed} jailed")

    # Create final visualization
    create_civil_violence_visualization(model, output_figs_dir)

    # Create GIF from saved images
    if save_gif:
        create_simulation_gif(simulation_steps_dir, steps)

    # Print final statistics
    print_final_statistics(model)

    return model


def save_step_image(model, step, step_dir):
    """
    Save a single step image showing agent positions as scatter points.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Collect agent positions
    quiescent_x, quiescent_y = [], []
    active_x, active_y = [], []
    jailed_x, jailed_y = [], []
    cops_x, cops_y = [], []
    
    # Iterate through all agents
    agents = [agent for cell in model.grid.coord_iter() for agent in cell[0]]
    for agent in agents:
        # Check if agent has a valid position
        if hasattr(agent, 'pos') and agent.pos is not None:
            if isinstance(agent, Citizen):
                if agent.jail_sentence > 0:
                    jailed_x.append(agent.pos[0])
                    jailed_y.append(agent.pos[1])
                elif agent.state.value == 1:  # Active
                    active_x.append(agent.pos[0])
                    active_y.append(agent.pos[1])
                else:
                    quiescent_x.append(agent.pos[0])
                    quiescent_y.append(agent.pos[1])
            elif isinstance(agent, Cop):
                cops_x.append(agent.pos[0])
                cops_y.append(agent.pos[1])
    
    # Plot agents as scatter points
    if quiescent_x:
        ax.scatter(quiescent_x, quiescent_y, c='lightblue', s=30, alpha=0.8, 
                  label=f'Quiescent ({len(quiescent_x)})', edgecolors='darkblue', linewidth=0.5)
    
    if active_x:
        ax.scatter(active_x, active_y, c='red', s=50, alpha=0.9, 
                  label=f'Active ({len(active_x)})', edgecolors='darkred', linewidth=0.5)
    
    if jailed_x:
        ax.scatter(jailed_x, jailed_y, c='orange', s=40, alpha=0.8, 
                  label=f'Jailed ({len(jailed_x)})', edgecolors='darkorange', linewidth=0.5)
    
    if cops_x:
        ax.scatter(cops_x, cops_y, c='black', s=60, alpha=1.0, 
                  label=f'Cops ({len(cops_x)})', edgecolors='gray', linewidth=0.5, marker='s')
    
    # Set up the plot
    ax.set_xlim(-0.5, model.grid.width - 0.5)
    ax.set_ylim(-0.5, model.grid.height - 0.5)
    ax.set_xlabel('Grid X', fontsize=12)
    ax.set_ylabel('Grid Y', fontsize=12)
    ax.set_title(f'Civil Violence Simulation - Step {step}', fontsize=16, fontweight='bold')
    
    # Add grid for better visualization
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xticks(range(0, model.grid.width, 5))
    ax.set_yticks(range(0, model.grid.height, 5))
    
    # Add statistics text box
    total_citizens = len(quiescent_x) + len(active_x) + len(jailed_x)
    active_percentage = (len(active_x) / total_citizens * 100) if total_citizens > 0 else 0
    
    stats_text = f'Total Citizens: {total_citizens}\nActive: {len(active_x)} ({active_percentage:.1f}%)\nJailed: {len(jailed_x)}\nCops: {len(cops_x)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='lightgray', alpha=0.9, edgecolor='black'))
    
    # Create legend
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Set aspect ratio to be equal
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{step_dir}/step_{step:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_simulation_gif(step_dir, total_steps):
    """
    Create a GIF animation from saved step images.
    """
    print("Creating GIF animation...")
    
    # Get all step images
    image_files = []
    for i in range(total_steps + 1):
        filename = f'{step_dir}/step_{i:03d}.png'
        if os.path.exists(filename):
            image_files.append(filename)
    
    if not image_files:
        print("No step images found!")
        return
    
    # Load images
    images = []
    for filename in image_files:
        img = Image.open(filename)
        images.append(img)
    
    # Create GIF in simulation_steps directory
    gif_filename = f'{step_dir}/civil_violence_simulation.gif'
    images[0].save(
        gif_filename,
        save_all=True,
        append_images=images[1:],
        duration=500,  # 500ms per frame
        loop=0  # Infinite loop
    )
    
    print(f"GIF animation saved as '{gif_filename}'")
    print(f"Total frames: {len(images)}")


def create_civil_violence_visualization(model, output_figs_dir):
    """
    Create visualization plots for the civil violence model.
    """
    # Get model data
    model_data = model.datacollector.get_model_vars_dataframe()

    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Population dynamics over time
    ax1.plot(model_data.index, model_data['Quiescent'], label='Quiescent Citizens', color='blue')
    ax1.plot(model_data.index, model_data['Active'], label='Active Citizens', color='red')
    ax1.plot(model_data.index, model_data['Jailed'], label='Jailed Citizens', color='orange')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Number of Citizens')
    ax1.set_title('Citizen Status Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Active vs Jailed citizens
    ax2.plot(model_data.index, model_data['Active'], label='Active', color='red', linewidth=2)
    ax2.plot(model_data.index, model_data['Jailed'], label='Jailed', color='orange', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Number of Citizens')
    ax2.set_title('Active vs Jailed Citizens')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Current state visualization using scatter plot
    # Collect agent positions
    quiescent_x, quiescent_y = [], []
    active_x, active_y = [], []
    jailed_x, jailed_y = [], []
    cops_x, cops_y = [], []
    
    # Iterate through all agents
    agents = [agent for cell in model.grid.coord_iter() for agent in cell[0]]
    for agent in agents:
        # Check if agent has a valid position
        if hasattr(agent, 'pos') and agent.pos is not None:
            if isinstance(agent, Citizen):
                if agent.jail_sentence > 0:
                    jailed_x.append(agent.pos[0])
                    jailed_y.append(agent.pos[1])
                elif agent.state.value == 1:  # Active
                    active_x.append(agent.pos[0])
                    active_y.append(agent.pos[1])
                else:
                    quiescent_x.append(agent.pos[0])
                    quiescent_y.append(agent.pos[1])
            elif isinstance(agent, Cop):
                cops_x.append(agent.pos[0])
                cops_y.append(agent.pos[1])
    
    # Plot agents as scatter points
    if quiescent_x:
        ax3.scatter(quiescent_x, quiescent_y, c='lightblue', s=20, alpha=0.7, 
                   label=f'Quiescent ({len(quiescent_x)})', edgecolors='darkblue', linewidth=0.3)
    
    if active_x:
        ax3.scatter(active_x, active_y, c='red', s=30, alpha=0.8, 
                   label=f'Active ({len(active_x)})', edgecolors='darkred', linewidth=0.3)
    
    if jailed_x:
        ax3.scatter(jailed_x, jailed_y, c='orange', s=25, alpha=0.7, 
                   label=f'Jailed ({len(jailed_x)})', edgecolors='darkorange', linewidth=0.3)
    
    if cops_x:
        ax3.scatter(cops_x, cops_y, c='black', s=35, alpha=1.0, 
                   label=f'Cops ({len(cops_x)})', edgecolors='gray', linewidth=0.3, marker='s')
    
    # Set up the plot
    ax3.set_xlim(-0.5, model.grid.width - 0.5)
    ax3.set_ylim(-0.5, model.grid.height - 0.5)
    ax3.set_xlabel('Grid X')
    ax3.set_ylabel('Grid Y')
    ax3.set_title('Final Agent Positions')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.legend(loc='upper right', fontsize=8)

    # Plot 4: Percentage of active citizens
    total_citizens = model_data['Total_Citizens']
    active_percentage = (model_data['Active'] / total_citizens * 100).fillna(0)

    ax4.plot(model_data.index, active_percentage, color='red', linewidth=2)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Percentage of Active Citizens (%)')
    ax4.set_title('Percentage of Active Citizens Over Time')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_figs_dir}/civil_violence_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'civil_violence_results.png'")


def print_final_statistics(model):
    """
    Print final simulation statistics.
    """
    quiescent = model.count_type(model, 'Quiescent')
    active = model.count_type(model, 'Active')
    jailed = model.count_type(model, 'Jailed')
    total_citizens = model.count_type(model, 'Citizen')
    cops = model.count_type(model, 'Cop')

    print(f"\nSimulation Results:")
    print(f"Total citizens: {total_citizens}")
    print(f"Total cops: {cops}")
    print(f"Final state:")
    print(f"  - Quiescent citizens: {quiescent} ({quiescent/total_citizens*100:.1f}%)")
    print(f"  - Active citizens: {active} ({active/total_citizens*100:.1f}%)")
    print(f"  - Jailed citizens: {jailed} ({jailed/total_citizens*100:.1f}%)")
    print(f"Government legitimacy: {model.legitimacy:.2f}")


def test_legitimacy_effects():
    """
    Test the effect of different legitimacy levels on civil violence.
    """
    print("\nTesting different legitimacy levels:")
    legitimacy_values = [0.2, 0.5, 0.8, 0.95]

    for legitimacy in legitimacy_values:
        model = CivilViolenceModel(legitimacy=legitimacy)

        # Run for fewer steps for comparison
        for _ in range(100):
            model.step()

        active = model.count_type(model, 'Active')
        total_citizens = model.count_type(model, 'Citizen')
        active_percentage = active / total_citizens * 100 if total_citizens > 0 else 0

        print(f"Legitimacy {legitimacy}: {active_percentage:.1f}% active citizens")


if __name__ == "__main__":
    # Run main simulation with GIF generation
    model = run_civil_violence_simulation(save_gif=True)

    # Test different legitimacy values
    test_legitimacy_effects()

    print("\nKey Findings:")
    print("1. 引爆点现象: Civil unrest shows tipping point behavior")
    print("2. 连锁反应: Cascading effects when critical mass is reached")
    print("3. 随机性影响: Small random events can trigger large-scale unrest")
    print("4. 动态可视化: GIF动画展示了社会动乱的演化过程")