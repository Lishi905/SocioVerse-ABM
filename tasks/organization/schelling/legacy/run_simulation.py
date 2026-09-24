# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
from schelling_model import SchellingModel
import matplotlib.patches as mpatches


def save_step_image(model, step, step_dir):
    """
    Save a single step image showing agent distribution at current state.
    """
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
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the state
    colors = ['red', 'blue', 'white']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    ax.imshow(agent_counts, cmap=cmap, norm=norm)
    ax.set_title(f'Schelling Segregation Model - Step {step}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    
    # Create legend
    red_patch = mpatches.Patch(color='red', label='Type 0 (Majority)')
    blue_patch = mpatches.Patch(color='blue', label='Type 1 (Minority)')
    white_patch = mpatches.Patch(color='white', label='Empty')
    ax.legend(handles=[red_patch, blue_patch, white_patch])
    
    # Add statistics text box
    total_agents = model.schedule.get_agent_count()
    happiness_ratio = model.happy / total_agents if total_agents > 0 else 0
    
    stats_text = f'Step: {step}\nTotal Agents: {total_agents}\nHappy Agents: {model.happy}\nHappiness Ratio: {happiness_ratio:.2%}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='lightgray', alpha=0.9, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(f'{step_dir}/step_{step:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_simulation_gif(step_dir, total_steps):
    """
    Create a GIF animation from saved step images.
    """
    print("正在创建GIF动画...")
    
    # Get all step images
    image_files = []
    for i in range(total_steps + 1):
        filename = f'{step_dir}/step_{i:03d}.png'
        if os.path.exists(filename):
            image_files.append(filename)
    
    if not image_files:
        print("未找到步骤图像！")
        return
    
    # Load images
    images = []
    for filename in image_files:
        img = Image.open(filename)
        images.append(img)
    
    # Create GIF
    gif_filename = f'{step_dir}/schelling_simulation.gif'
    images[0].save(
        gif_filename,
        save_all=True,
        append_images=images[1:],
        duration=500,  # 500ms per frame
        loop=0  # Infinite loop
    )
    
    print(f"GIF动画已保存为 '{gif_filename}'")
    print(f"总帧数: {len(images)}")


def run_schelling_simulation(steps=100, height=50, width=50, density=0.8, minority_pc=0.5, homophily=5, save_gif=True):
    """
    Run a Schelling segregation simulation and visualize the results.

    Args:
        steps: Number of steps to run
        height: Grid height
        width: Grid width
        density: What fraction of grid cells are occupied
        minority_pc: What fraction of agents are in the minority group
        homophily: How many similar neighbors agents want
        save_gif: Whether to save individual step images and create GIF
    """
    # Create directories for output
    simulation_steps_dir = "./raw_simulations/organization_model/schelling_segregation/simulation_steps"
    output_figs_dir = "./raw_simulations/organization_model/schelling_segregation/output_figs"
    
    if save_gif:
        if not os.path.exists(simulation_steps_dir):
            os.makedirs(simulation_steps_dir)
        if not os.path.exists(output_figs_dir):
            os.makedirs(output_figs_dir)
    
    # Create and run the model
    model = SchellingModel(height=height, width=width, density=density,
                          minority_pc=minority_pc, homophily=homophily)

    # 使用进度条显示仿真进度
    with tqdm(total=steps, desc="运行Schelling仿真", unit="步") as pbar:
        for i in range(steps):
            model.step()
            
            # Save image at regular intervals
            if save_gif and (i % 1 == 0):
                save_step_image(model, i, simulation_steps_dir)
            
            pbar.update(1)
            if not model.running:
                pbar.set_description(f"模型在第{i}步收敛")
                # 保存最后一步
                if save_gif:
                    save_step_image(model, i, simulation_steps_dir)
                break

    # Get the agent data
    agent_counts = np.zeros((model.grid.width, model.grid.height))

    for x in range(model.grid.width):
        for y in range(model.grid.height):
            cell_content = model.grid.get_cell_list_contents([(x, y)])
            if cell_content:
                agent_counts[x][y] = cell_content[0].type
            else:
                agent_counts[x][y] = -1  # Empty cell

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the final state
    colors = ['red', 'blue', 'white']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    ax1.imshow(agent_counts, cmap=cmap, norm=norm)
    ax1.set_title('Final Agent Distribution')
    ax1.set_xlabel('Grid X')
    ax1.set_ylabel('Grid Y')

    # Create legend
    red_patch = mpatches.Patch(color='red', label='Type 0 (Majority)')
    blue_patch = mpatches.Patch(color='blue', label='Type 1 (Minority)')
    white_patch = mpatches.Patch(color='white', label='Empty')
    ax1.legend(handles=[red_patch, blue_patch, white_patch])

    # Plot happiness over time
    model_data = model.datacollector.get_model_vars_dataframe()
    ax2.plot(model_data.index, model_data['Happy'])
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Number of Happy Agents')
    ax2.set_title('Agent Happiness Over Time')

    plt.tight_layout()
    plt.savefig(f'{output_figs_dir}/schelling_results.png', dpi=300, bbox_inches='tight')
    print("可视化已保存为 'schelling_results.png'")
    # plt.show()  # Commented out for headless environment

    # Create GIF from saved images
    if save_gif:
        create_simulation_gif(simulation_steps_dir, steps)

    # Print statistics
    total_agents = model.schedule.get_agent_count()
    final_happy = model.happy
    happiness_ratio = final_happy / total_agents if total_agents > 0 else 0

    print(f"仿真结果:")
    print(f"总代理数: {total_agents}")
    print(f"最终幸福代理数: {final_happy}")
    print(f"幸福度比例: {happiness_ratio:.2%}")
    print(f"同质性阈值: {homophily}")
    print(f"少数族群百分比: {minority_pc:.1%}")


if __name__ == "__main__":
    # Run simulation with default parameters
    print("正在运行Schelling隔离模型...")
    run_schelling_simulation(save_gif=True)

    # Run with different homophily values to show the effect
    print("\n测试不同同质性阈值的效果:")
    homophily_values = [1, 3, 5, 7]
    
    for homophily in tqdm(homophily_values, desc="测试同质性阈值", unit="个"):
        print(f"\n同质性阈值: {homophily}")
        model = SchellingModel(homophily=homophily)
        
        # 为每个同质性测试添加进度条
        with tqdm(total=100, desc=f"同质性={homophily}仿真", unit="步", leave=False) as pbar:
            for i in range(100):
                model.step()
                pbar.update(1)
                if not model.running:
                    pbar.set_description(f"同质性={homophily}在第{i}步收敛")
                    break
        
        total_agents = model.schedule.get_agent_count()
        happiness_ratio = model.happy / total_agents if total_agents > 0 else 0
        print(f"最终幸福度比例: {happiness_ratio:.2%}")