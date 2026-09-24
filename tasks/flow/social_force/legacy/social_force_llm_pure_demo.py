#!/usr/bin/env python3
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
纯LLM驱动的Social Force Model演示
- 不使用任何物理力
- LLM直接输出速度向量
- 每2秒统一决策（所有agent同时调用LLM）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons, Button
import matplotlib.patches as patches
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from social_force_llm_pure import PureLLMEngine

# Get PARAMETERS from the module
try:
    from social_force_core import PARAMETERS
except ImportError:
    # Fallback
    PARAMETERS = {
        "spawn_rate": 2.0,
        "min_spawn_distance": 1.0,
    }

class PureLLMDemo:
    """纯LLM驱动演示"""
    
    def __init__(self, scenario='bidirectional'):
        self.scenario = scenario
        self.paused = True
        self.started = False
        self.animation = None
        
        # 初始化纯LLM引擎
        self.engine = PureLLMEngine(scenario=scenario, decision_interval=2.0)
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.suptitle(f'Pure LLM Social Force Model ({scenario.title()}) - Ready', fontsize=16)
        
        # 设置可视化
        self.setup_visualization()
        self.setup_controls()
        
        plt.tight_layout()
        plt.show()
    
    def setup_visualization(self):
        """设置可视化"""
        self.ax.set_xlim(-2, self.engine.length + 2)
        self.ax.set_ylim(-2, self.engine.width + 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Distance (m)')
        self.ax.set_ylabel('Distance (m)')
        
        # 初始化可视化元素
        self.pedestrian_scatter = self.ax.scatter([], [], s=50, c='blue', alpha=0.7, zorder=5)
        self.velocity_arrows = []
        self.wall_patches = []
        
        # 绘制墙壁
        self.draw_walls()
    
    def draw_walls(self):
        """绘制墙壁"""
        # 清除旧的墙壁
        for patch in self.wall_patches:
            patch.remove()
        self.wall_patches.clear()
        
        # 绘制新墙壁
        for wall in self.engine.walls:
            x_coords = [wall.start[0], wall.end[0]]
            y_coords = [wall.start[1], wall.end[1]]
            line = self.ax.plot(x_coords, y_coords, 'k-', linewidth=2, zorder=1)[0]
            self.wall_patches.append(line)
    
    def setup_controls(self):
        """设置控制"""
        # Start/Stop按钮
        ax_start = plt.axes([0.02, 0.02, 0.12, 0.06])
        self.button_start = Button(ax_start, 'START', color='lightgreen')
        self.button_start.on_clicked(self.start_simulation)
        
        # Pause/Resume按钮
        ax_pause = plt.axes([0.15, 0.02, 0.10, 0.06])
        self.button_pause = Button(ax_pause, 'PAUSE', color='lightcoral')
        self.button_pause.on_clicked(self.toggle_pause)
        
        # Reset按钮
        ax_reset = plt.axes([0.26, 0.02, 0.10, 0.06])
        self.button_reset = Button(ax_reset, 'RESET', color='lightblue')
        self.button_reset.on_clicked(self.reset_simulation)
        
        # Scenario选择
        ax_scenario = plt.axes([0.49, 0.02, 0.15, 0.08])
        self.radio_scenario = RadioButtons(ax_scenario, ('bidirectional', 'bottleneck'))
        self.radio_scenario.on_clicked(self.change_scenario)
        
        # 状态显示
        self.status_text = self.ax.text(0.02, 0.95, 'Status: Ready', 
                                        transform=self.ax.transAxes,
                                        fontsize=10, verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def start_simulation(self, event):
        """开始模拟"""
        if not self.started:
            self.started = True
            self.paused = False
            self.button_start.label.set_text('STOP')
            self.button_start.color = 'lightcoral'
            self.animation = FuncAnimation(self.fig, self.update, interval=50, blit=False)
            self.fig.canvas.draw()
        else:
            self.stop_simulation()
    
    def stop_simulation(self):
        """停止模拟"""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        self.started = False
        self.paused = True
        self.button_start.label.set_text('START')
        self.button_start.color = 'lightgreen'
        
        # 自动导出指标
        try:
            export_dir = 'bench_exports'
            os.makedirs(export_dir, exist_ok=True)
            export_path = f"{export_dir}/sfm_pure_llm_{self.scenario}.csv"
            self.engine.export_metrics_csv(export_path)
            print(f"\n[EXPORT] Metrics exported to {export_path}")
            
            # 打印统计信息
            metrics = self.engine.get_metrics()
            print(f"[STATS] Steps: {self.engine.step_count} | Time: {self.engine.time:.2f}s")
            print(f"[STATS] LLM Decisions: {metrics['llm_decisions']} | Errors: {metrics['llm_errors']}")
            print(f"[STATS] Final Agents: {metrics['active_pedestrians']} | Avg Speed: {metrics['average_speed']:.3f} m/s")
        except Exception as e:
            print(f"[EXPORT] Failed to export metrics: {e}")
        
        self.status_text.set_text('Status: Stopped')
        self.fig.canvas.draw()
    
    def toggle_pause(self, event):
        """暂停/继续"""
        if not self.started:
            return
        
        self.paused = not self.paused
        if self.paused:
            if self.animation:
                self.animation.event_source.stop()
            self.button_pause.label.set_text('RESUME')
            self.button_pause.color = 'lightgreen'
            self.status_text.set_text('Status: Paused')
        else:
            self.animation = FuncAnimation(self.fig, self.update, interval=50, blit=False)
            self.button_pause.label.set_text('PAUSE')
            self.button_pause.color = 'lightcoral'
            self.status_text.set_text('Status: Running')
        self.fig.canvas.draw()
    
    def reset_simulation(self, event):
        """重置模拟"""
        self.stop_simulation()
        self.engine = PureLLMEngine(scenario=self.scenario, decision_interval=2.0)
        self.draw_walls()
        self.update(0)
        self.status_text.set_text('Status: Reset')
        self.fig.canvas.draw()
    
    def change_scenario(self, label):
        """切换场景"""
        self.stop_simulation()
        self.scenario = label
        self.engine = PureLLMEngine(scenario=label, decision_interval=2.0)
        self.draw_walls()
        self.fig.suptitle(f'Pure LLM Social Force Model ({label.title()}) - Ready', fontsize=16)
        self.update(0)
        self.fig.canvas.draw()
    
    def update(self, frame):
        """更新动画"""
        if not self.paused:
            self.engine.step()
        
        # 获取active行人
        active_pedestrians = self.engine.get_active_pedestrians()
        
        if not active_pedestrians:
            self.pedestrian_scatter.set_offsets(np.empty((0, 2)))
            # 清除箭头
            for arrow in self.velocity_arrows:
                arrow.remove()
            self.velocity_arrows.clear()
            return
        
        # 准备位置和颜色
        positions = np.array([p.position for p in active_pedestrians])
        colors = []
        arrow_colors = []
        
        for p in active_pedestrians:
            if p.spawn_direction == "left":
                colors.append('red')
                arrow_colors.append('red')
            elif p.spawn_direction == "right":
                colors.append('blue')
                arrow_colors.append('blue')
            else:
                colors.append('gray')
                arrow_colors.append('gray')
        
        # 更新散点图
        self.pedestrian_scatter.set_offsets(positions)
        self.pedestrian_scatter.set_color(colors)
        
        # 清除旧箭头
        for arrow in self.velocity_arrows:
            arrow.remove()
        self.velocity_arrows.clear()
        
        # 绘制新箭头（速度向量）
        for p, arrow_color in zip(active_pedestrians, arrow_colors):
            if np.linalg.norm(p.velocity) > 0.1:
                arrow = self.ax.arrow(
                    p.position[0], p.position[1],
                    p.velocity[0] * 0.5, p.velocity[1] * 0.5,
                    head_width=0.15, head_length=0.15,
                    fc=arrow_color, ec=arrow_color, alpha=0.7, zorder=4
                )
                self.velocity_arrows.append(arrow)
        
        # 更新状态
        metrics = self.engine.get_metrics()
        status = f"Status: Running | Agents: {metrics['active_pedestrians']} | "
        status += f"Speed: {metrics['average_speed']:.2f} m/s | "
        status += f"LLM Decisions: {metrics['llm_decisions']} | "
        status += f"Time: {self.engine.time:.1f}s | Steps: {self.engine.step_count}"
        self.status_text.set_text(status)
        
        # 定期导出指标（每100步）
        if self.engine.step_count > 0 and self.engine.step_count % 100 == 0:
            try:
                export_dir = 'bench_exports'
                os.makedirs(export_dir, exist_ok=True)
                export_path = f"{export_dir}/sfm_pure_llm_{self.scenario}.csv"
                self.engine.export_metrics_csv(export_path)
            except Exception:
                pass  # 静默失败，不影响模拟
        
        return [self.pedestrian_scatter] + self.velocity_arrows

if __name__ == "__main__":
    print("=" * 80)
    print("Pure LLM Social Force Model Demo")
    print("=" * 80)
    print("\nFeatures:")
    print("- No physics forces (pure LLM control)")
    print("- LLM outputs velocity vectors directly")
    print("- Unified decision every 2 seconds (all agents simultaneously)")
    print("- Collision resolution: agents placed in free spaces")
    print("\nControls:")
    print("- START: Begin simulation")
    print("- PAUSE: Pause/resume simulation")
    print("- RESET: Reset simulation")
    print("- Scenario: Switch between bidirectional and bottleneck")
    print("\nPedestrians are colored by spawn direction:")
    print("- Red: from left")
    print("- Blue: from right")
    print("\nArrows show velocity vectors (red=from left, blue=from right)")
    print("=" * 80)
    
    demo = PureLLMDemo(scenario='bidirectional')

