#!/usr/bin/env python3
"""
Social Force Model Traditional Demo
Traditional rule-based behavior only - no LLM integration
Two scenarios: bidirectional and bottleneck
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons, Button
import matplotlib.patches as patches
from social_force_core import SocialForceEngine, PARAMETERS
import os

class SocialForceTraditionalDemo:
    """Traditional demo with rule-based behavior only"""
    
    def __init__(self, scenario='bidirectional'):
        self.scenario = scenario
        self.mode = 'abm'  # Always traditional mode
        self.paused = True  # Start paused
        self.started = False  # Not started yet
        self.animation = None
        
        # Initialize engine with traditional mode only
        self.engine = SocialForceEngine(scenario=scenario, mode=self.mode)
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.suptitle(f'Social Force Model - Traditional ({scenario.title()}) - Ready to Start', fontsize=16)
        
        # Set up visualization
        self.setup_visualization()
        self.setup_controls()
        
        # Don't start animation automatically
        plt.tight_layout()
        plt.show()
    
    def setup_visualization(self):
        """Set up the main visualization"""
        self.ax.set_xlim(-2, self.engine.length + 2)
        self.ax.set_ylim(-2, self.engine.width + 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Distance (m)')
        self.ax.set_ylabel('Distance (m)')
        
        # Initialize visualization elements
        self.pedestrian_scatter = self.ax.scatter([], [], s=50, c='blue', alpha=0.7, zorder=5)
        self.velocity_arrows = []
        self.wall_patches = []
        
        # Draw walls
        self.draw_walls()
    
    def setup_controls(self):
        """Set up interactive controls"""
        # Start/Stop button (main control)
        ax_start = plt.axes([0.02, 0.02, 0.12, 0.06])
        self.button_start = Button(ax_start, 'START', color='lightgreen')
        self.button_start.on_clicked(self.start_simulation)
        
        # Pause/Resume button
        ax_pause = plt.axes([0.15, 0.02, 0.10, 0.06])
        self.button_pause = Button(ax_pause, 'PAUSE', color='lightcoral')
        self.button_pause.on_clicked(self.toggle_pause)
        
        # Reset button
        ax_reset = plt.axes([0.26, 0.02, 0.10, 0.06])
        self.button_reset = Button(ax_reset, 'RESET', color='lightblue')
        self.button_reset.on_clicked(self.reset_simulation)
        
        # Refresh button
        ax_refresh = plt.axes([0.37, 0.02, 0.10, 0.06])
        self.button_refresh = Button(ax_refresh, 'REFRESH', color='lightyellow')
        self.button_refresh.on_clicked(self.refresh_interface)
        
        # Scenario radio buttons (only bidirectional and bottleneck)
        ax_scenario = plt.axes([0.49, 0.02, 0.15, 0.08])
        self.radio_scenario = RadioButtons(ax_scenario, ('bidirectional', 'bottleneck'))
        self.radio_scenario.on_clicked(self.change_scenario)
        
        # Spawn rate slider
        ax_spawn = plt.axes([0.66, 0.02, 0.15, 0.03])
        self.slider_spawn = Slider(ax_spawn, 'Spawn Rate', 0.1, 2.0, valinit=0.5, valfmt='%.1f')
        self.slider_spawn.on_changed(self.update_spawn_rate)
        
        # Desired speed slider
        ax_speed = plt.axes([0.66, 0.06, 0.15, 0.03])
        self.slider_speed = Slider(ax_speed, 'Desired Speed', 0.5, 3.0, valinit=1.5, valfmt='%.1f')
        self.slider_speed.on_changed(self.update_desired_speed)
        
        # Max speed slider
        ax_max_speed = plt.axes([0.66, 0.10, 0.15, 0.03])
        self.slider_max_speed = Slider(ax_max_speed, 'Max Speed', 1.0, 5.0, valinit=3.0, valfmt='%.1f')
        self.slider_max_speed.on_changed(self.update_max_speed)
        
        # Status display
        ax_status = plt.axes([0.83, 0.02, 0.15, 0.10])
        ax_status.axis('off')
        self.status_text = ax_status.text(0.5, 0.5, 'Traditional Mode Ready\nClick START to begin', 
                                        ha='center', va='center', fontsize=10,
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    def start_simulation(self, event):
        """Start the simulation"""
        if not self.started:
            self.started = True
            self.paused = False
            self.button_start.label.set_text('STOP')
            self.button_start.color = 'lightcoral'
            self.button_pause.label.set_text('PAUSE')
            self.button_pause.color = 'lightcoral'
            
            # Start animation
            self.animation = FuncAnimation(
                self.fig, self.update, interval=50, blit=False, 
                cache_frame_data=False, save_count=1000
            )
            
            self.fig.suptitle(f'Social Force Model - Traditional ({self.scenario.title()}) - Running', fontsize=16)
            self.status_text.set_text('Traditional Mode Active\nSimulation Running')
            self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        else:
            # Stop simulation
            self.stop_simulation()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.started = False
        self.paused = True
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        # Auto-export metrics on stop
        try:
            export_dir = 'bench_exports'
            os.makedirs(export_dir, exist_ok=True)
            export_path = f"{export_dir}/sfm_traditional_{self.scenario}.csv"
            self.engine.export_metrics_csv(export_path)
            print(f"[EXPORT] Metrics appended to {export_path}")
        except Exception as e:
            print(f"[EXPORT] Failed to export metrics: {e}")
        
        self.button_start.label.set_text('START')
        self.button_start.color = 'lightgreen'
        self.button_pause.label.set_text('PAUSE')
        self.button_pause.color = 'lightcoral'
        
        self.fig.suptitle(f'Social Force Model - Traditional ({self.scenario.title()}) - Stopped', fontsize=16)
        self.status_text.set_text('Traditional Mode Ready\nClick START to begin')
        self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    def change_scenario(self, label):
        """Change simulation scenario"""
        if self.started:
            self.stop_simulation()
        
        self.scenario = label
        # Recreate engine with new scenario
        self.engine = SocialForceEngine(scenario=label, mode=self.mode)
        self.draw_walls()
        self.ax.set_xlim(-2, self.engine.length + 2)
        self.ax.set_ylim(-2, self.engine.width + 2)
        self.fig.suptitle(f'Social Force Model - Traditional ({label.title()}) - Ready to Start', fontsize=16)
        
        # Clear pedestrians
        self.engine.pedestrians.clear()
        self.engine.step_count = 0
    
    def update_spawn_rate(self, val):
        """Update spawn rate parameter"""
        self.engine.spawn_rate = val
    
    def update_desired_speed(self, val):
        """Update desired speed for all pedestrians"""
        for p in self.engine.pedestrians:
            p.desired_speed = val
    
    def update_max_speed(self, val):
        """Update max speed for all pedestrians"""
        for p in self.engine.pedestrians:
            p.max_speed = val
    
    def toggle_pause(self, event):
        """Toggle pause/resume"""
        if not self.started:
            return
        
        self.paused = not self.paused
        if self.paused:
            self.button_pause.label.set_text('RESUME')
            self.button_pause.color = 'lightgreen'
            self.fig.suptitle(f'Social Force Model - Traditional ({self.scenario.title()}) - Paused', fontsize=16)
            self.status_text.set_text('Traditional Mode Active\nSimulation Paused')
            self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="orange"))
        else:
            self.button_pause.label.set_text('PAUSE')
            self.button_pause.color = 'lightcoral'
            self.fig.suptitle(f'Social Force Model - Traditional ({self.scenario.title()}) - Running', fontsize=16)
            self.status_text.set_text('Traditional Mode Active\nSimulation Running')
            self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    def reset_simulation(self, event):
        """Reset the simulation"""
        if self.started:
            self.stop_simulation()
        
        self.engine.pedestrians.clear()
        self.engine.step_count = 0
        self.fig.suptitle(f'Social Force Model - Traditional ({self.scenario.title()}) - Ready to Start', fontsize=16)
    
    def refresh_interface(self, event):
        """Refresh the interface and update all visual elements"""
        # Force redraw of walls
        self.draw_walls()
        
        # Update axis limits
        self.ax.set_xlim(-2, self.engine.length + 2)
        self.ax.set_ylim(-2, self.engine.width + 2)
        
        # Clear and redraw pedestrians if any exist
        if self.engine.pedestrians:
            active_pedestrians = [p for p in self.engine.pedestrians if p.active]
            positions = np.array([[p.position[0], p.position[1]] for p in active_pedestrians])
            if len(positions) > 0:
                # Color by spawn direction: left (red), right (blue)
                colors = []
                for p in active_pedestrians:
                    if p.spawn_direction == "left":
                        colors.append('red')  # From left, moving right
                    elif p.spawn_direction == "right":
                        colors.append('blue')  # From right, moving left
                    else:
                        # Fallback for pedestrians without spawn_direction (shouldn't happen)
                        colors.append('gray')
                
                self.pedestrian_scatter.set_offsets(positions)
                self.pedestrian_scatter.set_color(colors)
        
        # Update status display
        if self.started:
            if self.paused:
                self.status_text.set_text('Traditional Mode Active\nSimulation Paused')
                self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="orange"))
            else:
                self.status_text.set_text('Traditional Mode Active\nSimulation Running')
                self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        else:
            self.status_text.set_text('Traditional Mode Ready\nClick START to begin')
            self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        # Force figure refresh
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        print(f"[REFRESH] Interface refreshed - Scenario: {self.scenario}, Started: {self.started}, Paused: {self.paused}")
    
    def draw_walls(self):
        """Draw walls for current scenario"""
        # Clear existing wall patches
        for patch in self.wall_patches:
            patch.remove()
        self.wall_patches.clear()
        
        # Draw walls based on scenario
        for wall in self.engine.walls:
            if wall.start[0] == wall.end[0]:  # Vertical wall
                x_start = wall.start[0]
                y_start = min(wall.start[1], wall.end[1])
                width = 0.1
                height = max(wall.start[1], wall.end[1]) - y_start
            else:  # Horizontal wall
                x_start = min(wall.start[0], wall.end[0])
                y_start = wall.start[1]
                width = max(wall.start[0], wall.end[0]) - x_start
                height = 0.1
            
            wall_patch = patches.Rectangle((x_start, y_start), width, height, 
                                         facecolor='black', edgecolor='black', zorder=3)
            self.ax.add_patch(wall_patch)
            self.wall_patches.append(wall_patch)
    
    def update(self, frame):
        """Update animation frame"""
        if not self.started or self.paused:
            return [self.pedestrian_scatter] + self.velocity_arrows
        
        self.engine.step()
        
        # Update pedestrian positions
        if self.engine.pedestrians:
            active_pedestrians = [p for p in self.engine.pedestrians if p.active]
            positions = np.array([[p.position[0], p.position[1]] for p in active_pedestrians])
            
            if len(positions) > 0:
                # Color by spawn direction: left (red), right (blue)
                colors = []
                for p in active_pedestrians:
                    if p.spawn_direction == "left":
                        colors.append('red')  # From left, moving right
                    elif p.spawn_direction == "right":
                        colors.append('blue')  # From right, moving left
                    else:
                        # Fallback for pedestrians without spawn_direction (shouldn't happen)
                        colors.append('gray')
                
                self.pedestrian_scatter.set_offsets(positions)
                self.pedestrian_scatter.set_color(colors)
                
                # Update velocity arrows
                for arrow in self.velocity_arrows:
                    arrow.remove()
                self.velocity_arrows.clear()
                
                for p in self.engine.pedestrians:
                    if p.active and np.linalg.norm(p.velocity) > 0.1:
                        # Set arrow color based on spawn direction
                        if p.spawn_direction == "left":
                            arrow_color = 'red'  # From left, moving right
                        elif p.spawn_direction == "right":
                            arrow_color = 'blue'  # From right, moving left
                        else:
                            arrow_color = 'gray'  # Fallback
                        
                        arrow = self.ax.arrow(p.position[0], p.position[1], p.velocity[0] * 0.5, p.velocity[1] * 0.5,
                                           head_width=0.1, head_length=0.1, 
                                           fc=arrow_color, ec=arrow_color, alpha=0.7, zorder=4)
                        self.velocity_arrows.append(arrow)
        
        return [self.pedestrian_scatter] + self.velocity_arrows

def main():
    """Main function to run the traditional demo"""
    print("Social Force Model Traditional Demo")
    print("Controls:")
    print("- Use sliders to adjust spawn rate, desired speed, and max speed")
    print("- Use radio buttons to switch scenarios")
    print("- Click Pause/Resume to control simulation")
    print("- Click Reset to restart simulation")
    print("- Pedestrians are colored by spawn direction (red=from left, blue=from right)")
    print("- Arrows show velocity direction (red=from left, blue=from right)")
    print("- Traditional rule-based behavior only")
    
    demo = SocialForceTraditionalDemo(scenario='bidirectional')

if __name__ == "__main__":
    main()
