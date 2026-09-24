"""
Traditional Boids Demo
Classic flocking behavior without LLM integration.
Simple import from consolidated boids_core.py following Sugarscape's pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib.widgets import Slider, Button
from boids_core import BoidsEngine


class TraditionalBoidsDemo:
    """Demo of traditional Boids flocking behavior."""
    
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Initialize traditional simulation
        self.engine = BoidsEngine(
            width=800, height=600, num_boids=40,
            mode="traditional", scenario="open_field"
        )
        
        self.paused = False
        self.setup_plot()
        self.setup_controls()
        
        # Animation (double speed)
        self.ani = animation.FuncAnimation(
            self.fig, self.update, interval=25, blit=False, cache_frame_data=False
        )
    
    def setup_plot(self):
        """Setup the main plot."""
        self.ax.set_xlim(0, self.engine.width)
        self.ax.set_ylim(0, self.engine.height)
        self.ax.set_aspect('equal')
        self.ax.set_title('Traditional Boids Flocking Demo')
        self.ax.grid(True, alpha=0.3)
        
        # Initialize boids scatter with initial positions
        initial_positions = self.engine.get_positions()
        if initial_positions:
            x_pos = [pos[0] for pos in initial_positions]
            y_pos = [pos[1] for pos in initial_positions]
            colors = [pos[2] for pos in initial_positions]
            self.boids_scatter = self.ax.scatter(x_pos, y_pos, c=colors, s=30, alpha=0.8)
        else:
            self.boids_scatter = self.ax.scatter([], [], c=[], s=30, alpha=0.8)
    
    def setup_controls(self):
        """Setup control widgets."""
        # Pause button
        ax_pause = plt.axes([0.02, 0.02, 0.08, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_pause.on_clicked(self.toggle_pause)
        
        # Reset button
        ax_reset = plt.axes([0.12, 0.02, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset)
        
        # Separation slider
        ax_sep = plt.axes([0.02, 0.08, 0.15, 0.02])
        self.slider_sep = Slider(ax_sep, 'Separation', 0.0, 3.0, valinit=1.0)
        self.slider_sep.on_changed(self.update_separation)
        
        # Alignment slider
        ax_align = plt.axes([0.02, 0.11, 0.15, 0.02])
        self.slider_align = Slider(ax_align, 'Alignment', 0.0, 3.0, valinit=1.0)
        self.slider_align.on_changed(self.update_alignment)
        
        # Cohesion slider
        ax_coh = plt.axes([0.02, 0.14, 0.15, 0.02])
        self.slider_coh = Slider(ax_coh, 'Cohesion', 0.0, 3.0, valinit=1.0)
        self.slider_coh.on_changed(self.update_cohesion)
        
        # Speed slider
        ax_speed = plt.axes([0.25, 0.08, 0.15, 0.02])
        self.slider_speed = Slider(ax_speed, 'Max Speed', 1.0, 8.0, valinit=6.0)
        self.slider_speed.on_changed(self.update_speed)
    
    def update(self, frame):
        """Update animation frame."""
        if not self.paused:
            self.engine.step()
        
        # Get positions and colors
        positions = self.engine.get_positions()
        if positions:
            x_pos = [pos[0] for pos in positions]
            y_pos = [pos[1] for pos in positions]
            colors = [pos[2] for pos in positions]
            
            # Update scatter plot
            self.boids_scatter.set_offsets(np.column_stack([x_pos, y_pos]))
            self.boids_scatter.set_facecolors(colors)
        
        # Update title
        self.ax.set_title(f'Traditional Boids - Time: {self.engine.time:.1f}s - Boids: {len(self.engine.boids)}')
        
        return self.boids_scatter
    
    def toggle_pause(self, event):
        """Toggle pause."""
        self.paused = not self.paused
        self.btn_pause.label.set_text('Resume' if self.paused else 'Pause')
    
    def reset(self, event):
        """Reset simulation."""
        # Auto-export metrics before reset
        try:
            export_dir = 'bench_exports'
            os.makedirs(export_dir, exist_ok=True)
            export_path = f"{export_dir}/boids_traditional.csv"
            self.engine.export_metrics_csv(export_path)
            print(f"[EXPORT] Metrics appended to {export_path}")
        except Exception as e:
            print(f"[EXPORT] Failed to export metrics: {e}")
        self.engine = BoidsEngine(
            width=800, height=600, num_boids=40,
            mode="traditional", scenario="open_field"
        )
        self.update_sliders()
    
    def update_separation(self, val):
        """Update separation weight."""
        for boid in self.engine.boids:
            boid.separation_weight = val
    
    def update_alignment(self, val):
        """Update alignment weight."""
        for boid in self.engine.boids:
            boid.alignment_weight = val
    
    def update_cohesion(self, val):
        """Update cohesion weight."""
        for boid in self.engine.boids:
            boid.cohesion_weight = val
    
    def update_speed(self, val):
        """Update max speed."""
        for boid in self.engine.boids:
            boid.max_speed = val
    
    def update_sliders(self):
        """Update slider values."""
        if self.engine.boids:
            boid = self.engine.boids[0]
            self.slider_sep.set_val(boid.separation_weight)
            self.slider_align.set_val(boid.alignment_weight)
            self.slider_coh.set_val(boid.cohesion_weight)
            self.slider_speed.set_val(boid.max_speed)
    
    def show(self):
        """Show the demo."""
        plt.show()


def main():
    """Run the traditional demo."""
    demo = TraditionalBoidsDemo()
    demo.show()


if __name__ == "__main__":
    main()
