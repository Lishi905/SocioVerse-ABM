"""
LLM-Enhanced Boids Demo
Boids with intelligent LLM-controlled behavior.
Simple import from consolidated boids_core.py following Sugarscape's pattern.
Enhanced error monitoring for proper ABM-LLM comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib.widgets import Slider, Button, RadioButtons
from boids_core import BoidsEngine


class LLMBoidsDemo:
    """Demo of LLM-enhanced Boids behavior."""
    
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25, left=0.1)
        
        # Initialize LLM simulation
        self.engine = BoidsEngine(
            width=800, height=600, num_boids=50,
            mode="llm", scenario="open_field"
        )
        self.engine.backend = "openai"
        self.engine.debug = True
        
        self.paused = False
        self.show_behavior_info = False
        self.setup_plot()
        self.setup_controls()
        
        # Animation (double speed)
        self.ani = animation.FuncAnimation(
            self.fig, self.update, interval=50, blit=False, cache_frame_data=False
        )
    
    def setup_plot(self):
        """Setup the main plot."""
        self.ax.set_xlim(0, self.engine.width)
        self.ax.set_ylim(0, self.engine.height)
        self.ax.set_aspect('equal')
        self.ax.set_title('LLM-Enhanced Boids Demo')
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
        
        # Initialize behavior info text
        self.behavior_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                        verticalalignment='top', fontsize=10,
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self.behavior_text.set_visible(False)
    
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
        
        # Toggle behavior info
        ax_info = plt.axes([0.22, 0.02, 0.12, 0.04])
        self.btn_info = Button(ax_info, 'Hide Info')
        self.btn_info.on_clicked(self.toggle_info)
        
        # Scenario selection
        ax_scenario = plt.axes([0.02, 0.08, 0.15, 0.08])
        self.radio_scenario = RadioButtons(ax_scenario, ('Open Field', 'Predator Prey'))
        self.radio_scenario.on_clicked(self.change_scenario)
        
        # LLM decision interval slider
        ax_interval = plt.axes([0.02, 0.17, 0.15, 0.02])
        self.slider_interval = Slider(ax_interval, 'LLM Interval', 10, 200, valinit=50)
        self.slider_interval.on_changed(self.update_interval)
        
        # Base behavior parameters (these will be modulated by LLM)
        ax_sep = plt.axes([0.25, 0.08, 0.15, 0.02])
        self.slider_sep = Slider(ax_sep, 'Base Separation', 0.0, 3.0, valinit=1.0)
        self.slider_sep.on_changed(self.update_base_separation)
        
        ax_align = plt.axes([0.25, 0.11, 0.15, 0.02])
        self.slider_align = Slider(ax_align, 'Base Alignment', 0.0, 3.0, valinit=1.0)
        self.slider_align.on_changed(self.update_base_alignment)
        
        ax_coh = plt.axes([0.25, 0.14, 0.15, 0.02])
        self.slider_coh = Slider(ax_coh, 'Base Cohesion', 0.0, 3.0, valinit=1.0)
        self.slider_coh.on_changed(self.update_base_cohesion)
        
        # Speed slider
        ax_speed = plt.axes([0.25, 0.17, 0.15, 0.02])
        self.slider_speed = Slider(ax_speed, 'Base Speed', 1.0, 8.0, valinit=6.0)
        self.slider_speed.on_changed(self.update_base_speed)
    
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
        
        # Update behavior info
        if self.show_behavior_info:
            self.behavior_text.set_visible(True)
            self.update_behavior_info()
        else:
            self.behavior_text.set_visible(False)
        
        # Update title
        mode_text = "LLM" if self.engine.mode == "llm" else "Traditional"
        backend_text = f" ({self.engine.backend})" if self.engine.backend else ""
        self.ax.set_title(f'{mode_text} Boids{backend_text} - Time: {self.engine.time:.1f}s - Boids: {len(self.engine.boids)}')
        
        return self.boids_scatter, self.behavior_text
    
    def update_behavior_info(self):
        """Update behavior information display with comprehensive LLM monitoring."""
        if not self.engine.boids:
            return
        
        # Sample a few boids for behavior info
        sample_boids = self.engine.boids[:3]
        
        info_lines = ["LLM Behavior Info (NO FALLBACKS):"]
        for i, boid in enumerate(sample_boids):
            info_lines.append(f"Boid {boid.id}: Bold={boid.boldness:.1f}, "
                            f"Curious={boid.curiosity:.1f}, "
                            f"Social={boid.social_preference:.1f}")
        
        # Add comprehensive LLM status
        if hasattr(self.engine, 'llm_responses') and self.engine.llm_responses:
            total_responses = len(self.engine.llm_responses)
            
            # Count successful vs failed responses
            successful_responses = sum(1 for a in self.engine.llm_responses.values() 
                                     if not (isinstance(a, dict) and "__error__" in a))
            error_responses = sum(1 for a in self.engine.llm_responses.values() 
                                if isinstance(a, dict) and "__error__" in a)
            
            info_lines.append(f"✅ Successful LLM: {successful_responses}")
            if error_responses > 0:
                info_lines.append(f"❌ Failed LLM: {error_responses}")
            
            # Show which specific agents have errors
            if error_responses > 0:
                error_agents = [str(boid_id) for boid_id, action in self.engine.llm_responses.items()
                              if isinstance(action, dict) and "__error__" in action]
                info_lines.append(f"Failed agents: {', '.join(error_agents[:5])}")
        
        # Add queue info
        if hasattr(self.engine, 'llm_request_queue'):
            queue_size = self.engine.llm_request_queue.qsize()
            info_lines.append(f"Pending requests: {queue_size}")
        
        # Show agents without any LLM response
        total_boids = len(self.engine.boids)
        agents_with_response = len(self.engine.llm_responses) if hasattr(self.engine, 'llm_responses') else 0
        agents_without_response = total_boids - agents_with_response
        if agents_without_response > 0:
            info_lines.append(f"⚠ No LLM response: {agents_without_response}")
        
        # Show LLM decision interval
        if hasattr(self.engine, 'llm_decision_tick_interval'):
            info_lines.append(f"LLM interval: {self.engine.llm_decision_tick_interval} steps")
        
        # Show performance statistics
        if hasattr(self.engine, 'llm_performance_stats'):
            stats = self.engine.llm_performance_stats
            info_lines.append(f"✅ Requests: {stats['requests_sent']}")
            info_lines.append(f"✅ Responses: {stats['responses_received']}")
            if stats['errors_encountered'] > 0:
                info_lines.append(f"❌ Errors: {stats['errors_encountered']}")
            if stats['queue_full_skips'] > 0:
                info_lines.append(f"⚠️ Throttled: {stats['queue_full_skips']}")
            info_lines.append(f"🔧 Policy created: {'Yes' if stats['policy_created'] else 'No'}")
        
        self.behavior_text.set_text('\n'.join(info_lines))
    
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
            export_path = f"{export_dir}/boids_llm.csv"
            self.engine.export_metrics_csv(export_path)
            print(f"[EXPORT] Metrics appended to {export_path}")
        except Exception as e:
            print(f"[EXPORT] Failed to export metrics: {e}")
        current_scenario = self.engine.scenario
        
        self.engine = BoidsEngine(
            width=800, height=600, num_boids=50,
            mode="llm", scenario=current_scenario
        )
        self.engine.backend = "openai"
        self.engine.debug = True
        
        self.update_sliders()
    
    def toggle_info(self, event):
        """Toggle behavior info display."""
        self.show_behavior_info = not self.show_behavior_info
        self.btn_info.label.set_text('Show Info' if not self.show_behavior_info else 'Hide Info')
        if not self.show_behavior_info:
            self.behavior_text.set_text('')
            self.behavior_text.set_visible(False)
        else:
            self.behavior_text.set_visible(True)
        
    def change_scenario(self, label):
        """Change scenario."""
        scenario_map = {'Open Field': 'open_field', 'Predator Prey': 'predator_prey'}
        new_scenario = scenario_map[label]
        
        if new_scenario != self.engine.scenario:
            self.engine.scenario = new_scenario
            self.engine._setup_scenario()
    
    def update_interval(self, val):
        """Update LLM decision interval."""
        self.engine.llm_decision_tick_interval = int(val)
    
    def update_base_separation(self, val):
        """Update base separation weight."""
        for boid in self.engine.boids:
            # Store as base value, LLM will modulate from this
            boid.separation_weight = val
    
    def update_base_alignment(self, val):
        """Update base alignment weight."""
        for boid in self.engine.boids:
            boid.alignment_weight = val
    
    def update_base_cohesion(self, val):
        """Update base cohesion weight."""
        for boid in self.engine.boids:
            boid.cohesion_weight = val
    
    def update_base_speed(self, val):
        """Update base max_speed for all boids."""
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
        
        self.slider_interval.set_val(self.engine.llm_decision_tick_interval)
    
    def show(self):
        """Show the demo."""
        plt.show()


def main():
    """Run the LLM demo."""
    print("Starting LLM-Enhanced Boids Demo...")
    print("Make sure you have OPENAI_API_KEY set in your environment or OPEN_AI_KEY.txt file in the Boids directory.")
    
    try:
        demo = LLMBoidsDemo()
        demo.show()
    except Exception as e:
        print(f"Error starting demo: {e}")
        print("This might be due to missing OpenAI API key or openai package.")
        print("To install openai package: pip install openai")
        print("To set API key: export OPENAI_API_KEY='sk-...' or create OPEN_AI_KEY.txt file in the Boids directory")


if __name__ == "__main__":
    main()
