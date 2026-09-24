#!/usr/bin/env python3
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
Social Force Model Core Engine with Embedded OpenAI Policy
Consolidated implementation with performance optimizations and LLM integration
"""

import numpy as np
import math
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import os
import threading
import queue
import time
import json

# ========================================================================
# EMBEDDED OPENAI POLICY CLASS
# ========================================================================

class LLMPolicy:
    """Base class for LLM policies."""
    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

SYSTEM_PROMPT = (
    "You control a pedestrian trying to pass through a crowded doorway. "
    "Your goal is to get through to the other side at a comfortable pace. "
    "Others are also trying to pass through in various directions. "
    "CRITICAL: Navigate toward the doorway OPENING (not the walls). "
    "Use doorway_opening_top and doorway_opening_bottom to find the clear passage. "
    "Use direction_bias to steer toward the doorway opening and avoid walls. "
    "Make strong direction changes (angle_offset ±90°) when needed to navigate properly. "
    "Return ONLY a compact JSON action with fields: speed_modulation.factor, "
    "direction_bias.angle_offset, direction_bias.strength, social_parameters.personal_space, "
    "social_parameters.assertiveness, tactical_intent.mode. Keep values within reasonable bounds."
)

DEFAULT_ACTION: Dict[str, Any] = {
    "speed_modulation": {"factor": 1.0, "reason": "normal"},
    "direction_bias": {"angle_offset": 0.0, "strength": 0.0},
    "social_parameters": {"personal_space": 1.0, "assertiveness": 1.0},
    "tactical_intent": {"mode": "normal", "target_agent": None},
}

class OpenAIPolicy(LLMPolicy):
    """OpenAI-backed policy with performance optimizations."""

    def __init__(self, model: str = "gpt-4o-2024-08-06"):
        # Allow overriding via environment variable
        env_model = os.getenv("SFM_LLM_MODEL")
        self.model = env_model if env_model else model
        # Load API key
        api_key = self._get_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set (env or OPENAI_API_KEY.txt)")
        
        try:
            import openai
            openai.api_key = api_key
        except ImportError:
            raise RuntimeError("OpenAI package not available. Install with: pip install openai")

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment or file."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                # Try loading from file (relative to project root)
                key_file = 'OPENAI_API_KEY.txt'
                if os.path.exists(key_file):
                    with open(key_file, 'r') as f:
                        api_key = f.read().strip()
            except Exception:
                pass
        return api_key

    def _call_openai(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Make OpenAI API call with timeout wrapper and 429-aware retries."""
        try:
            import openai
            
            # Trim observation to essential fields to keep prompt small
            obs_trim = {
                "self_state": observation.get("self_state", {}),
                "crowd_metrics": observation.get("crowd_metrics", {}),
                "environment": {
                    k: observation.get("environment", {}).get(k)
                    for k in ("doorway_distance", "doorway_bearing", "nearest_wall_distance", "nearest_wall_direction")
                },
                "scenario_context": observation.get("scenario_context", {}),
                "bottleneck_context": observation.get("bottleneck_context", {}),
            }

            user_prompt = (
                "Observation:\n" + json.dumps(obs_trim, separators=(",", ":")) +
                "\nReturn a JSON action with numeric values within:" 
                " speed_modulation.factor [0.3,1.0];"
                " direction_bias.angle_offset [-1.57,1.57]; direction_bias.strength [0.8,1.0];"
                " social_parameters.personal_space [0.8,1.5]; social_parameters.assertiveness [0.5,2.0];"
                " tactical_intent.mode in {normal,yield,push,follow}."
                "\nRemember: Use doorway_opening_top and doorway_opening_bottom to navigate through the clear passage, not the walls!"
            )

            # Retry with exponential backoff (handles 429 and transient errors)
            max_retries = 3
            backoff = 1.0
            last_err = None
            for attempt in range(max_retries + 1):
                result = [None]
                exception = [None]
                
                def api_call():
                    try:
                        # Increase max_tokens for reasoning models
                        max_tokens = 1000  # Default
                        if self.model in ['gpt-5-2025-08-07', 'deepseek-r1-0528', 'deepseek-r1', 'qwen3-235b-a22b-thinking-2507']:
                            max_tokens = 2000  # Reasoning models need more tokens
                        
                        response = openai.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.2,
                            max_tokens=max_tokens,
                            timeout=15.0,
                        )
                        content = response.choices[0].message.content or "{}"
                        
                        # Handle reasoning mode outputs (extract final answer after thinking)
                        if self.model in ['deepseek-r1', 'deepseek-r1-0528', 'gpt-5-2025-08-07']:
                            # DeepSeek R1 and GPT-5 use </think> tag
                            if '</think>' in content:
                                content = content.split('</think>')[-1].strip()
                        
                        if self.model == 'qwen3-235b-a22b-thinking-2507':
                            # Qwen thinking mode
                            if '</think>' in content:
                                content = content.split('</think>')[-1].strip()
                            elif '【结论】' in content:
                                content = content.split('【结论】')[-1].strip()
                        
                        start = content.find("{")
                        end = content.rfind("}")
                        if start == -1 or end == -1:
                            result[0] = DEFAULT_ACTION
                        else:
                            parsed = json.loads(content[start:end+1])
                            result[0] = parsed if isinstance(parsed, dict) else DEFAULT_ACTION
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=api_call)
                thread.daemon = True
                thread.start()
                thread.join(timeout=30.0)
                
                if thread.is_alive():
                    last_err = TimeoutError("LLM API call timed out")
                elif exception[0]:
                    last_err = exception[0]
                else:
                    return result[0]
                
                # Detect rate limit (429) or transient; backoff then retry
                msg = str(last_err)
                is_rate_limited = ("429" in msg) or ("Rate limit" in msg) or ("rate" in msg.lower())
                if attempt < max_retries and (is_rate_limited or True):
                    time.sleep(backoff + random.uniform(0, 0.5))
                    backoff *= 2
                    continue
                break
            
            # No fallback: propagate failure so the engine can handle/abort
            raise RuntimeError("LLM call failed after retries")
            
        except Exception as e:
            # No fallback: propagate the original error
            raise

    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return self._call_openai(observation)

    def decide_batch(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process each agent individually to avoid token limits."""
        results = []
        
        for i, obs in enumerate(observations):
            try:
                # Individual calls with timeout protection
                action = self._call_openai(obs)
                results.append(action)
            except Exception as e:
                # No fallback: propagate failure to caller
                raise
        
        return results

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def _time_to_collision(r_ab: np.ndarray, v_ab: np.ndarray, epsilon: float = 0.01, t_max: float = 10.0) -> float:
    """Estimate time to collision along approach direction with safeguards."""
    v_rel_mag = float(np.linalg.norm(v_ab))
    if v_rel_mag < epsilon:
        return t_max
    approach_dist = -float(np.dot(r_ab, v_ab)) / v_rel_mag
    if approach_dist <= 0:
        return t_max
    ttc = approach_dist / v_rel_mag
    return float(min(ttc, t_max))

def _distance_to_segment(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> Tuple[float, np.ndarray]:
    """Perpendicular distance to a finite segment with clamped projection."""
    segment = seg_end - seg_start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-9:
        return float(np.linalg.norm(point - seg_start)), seg_start.copy()
    t = float(np.dot(point - seg_start, segment) / denom)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = seg_start + t * segment
    return float(np.linalg.norm(point - nearest)), nearest

# ========================================================================
# GLOBAL PARAMETERS
# ========================================================================

PARAMETERS = {
    # Force parameters (Helbing & Molnár 1995)
    "V_0": 2.1,              # m²/s² - Pedestrian repulsion strength (V⁰_αβ)
    "sigma": 0.3,            # m - Pedestrian interaction range (σ)
    "U_0": 10.0,             # m²/s² - Wall repulsion strength (U⁰_αB)
    "R": 0.2,                # m - Pedestrian radius / wall interaction range
    "tau": 0.5,              # s - Relaxation time (τ_α)
    "v_0": 1.34,             # m/s - Mean desired walking speed (μ)
    "v_0_sigma": 0.26,       # m/s - Std dev of desired speed (σ)
    "v_max": 3.0,            # m/s - Maximum allowed speed
    "delta_t_predict": 2.0,  # s - Look-ahead time for elliptical interaction (Δt)
    
    # View angle parameters (Helbing & Molnár 1995)
    "field_of_view": 200.0,  # degrees - Effective field of view (2φ)
    "behind_factor": 0.5,    # [-] - Weight for objects behind pedestrian (c)
    
    # Noise parameters
    "noise_sigma": 0.1,      # m/s² - Standard deviation of fluctuations
    
    # Spawning parameters
    "spawn_rate": 2.0,       # pedestrians/second
    "min_spawn_distance": 1.0,  # m - Minimum distance between spawned pedestrians
    
    # Environment
    "walkway_width": 10.0,   # m - Width of the walkway
    "walkway_length": 50.0,  # m - Length of the walkway
    "dt": 0.05,              # s - Time step
}

# ========================================================================
# DATA CLASSES
# ========================================================================

@dataclass
class Wall:
    """Represents a wall boundary"""
    start: Tuple[float, float]
    end: Tuple[float, float]
    name: str

# ========================================================================
# PEDESTRIAN CLASS
# ========================================================================

class Pedestrian:
    """Individual pedestrian in the Social Force Model"""
    
    def __init__(self, position: np.ndarray, destination: np.ndarray, 
                 scenario: str = "bidirectional", mode: str = "abm"):
        self.position = position.copy()
        self.velocity = np.zeros(2)
        self.preferred_velocity = np.zeros(2)  # w_α in paper: preferred/desired velocity vector
        self.destination = destination.copy()
        self.scenario = scenario
        self.mode = mode
        self.active = True
        self.speed = 0.0
        
        # Individual parameters (Helbing & Molnár 1995)
        # Desired speed: Gaussian distribution μ=1.34 m/s, σ=0.26 m/s
        self.desired_speed = max(0.5, np.random.normal(
            PARAMETERS["v_0"], 
            PARAMETERS.get("v_0_sigma", 0.26)
        ))
        self.desired_speed_base = float(self.desired_speed)
        
        # Relaxation time: around 0.5s with variation
        self.relaxation_time = PARAMETERS["tau"] * random.uniform(0.8, 1.2)
        self.relaxation_time_base = float(self.relaxation_time)
        
        # Pedestrian radius
        self.radius = PARAMETERS["R"] * random.uniform(0.9, 1.1)
        
        # Maximum speed (1.3x desired speed as in paper)
        self.max_speed = min(self.desired_speed * 1.3, PARAMETERS["v_max"])
        
        # View angle parameters
        self.field_of_view = np.radians(PARAMETERS["field_of_view"])  # 200 degrees in radians
        self.behind_factor = PARAMETERS["behind_factor"]  # c = 0.5
        
        # Ellipse prediction time
        self.delta_t_predict = PARAMETERS["delta_t_predict"]  # Δt = 2.0s
        
        # LLM modulation state (only used in LLM mode)
        self.action_decay = 0.0
        self.cached_action: Optional[Dict[str, Any]] = None
        self.direction_bias_angle = 0.0
        self.direction_bias_strength = 0.0
        self.last_action: Optional[Dict[str, Any]] = None
        self.id: Optional[int] = None
        
        # Track initial spawn direction for visualization
        # "left" if spawned from left (moving right), "right" if spawned from right (moving left)
        self.spawn_direction: Optional[str] = None
    
    def calculate_desired_velocity(self) -> np.ndarray:
        """
        Calculate desired velocity vector towards destination.
        Paper: e_α points in the direction to the next destination.
        Returns: v⁰_α * e_α (desired speed times unit direction vector)
        """
        direction = self.destination - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:  # Close to destination
            return np.zeros(2)
        
        # Normalize direction to get unit vector e_α
        direction = direction / distance

        # Apply direction bias ONLY in LLM mode and when action_decay > 0
        # In pure ABM mode, direction_bias_strength should always be 0
        if (self.mode == "llm" and 
            self.action_decay > 0.0 and 
            self.direction_bias_strength > 0.0 and 
            abs(self.direction_bias_angle) > 0.0):
            # Rotate direction by bias angle
            c = math.cos(self.direction_bias_angle)
            s = math.sin(self.direction_bias_angle)
            rot_x = c * direction[0] - s * direction[1]
            rot_y = s * direction[0] + c * direction[1]
            biased = np.array([rot_x, rot_y])
            # Blend biased and original direction based on strength
            direction = self.direction_bias_strength * biased + (1.0 - self.direction_bias_strength) * direction
            # Renormalize to unit vector
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm

        return direction * self.desired_speed
    
    def calculate_social_force(self, other_pedestrians: List['Pedestrian']) -> np.ndarray:
        """
        Calculate anisotropic elliptical repulsive force (Helbing & Molnár 1995).
        
        Paper formula:
        - Ellipse semiminor axis: 2b = sqrt((||r_αβ|| + ||r_αβ - v_β*Δt*e_β||)² - (v_β*Δt)²)
        - Repulsive potential: V_αβ(b) = V⁰_αβ * exp(-b/σ)
        - View-dependent weight: w(e, f) = 1 if e·f ≥ ||f||*cos(φ), else c
        """
        total_force = np.zeros(2)
        if not other_pedestrians:
            return total_force

        # Get own heading direction e_α (prefer velocity, fallback to goal direction)
        if np.linalg.norm(self.velocity) > 0.1:
            e_alpha = self.velocity / np.linalg.norm(self.velocity)
        else:
            goal_vec = self.destination - self.position
            goal_dist = np.linalg.norm(goal_vec)
            if goal_dist > 1e-8:
                e_alpha = goal_vec / goal_dist
            else:
                e_alpha = np.array([1.0, 0.0])  # Default forward direction

        delta_t = self.delta_t_predict  # Δt = 2.0s as in paper

        for other in other_pedestrians:
            if other is self or not other.active:
                continue

            # Relative position vector r_αβ = r_α - r_β
            r_ab = self.position - other.position
            norm_r = np.linalg.norm(r_ab)
            if norm_r < 1e-8:
                continue

            # Other pedestrian's velocity and direction
            v_beta = np.linalg.norm(other.velocity)
            if v_beta > 1e-8:
                e_beta = other.velocity / v_beta
            else:
                # Fallback: use goal direction for stationary pedestrian
                to_goal = other.destination - other.position
                goal_dist = np.linalg.norm(to_goal)
                if goal_dist > 1e-8:
                    e_beta = to_goal / goal_dist
                else:
                    e_beta = np.array([1.0, 0.0])

            # Predict other's future position: step = v_β * Δt * e_β
            step = v_beta * delta_t * e_beta

            # Calculate ellipse semiminor axis b (Paper formula 4)
            # 2b = sqrt((||r_ab|| + ||r_ab - step||)² - (v_β*Δt)²)
            r_ab_minus_step = r_ab - step
            norm_r_minus = np.linalg.norm(r_ab_minus_step)
            v_beta_dt = v_beta * delta_t
            
            # Calculate 2b
            two_b_squared = (norm_r + norm_r_minus) ** 2 - v_beta_dt ** 2
            if two_b_squared < 0.0:
                two_b_squared = 0.0
            b = 0.5 * math.sqrt(two_b_squared)

            # Calculate repulsive force magnitude
            # V_αβ(b) = V⁰_αβ * exp(-b/σ)
            # Force magnitude = gradient magnitude ≈ V⁰_αβ * exp(-b/σ) / σ
            force_magnitude = PARAMETERS["V_0"] * math.exp(-b / PARAMETERS["sigma"]) / PARAMETERS["sigma"]

            # Calculate force direction (gradient direction)
            # Approximate as direction from other to self
            force_direction = r_ab / norm_r

            # Calculate force vector (before view-dependent weighting)
            force_vector = force_magnitude * force_direction

            # Apply view-dependent weight w(e_α, -f_αβ)
            # Paper: w(e, f) = 1 if e·f ≥ ||f||*cos(φ), else c
            # where 2φ = field_of_view (200 degrees), c = behind_factor (0.5)
            view_weight = self._calculate_view_weight(e_alpha, -force_direction)
            force_vector = view_weight * force_vector

            # Handle overlap (very close pedestrians)
            effective_distance = norm_r - (self.radius + other.radius)
            if effective_distance < 0:
                # Strong repulsion when overlapping
                overlap_force = 100.0 * force_direction
                force_vector = force_vector + overlap_force

            total_force += force_vector

        return total_force
    
    def _calculate_view_weight(self, my_direction: np.ndarray, force_direction: np.ndarray) -> float:
        """
        Calculate view-dependent weight w(e, f) per Helbing & Molnár (1995).
        
        Paper formula:
        w(e, f) = 1 if e·f ≥ ||f||*cos(φ), else c
        
        where:
        - e: pedestrian's heading direction
        - f: force direction
        - 2φ: field of view (200 degrees)
        - c: behind factor (0.5)
        """
        # Normalize directions
        my_dir_norm = np.linalg.norm(my_direction)
        force_dir_norm = np.linalg.norm(force_direction)
        
        if my_dir_norm < 1e-8 or force_dir_norm < 1e-8:
            return 1.0
        
        my_dir_unit = my_direction / my_dir_norm
        force_dir_unit = force_direction / force_dir_norm
        
        # Calculate dot product e·f
        dot_product = np.dot(my_dir_unit, force_dir_unit)
        
        # cos(φ) where φ = field_of_view / 2
        cos_half_fov = math.cos(self.field_of_view / 2.0)
        
        # Apply paper formula
        if dot_product >= cos_half_fov:
            return 1.0  # In field of view
        else:
            return self.behind_factor  # Behind pedestrian
    
    def calculate_wall_force(self, walls: List[Wall]) -> np.ndarray:
        """
        Calculate repulsive force from walls (Helbing & Molnár 1995).
        
        Paper formula:
        U_αB(r) = U⁰_αB * exp(-r/R)
        Force magnitude = U⁰_αB * exp(-r/R) / R
        """
        total_force = np.zeros(2)
        
        # Get own heading direction for view-dependent weighting
        if np.linalg.norm(self.velocity) > 0.1:
            e_alpha = self.velocity / np.linalg.norm(self.velocity)
        else:
            goal_vec = self.destination - self.position
            goal_dist = np.linalg.norm(goal_vec)
            if goal_dist > 1e-8:
                e_alpha = goal_vec / goal_dist
            else:
                e_alpha = np.array([1.0, 0.0])
        
        for wall in walls:
            # Find closest point on wall segment
            wall_start = np.array(wall.start)
            wall_end = np.array(wall.end)
            wall_vector = wall_end - wall_start
            wall_length = np.linalg.norm(wall_vector)
            
            if wall_length < 0.01:
                continue
                
            # Project pedestrian position onto wall line
            wall_unit = wall_vector / wall_length
            to_pedestrian = self.position - wall_start
            projection_length = np.dot(to_pedestrian, wall_unit)
            
            # Clamp projection to wall endpoints
            projection_length = max(0, min(wall_length, projection_length))
            closest_point = wall_start + projection_length * wall_unit
            
            # Distance vector from wall to pedestrian
            distance_vector = self.position - closest_point
            distance = np.linalg.norm(distance_vector)
            
            if distance < 0.01:
                # Too close, use strong repulsion
                direction = np.array([1.0, 0.0]) if distance < 0.001 else distance_vector / distance
                force_magnitude = 100.0
                force_vector = force_magnitude * direction
            else:
                # Effective distance (accounting for pedestrian radius)
                effective_distance = distance - self.radius
                
                if effective_distance < 0:
                    # Overlapping with wall - strong repulsion
                    direction = distance_vector / distance
                    force_magnitude = 100.0
                else:
                    # Normal wall repulsion (Paper formula)
                    # U_αB(r) = U⁰_αB * exp(-r/R)
                    # Force = gradient magnitude ≈ U⁰_αB * exp(-r/R) / R
                    direction = distance_vector / distance
                    force_magnitude = PARAMETERS["U_0"] * math.exp(-effective_distance / PARAMETERS["R"]) / PARAMETERS["R"]
                
                force_vector = force_magnitude * direction
                
                # Apply view-dependent weight (same as social force)
                view_weight = self._calculate_view_weight(e_alpha, -direction)
                force_vector = view_weight * force_vector
            
            total_force += force_vector
        
        return total_force
    
    def apply_force(self, force: np.ndarray):
        """
        Apply force and update position according to Helbing & Molnár (1995) motion equations.
        
        Paper equations:
        dw_α/dt = F_α(t) + fluctuations
        v_α(t) = w_α(t) * g(v_max_α / ||w_α||)  where g(x) = 1 if x >= 1, else x
        dr_α/dt = v_α(t)
        
        where:
        - w_α: preferred/desired velocity vector
        - F_α: total force (desired + social + wall)
        - v_α: actual velocity
        """
        dt = PARAMETERS["dt"]
        
        # 1. Calculate desired force F⁰_α = (1/τ_α) * (v⁰_α * e_α - v_α)
        desired_velocity = self.calculate_desired_velocity()
        desired_force = (desired_velocity - self.velocity) / self.relaxation_time
        
        # 2. Total force F_α(t) = F⁰_α + Σ_β F_αβ + Σ_B F_αB
        # Note: 'force' parameter contains social + wall forces from step()
        # Here we add the desired force to get total force
        total_force = desired_force + force
        
        # 3. Add fluctuations (Gaussian white noise)
        noise_std = PARAMETERS.get("noise_sigma", 0.1)  # σ_noise
        fluctuations = np.random.normal(0.0, noise_std, 2) * math.sqrt(dt)
        total_force = total_force + fluctuations
        
        # 4. Update preferred velocity w_α: dw_α/dt = F_α(t) + fluctuations
        # Using Euler integration
        self.preferred_velocity += total_force * dt
        
        # 5. Calculate actual velocity v_α from preferred velocity w_α
        # v_α(t) = w_α(t) * g(v_max_α / ||w_α||)
        w_norm = np.linalg.norm(self.preferred_velocity)
        
        if w_norm > 1e-8:
            # g function: g(x) = 1 if x >= 1, else x
            x = self.max_speed / w_norm
            g_value = 1.0 if x >= 1.0 else x
            
            # Apply g function to limit speed
            self.velocity = self.preferred_velocity * g_value
        else:
            # If preferred velocity is near zero, actual velocity is zero
            self.velocity = np.zeros(2)
        
        # 6. Ensure velocity doesn't exceed maximum (safety check)
        v_norm = np.linalg.norm(self.velocity)
        if v_norm > self.max_speed and v_norm > 1e-8:
            self.velocity = self.velocity * (self.max_speed / v_norm)
            v_norm = self.max_speed
        
        # 7. Update position: dr_α/dt = v_α(t)
        self.position += self.velocity * dt
        
        # 8. Update speed for metrics
        self.speed = v_norm
    
    def check_boundaries(self, width: float, length: float):
        """Check if pedestrian has reached boundaries or destination"""
        # Check if reached destination
        distance_to_dest = np.linalg.norm(self.position - self.destination)
        if distance_to_dest < 0.5:
            self.active = False
            return
        
        # Check if exited simulation area
        if (self.position[0] < -1.0 or self.position[0] > length + 1.0 or
            self.position[1] < -1.0 or self.position[1] > width + 1.0):
            self.active = False
    
    def step(self, other_pedestrians: List['Pedestrian'], walls: List[Wall]):
        """
        Execute one simulation step according to Helbing & Molnár (1995).
        
        For ABM mode: Pure physics-based forces
        For LLM mode: Apply LLM modulation first, then physics forces
        """
        if not self.active:
            return
        
        # Apply cached LLM action with decay between decisions (only in LLM mode)
        if self.mode == "llm" and self.cached_action is not None and self.action_decay > 0.0:
            self._apply_action(self.cached_action, strength=self.action_decay)
            # Exponential decay toward defaults
            self.action_decay *= 0.9
        
        # In ABM mode, ensure no LLM bias remains
        if self.mode == "abm":
            # Reset any lingering LLM effects
            self.direction_bias_strength = 0.0
            self.direction_bias_angle = 0.0
        
        # Calculate forces (desired force is calculated inside apply_force)
        # Paper: F_α(t) = F⁰_α + Σ_β F_αβ + Σ_B F_αB
        social_force = self.calculate_social_force(other_pedestrians)
        wall_force = self.calculate_wall_force(walls)
        
        # External forces (social + wall)
        # Desired force F⁰_α will be calculated inside apply_force
        external_forces = social_force + wall_force
        
        # Apply forces and update state (desired force added inside)
        self.apply_force(external_forces)
        
        # Check boundaries
        self.check_boundaries(PARAMETERS["walkway_width"], PARAMETERS["walkway_length"])

    def build_observation(self, engine: 'SocialForceEngine') -> Dict[str, Any]:
        """Build observation dict for LLM policy (complex observation building preserved)."""
        pos = self.position.copy()
        vel = self.velocity.copy()
        speed = float(np.linalg.norm(vel))
        goal_vec = self.destination - self.position
        goal_dist = float(np.linalg.norm(goal_vec))
        e_alpha = goal_vec / goal_dist if goal_dist > 1e-8 else np.zeros(2)

        # Local neighbors within FOV 200° around velocity (fallback goal dir)
        heading = vel / np.linalg.norm(vel) if np.linalg.norm(vel) > 0.1 else e_alpha
        def angle_to(vec: np.ndarray) -> float:
            a = math.atan2(vec[1], vec[0])
            b = math.atan2(heading[1], heading[0])
            d = (a - b + math.pi) % (2 * math.pi) - math.pi
            return d

        local: List[Dict[str, Any]] = []
        for other in engine.pedestrians:
            if other is self or not other.active:
                continue
            r = other.position - pos
            d = float(np.linalg.norm(r))
            if d < 1e-6:
                continue
            ang = angle_to(r)
            if abs(ang) > math.radians(100):
                continue
            dv = other.velocity - vel
            # Simple TTC (local helper)
            ttc = _time_to_collision(r, dv)
            is_same = bool(np.dot(other.velocity, e_alpha) >= 0)
            local.append({
                "relative_position": [float(r[0]), float(r[1])],
                "distance": d,
                "relative_velocity": [float(dv[0]), float(dv[1])],
                "bearing": float(ang),
                "time_to_collision": float(ttc),
                "is_same_direction": is_same,
            })
        # Sort and keep k=7
        local.sort(key=lambda x: (x["distance"], abs(x["bearing"])))
        local = local[:7]

        # Environment distances: nearest wall and doorway metrics (bottleneck assumption)
        nearest_wall_distance = 1e9
        nearest_wall_dir = np.zeros(2)
        for w in engine.walls:
            d, nearest = _distance_to_segment(pos, np.array(w.start), np.array(w.end))
            if d < nearest_wall_distance:
                nearest_wall_distance = d
                vec = pos - nearest
                nrm = np.linalg.norm(vec)
                if nrm > 1e-8:
                    nearest_wall_dir = vec / nrm

        # Doorway center and posts (only for bottleneck scenario)
        door_center = np.array([engine.length * 0.5, engine.width * 0.5])
        door_vec = door_center - pos
        doorway_distance = float(np.linalg.norm(door_vec))
        doorway_bearing = angle_to(door_vec) if doorway_distance > 1e-8 else 0.0
        half_gap = 0.6  # 1.2 m doorway current default
        door_top = np.array([engine.length * 0.5, engine.width * 0.5 + half_gap])
        door_bottom = np.array([engine.length * 0.5, engine.width * 0.5 - half_gap])
        edges = []
        for p in (door_bottom, door_top):
            vec = p - pos
            edges.append({
                "distance": float(np.linalg.norm(vec)),
                "bearing": angle_to(vec),
            })

        # Crowd metrics (simple): density within 2 m and forward density
        radius = 2.0
        forward_density = 0.0
        local_count = 0
        for other in engine.pedestrians:
            if other is self or not other.active:
                continue
            r = other.position - pos
            d = np.linalg.norm(r)
            if d <= radius:
                local_count += 1
                if np.dot(r, heading) > 0:
                    forward_density += 1
        local_area = math.pi * radius * radius
        local_density = float(local_count / local_area)
        forward_density = float(forward_density / (local_area / 2.0))  # half-disk

        at_doorway = bool(doorway_distance < 2.0)
        side = "center"
        if pos[1] < engine.width * 0.33:
            side = "bottom" if engine.width > 0 else "left"
        elif pos[1] > engine.width * 0.66:
            side = "top" if engine.width > 0 else "right"
        
        # Visual bottleneck detection: doorway must be within field of view
        doorway_in_fov = False
        if doorway_distance > 1e-8:  # Avoid division by zero
            doorway_fov_angle = angle_to(door_vec)
            doorway_in_fov = abs(doorway_fov_angle) <= math.radians(100)  # 200° total FOV
        
        bottleneck_exists = bool(doorway_in_fov and doorway_distance < 15.0 and engine.scenario == "bottleneck")
        bottleneck_width = 1.2  # meters - the narrow doorway width
        bottleneck_congestion = float(min(1.0, local_density / 0.8))  # congestion at bottleneck

        obs: Dict[str, Any] = {
            "self_state": {
                "position": [float(pos[0]), float(pos[1])],
                "velocity": [float(vel[0]), float(vel[1])],
                "speed": speed,
                "desired_speed_default": float(self.desired_speed_base),
                "distance_to_goal": goal_dist,
                "goal_direction": [float(e_alpha[0]), float(e_alpha[1])],
                "time_in_simulation": float(engine.time),
            },
            "local_agents": local,
            "environment": {
                "nearest_wall_distance": float(nearest_wall_distance),
                "nearest_wall_direction": [float(nearest_wall_dir[0]), float(nearest_wall_dir[1])],
                "doorway_distance": doorway_distance,
                "doorway_bearing": float(doorway_bearing),
                "door_edges": edges,
            },
            "crowd_metrics": {
                "local_density": local_density,
                "forward_density": forward_density,
                "doorway_congestion": float(min(1.0, local_density / 1.0)),
                "flow_direction": [float(heading[0]), float(heading[1])],
            },
            "scenario_context": {
                "at_doorway": at_doorway,
                "in_queue": bool(local_density > 0.3),
                "side_of_corridor": side,
            },
            "bottleneck_context": {
                "bottleneck_exists": bottleneck_exists,
                "bottleneck_width": bottleneck_width,
                "bottleneck_congestion": bottleneck_congestion,
                "doorway_distance": doorway_distance,
                "doorway_bearing": float(doorway_bearing),
                "doorway_in_field_of_view": doorway_in_fov,
                "doorway_fov_angle": float(doorway_fov_angle) if doorway_distance > 1e-8 else 0.0,
                "need_to_turn_to_see_bottleneck": not doorway_in_fov and doorway_distance < 15.0,
                "doorway_center": [float(door_center[0]), float(door_center[1])],
                "doorway_opening_top": [float(door_top[0]), float(door_top[1])],
                "doorway_opening_bottom": [float(door_bottom[0]), float(door_bottom[1])],
                "door_edges": edges,
            },
        }
        return obs

    def _apply_action(self, action: Dict[str, Any], strength: float = 1.0):
        """Apply bounded modulation from action, interpolating toward target."""
        ACTION_BOUNDS = {
            "speed_modulation": {"factor": (0.3, 1.0)},
            "direction_bias": {"angle_offset": (-math.pi/2, math.pi/2), "strength": (0.0, 1.0)},
            "social_parameters": {"personal_space": (0.8, 1.5), "assertiveness": (0.5, 2.0)},
        }

        # Speed modulation
        factor = float(action.get("speed_modulation", {}).get("factor", 1.0))
        lo, hi = ACTION_BOUNDS["speed_modulation"]["factor"]
        factor = min(max(factor, lo), hi)
        target_speed = self.desired_speed_base * factor
        # rate limit toward target
        delta = target_speed - self.desired_speed
        max_rate = 0.2 * self.desired_speed_base
        delta = max(-max_rate, min(max_rate, delta))
        self.desired_speed += delta * strength

        # Direction bias
        db = action.get("direction_bias", {})
        ang = float(db.get("angle_offset", 0.0))
        st = float(db.get("strength", 0.0))
        ang_lo, ang_hi = ACTION_BOUNDS["direction_bias"]["angle_offset"]
        st_lo, st_hi = ACTION_BOUNDS["direction_bias"]["strength"]
        self.direction_bias_angle = max(ang_lo, min(ang_hi, ang))
        self.direction_bias_strength = max(st_lo, min(st_hi, st))

# ========================================================================
# SOCIAL FORCE ENGINE
# ========================================================================

class SocialForceEngine:
    """Core Social Force Model simulation engine with performance optimizations"""
    
    def __init__(self, scenario: str = "bidirectional", 
                 width: float = 10.0, length: float = 50.0,
                 mode: str = "abm", seed: Optional[int] = None):
        """Initialize Social Force Model simulation engine"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.scenario = scenario
        self.width = width
        self.length = length
        self.mode = mode
        self.backend = None
        # Allow selecting OpenAI backend via mode or env
        try:
            import os
            if mode == "llm_openai" or os.getenv("SFM_LLM_BACKEND", "").lower() == "openai":
                self.backend = "openai"
                self.mode = "llm"
        except Exception:
            pass
        # Debug toggle for LLM decisions
        self.debug = False
        try:
            self.debug = os.getenv("SFM_LLM_DEBUG", "0") == "1"
        except Exception:
            self.debug = False
        # Strict mode: abort on LLM errors (no fallback) – default OFF
        self.strict = False
        try:
            # Allow override, but keep strict by default
            if os.getenv("SFM_LLM_STRICT", "").strip() != "":
                self.strict = os.getenv("SFM_LLM_STRICT", "0") == "1"
        except Exception:
            self.strict = True
        
        # Update parameters
        PARAMETERS["walkway_width"] = width
        PARAMETERS["walkway_length"] = length
        
        # Simulation state
        self.pedestrians: List[Pedestrian] = []
        self.walls: List[Wall] = []
        self.time = 0.0
        self.step_count = 0
        
        # Spawning
        self.last_spawn_time = 0.0
        self.spawn_counter = 0
        self.llm_decision_tick_interval = 100  # with dt=0.05 => 5s, reduce request rate
        self.max_llm_agents_per_tick = 4  # cap per-tick agent decisions
        
        # Dynamically adjust rate limit for reasoning models
        self.llm_model_name = os.environ.get("SFM_LLM_MODEL", "gpt-4o-2024-08-06")
        self.min_seconds_between_llm_requests = 10.0 if any(r in self.llm_model_name.lower() 
                                                              for r in ['r1', 'thinking', 'reasoning', 'gpt-5']) else 3.0
        
        self._last_llm_request_time = 0.0
        
        # Create environment walls
        self.create_walls()
        
        # Initialize background LLM processing with performance optimizations
        self.llm_request_queue = queue.Queue(maxsize=100)  # Throttling
        self.llm_response_queue = queue.Queue()
        self.llm_worker_thread = None
        self.pending_llm_request = None
        self.llm_policy = None  # Reusable policy instance
        self.llm_performance_stats = {
            "requests_sent": 0,
            "responses_received": 0,
            "errors_encountered": 0,
            "policy_created": False,
            "queue_full_skips": 0
        }
        # Metrics buffers
        self._lane_counts: List[float] = []
        self._lane_counts_per_width: Dict[float, List[float]] = {}
        # LLM experiment metrics
        self._llm_iterations: int = 0
        self._llm_actions_total: int = 0
        self._llm_actions_per_iteration: List[int] = []
        self._llm_unique_agent_ids: set[int] = set()
        
        # Spawn initial pedestrians
        self.spawn_initial_pedestrians(45)  # Increased from 15 to 45 (3x)
    
    def create_walls(self):
        """Create wall boundaries for the scenario"""
        if self.scenario == "bidirectional":
            # Simple rectangular walkway
            self.walls = [
                Wall((0, 0), (self.length, 0), "bottom_wall"),
                Wall((0, self.width), (self.length, self.width), "top_wall")
            ]
        elif self.scenario == "bottleneck":
            # NARROW DOORWAY BOTTLENECK: Horizontal walls with vertical gap
            bottleneck_position = self.length * 0.5  # Middle of corridor
            doorway_width = 1.2  # meters - very narrow doorway
            
            # Calculate the center of the walkway
            walkway_center = self.width / 2
            
            # Calculate doorway gap positions
            door_top = walkway_center + doorway_width / 2
            door_bottom = walkway_center - doorway_width / 2
            
            self.walls = [
                # Bottom wall - horizontal line with gap
                Wall((0, 0), (bottleneck_position, 0), "bottom_wall_left"),
                Wall((bottleneck_position, 0), (self.length, 0), "bottom_wall_right"),
                
                # Top wall - horizontal line with gap
                Wall((0, self.width), (bottleneck_position, self.width), "top_wall_left"),
                Wall((bottleneck_position, self.width), (self.length, self.width), "top_wall_right"),
                
                # Vertical walls to create the doorway gap
                Wall((bottleneck_position, 0), (bottleneck_position, door_bottom), "left_door_wall"),
                Wall((bottleneck_position, door_top), (bottleneck_position, self.width), "right_door_wall")
            ]
        elif self.scenario == "open_space":
            # Open space with boundary walls
            self.walls = [
                Wall((0, 0), (self.length, 0), "bottom_wall"),
                Wall((0, self.width), (self.length, self.width), "top_wall"),
                Wall((0, 0), (0, self.width), "left_wall"),
                Wall((self.length, 0), (self.length, self.width), "right_wall")
            ]
    
    def _llm_worker_thread(self):
        """Background thread worker for LLM API calls with performance optimizations"""
        while True:
            try:
                # Get request from queue
                request = self.llm_request_queue.get(timeout=1.0)
                if request is None:  # Shutdown signal
                    break
                
                llm_agents, observations, agent_obs_map = request
                
                # Make the API call with timeout wrapper
                try:
                    if self.debug:
                        print(f"[LLM worker] Making API call for {len(observations)} agents")
                    
                    # Create policy instance once and reuse it
                    if self.llm_policy is None:
                        if self.debug:
                            print("[LLM] Creating OpenAIPolicy instance...")
                        self.llm_policy = OpenAIPolicy()
                        self.llm_performance_stats["policy_created"] = True
                        if self.debug:
                            print("[LLM] OpenAIPolicy created successfully")
                    
                    # Use timeout wrapper for API calls
                    result = [None]
                    exception = [None]
                    
                    def api_call():
                        try:
                            actions = self.llm_policy.decide_batch(observations)
                            result[0] = actions
                        except Exception as e:
                            exception[0] = e
                    
                    # Start API call in thread with timeout
                    thread = threading.Thread(target=api_call)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=30.0)  # extended timeout
                    
                    if thread.is_alive():
                        # Thread is still running, API call timed out
                        raise TimeoutError("LLM API call timed out")
                    
                    if exception[0]:
                        raise exception[0]
                    
                    actions = result[0]
                    
                    if self.debug:
                        print(f"[LLM worker] API call successful, got {len(actions)} actions")
                    
                    # Create agent ID to action mapping
                    agent_action_map = {}
                    if len(actions) != len(agent_obs_map):
                        raise RuntimeError(f"Action count mismatch: got {len(actions)} actions for {len(agent_obs_map)} agents")
                    
                    for (agent_id, obs), action in zip(agent_obs_map.items(), actions):
                        agent_action_map[agent_id] = action
                    
                    if self.debug:
                        print(f"[LLM worker] Putting response in queue: {agent_action_map}")
                    self.llm_response_queue.put(('success', agent_action_map))
                    self.llm_performance_stats["responses_received"] += 1
                    # Record experiment metrics for this iteration
                    try:
                        num_actions = len(actions)
                        self._llm_iterations += 1
                        self._llm_actions_total += num_actions
                        self._llm_actions_per_iteration.append(num_actions)
                        for agent_id in agent_obs_map.keys():
                            try:
                                self._llm_unique_agent_ids.add(int(agent_id))
                            except Exception:
                                pass
                    except Exception:
                        pass
                    
                except Exception as e:
                    print(f"[LLM worker] API call failed: {e}")  # Always print errors
                    self.llm_response_queue.put(('error', str(e)))
                    self.llm_performance_stats["errors_encountered"] += 1
                
                self.llm_request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.debug:
                    print(f"[LLM worker ERROR] {e}")
                continue
    
    def _start_llm_worker(self):
        """Start the background LLM worker thread"""
        if self.llm_worker_thread is None or not self.llm_worker_thread.is_alive():
            self.llm_worker_thread = threading.Thread(target=self._llm_worker_thread, daemon=True)
            self.llm_worker_thread.start()
    
    def spawn_initial_pedestrians(self, count: int):
        """Spawn initial pedestrians for the simulation"""
        for i in range(count):
            self.spawn_pedestrian()
            self.time += 0.5  # Spread spawn times
    
    def spawn_pedestrian(self):
        """Spawn a new pedestrian at random entrance"""
        # Determine spawn side (left or right)
        spawn_side = "left" if random.random() < 0.5 else "right"
        
        if spawn_side == "left":
            spawn_x = -0.5
            destination_x = self.length + 0.5
        else:
            spawn_x = self.length + 0.5
            destination_x = -0.5
        
        # For bottleneck scenario: spawn at random Y but aim for doorway center
        if self.scenario == "bottleneck":
            spawn_y = random.uniform(0.5, self.width - 0.5)
            doorway_center_y = self.width / 2  # Center of the doorway
            destination_y = doorway_center_y
        else:
            # Original logic for other scenarios
            spawn_y = random.uniform(0.5, self.width - 0.5)
            destination_y = spawn_y
        
        # Check minimum distance from existing pedestrians
        spawn_pos = np.array([spawn_x, spawn_y])
        min_distance = PARAMETERS["min_spawn_distance"]
        
        too_close = any(
            np.linalg.norm(spawn_pos - p.position) < min_distance 
            for p in self.pedestrians if p.active
        )
        
        if not too_close:
            destination = np.array([destination_x, destination_y])
            pedestrian = Pedestrian(spawn_pos, destination, self.scenario, self.mode)
            pedestrian.id = int(self.spawn_counter)
            # Record spawn direction for visualization
            pedestrian.spawn_direction = spawn_side
            self.pedestrians.append(pedestrian)
            self.spawn_counter += 1
    
    def step(self):
        """Execute one simulation step with performance optimizations"""
        self.step_count += 1
        self.time += PARAMETERS["dt"]
        
        # Process LLM responses on every step (not just decision intervals)
        if self.mode == "llm":
            # Check for completed responses first
            try:
                while not self.llm_response_queue.empty():
                    status, data = self.llm_response_queue.get_nowait()
                    if status == 'success':
                        # Use agent ID mapping instead of zipping with current agents
                        agent_action_map = data  # Now contains {agent_id: action}
                        if self.debug:
                            print(f"[LLM response] Processing agent_action_map: {agent_action_map}")
                        for agent_id, action in agent_action_map.items():
                            # Find the agent by ID
                            agent = next((p for p in self.pedestrians if p.id == agent_id and p.active), None)
                            if agent:
                                if self.debug:
                                    print(f"[LLM response] Found agent {agent_id}, applying action: {action}")
                                agent.cached_action = action
                                agent.action_decay = 1.0
                                agent.last_action = action
                    elif status == 'error':
                        if self.debug:
                            print(f"[LLM ERROR] {data}")
                        if self.strict:
                            raise RuntimeError(f"LLM API error: {data}")
                        # Non-strict mode: ignore error and continue
                    self.pending_llm_request = None
            except queue.Empty:
                pass
        
        # LLM batched decisions every interval with throttling
        if (self.mode == "llm") and (self.step_count % self.llm_decision_tick_interval == 0):
            llm_agents = [p for p in self.pedestrians if p.active and p.mode == "llm"]
            if llm_agents:
                # Start new request if none pending and queue is not full
                if self.pending_llm_request is None:
                    # Enforce min interval between requests
                    if (self.time - self._last_llm_request_time) < self.min_seconds_between_llm_requests:
                        # Skip this tick to respect rate limits
                        pass
                    else:
                        try:
                            # Create agent ID mapping for observations (cap per tick)
                            selected = llm_agents[: self.max_llm_agents_per_tick]
                            agent_obs_map = {}
                            for p in selected:
                                agent_obs_map[p.id] = p.build_observation(self)
                        
                            observations = list(agent_obs_map.values())
                            self._start_llm_worker()
                        
                            # Use put_nowait to avoid blocking, and handle queue full
                            self.llm_request_queue.put_nowait((selected, observations, agent_obs_map))
                            self.llm_performance_stats["requests_sent"] += 1
                            self.pending_llm_request = (selected, observations, agent_obs_map)
                            self._last_llm_request_time = self.time
                        
                            if self.debug:
                                print(f"[LLM] queued request for {len(selected)} agents")
                        except queue.Full:
                            # Queue is full, skip this request
                            self.llm_performance_stats["queue_full_skips"] += 1
                            if self.debug:
                                print(f"[LLM THROTTLE] Queue full, skipping request for {len(llm_agents)} agents")
        
        # Update all active pedestrians
        for pedestrian in self.pedestrians:
            if pedestrian.active:
                pedestrian.step(self.pedestrians, self.walls)
        
        # Remove inactive pedestrians
        self.pedestrians = [p for p in self.pedestrians if p.active]
        
        # Spawn new pedestrians
        time_since_spawn = self.time - self.last_spawn_time
        if time_since_spawn >= 1.0 / PARAMETERS["spawn_rate"]:
            self.spawn_pedestrian()
            self.last_spawn_time = self.time

        # Update lane metrics (bidirectional only)
        try:
            self.update_lane_metrics()
        except Exception:
            pass
    
    def get_active_pedestrians(self) -> List[Pedestrian]:
        """Get list of active pedestrians"""
        return [p for p in self.pedestrians if p.active]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current simulation metrics"""
        active_pedestrians = self.get_active_pedestrians()
        
        if not active_pedestrians:
            return {
                "total_pedestrians": len(self.pedestrians),
                "active_pedestrians": 0,
                "average_speed": 0.0,
                "density": 0.0,
                "flow_rate": 0.0
            }
        
        avg_speed = np.mean([p.speed for p in active_pedestrians])
        density = len(active_pedestrians) / (self.width * self.length)
        flow_rate = avg_speed * density
        
        return {
            "total_pedestrians": len(self.pedestrians),
            "active_pedestrians": len(active_pedestrians),
            "average_speed": avg_speed,
            "density": density,
            "flow_rate": flow_rate
        }

    # --------------------------------------------------------------------
    # Lane formation metric (bidirectional scenario)
    # N(W) ~ 0.36 m^-1 * W + 0.59
    # We estimate lane count by clustering lateral positions by direction
    # and counting distinct bands in y for each direction, then averaging.
    # --------------------------------------------------------------------
    def _estimate_lane_count(self) -> Optional[float]:
        if self.scenario != "bidirectional":
            return None
        active = self.get_active_pedestrians()
        if not active:
            return None
        # Separate by direction using x-velocity sign (fallback to destination vector)
        left_to_right = []
        right_to_left = []
        for p in active:
            vx = p.velocity[0]
            # Fallback if nearly zero velocity: use destination direction
            if abs(vx) < 1e-3:
                dir_sign = 1.0 if (p.destination[0] - p.position[0]) >= 0 else -1.0
            else:
                dir_sign = 1.0 if vx >= 0 else -1.0
            if dir_sign >= 0:
                left_to_right.append(p.position[1])
            else:
                right_to_left.append(p.position[1])

        def count_bands(y_vals: List[float], bandwidth: float = 0.8) -> int:
            if not y_vals:
                return 0
            y_sorted = sorted(y_vals)
            bands = 1
            band_start = y_sorted[0]
            for y in y_sorted[1:]:
                if y - band_start > bandwidth:
                    bands += 1
                    band_start = y
            return bands

        bands_lr = count_bands(left_to_right)
        bands_rl = count_bands(right_to_left)
        # Average lane count across directions
        return 0.5 * (bands_lr + bands_rl)

    def update_lane_metrics(self):
        lane_estimate = self._estimate_lane_count()
        if lane_estimate is None:
            return
        self._lane_counts.append(lane_estimate)
        # Track per-width as well for multi-run aggregation
        w = float(self.width)
        if w not in self._lane_counts_per_width:
            self._lane_counts_per_width[w] = []
        self._lane_counts_per_width[w].append(lane_estimate)

    def get_lane_metrics(self) -> Dict[str, Any]:
        if not self._lane_counts:
            return {
                "walkway_width": float(self.width),
                "lane_count_mean": None,
                "lane_count_std": None,
                "expected_by_scaling": 0.36 * float(self.width) + 0.59
            }
        arr = np.array(self._lane_counts, dtype=float)
        return {
            "walkway_width": float(self.width),
            "lane_count_mean": float(np.mean(arr)),
            "lane_count_std": float(np.std(arr)),
            "expected_by_scaling": 0.36 * float(self.width) + 0.59
        }

    def export_metrics_csv(self, path: str):
        import csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lane = self.get_lane_metrics()
        core = self.get_metrics()
        fields = {
            "scenario": self.scenario,
            "width": float(self.width),
            "length": float(self.length),
            "time": float(self.time),
            **core,
            **lane,
        }
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fields.keys()))
            if write_header:
                w.writeheader()
            w.writerow(fields)
    
    def get_llm_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive LLM debugging information"""
        active_pedestrians = self.get_active_pedestrians()
        total_boids = len(active_pedestrians)
        
        # Count responses
        total_responses = self.llm_performance_stats["responses_received"]
        successful_responses = self.llm_performance_stats["responses_received"]
        error_responses = self.llm_performance_stats["errors_encountered"]
        
        return {
            "total_pedestrians": total_boids,
            "total_responses": total_responses,
            "successful_responses": successful_responses,
            "error_responses": error_responses,
            "no_response_count": total_boids - total_responses,
            "pending_requests": self.llm_request_queue.qsize(),
            "llm_interval": self.llm_decision_tick_interval,
            "backend": self.backend,
            "performance_stats": self.llm_performance_stats
        }
    
    def get_llm_experiment_stats(self) -> Dict[str, Any]:
        """Aggregate LLM experiment stats for experiments:
        - total requests
        - agent num (unique)
        - actions per iter (summary)
        - iterations (decision rounds with responses)
        """
        import numpy as _np
        iterations = int(self._llm_iterations)
        actions_total = int(self._llm_actions_total)
        actions_per_iter = list(self._llm_actions_per_iteration)
        unique_agents = len(self._llm_unique_agent_ids)
        active_agents = len(self.get_active_pedestrians())
        sent = int(self.llm_performance_stats.get("requests_sent", 0))
        received = int(self.llm_performance_stats.get("responses_received", 0))
        errors = int(self.llm_performance_stats.get("errors_encountered", 0))
        queue_skips = int(self.llm_performance_stats.get("queue_full_skips", 0))
        if actions_per_iter:
            arr = _np.array(actions_per_iter, dtype=float)
            mean_actions = float(_np.mean(arr))
            std_actions = float(_np.std(arr))
            min_actions = int(_np.min(arr))
            max_actions = int(_np.max(arr))
        else:
            mean_actions = std_actions = 0.0
            min_actions = max_actions = 0
        return {
            "total_requests_sent": sent,
            "total_responses_received": received,
            "total_errors": errors,
            "queue_full_skips": queue_skips,
            "iterations": iterations,
            "total_actions": actions_total,
            "actions_per_iteration_mean": mean_actions,
            "actions_per_iteration_std": std_actions,
            "actions_per_iteration_min": min_actions,
            "actions_per_iteration_max": max_actions,
            "unique_llm_agents": unique_agents,
            "current_active_agents": active_agents,
            "llm_decision_tick_interval": int(self.llm_decision_tick_interval),
            "max_llm_agents_per_tick": int(self.max_llm_agents_per_tick),
        }
