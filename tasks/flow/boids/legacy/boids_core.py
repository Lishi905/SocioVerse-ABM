"""



Boids Flocking Simulation Core Engine

Implements classic Boids algorithm with separation, alignment, and cohesion forces.

Supports both traditional rule-based and LLM-controlled behaviors.

Embedded OpenAI policy following Sugarscape's architectural simplicity.

"""

import numpy as np

import math

import random

import time

import json

import os

import threading

import queue

from typing import List, Dict, Any, Optional, Tuple, Set, Callable

from dataclasses import dataclass

# ============================================================================

# CONSTANTS AND SCHEMAS (from schemas.py)

# ============================================================================

# Action parameter bounds for LLM control

ACTION_BOUNDS = {

    "boldness": (0.1, 3.0),           # Overall behavior multiplier

    "curiosity": (0.1, 3.0),          # Exploration tendency

    "social_preference": (0.1, 3.0),  # Flocking vs independence

    "separation_weight": (0.0, 5.0),  # Avoid crowding force

    "alignment_weight": (0.0, 5.0),   # Match neighbor velocities

    "cohesion_weight": (0.0, 5.0),    # Move toward flock center

    "exploration_bias": ((-0.5, 0.5), (-0.5, 0.5))  # Random exploration direction

}

# Safety limits to prevent extreme behaviors

SAFETY_LIMITS = {

    "max_speed_multiplier": 2.0,      # Max speed increase

    "min_speed_multiplier": 0.1,      # Min speed decrease

    "max_force_multiplier": 3.0,      # Max force increase

    "min_force_multiplier": 0.1,      # Min force decrease

}

# Observation value ranges for normalization

OBSERVATION_RANGES = {

    "position": (0, 1000),            # Field dimensions

    "velocity": (-10, 10),            # Velocity components

    "speed": (0, 10),                 # Speed magnitude

    "direction": (-math.pi, math.pi), # Direction angle

    "distance": (0, 200),             # Distance to neighbors

    "local_density": (0, 20),         # Nearby boid count

    "edge_distance": (0, 500),        # Distance to field edge

    "time": (0, 10000),               # Simulation time

}

# ============================================================================

# OPENAI POLICY (embedded, like LLM_based.py in Sugarscape)

# ============================================================================

try:

    import openai

except ImportError:

    print("Warning: openai package not installed. LLM mode will not work.")

    openai = None

class OpenAIPolicy:

    """OpenAI-powered policy for intelligent boid behavior."""

    

    def __init__(self):

        if openai is None:

            raise RuntimeError("OpenAI package not available. Install with: pip install openai")

        

        # Load API key

        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:

            # Try loading from file (relative to project root)

            key_file = 'OPEN_AI_KEY.txt'

            if os.path.exists(key_file):

                with open(key_file, 'r') as f:

                    api_key = f.read().strip()

        

        if not api_key:

            raise RuntimeError("OPENAI_API_KEY not set (env or OPEN_AI_KEY.txt)")

        

        openai.api_key = api_key

        

        # Model configuration

        self.model = "gpt-4o-mini"

        self.max_tokens = 150

        self.temperature = 0.7

        

        # System prompt for individual bird behavior (unified format)
        self.system_prompt = """Task Description: You are a bird in a flock simulation. Your objective is to stay with the group while maintaining a comfortable distance from other birds. You must make direct movement decisions based on separation (avoid crowding), alignment (match nearby birds' direction/speed), cohesion (move toward flock center), and wall avoidance.

Action Rules: You must output your desired velocity change as a vector. Think like a real bird deciding which direction to fly and how fast. Make realistic bird decisions: smooth movements, gradual adjustments, and natural flocking behavior. Action space: velocity_change [x, y] components in range (-2.0 to 2.0), speed_modulation factor (0.5 to 2.0), confidence (0.1 to 1.0).

Response Format: Respond with ONLY a JSON object containing: {"velocity_change": [x, y], "speed_modulation": factor, "confidence": value}"""

    

    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:

        """Make decision using OpenAI API."""

        try:

            # Build user prompt

            user_prompt = self._build_user_prompt(observation)

            

            # Make API call with aggressive timeout

            response = openai.chat.completions.create(

                model=self.model,

                messages=[

                    {"role": "system", "content": self.system_prompt},

                    {"role": "user", "content": user_prompt}

                ],

                max_tokens=self.max_tokens,

                temperature=self.temperature,

                timeout=5.0  # Reduced timeout for faster failure detection

            )

            

            # Parse response

            content = response.choices[0].message.content.strip()

            

            # Extract JSON from response

            if content.startswith('```json'):

                content = content[7:]

            if content.endswith('```'):

                content = content[:-3]

            

            action = json.loads(content)

            

            # Validate and clamp action parameters

            action = self._validate_action(action)

            

            return action

            

        except Exception as e:

            # Surface the error to caller; do not pretend success

            raise RuntimeError(f"OpenAI API call failed: {e}")

    

    def decide_batch(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        """Process multiple observations individually to avoid token limits."""

        actions = []

        for obs in observations:

            action = self.decide(obs)

            actions.append(action)

            # Small delay to avoid rate limits

            time.sleep(0.1)

        return actions

    

    def _build_user_prompt(self, observation: Dict[str, Any]) -> str:

        """Build user prompt from observation."""

        self_state = observation.get("self_state", {})

        flock_context = observation.get("flock_context", {})

        environment = observation.get("environment", {})

        nearby_agents = observation.get("nearby_agents", [])

        

        # Extract key information

        position = self_state.get("position", [0, 0])

        velocity = self_state.get("velocity", [0, 0])

        speed = self_state.get("speed", 0)

        color = self_state.get("color", "blue")

        

        local_density = flock_context.get("local_density", 0)

        flock_size = flock_context.get("flock_size", 0)

        flock_center = flock_context.get("flock_center", [0, 0])

        avg_velocity = flock_context.get("average_velocity", [0, 0])

        

        edge_distance = environment.get("edge_distance", 100)

        scenario = environment.get("scenario", "open_field")

        sim_time = environment.get("time", 0)

        

        # Build user prompt in unified format
        attributes = f"""Position: [{position[0]:.1f}, {position[1]:.1f}]
Velocity: [{velocity[0]:.1f}, {velocity[1]:.1f}]
Speed: {speed:.1f}
Color: {color}
Nearby birds: {len(nearby_agents)}
Flock center: [{flock_center[0]:.1f}, {flock_center[1]:.1f}]
Average flock velocity: [{avg_velocity[0]:.1f}, {avg_velocity[1]:.1f}]
Local density: {local_density:.1f}
Nearby birds details:
{self._format_nearby_birds(nearby_agents)}"""

        env_configs = f"""Distance to nearest boundary: {edge_distance:.1f}
Scenario: {scenario}
Time: {sim_time:.1f}"""

        prompt = f"""Agent Information: {attributes}

Environment Settings: {env_configs}"""

        

        return prompt

    

    def _format_nearby_birds(self, nearby_agents: list) -> str:

        """Format nearby birds information for the prompt."""

        if not nearby_agents:

            return "No birds nearby"

        

        bird_info = []

        for i, agent in enumerate(nearby_agents[:5]):  # Limit to 5 nearest birds

            pos = agent.get("position", [0, 0])

            vel = agent.get("velocity", [0, 0])

            dist = agent.get("distance", 0)

            bird_info.append(f"  Bird {i+1}: pos[{pos[0]:.1f}, {pos[1]:.1f}], vel[{vel[0]:.1f}, {vel[1]:.1f}], distance={dist:.1f}")

        

        return "\n".join(bird_info)

    

    def _validate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:

        """Validate and clamp action parameters to safe bounds."""

        validated = {}

        

        # Validate velocity_change

        if "velocity_change" in action:

            val = action["velocity_change"]

            if isinstance(val, list) and len(val) == 2:

                validated["velocity_change"] = [

                    max(-2.0, min(2.0, float(val[0]))),

                    max(-2.0, min(2.0, float(val[1])))

                ]

            else:

                validated["velocity_change"] = [0.0, 0.0]

        else:

            validated["velocity_change"] = [0.0, 0.0]

        

        # Validate speed_modulation

        if "speed_modulation" in action:

            val = action["speed_modulation"]

            if isinstance(val, (int, float)):

                validated["speed_modulation"] = max(0.5, min(2.0, float(val)))

            else:

                validated["speed_modulation"] = 1.0

        else:

            validated["speed_modulation"] = 1.0

        

        # Validate confidence

        if "confidence" in action:

            val = action["confidence"]

            if isinstance(val, (int, float)):

                validated["confidence"] = max(0.1, min(1.0, float(val)))

            else:

                validated["confidence"] = 1.0

        else:

            validated["confidence"] = 1.0

        

        return validated

# ============================================================================

# BOID DATA STRUCTURE

# ============================================================================

@dataclass

class Boid:

    """Individual boid with position, velocity, and behavior parameters."""

    id: int

    x: float

    y: float

    vx: float

    vy: float

    color: str = 'blue'

    

    # Boids parameters

    separation_weight: float = 1.0

    alignment_weight: float = 1.0

    cohesion_weight: float = 1.0

    max_speed: float = 6.0  # Increased from 4.0 to 6.0 (50% faster)

    max_force: float = 0.5

    perception_radius: float = 50.0

    separation_radius: float = 25.0

    

    # LLM-controllable parameters

    boldness: float = 1.0  # Multiplier for all forces

    curiosity: float = 1.0  # Tendency to explore

    social_preference: float = 1.0  # Preference for flocking vs independence



class LLMDecisionCache:

    """Cache LLM decisions for reuse across multiple simulation steps."""



    def __init__(self, persistence_steps: int = 20):

        self.persistence_steps = max(1, persistence_steps)

        self._store: Dict[int, Tuple[Dict[str, Any], int]] = {}



    def set_persistence(self, persistence_steps: int) -> None:

        self.persistence_steps = max(1, persistence_steps)



    def store_action(self, agent_id: int, action: Dict[str, Any], current_step: int) -> None:

        expiry_step = current_step + self.persistence_steps

        self._store[agent_id] = (action.copy(), expiry_step)



    def get_action(self, agent_id: int, current_step: int) -> Optional[Dict[str, Any]]:

        cached = self._store.get(agent_id)

        if not cached:

            return None

        action, expiry_step = cached

        if current_step <= expiry_step:

            return action.copy()

        self._store.pop(agent_id, None)

        return None

    def active_agent_ids(self, current_step: int) -> Set[int]:
        active: Set[int] = set()
        expired: List[int] = []
        for agent_id, (_, expiry_step) in self._store.items():
            if current_step <= expiry_step:
                active.add(agent_id)
            else:
                expired.append(agent_id)
        for agent_id in expired:
            self._store.pop(agent_id, None)
        return active

# ============================================================================

# MAIN BOIDS ENGINE

# ============================================================================

class BoidsEngine:

    """Main simulation engine for Boids flocking behavior."""

    

    def __init__(self, width: int = 800, height: int = 600, num_boids: int = 50, 

                 mode: str = "traditional", scenario: str = "open_field"):

        self.width = width

        self.height = height

        self.num_boids = num_boids

        self.mode = mode

        self.scenario = scenario

        

        # Simulation parameters (double speed but stable)

        self.dt = 0.05

        self.time = 0.0

        self.step_count = 0  # integer tick counter for scheduling

        self.boids: List[Boid] = []

        

        # LLM integration

        self.backend = None

        self.debug = False

        self.llm_decision_tick_interval = 20  # LLM decides every 20 steps (~1s at dt=0.05)

        self.llm_request_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues

        self.llm_response_queue = queue.Queue()

        self.llm_worker_thread = None

        self.llm_responses = {}  # agent_id -> action

        self.llm_policy = None  # Reusable policy instance

        self.llm_performance_stats = {

            "requests_sent": 0,

            "responses_received": 0,

            "errors_encountered": 0,

            "policy_created": False,

            "queue_full_skips": 0,

            "queue_throttle_skips": 0

        }

        self.debug_missing = False

        self.freeze_on_missing_action = True

        self.llm_pending_requests: Set[int] = set()

        self.llm_response_timestamps: Dict[int, int] = {}

        self.llm_decision_cache = LLMDecisionCache(

            persistence_steps=self.llm_decision_tick_interval

        )

        self._start_llm_worker()

        

        # Initialize boids

        self._spawn_boids()

        

        # Scenario-specific setup

        self._setup_scenario()

    

    def _spawn_boids(self):

        """Spawn initial boids with random positions and velocities."""

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        

        for i in range(self.num_boids):

            # Random position

            x = random.uniform(50, self.width - 50)

            y = random.uniform(50, self.height - 50)

            

            # Random velocity

            angle = random.uniform(0, 2 * math.pi)

            speed = random.uniform(1.0, 2.0)

            vx = speed * math.cos(angle)

            vy = speed * math.sin(angle)

            

            # Assign color based on position (left/right sides)

            color = 'blue' if x < self.width / 2 else 'red'

            

            boid = Boid(

                id=i,

                x=x, y=y,

                vx=vx, vy=vy,

                color=color

            )

            self.boids.append(boid)

    

    def _setup_scenario(self):

        """Setup scenario-specific parameters."""

        if self.scenario == "predator_prey":

            # Some boids become predators (red), others prey (blue)

            for i, boid in enumerate(self.boids):

                if i < self.num_boids // 4:  # 25% predators

                    boid.color = 'red'

                    boid.max_speed = 4.0

                    boid.separation_weight = 0.5

                    boid.cohesion_weight = 0.3

                else:

                    boid.color = 'blue'

                    boid.max_speed = 3.5

                    boid.separation_weight = 2.0

    

    def step(self):

        """Execute one simulation step."""

        if self.mode == "llm":

            self._process_llm_responses()

            if self.step_count % self.llm_decision_tick_interval == 0:

                self._request_llm_decisions()

        

        # Update boid positions and velocities

        for boid in self.boids:

            self._update_boid(boid)

        

        self.time += self.dt

        self.step_count += 1

    

    def _update_boid(self, boid: Boid):

        """Update individual boid using traditional rules or pure LLM control."""

        speed_mod = 1.0
        separation = self._calculate_separation(boid)
        alignment = self._calculate_alignment(boid)
        cohesion = self._calculate_cohesion(boid)
        trad_force_x = (
            separation[0] * boid.separation_weight +
            alignment[0] * boid.alignment_weight +
            cohesion[0] * boid.cohesion_weight
        ) * boid.boldness

        trad_force_y = (
            separation[1] * boid.separation_weight +
            alignment[1] * boid.alignment_weight +
            cohesion[1] * boid.cohesion_weight
        ) * boid.boldness

        if self.mode == "llm":

            action = self.llm_responses.get(boid.id)

            if isinstance(action, dict) and "__error__" in action:

                if self.debug:

                    print(f"[LLM ERROR] Agent {boid.id}: {action['__error__']}")

                self._halt_boid(boid)

                return

            if action is None:
                action = self.llm_decision_cache.get_action(boid.id, self.step_count)
            else:
                self.llm_responses.pop(boid.id, None)

            if not action:
                if self.debug_missing:
                    print(f"[LLM MISSING] Agent {boid.id} has no recent LLM action - halting")
                if self.freeze_on_missing_action:
                    self._halt_boid(boid)
                    return
                total_force_x = trad_force_x
                total_force_y = trad_force_y
            else:
                total_force_x, total_force_y, speed_mod = self._force_from_llm_action(boid, action)
        else:
            total_force_x = trad_force_x
            total_force_y = trad_force_y

        

        # Limit force

        force_magnitude = math.sqrt(total_force_x**2 + total_force_y**2)

        if force_magnitude > boid.max_force:

            total_force_x = (total_force_x / force_magnitude) * boid.max_force

            total_force_y = (total_force_y / force_magnitude) * boid.max_force

        

        # Update velocity

        boid.vx += total_force_x * self.dt

        boid.vy += total_force_y * self.dt

        if self.mode == "llm":

            boid.vx *= speed_mod

            boid.vy *= speed_mod

        

        # Limit speed (both minimum and maximum in LLM mode to avoid freezing)

        speed = math.sqrt(boid.vx**2 + boid.vy**2)

        if self.mode == "llm":

            # Gentle minimum speed floor to prevent complete freezing in LLM runs

            min_speed = 0.5

            if 0.0 < speed < min_speed:

                scale = min_speed / speed

                boid.vx *= scale

                boid.vy *= scale

                speed = min_speed

        if speed > boid.max_speed:

            boid.vx = (boid.vx / speed) * boid.max_speed

            boid.vy = (boid.vy / speed) * boid.max_speed

        

        # Update position

        boid.x += boid.vx * self.dt

        boid.y += boid.vy * self.dt

        

        # Boundary conditions (wrap around)

        if boid.x < 0:

            boid.x = self.width

        elif boid.x > self.width:

            boid.x = 0

        

        if boid.y < 0:

            boid.y = self.height

        elif boid.y > self.height:

            boid.y = 0

    

    def _calculate_separation(self, boid: Boid) -> Tuple[float, float]:

        """Calculate separation force to avoid crowding."""

        steer_x, steer_y = 0.0, 0.0

        count = 0

        

        for other in self.boids:

            if other.id == boid.id:

                continue

            

            distance = math.sqrt((boid.x - other.x)**2 + (boid.y - other.y)**2)

            if distance < boid.separation_radius:

                # Calculate vector pointing away from neighbor

                diff_x = boid.x - other.x

                diff_y = boid.y - other.y

                

                # Weight by distance (closer = stronger force)

                weight = 1.0 / (distance + 0.1)

                steer_x += diff_x * weight

                steer_y += diff_y * weight

                count += 1

        

        if count > 0:

            steer_x /= count

            steer_y /= count

            

            # Normalize and scale

            magnitude = math.sqrt(steer_x**2 + steer_y**2)

            if magnitude > 0:

                steer_x = (steer_x / magnitude) * boid.max_force

                steer_y = (steer_y / magnitude) * boid.max_force

        

        return steer_x, steer_y

    

    def _calculate_alignment(self, boid: Boid) -> Tuple[float, float]:

        """Calculate alignment force to match neighbor velocities."""

        steer_x, steer_y = 0.0, 0.0

        count = 0

        

        for other in self.boids:

            if other.id == boid.id:

                continue

            

            distance = math.sqrt((boid.x - other.x)**2 + (boid.y - other.y)**2)

            if distance < boid.perception_radius:

                steer_x += other.vx

                steer_y += other.vy

                count += 1

        

        if count > 0:

            steer_x /= count

            steer_y /= count

            

            # Normalize and scale

            magnitude = math.sqrt(steer_x**2 + steer_y**2)

            if magnitude > 0:

                steer_x = (steer_x / magnitude) * boid.max_force

                steer_y = (steer_y / magnitude) * boid.max_force

        

        return steer_x, steer_y

    

    def _calculate_cohesion(self, boid: Boid) -> Tuple[float, float]:

        """Calculate cohesion force to move toward center of neighbors."""

        center_x, center_y = 0.0, 0.0

        count = 0

        

        for other in self.boids:

            if other.id == boid.id:

                continue

            

            distance = math.sqrt((boid.x - other.x)**2 + (boid.y - other.y)**2)

            if distance < boid.perception_radius:

                center_x += other.x

                center_y += other.y

                count += 1

        

        # Initialize return values

        steer_x, steer_y = 0.0, 0.0

        

        if count > 0:

            center_x /= count

            center_y /= count

            

            # Calculate steering force toward center

            steer_x = center_x - boid.x

            steer_y = center_y - boid.y

            

            # Normalize and scale

            magnitude = math.sqrt(steer_x**2 + steer_y**2)

            if magnitude > 0:

                steer_x = (steer_x / magnitude) * boid.max_force

                steer_y = (steer_y / magnitude) * boid.max_force

        

        return steer_x, steer_y

    def _force_from_llm_action(self, boid: Boid, action: Dict[str, Any]) -> Tuple[float, float, float]:

        """Convert raw LLM output into a force vector and speed modulation factor."""

        velocity_change = action.get("velocity_change", [0.0, 0.0])

        if not (isinstance(velocity_change, list) and len(velocity_change) == 2):

            velocity_change = [0.0, 0.0]

        try:

            force_x = float(velocity_change[0])

            force_y = float(velocity_change[1])

        except (TypeError, ValueError):

            force_x = 0.0

            force_y = 0.0

        confidence = action.get("confidence", 1.0)

        if isinstance(confidence, (int, float)):

            confidence = max(0.0, min(1.0, float(confidence)))

        else:

            confidence = 1.0

        force_x *= confidence

        force_y *= confidence

        magnitude = math.sqrt(force_x**2 + force_y**2)

        if magnitude > boid.max_force:

            force_x = (force_x / magnitude) * boid.max_force

            force_y = (force_y / magnitude) * boid.max_force

        speed_mod = action.get("speed_modulation", 1.0)

        if not isinstance(speed_mod, (int, float)):

            speed_mod = 1.0

        speed_mod = max(0.5, min(2.0, float(speed_mod)))

        return force_x, force_y, speed_mod

    

    def _apply_llm_action(self, boid: Boid, action: Dict[str, Any], 

                         separation: Tuple[float, float], 

                         alignment: Tuple[float, float], 

                         cohesion: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:

        """Apply LLM action as direct velocity changes (bird behavior)."""

        

        # Get LLM's desired velocity change

        if "velocity_change" in action:

            vel_change = action["velocity_change"]

            if isinstance(vel_change, list) and len(vel_change) == 2:

                llm_dx, llm_dy = vel_change

                

                # Apply the LLM's velocity change directly to the forces

                # This overrides the traditional boids forces with LLM decision

                confidence = action.get("confidence", 1.0)

                speed_mod = action.get("speed_modulation", 1.0)

                

                # Scale the LLM decision by confidence

                llm_dx *= confidence

                llm_dy *= confidence

                

                # Apply speed modulation

                current_speed = math.sqrt(boid.vx**2 + boid.vy**2)

                if current_speed > 0:

                    speed_factor = speed_mod

                    # Normalize current velocity and apply speed factor

                    norm_vx = boid.vx / current_speed

                    norm_vy = boid.vy / current_speed

                    

                    # Combine LLM direction change with speed modulation

                    final_vx = (norm_vx * speed_factor + llm_dx) * 0.5

                    final_vy = (norm_vy * speed_factor + llm_dy) * 0.5

                else:

                    # If no current velocity, use LLM decision directly

                    final_vx = llm_dx

                    final_vy = llm_dy

                

                # Apply the LLM decision as the primary force

                # Blend with traditional forces based on confidence

                sep_x, sep_y = separation

                align_x, align_y = alignment

                coh_x, coh_y = cohesion

                

                # Low confidence = more traditional behavior

                # High confidence = more LLM control

                traditional_weight = 1.0 - confidence

                llm_weight = confidence

                

                final_sep_x = sep_x * traditional_weight + final_vx * llm_weight

                final_sep_y = sep_y * traditional_weight + final_vy * llm_weight

                

                # Apply some alignment and cohesion for natural flocking

                final_align_x = align_x * traditional_weight * 0.3

                final_align_y = align_y * traditional_weight * 0.3

                final_coh_x = coh_x * traditional_weight * 0.3

                final_coh_y = coh_y * traditional_weight * 0.3

                

                return (final_sep_x, final_sep_y), (final_align_x, final_align_y), (final_coh_x, final_coh_y)

        

        # If no valid LLM action, return traditional forces

        return separation, alignment, cohesion

    

    # ========================================================================

    # LLM INTEGRATION METHODS (async threading)

    # ========================================================================

    

    def _start_llm_worker(self):

        """Start background thread for LLM processing."""

        if self.mode != "llm":

            return

        

        self.llm_worker_thread = threading.Thread(target=self._llm_worker_thread, daemon=True)

        self.llm_worker_thread.start()

    

    def _llm_worker_thread(self):

        """Background thread for processing LLM requests."""

        while True:

            try:

                # Get request from queue

                request = self.llm_request_queue.get(timeout=1.0)

                if request is None:

                    break

                

                agent_id, observation = request

                if self.debug:

                    print(f"[LLM WORKER] Processing request for agent {agent_id}")

                

                # Load and use LLM policy

                try:

                    if self.backend == "openai":

                        # Create policy instance once and reuse it

                        if self.llm_policy is None:

                            if self.debug:

                                print("[LLM] Creating OpenAIPolicy instance...")

                            try:

                                self.llm_policy = OpenAIPolicy()

                                self.llm_performance_stats["policy_created"] = True

                                if self.debug:

                                    print("[LLM] OpenAIPolicy created successfully")

                            except Exception as e:

                                if self.debug:

                                    print(f"[LLM] Failed to create OpenAIPolicy: {e}")

                                raise

                        

                        if self.debug:

                            print(f"[LLM] Calling decide() for agent {agent_id}")

                        

                        action = self.llm_policy.decide(observation)

                        

                        if self.debug:

                            print(f"[LLM] Received response for agent {agent_id}")

                    else:

                        # No fallback: if backend not set to openai in LLM mode, raise

                        raise RuntimeError("LLM backend not configured (expected 'openai')")

                    

                    # Put response in queue

                    self.llm_response_queue.put((agent_id, action))

                    self.llm_performance_stats["responses_received"] += 1

                    

                except Exception as e:

                    # Surface error result to main thread; do not substitute actions

                    if self.debug:

                        print(f"[LLM ERROR] Agent {agent_id}: {type(e).__name__}: {e}")

                    self.llm_response_queue.put((agent_id, {"__error__": str(e)}))

                    self.llm_performance_stats["errors_encountered"] += 1

                

                self.llm_request_queue.task_done()

                

            except queue.Empty:

                continue

            except Exception as e:

                if self.debug:

                    print(f"[ERROR] LLM worker thread error: {e}")

                break

    

    def _request_llm_decisions(self):

        """Request LLM decisions for agents that need fresh guidance."""

        queue_size = self.llm_request_queue.qsize()

        if queue_size >= len(self.boids):

            self.llm_performance_stats["queue_throttle_skips"] += 1

            if self.debug:

                print(f"[LLM THROTTLE] Skipping requests; queue size {queue_size} >= boid count {len(self.boids)}")

            return

        if self.debug:

            print(f"[LLM REQUEST] Evaluating boids for new LLM decisions (queue size: {queue_size})")

        self.llm_decision_cache.set_persistence(self.llm_decision_tick_interval)

        for boid in self.boids:

            agent_id = boid.id

            if agent_id in self.llm_pending_requests:

                continue

            last_step = self.llm_response_timestamps.get(agent_id, -self.llm_decision_tick_interval)

            steps_since = self.step_count - last_step

            if steps_since < self.llm_decision_tick_interval:

                continue

            observation = self._build_observation(boid)

            try:

                self.llm_request_queue.put_nowait((agent_id, observation))

                self.llm_pending_requests.add(agent_id)

                self.llm_performance_stats["requests_sent"] += 1

            except queue.Full:

                self.llm_performance_stats["queue_full_skips"] += 1

                if self.debug:

                    print(f"[LLM THROTTLE] Queue full, stopping enqueue at agent {agent_id}")

                break

    

    def _build_observation(self, boid: Boid) -> Dict[str, Any]:

        """Build observation for LLM decision making."""

        # Find nearby boids

        nearby_boids = []

        local_density = 0

        

        for other in self.boids:

            if other.id == boid.id:

                continue

            

            distance = math.sqrt((boid.x - other.x)**2 + (boid.y - other.y)**2)

            if distance < boid.perception_radius:

                nearby_boids.append({

                    "distance": distance,

                    "position": [other.x, other.y],

                    "velocity": [other.vx, other.vy],

                    "relative_x": other.x - boid.x,

                    "relative_y": other.y - boid.y,

                    "color": other.color,

                    "id": other.id

                })

                local_density += 1.0 / (distance + 1.0)

        

        # Calculate flock center and average velocity

        if nearby_boids:

            flock_center_x = sum(b["relative_x"] for b in nearby_boids) / len(nearby_boids)

            flock_center_y = sum(b["relative_y"] for b in nearby_boids) / len(nearby_boids)

            follow_center_x = boid.x + flock_center_x

            follow_center_y = boid.y + flock_center_y

            

            avg_velocity_x = sum(b["velocity"][0] for b in nearby_boids) / len(nearby_boids)

            avg_velocity_y = sum(b["velocity"][1] for b in nearby_boids) / len(nearby_boids)

        else:

            follow_center_x = follow_center_y = boid.x

            flock_center_x = flock_center_y = 0

            avg_velocity_x = avg_velocity_y = 0

        

        # Calculate current speed and direction

        current_speed = math.sqrt(boid.vx**2 + boid.vy**2)

        current_direction = math.atan2(boid.vy, boid.vx) if current_speed > 0.1 else 0

        

        # Environment info

        edge_distance = min(boid.x, self.width - boid.x, boid.y, self.height - boid.y)

        

        observation = {

            "agent_id": boid.id,

            "self_state": {

                "position": [boid.x, boid.y],

                "velocity": [boid.vx, boid.vy],

                "speed": current_speed,

                "direction": current_direction,

                "color": boid.color

            },

            "nearby_agents": nearby_boids[:5],  # Limit to 5 nearest

            "flock_context": {

                "flock_center": [follow_center_x, follow_center_y],

                "average_velocity": [avg_velocity_x, avg_velocity_y],

                "local_density": local_density,

                "flock_size": len(nearby_boids)

            },

            "environment": {

                "field_size": [self.width, self.height],

                "edge_distance": edge_distance,

                "scenario": self.scenario,

                "time": self.time

            },

            "behavior_params": {

                "separation_weight": boid.separation_weight,

                "alignment_weight": boid.alignment_weight,

                "cohesion_weight": boid.cohesion_weight,

                "boldness": boid.boldness,

                "curiosity": boid.curiosity,

                "social_preference": boid.social_preference

            }

        }

        

        return observation

    

    def _process_llm_responses(self):

        """Process LLM responses and update boid behaviors - NO FALLBACKS."""

        while not self.llm_response_queue.empty():

            try:

                agent_id, action = self.llm_response_queue.get_nowait()

                self.llm_pending_requests.discard(agent_id)

                # Store ALL responses (including errors) for true benchmarking

                if isinstance(action, dict) and "__error__" in action:

                    if self.debug:

                        print(f"[LLM ERROR STORED] Agent {agent_id}: {action['__error__']}")

                    # Keep error responses to track failures - don't delete them

                    self.llm_responses[agent_id] = action

                    self.llm_response_timestamps[agent_id] = self.step_count

                else:

                    self.llm_responses[agent_id] = action

                    self.llm_decision_cache.store_action(agent_id, action, self.step_count)

                    self.llm_response_timestamps[agent_id] = self.step_count

                self.llm_response_queue.task_done()

            except queue.Empty:

                break

    

    # ========================================================================

    # UTILITY METHODS

    # ========================================================================

    

    def get_positions(self):

        """Get current positions of all boids."""

        return [(boid.x, boid.y, boid.color) for boid in self.boids]

    

    def get_velocities(self):

        """Get current velocities of all boids."""

        return [(boid.vx, boid.vy) for boid in self.boids]



    def run_steps(

        self,

        steps: int,

        show_progress: bool = False,

        desc: Optional[str] = None,

        after_step: Optional[Callable[[int], None]] = None,

    ) -> None:

        """Advance the simulation multiple steps, optionally with a tqdm progress bar.



        Parameters

        ----------

        steps: int

            Number of simulation steps to execute.

        show_progress: bool

            If True, display a tqdm progress bar (requires tqdm to be installed).

        desc: Optional[str]

            Optional description string for the tqdm progress bar.

        after_step: Optional[Callable[[int], None]]

            Callback invoked after each step with the current zero-based index.

        """



        iterator: Any = range(steps)



        if show_progress:

            try:

                from tqdm import tqdm  # type: ignore



                iterator = tqdm(iterator, desc=desc or "Boids Simulation", unit="step")

            except ImportError:

                if self.debug:

                    print("[LLM INFO] tqdm not available; continuing without progress bar")



        for idx in iterator:

            self.step()

            if after_step is not None:

                after_step(idx)

    

    def add_boid(self, x: float, y: float, color: str = 'blue'):

        """Add a new boid to the simulation."""

        boid_id = max([b.id for b in self.boids], default=-1) + 1

        boid = Boid(

            id=boid_id,

            x=x, y=y,

            vx=random.uniform(-1, 1),

            vy=random.uniform(-1, 1),

            color=color

        )

        self.boids.append(boid)

    

    def remove_boid(self, boid_id: int):

        """Remove a boid from the simulation."""

        self.boids = [b for b in self.boids if b.id != boid_id]

    

    def get_llm_debug_info(self):

        """Get comprehensive LLM debugging information."""

        if self.mode != "llm":

            return {"error": "Not in LLM mode"}

        

        total_boids = len(self.boids)

        error_responses = sum(1 for a in self.llm_responses.values() 

                            if isinstance(a, dict) and "__error__" in a)



        active_agents = self.llm_decision_cache.active_agent_ids(self.step_count)

        successful_responses = len(active_agents)

        total_responses = successful_responses + error_responses

 

        # Get specific error details

        error_details = []

        for boid_id, action in self.llm_responses.items():

            if isinstance(action, dict) and "__error__" in action:

                error_details.append({

                    "agent_id": boid_id,

                    "error": action["__error__"]

                })

        no_response_count = max(0, total_boids - successful_responses - error_responses)



        return {

            "total_boids": total_boids,

            "total_responses": total_responses,

            "successful_responses": successful_responses,

            "error_responses": error_responses,

            "no_response_count": no_response_count,

            "cached_agents": successful_responses,

            "pending_requests": self.llm_request_queue.qsize(),

            "error_details": error_details,

            "llm_interval": self.llm_decision_tick_interval,

            "backend": self.backend,

            "performance_stats": self.llm_performance_stats

        }

    def _halt_boid(self, boid: Boid):

        """Stop a boid in place when LLM guidance is unavailable or invalid."""

        boid.vx = 0.0

        boid.vy = 0.0


