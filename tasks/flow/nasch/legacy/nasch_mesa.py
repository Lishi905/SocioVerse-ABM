# NOTE (SocioVerse-ABM public release): legacy snapshot kept for provenance; only the
# missing-API-key message was changed (it names OPENAI_API_KEY). Not maintained (see the
# README in this legacy folder). Comments and messages may be in the original authors'
# language (Chinese).
"""
NaSch Traffic Model with Mesa Framework
======================================

Agent-based implementation of the Nagel-Schreckenberg traffic model
with pluggable behavior engines (ABM and LLM).

This implementation follows the Sugarscape architecture pattern,
allowing vehicles to use either traditional rule-based behavior
or LLM-driven decision making.
"""

import random
import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation, SimultaneousActivation
from mesa.datacollection import DataCollector
import os
import csv
import queue
import threading
import time
import pickle
import json
from datetime import datetime

from behavior_engine.ABM_agent import ABM_Agent
from behavior_engine.LLM_agent import LLM_Agent


class Vehicle(Agent):
    """
    Vehicle agent in the NaSch traffic model.
    
    Each vehicle can use either ABM (rule-based) or LLM (AI-driven) behavior.
    The action space remains unchanged: speed values in [0, max_speed].
    """
    
    def __init__(self, unique_id, model, position, speed=0, max_speed=5, mode="abm"):
        super().__init__(unique_id, model)
        self.position = position
        self.speed = speed
        self.new_speed = speed  # For parallel updates
        self.max_speed = max_speed
        self.mode = mode
        
        # Behavior engines (lazy initialization)
        self.abm_agent = None
        self.llm_agent = None
        
        # LLM decision caching
        self.cached_decision = None  # Cached speed decision from LLM
        self.decision_expiry_step = 0  # When the cached decision expires
        self.last_llm_decision_step = -1  # Last step when LLM was consulted
        self.using_fallback = False
    
    def _calculate_gap_ahead(self):
        """
        Calculate gap (number of empty cells) to next vehicle ahead.
        Returns the number of empty cells between current position and next vehicle.
        """
        for distance in range(1, self.model.L):
            next_pos = (self.position + distance) % self.model.L
            if self.model.grid.get_cell_list_contents([next_pos, 0]):
                # Gap is distance - 1 (number of empty cells, not including next vehicle's cell)
                return distance - 1
        # No vehicle ahead - gap is the entire road
        return self.model.L - 1
    
    def _build_observation(self):
        """
        Build observation - only what the agent can actually observe.
        Minimal observation space following the design principle.
        """
        return {
            "current_speed": self.speed,
            "gap_ahead": self._calculate_gap_ahead(),
        }
    
    def calculate_new_speed(self):
        """
        Phase 1: Calculate new speed based on current state (parallel update).
        All vehicles calculate speeds simultaneously before any movement.
        """
        # Check for cached LLM decision first
        if self.mode == "llm" and self.cached_decision is not None:
            # decision_expiry_step是应用决策时的step_count + cache_duration
            # 在同步模式下，run_synchronous_llm_step()在step()之前调用
            # 应用决策时step_count是N，decision_expiry_step = N + cache_duration
            # 然后step()中step_count变成N+1
            # 检查时：如果cache_duration=1，N+1 <= N+1 应该是True（使用<=而不是<）
            if self.model.step_count <= self.decision_expiry_step:
                # Use cached decision (from LLM)
                self.new_speed = self.cached_decision
                self.using_fallback = False
                return
            else:
                # Cache expired, clear it (will request new LLM decision)
                # 在同步模式下，这不应该发生（应该已经获取了新决策）
                if self.model.synchronous_mode:
                    # 同步模式下，缓存过期说明有问题
                    # 可能是cache_duration设置不当，或者没有及时获取新决策
                    pass
                self.cached_decision = None
        
        # Build observation (what agent can see) - uses OLD positions
        observation = self._build_observation()
        
        # Add model parameters (not observations, but needed for decision)
        observation["max_speed"] = self.max_speed
        observation["randomization_prob"] = self.model.p
        
        # Decision making with pluggable behavior engine
        if self.mode == "abm":
            # Traditional rule-based behavior
            if self.abm_agent is None:
                self.abm_agent = ABM_Agent(model="nasch", scenario="nasch")
            self.abm_agent.update_attributes(observation)
            action = self.abm_agent.take_actions()
            new_speed = action["speed"]
            
        elif self.mode == "llm":
            # PURE LLM MODE: MUST use LLM decisions only, NO FALLBACK ALLOWED
            # 在同步模式下，每一步都应该有LLM决策（100%控制率）
            if self.cached_decision is None:
                # 如果没有LLM决策，检查是否是同步模式
                if self.model.synchronous_mode:
                    # 同步模式下，如果没有决策，说明有问题（应该在上一步已经获取）
                    # 抛出异常，不允许使用保守行为
                    raise RuntimeError(
                        f"Vehicle {self.unique_id}: No LLM decision available in synchronous mode! "
                        f"This should not happen. Check that run_synchronous_llm_step() is called before step()."
                    )
                else:
                    raise RuntimeError(
                        f"Vehicle {self.unique_id}: missing LLM decision. "
                        "NaSch LLM mode is barrier-synchronous only."
                    )
            else:
                # Use LLM decision (this is the real behavior - 100% control rate)
                new_speed = self.cached_decision
                self.using_fallback = False
                if hasattr(self, '_waiting_for_llm'):
                    self._waiting_for_llm = False
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Store new speed (but don't move yet - that happens in move())
        self.new_speed = max(0, min(int(new_speed), self.max_speed))
    
    def apply_llm_decision(self, decision_speed, cache_duration=None):
        """
        Apply a new LLM decision and cache it for future steps.
        
        Args:
            decision_speed: The speed decision from LLM
            cache_duration: Number of steps to cache this decision (optional)
        """
        if cache_duration is None:
            cache_duration = getattr(self.model, "llm_cache_duration", 50)

        self.cached_decision = max(0, min(int(decision_speed), self.max_speed))
        # decision_expiry_step: 在当前step_count基础上，缓存cache_duration步
        # 在同步模式下，run_synchronous_llm_step()在step()之前调用
        # 应用决策时step_count是N，然后step()中step_count会变成N+1
        # 所以decision_expiry_step应该设置为 (N+1) + (cache_duration-1) = N + cache_duration
        # 但实际上，由于step()会在之后调用，step_count会增加1
        # 所以如果我们想让决策在当前step和下一个step都有效，应该设置为N + cache_duration + 1
        # 但更简单的方法是：在同步模式下，如果cache_duration=1，设置为step_count + 2（覆盖当前step和下一个step）
        if self.model.synchronous_mode and cache_duration == 1:
            self.decision_expiry_step = self.model.step_count + 1
        else:
            # 异步模式或cache_duration > 1：正常设置
            self.decision_expiry_step = self.model.step_count + cache_duration
        self.last_llm_decision_step = self.model.step_count
        self.using_fallback = False
        stats = getattr(self.model, "llm_control_stats", None)
        if stats is not None:
            stats["llm_decisions"] = stats.get("llm_decisions", 0) + 1
    
    def move(self):
        """
        Phase 2: Move vehicle using calculated speed (parallel update).
        All vehicles move simultaneously after all speeds are calculated.
        """
        # Update speed and position
        self.speed = self.new_speed
        new_position = (self.position + self.speed) % self.model.L
        self.model.grid.move_agent(self, (new_position, 0))
        self.position = new_position
    
    def step(self):
        """
        DEPRECATED: Use calculate_new_speed() + move() for proper parallel updates.
        Kept for backward compatibility but should not be used.
        """
        self.calculate_new_speed()
        self.move()
    

class NaSchModel(Model):
    """
    NaSch traffic model with Mesa framework.
    
    Supports both ABM and LLM behavior modes for vehicles.
    Maintains the original NaSch physics while enabling intelligent decision making.
    """
    
    def __init__(self, L=250, rho=0.20, vmax=5, p=0.50, mode="abm", seed=None):
        super().__init__()
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        # Model parameters
        self.L = L  # Road length
        self.rho = rho  # Vehicle density
        self.vmax = vmax  # Maximum speed
        self.p = p  # Randomization probability
        self.mode = mode  # Behavior mode: "abm" or "llm"
        
        # Mesa components
        self.grid = MultiGrid(L, 1, torus=True)  # 1D ring road
        # Use SimultaneousActivation for parallel updates (required for NaSch)
        self.schedule = SimultaneousActivation(self)
        # Initialize schedule steps to 0 (DataCollector uses schedule.steps)
        self.schedule.steps = 0
        
        # LLM integration parameters (similar to Social Force Model)
        self.llm_decision_tick_interval = 10  # Default; may be overridden below
        self.max_llm_agents_per_tick = 4  # Max vehicles per decision round
        self.llm_max_concurrent = int(os.environ.get("LLM_MAX_CONCURRENT", "2"))  # Max concurrent API calls (default: 2 to avoid rate limits)
        
        # Get LLM model name from environment (set by runner scripts)
        self.llm_model_name = os.environ.get("LLM_MODEL_NAME", "gpt-4o")
        
        # Adjust rate limiting for reasoning models (slower)
        self.min_seconds_between_llm_requests = 10.0 if any(r in self.llm_model_name.lower() 
                                                             for r in ['r1', 'thinking', 'reasoning', 'gpt-5']) else 3.0
        self._last_llm_request_time = 0.0
        self.step_count = 0  # Track simulation steps for throttling
        self._llm_vehicle_index = 0  # Round-robin index for covering all vehicles
        self.llm_cache_duration = 1
        self.llm_warmup_steps = 0
        self.llm_control_stats = {
            "total_decisions": 0,
            "llm_decisions": 0,
            "waiting_for_llm": 0,  # Vehicles waiting for LLM decision (no fallback)
        }
        self.last_llm_step_debug = []
        
        if mode == "llm":
            # Default: Aggressive cadence to ensure LLM meaningfully controls vehicles
            # Note: These defaults can be overridden by run_llm_simulation() parameters
            self.llm_decision_tick_interval = 1  # 默认每5步触发一次
            self.llm_cache_duration = 1  # Fresh LLM decision required every step.
            self.max_llm_agents_per_tick = 10**9  # 默认每批处理20个车辆
            self.llm_warmup_steps = 0  # 默认预热3步
            # Faster throttle for lightweight models (overridden for reasoning ones)
            if not any(r in self.llm_model_name.lower() for r in ['r1', 'thinking', 'reasoning', 'gpt-5']):
                self.min_seconds_between_llm_requests = 0.5  # 默认0.5秒间隔
        
        # LLM async processing system
        self.llm_request_queue = queue.Queue(maxsize=100)  # Throttling
        self.llm_response_queue = queue.Queue()
        self.llm_worker_thread = None
        self.pending_llm_request = None
        self.llm_performance_stats = {
            "requests_sent": 0,
            "responses_received": 0,
            "errors_encountered": 0,
            "queue_full_skips": 0
        }
        self.debug = False  # Set to True for LLM debugging
        self.synchronous_mode = True  # Canonical mode: barrier-synchronous only.
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Vehicles": lambda m: len(m.schedule.agents),
                "Avg_Speed": lambda m: np.mean([a.speed for a in m.schedule.agents]) if m.schedule.agents else 0,
                "Max_Speed": lambda m: max([a.speed for a in m.schedule.agents]) if m.schedule.agents else 0,
                "Min_Speed": lambda m: min([a.speed for a in m.schedule.agents]) if m.schedule.agents else 0,
            },
            agent_reporters={
                "Speed": "speed",
                "Position": "position",
            }
        )
        
        # Initialize vehicles
        self._spawn_vehicles()
        
        # Collect initial state (step 0)
        self.datacollector.collect(self)
    
    def _spawn_vehicles(self):
        """Spawn vehicles on the road"""
        N = int(self.rho * self.L)
        positions = random.sample(range(self.L), N)
        
        for i, pos in enumerate(positions):
            vehicle = Vehicle(
                unique_id=i,
                model=self,
                position=pos,
                speed=random.randint(0, self.vmax),
                max_speed=self.vmax,
                mode=self.mode
            )
            self.schedule.add(vehicle)
            self.grid.place_agent(vehicle, (pos, 0))

    def _take_vehicle_snapshot(self):
        """Freeze all vehicle states before computing any decisions."""
        return {
            v.unique_id: {
                "position": int(v.position),
                "speed": int(v.speed),
                "max_speed": int(v.max_speed),
            }
            for v in self.schedule.agents
        }

    def _gap_from_snapshot(self, position, snapshot):
        positions = {state["position"] for state in snapshot.values()}
        for distance in range(1, self.L):
            if (position + distance) % self.L in positions:
                return distance - 1
        return self.L - 1

    def _observation_from_snapshot(self, vehicle, snapshot):
        state = snapshot[vehicle.unique_id]
        return {
            "current_speed": state["speed"],
            "gap_ahead": self._gap_from_snapshot(state["position"], snapshot),
            "max_speed": state["max_speed"],
            "randomization_prob": self.p,
        }

    def _nasch_valid_outputs(self, current_speed, gap_ahead, vmax):
        v_temp = min(int(current_speed) + 1, int(vmax))
        v_safe = min(v_temp, int(gap_ahead))
        v_hesitate = max(v_safe - 1, 0)
        return sorted({int(v_safe), int(v_hesitate)})

    def _compute_abm_speed_from_snapshot(self, vehicle, snapshot):
        observation = self._observation_from_snapshot(vehicle, snapshot)
        if vehicle.abm_agent is None:
            vehicle.abm_agent = ABM_Agent(model="nasch", scenario="nasch")
        vehicle.abm_agent.update_attributes(observation)
        action = vehicle.abm_agent.take_actions()
        return max(0, min(int(action["speed"]), vehicle.max_speed))
    
    def _llm_worker_thread(self):
        """Background thread worker for LLM API calls"""
        print("[LLM worker] Thread started")
        while True:
            try:
                # Get request from queue
                request = self.llm_request_queue.get(timeout=1.0)
                if request is None:  # Shutdown signal
                    print("[LLM worker] Shutdown signal received")
                    break
                
                selected_vehicles, observations, vehicle_obs_map = request
                print(f"[LLM worker] Processing request for {len(observations)} vehicles (model: {self.llm_model_name})")
                
                # Make the API call
                try:
                    if self.debug:
                        print(f"[LLM worker] Making API call for {len(observations)} vehicles")
                    
                    # Process each vehicle's LLM decision
                    vehicle_action_map = {}
                    for vehicle_id, obs in vehicle_obs_map.items():
                        vehicle = next((v for v in selected_vehicles if v.unique_id == vehicle_id), None)
                        if vehicle is None:
                            print(f"[LLM worker WARNING] Vehicle {vehicle_id} not found in selected vehicles")
                            continue
                        
                        try:
                            # Get LLM agent for this vehicle
                            if vehicle.llm_agent is None:
                                print(f"[LLM worker] Creating LLM agent for vehicle {vehicle_id} (model: {self.llm_model_name})")
                                try:
                                    vehicle.llm_agent = LLM_Agent(model=self.llm_model_name, scenario="nasch")
                                except Exception as agent_e:
                                    error_msg = f"Failed to create LLM agent for vehicle {vehicle_id}: {agent_e}"
                                    print(f"[LLM worker ERROR] {error_msg}")
                                    raise RuntimeError(error_msg)
                            
                            if self.debug:
                                print(f"[LLM worker] Vehicle {vehicle_id}: Updating attributes: {obs}")
                            
                            print(f"[LLM worker] Vehicle {vehicle_id}: Calling update_attributes...")
                            vehicle.llm_agent.update_attributes(obs)
                            print(f"[LLM worker] Vehicle {vehicle_id}: Calling take_actions (this may take time)...")
                            
                            import time as time_module
                            
                            start_time = time_module.time()
                            
                            # Call take_actions directly (OpenAI client already has timeout)
                            # The client timeout (5s connect, 30s read) should be sufficient
                            # If it hangs, we'll get a timeout error from the client
                            try:
                                action = vehicle.llm_agent.take_actions()
                                elapsed = time_module.time() - start_time
                                print(f"[LLM worker] Vehicle {vehicle_id}: Got action in {elapsed:.2f}s: {action}")
                            except Exception as api_e:
                                elapsed = time_module.time() - start_time
                                error_msg = f"LLM API call failed for vehicle {vehicle_id} after {elapsed:.2f}s: {api_e}"
                                print(f"[LLM worker ERROR] {error_msg}")
                                import traceback
                                print(f"[LLM worker ERROR] Traceback: {traceback.format_exc()}")
                                raise RuntimeError(error_msg) from api_e
                            
                            if self.debug:
                                print(f"[LLM worker] Vehicle {vehicle_id}: Got action: {action}")
                            
                            # Validate action response
                            if action is None:
                                raise RuntimeError(f"LLM returned None action for vehicle {vehicle_id}")
                            
                            if not isinstance(action, dict):
                                raise RuntimeError(f"LLM returned invalid action type for vehicle {vehicle_id}: {type(action)}, value: {action}")
                            
                            decision_speed = action.get("speed", None)
                            if decision_speed is None:
                                raise RuntimeError(f"LLM action missing 'speed' key for vehicle {vehicle_id}: {action}")
                            
                            try:
                                decision_speed = int(decision_speed)
                            except (ValueError, TypeError):
                                raise RuntimeError(f"LLM returned invalid speed value for vehicle {vehicle_id}: {decision_speed} (type: {type(decision_speed)})")
                            
                            # Clamp speed to valid range
                            decision_speed = max(0, min(int(decision_speed), vehicle.max_speed))
                            vehicle_action_map[vehicle_id] = decision_speed
                            print(f"[LLM worker] Vehicle {vehicle_id}: Decision speed = {decision_speed}")
                            
                        except Exception as veh_e:
                            # NO FALLBACK: Raise error if LLM call fails
                            import traceback
                            error_msg = f"Vehicle {vehicle_id} LLM call failed: {veh_e}\n{traceback.format_exc()}"
                            print(f"[LLM worker ERROR] {error_msg}")
                            # Do NOT use fallback - raise error to stop simulation
                            raise RuntimeError(f"LLM decision required but failed for vehicle {vehicle_id}: {veh_e}")
                    
                    print(f"[LLM worker] ✅ API calls successful, got {len(vehicle_action_map)} decisions")
                    
                    self.llm_response_queue.put(('success', vehicle_action_map))
                    self.llm_performance_stats["responses_received"] += 1
                    print(f"[LLM worker] Response queued, total received: {self.llm_performance_stats['responses_received']}")
                    
                except Exception as e:
                    import traceback
                    error_msg = f"API call failed: {e}\n{traceback.format_exc()}"
                    print(f"[LLM worker ERROR] {error_msg}")
                    # Put error in queue only once
                    try:
                        self.llm_response_queue.put(('error', error_msg), timeout=1.0)
                        self.llm_performance_stats["errors_encountered"] += 1
                        print(f"[LLM worker] Error response queued")
                    except Exception as queue_e:
                        print(f"[LLM worker ERROR] Failed to queue error response: {queue_e}")
                
                finally:
                    # Always mark task as done, even if there was an error
                    self.llm_request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                # Log all exceptions (not just in debug mode) to help diagnose issues
                import traceback
                error_msg = f"[LLM worker ERROR] Unexpected error in worker thread: {e}\n{traceback.format_exc()}"
                print(error_msg)
                # Try to put error in queue so main thread knows something went wrong
                try:
                    self.llm_response_queue.put(('error', str(e)), timeout=0.1)
                    self.llm_performance_stats["errors_encountered"] = self.llm_performance_stats.get("errors_encountered", 0) + 1
                except:
                    pass  # If queue is full, just continue
                continue
    
    def _start_llm_worker(self):
        """Start the background LLM worker thread"""
        if self.llm_worker_thread is None or not self.llm_worker_thread.is_alive():
            print(f"[LLM] Starting worker thread (model: {self.llm_model_name})")
            self.llm_worker_thread = threading.Thread(target=self._llm_worker_thread, daemon=True)
            self.llm_worker_thread.start()
            print(f"[LLM] Worker thread started: {self.llm_worker_thread.is_alive()}")
        else:
            print(f"[LLM] Worker thread already running")
    
    def step(self):
        """
        Execute one barrier-synchronous simulation step.
        FREEZE all vehicle states, COMPUTE all speeds, then COMMIT moves together.
        """
        self.step_count += 1
        snapshot = self._take_vehicle_snapshot()

        if self.mode == "abm":
            for agent in self.schedule.agents:
                agent.new_speed = self._compute_abm_speed_from_snapshot(agent, snapshot)
        elif self.mode == "llm":
            for agent in self.schedule.agents:
                if agent.cached_decision is None or self.step_count > agent.decision_expiry_step:
                    raise RuntimeError(
                        f"Vehicle {agent.unique_id}: missing fresh LLM decision at step {self.step_count}"
                    )
                agent.new_speed = max(0, min(int(agent.cached_decision), agent.max_speed))
                self.llm_control_stats["total_decisions"] = self.llm_control_stats.get("total_decisions", 0) + 1
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        for agent in self.schedule.agents:
            agent.move()
        
        self._steps = self.step_count
        self.datacollector.collect(self)
        
        if self.mode == "llm" and self.debug and self.step_count % 5 == 0:
            self._print_llm_coverage_stats()
    def _force_llm_request_all_vehicles(self):
        """Force a full sweep of LLM requests across all vehicles (warm-up)."""
        llm_vehicles = [v for v in self.schedule.agents if v.mode == "llm"]
        if not llm_vehicles:
            return

        self._start_llm_worker()

        batches_sent = 0
        for start_idx in range(0, len(llm_vehicles), self.max_llm_agents_per_tick):
            batch = llm_vehicles[start_idx:start_idx + self.max_llm_agents_per_tick]
            vehicle_obs_map = {}
            for v in batch:
                obs = v._build_observation()
                obs["max_speed"] = v.max_speed
                obs["randomization_prob"] = self.p
                obs["step"] = self.step_count
                vehicle_obs_map[v.unique_id] = obs

            try:
                self.llm_request_queue.put_nowait((batch, list(vehicle_obs_map.values()), vehicle_obs_map))
                self.llm_performance_stats["requests_sent"] += 1
                batches_sent += 1
            except queue.Full:
                if self.debug:
                    print("[LLM WARMUP] Queue full, skipping batch")
                break

        if batches_sent:
            self._last_llm_request_time = time.time()
            if self.debug:
                print(f"[LLM WARMUP] Step {self.step_count}: queued {batches_sent} batches")

    def _print_llm_coverage_stats(self):
        """Diagnostic printout of LLM control usage (NO FALLBACK)."""
        stats = self.llm_control_stats
        total = stats.get("total_decisions", 0)
        if total == 0:
            return

        llm_rate = stats.get("llm_decisions", 0) / total * 100.0
        waiting_rate = (total - stats.get("llm_decisions", 0)) / total * 100.0

        llm_controlled = sum(1 for v in self.schedule.agents
                             if v.mode == "llm" and v.cached_decision is not None)
        total_llm = sum(1 for v in self.schedule.agents if v.mode == "llm")

        print(f"\n[LLM COVERAGE] Step {self.step_count}")
        print(f"  Active LLM control: {llm_controlled}/{total_llm}")
        print(f"  Decision split: LLM {llm_rate:.1f}% | Waiting for LLM {waiting_rate:.1f}% (NO FALLBACK)")
        print(f"  API stats: sent={self.llm_performance_stats['requests_sent']}, "
              f"received={self.llm_performance_stats['responses_received']}, "
              f"errors={self.llm_performance_stats['errors_encountered']}")

    def run_synchronous_llm_step(self):
        """
        Blocking LLM update that waits for fresh decisions (synchronous mode).
        使用并发模式：所有车辆同时调用LLM API，等待所有响应到达后统一更新状态。
        """
        if self.mode != "llm":
            return

        llm_vehicles = [v for v in self.schedule.agents if v.mode == "llm"]
        if not llm_vehicles:
            return

        import time as time_module
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 为所有车辆构建观察
        snapshot = self._take_vehicle_snapshot()
        vehicle_obs_map = {}
        for v in llm_vehicles:
            obs = self._observation_from_snapshot(v, snapshot)
            obs["step"] = self.step_count
            vehicle_obs_map[v.unique_id] = obs

        # 并发处理函数：为单个车辆获取LLM决策（带重试机制）
        def get_llm_decision(vehicle_id, max_retries=3):
            """为单个车辆获取LLM决策（并发执行，带速率限制重试）"""
            vehicle = next((v for v in llm_vehicles if v.unique_id == vehicle_id), None)
            if vehicle is None:
                return None, RuntimeError(f"Vehicle {vehicle_id} not found")
            
            for retry in range(max_retries):
                try:
                    # 创建LLM agent（如果还没有）
                    if vehicle.llm_agent is None:
                        vehicle.llm_agent = LLM_Agent(model=self.llm_model_name, scenario="nasch")
                    
                    # 获取观察
                    obs = vehicle_obs_map[vehicle_id]
                    
                    # 更新属性并调用LLM
                    vehicle.llm_agent.update_attributes(obs)
                    action = vehicle.llm_agent.take_actions()  # 阻塞调用，等待响应
                    
                    # 验证响应
                    if action is None:
                        raise RuntimeError(f"LLM returned None action for vehicle {vehicle_id}")
                    if not isinstance(action, dict):
                        raise RuntimeError(f"LLM returned invalid action type: {type(action)}")
                    
                    decision_speed = action.get("speed", None)
                    if decision_speed is None:
                        raise RuntimeError(f"LLM action missing 'speed' key: {action}")
                    
                    # 限制速度范围
                    decision_speed = max(0, min(int(decision_speed), vehicle.max_speed))
                    valid_outputs = self._nasch_valid_outputs(
                        obs["current_speed"], obs["gap_ahead"], obs["max_speed"]
                    )
                    debug_entry = {
                        "step": int(self.step_count),
                        "vehicle_id": int(vehicle_id),
                        "id": int(vehicle_id),
                        "raw_response": str(action.get("raw_response", "")),
                        "parsed_speed": int(action.get("parsed_speed", decision_speed)),
                        "parse_strategy": str(action.get("parse_strategy", "unknown")),
                        "v_current": int(obs["current_speed"]),
                        "gap_ahead": int(obs["gap_ahead"]),
                        "vmax": int(obs["max_speed"]),
                        "p": float(obs["randomization_prob"]),
                        "abm_expected": int(valid_outputs[-1]),
                        "abm_valid_outputs": valid_outputs,
                        "compliant": int(decision_speed) in valid_outputs,
                    }
                    if "error" in action:
                        debug_entry["parse_error"] = str(action["error"])
                    return vehicle_id, {"speed": decision_speed, "debug": debug_entry}
                    
                except Exception as exc:
                    # 检查是否是速率限制错误（429）
                    error_str = str(exc)
                    if "429" in error_str or "rate_limit" in error_str.lower() or "Rate limit" in error_str:
                        if retry < max_retries - 1:
                            # 提取等待时间（如果有）
                            wait_time = 1.0  # 默认等待1秒
                            if "try again in" in error_str:
                                try:
                                    # 尝试提取等待时间（例如 "try again in 72ms"）
                                    import re
                                    match = re.search(r'try again in (\d+)(ms|s)', error_str)
                                    if match:
                                        wait_amt = float(match.group(1))
                                        unit = match.group(2)
                                        if unit == "ms":
                                            wait_time = wait_amt / 1000.0
                                        else:
                                            wait_time = wait_amt
                                    wait_time = max(wait_time, 0.1)  # 至少等待0.1秒
                                except:
                                    wait_time = 1.0 + retry * 0.5  # 指数退避
                            
                            time_module.sleep(wait_time)
                            continue  # 重试
                        else:
                            # 最后一次重试也失败
                            return vehicle_id, exc
                    else:
                        # 非速率限制错误，直接返回
                        return vehicle_id, exc
            
            # 所有重试都失败
            return vehicle_id, RuntimeError(f"Failed after {max_retries} retries for vehicle {vehicle_id}")

        # 使用ThreadPoolExecutor并发处理所有车辆
        # 最大并发数：使用配置的并发数（默认2，避免速率限制）
        max_workers = min(len(llm_vehicles), self.llm_max_concurrent)
        start_time = time_module.time()
        
        print(f"[LLM CONCURRENT] Step {self.step_count}: 并发处理 {len(llm_vehicles)} 个车辆 (最大并发: {max_workers})")
        
        vehicle_decisions = {}  # {vehicle_id: decision_speed}
        vehicle_debug = {}
        vehicle_errors = {}     # {vehicle_id: error}
        
        # 并发执行所有车辆的LLM调用
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(get_llm_decision, vehicle_id): vehicle_id 
                for vehicle_id in vehicle_obs_map.keys()
            }
            
            # 等待所有任务完成
            for future in as_completed(futures):
                vehicle_id = futures[future]
                try:
                    result_vehicle_id, result = future.result()
                    if isinstance(result, Exception):
                        vehicle_errors[result_vehicle_id] = result
                    else:
                        vehicle_decisions[result_vehicle_id] = result["speed"]
                        vehicle_debug[result_vehicle_id] = result["debug"]
                except Exception as exc:
                    vehicle_errors[vehicle_id] = exc
        
        elapsed = time_module.time() - start_time
        if len(vehicle_decisions) != len(llm_vehicles):
            missing = sorted({v.unique_id for v in llm_vehicles} - set(vehicle_decisions))
            raise RuntimeError(f"LLM decision barrier incomplete; missing vehicles: {missing}")
        
        # 检查错误
        if vehicle_errors:
            error_msg = f"[LLM CONCURRENT ERROR] {len(vehicle_errors)} 个车辆失败:\n"
            rate_limit_errors = []
            other_errors = []
            
            for vehicle_id, error in vehicle_errors.items():
                error_str = str(error)
                error_msg += f"  Vehicle {vehicle_id}: {error}\n"
                # 检查是否是速率限制错误
                if "429" in error_str or "rate limit" in error_str.lower() or "rate_limit" in error_str.lower():
                    rate_limit_errors.append((vehicle_id, error))
                else:
                    other_errors.append((vehicle_id, error))
            
            print(error_msg)
            
            # 如果主要是速率限制错误，给出更明确的提示
            if rate_limit_errors and len(rate_limit_errors) >= len(vehicle_errors) * 0.5:
                print("\n" + "!" * 80)
                print("!" * 80)
                print("⚠️  ⚠️  ⚠️  检测到速率限制错误！ ⚠️  ⚠️  ⚠️")
                print("!" * 80)
                print(f"失败的车辆数: {len(rate_limit_errors)}/{len(vehicle_errors)}")
                print(f"当前并发数: {max_workers}")
                print(f"当前步数: {self.step_count}")
                print("!" * 80)
                print("建议:")
                print("  1. 降低并发数（当前配置: LLM_MAX_CONCURRENT=2）")
                print("  2. 等待1-2分钟后重试")
                print("  3. 检查API配额和使用情况")
                print("!" * 80)
                print("!" * 80 + "\n")
            
            # 在同步模式中，如果任何车辆失败，抛出异常（严格模式，不使用fallback）
            raise RuntimeError(f"LLM concurrent calls failed for {len(vehicle_errors)} vehicles: {list(vehicle_errors.values())[0]}")
        
        # 统一应用到所有车辆（等所有决策都到达后）
        print(f"[LLM CONCURRENT] ✅ 所有 {len(vehicle_decisions)} 个决策已到达 (耗时: {elapsed:.2f}秒)")
        self.last_llm_step_debug = [
            vehicle_debug[vehicle_id]
            for vehicle_id in sorted(vehicle_debug)
        ]
        for vehicle_id, decision_speed in vehicle_decisions.items():
            vehicle = next((v for v in llm_vehicles if v.unique_id == vehicle_id), None)
            if vehicle:
                # 在同步模式下，step()会在之后调用，step_count会增加1
                # 所以decision_expiry_step应该设置为step_count + cache_duration
                # 这样在step()中step_count增加后，检查时step_count < decision_expiry_step仍然为True
                vehicle.apply_llm_decision(decision_speed, cache_duration=self.llm_cache_duration)
                if self.debug:
                    print(f"  Vehicle {vehicle_id}: speed = {decision_speed}, expiry_step = {vehicle.decision_expiry_step}")
        
        # 更新统计
        self.llm_performance_stats["responses_received"] = self.llm_performance_stats.get("responses_received", 0) + len(vehicle_decisions)
        self.llm_performance_stats["requests_sent"] = self.llm_performance_stats.get("requests_sent", 0) + len(llm_vehicles)

    def get_vehicle_positions(self):
        """Get current vehicle positions for visualization"""
        positions = []
        for agent in self.schedule.agents:
            positions.append(agent.position)
        return positions
    
    def get_vehicle_speeds(self):
        """Get current vehicle speeds for analysis"""
        speeds = []
        for agent in self.schedule.agents:
            speeds.append(agent.speed)
        return speeds
    
    def get_llm_performance_stats(self):
        """Get LLM performance statistics"""
        return self.llm_performance_stats.copy() if self.mode == "llm" else {}

    def export_metrics_csv(self, path: str):
        """Export model-level metrics to CSV (append)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df_model = self.datacollector.get_model_vars_dataframe()
        last = df_model.iloc[-1] if not df_model.empty else None
        fields = {
            "mode": self.mode,
            "L": int(self.L),
            "rho": float(self.rho),
            "vmax": int(self.vmax),
            "p": float(self.p),
            "steps": int(len(df_model))
        }
        if last is not None:
            fields.update({
                "Total_Vehicles": float(last.get("Total_Vehicles", 0)),
                "Avg_Speed": float(last.get("Avg_Speed", 0)),
                "Max_Speed": float(last.get("Max_Speed", 0)),
                "Min_Speed": float(last.get("Min_Speed", 0)),
            })
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fields.keys()))
            if write_header:
                w.writeheader()
            w.writerow(fields)
    
    def save_checkpoint(self, checkpoint_path: str):
        """
        Save model state to a checkpoint file.
        
        Args:
            checkpoint_path: Path to save the checkpoint file
        """
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Collect current state
        checkpoint_data = {
            # Model parameters
            "L": self.L,
            "rho": self.rho,
            "vmax": self.vmax,
            "p": self.p,
            "mode": self.mode,
            "step_count": self.step_count,
            "schedule_steps": self.schedule.steps,
            
            # LLM parameters
            "llm_decision_tick_interval": self.llm_decision_tick_interval,
            "llm_cache_duration": self.llm_cache_duration,
            "llm_max_concurrent": self.llm_max_concurrent,
            "llm_model_name": self.llm_model_name,
            "synchronous_mode": getattr(self, 'synchronous_mode', False),
            "llm_control_stats": self.llm_control_stats.copy(),
            "llm_performance_stats": self.llm_performance_stats.copy(),
            
            # Vehicle states
            "vehicles": []
        }
        
        # Save vehicle states
        for agent in self.schedule.agents:
            vehicle_data = {
                "unique_id": agent.unique_id,
                "position": agent.position,
                "speed": agent.speed,
                "new_speed": agent.new_speed,
                "max_speed": agent.max_speed,
                "mode": agent.mode,
                "cached_decision": agent.cached_decision,
                "decision_expiry_step": agent.decision_expiry_step,
                "last_llm_decision_step": agent.last_llm_decision_step,
            }
            checkpoint_data["vehicles"].append(vehicle_data)
        
        # Save DataCollector data
        # Reset index so Step/AgentID become columns (needed for proper restoration)
        df_model = self.datacollector.get_model_vars_dataframe()
        if not df_model.empty:
            checkpoint_data["datacollector_model_vars"] = df_model.reset_index().to_dict('records')
        else:
            checkpoint_data["datacollector_model_vars"] = []
        
        df_agent = self.datacollector.get_agent_vars_dataframe()
        if not df_agent.empty:
            checkpoint_data["datacollector_agent_vars"] = df_agent.reset_index().to_dict('records')
        else:
            checkpoint_data["datacollector_agent_vars"] = []
        
        # Save metadata
        checkpoint_data["checkpoint_time"] = datetime.now().isoformat()
        checkpoint_data["checkpoint_version"] = "1.0"
        
        # Save to file
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"💾 检查点已保存: {checkpoint_path} (步数: {self.step_count})")
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str, seed=None):
        """
        Load model state from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            seed: Random seed (optional, for reproducibility)
        
        Returns:
            NaSchModel: Restored model instance
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        # Load checkpoint data
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        print(f"📂 加载检查点: {checkpoint_path}")
        print(f"   步数: {checkpoint_data['step_count']}")
        print(f"   车辆数: {len(checkpoint_data['vehicles'])}")
        print(f"   保存时间: {checkpoint_data.get('checkpoint_time', '未知')}")
        
        # Create model with saved parameters
        model = NaSchModel(
            L=checkpoint_data["L"],
            rho=checkpoint_data["rho"],
            vmax=checkpoint_data["vmax"],
            p=checkpoint_data["p"],
            mode=checkpoint_data["mode"],
            seed=seed
        )
        
        # Restore LLM parameters
        model.llm_decision_tick_interval = checkpoint_data.get("llm_decision_tick_interval", 50)
        model.llm_cache_duration = checkpoint_data.get("llm_cache_duration", model.llm_decision_tick_interval)
        model.llm_max_concurrent = checkpoint_data.get("llm_max_concurrent", 2)
        model.llm_model_name = checkpoint_data.get("llm_model_name", "gpt-4o")
        model.synchronous_mode = checkpoint_data.get("synchronous_mode", False)
        model.llm_control_stats = checkpoint_data.get("llm_control_stats", {
            "total_decisions": 0,
            "llm_decisions": 0,
            "waiting_for_llm": 0,
        })
        model.llm_performance_stats = checkpoint_data.get("llm_performance_stats", {
            "requests_sent": 0,
            "responses_received": 0,
            "errors_encountered": 0,
            "queue_full_skips": 0,
        })
        
        # Clear existing vehicles
        model.schedule = SimultaneousActivation(model)
        model.grid = MultiGrid(model.L, 1, torus=True)
        
        # Restore vehicles
        for vehicle_data in checkpoint_data["vehicles"]:
            vehicle = Vehicle(
                unique_id=vehicle_data["unique_id"],
                model=model,
                position=vehicle_data["position"],
                speed=vehicle_data["speed"],
                max_speed=vehicle_data["max_speed"],
                mode=vehicle_data["mode"]
            )
            vehicle.new_speed = vehicle_data.get("new_speed", vehicle_data["speed"])
            vehicle.cached_decision = vehicle_data.get("cached_decision", None)
            vehicle.decision_expiry_step = vehicle_data.get("decision_expiry_step", 0)
            vehicle.last_llm_decision_step = vehicle_data.get("last_llm_decision_step", -1)
            
            model.schedule.add(vehicle)
            model.grid.place_agent(vehicle, (vehicle.position, 0))
        
        # Restore step count
        model.step_count = checkpoint_data["step_count"]
        model.schedule.steps = checkpoint_data.get("schedule_steps", checkpoint_data["step_count"])
        
        # Restore DataCollector historical data from checkpoint
        # Store as model attributes so visualization functions can combine them
        import pandas as pd
        if "datacollector_model_vars" in checkpoint_data and checkpoint_data["datacollector_model_vars"]:
            try:
                checkpoint_model_df = pd.DataFrame(checkpoint_data["datacollector_model_vars"])
                # Handle both old format (no Step column) and new format (Step is a column)
                if "Step" in checkpoint_model_df.columns:
                    # New format: Step is a column, set it as index
                    checkpoint_model_df.set_index("Step", inplace=True)
                else:
                    # Old format: Step information was lost in to_dict('records')
                    # Try to reconstruct Step from the number of records
                    # Assume steps start from 0 and are consecutive
                    num_steps = len(checkpoint_model_df)
                    checkpoint_step_count = checkpoint_data.get("step_count", num_steps)
                    # Create Step index: assume data starts from step 0 or from (checkpoint_step_count - num_steps)
                    start_step = max(0, checkpoint_step_count - num_steps)
                    checkpoint_model_df.index = pd.RangeIndex(start=start_step, stop=start_step + num_steps, name="Step")
                    print(f"  注意: 旧格式检查点，从步数 {start_step} 开始推断")
                model.checkpoint_model_vars_df = checkpoint_model_df
                print(f"✅ 已恢复检查点模型变量数据: {len(checkpoint_model_df)} 步")
            except Exception as e:
                print(f"⚠️  恢复检查点模型变量数据失败: {e}")
                import traceback
                traceback.print_exc()
                model.checkpoint_model_vars_df = None
        else:
            model.checkpoint_model_vars_df = None
        
        if "datacollector_agent_vars" in checkpoint_data and checkpoint_data["datacollector_agent_vars"]:
            try:
                checkpoint_agent_df = pd.DataFrame(checkpoint_data["datacollector_agent_vars"])
                # Agent vars have MultiIndex (Step, AgentID) - after reset_index, both become columns
                # Find the index columns (Step and the agent ID column)
                index_cols = []
                if "Step" in checkpoint_agent_df.columns:
                    index_cols.append("Step")
                # Find agent ID column (could be AgentID, unique_id, or similar)
                agent_id_col = None
                for col in ["AgentID", "unique_id", "Agent"]:
                    if col in checkpoint_agent_df.columns:
                        agent_id_col = col
                        break
                
                if len(index_cols) >= 1 and agent_id_col:
                    # New format: both Step and AgentID are columns
                    index_cols.append(agent_id_col)
                    checkpoint_agent_df.set_index(index_cols, inplace=True)
                    model.checkpoint_agent_vars_df = checkpoint_agent_df
                    print(f"✅ 已恢复检查点代理变量数据: {len(checkpoint_agent_df)} 条记录")
                elif "Step" in checkpoint_agent_df.columns:
                    # Only Step available (unlikely but possible)
                    checkpoint_agent_df.set_index("Step", inplace=True)
                    model.checkpoint_agent_vars_df = checkpoint_agent_df
                    print(f"✅ 已恢复检查点代理变量数据: {len(checkpoint_agent_df)} 条记录（仅Step索引）")
                else:
                    # Old format: Step and AgentID information was lost in to_dict('records')
                    # We can't perfectly reconstruct MultiIndex without knowing the structure
                    # For now, skip restoration of old format agent vars
                    print(f"⚠️  检查点代理变量数据为旧格式（缺少索引信息），无法恢复")
                    model.checkpoint_agent_vars_df = None
            except Exception as e:
                print(f"⚠️  恢复检查点代理变量数据失败: {e}")
                import traceback
                traceback.print_exc()
                model.checkpoint_agent_vars_df = None
        else:
            model.checkpoint_agent_vars_df = None
        
        # Collect current state to DataCollector (for continuation)
        model.datacollector.collect(model)
        
        print(f"✅ 检查点加载完成: 步数 {model.step_count}, 车辆数 {len(model.schedule.agents)}")
        if model.checkpoint_model_vars_df is not None or model.checkpoint_agent_vars_df is not None:
            print(f"📊 检查点数据已恢复，可视化时将合并历史数据")
        
        return model


def run_simulation(L=200, rho=0.25, vmax=5, p=0.2, steps=100, mode="abm", seed=42):
    """
    Run a NaSch simulation with specified parameters.
    
    Args:
        L: Road length
        rho: Vehicle density
        vmax: Maximum speed
        p: Randomization probability
        steps: Number of simulation steps
        mode: "abm" or "llm"
        seed: Random seed
    
    Returns:
        NaSchModel: The completed simulation model
    """
    model = NaSchModel(L=L, rho=rho, vmax=vmax, p=p, mode=mode, seed=seed)
    
    for _ in range(steps):
        model.step()
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Running NaSch simulation...")
    
    # ABM mode
    print("Running ABM mode...")
    abm_model = run_simulation(L=100, rho=0.3, vmax=5, p=0.2, steps=50, mode="abm")
    print(f"ABM - Final average speed: {abm_model.datacollector.get_model_vars_dataframe()['Avg_Speed'].iloc[-1]:.2f}")
    
    # LLM mode (requires API key)
    try:
        print("Running LLM mode...")
        llm_model = run_simulation(L=100, rho=0.3, vmax=5, p=0.2, steps=50, mode="llm")
        print(f"LLM - Final average speed: {llm_model.datacollector.get_model_vars_dataframe()['Avg_Speed'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"LLM mode failed (likely API key issue): {e}")
        print("Please ensure OPENAI_API_KEY is set to a valid API key")
