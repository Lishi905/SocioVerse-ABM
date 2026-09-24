#!/usr/bin/env python3
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
纯LLM驱动的Social Force Model - 无物理力版本
- LLM直接输出速度向量
- 每2秒统一决策（所有agent同时调用LLM）
- 不使用任何物理力
- 碰撞解决：放在空闲位置
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import base classes and helpers from original
try:
    from social_force_core import (
        Wall, PARAMETERS, _time_to_collision, _distance_to_segment
    )
except ImportError:
    # Fallback: define locally if import fails
    from dataclasses import dataclass
    
    @dataclass
    class Wall:
        start: Tuple[float, float]
        end: Tuple[float, float]
        name: str
    
    PARAMETERS = {
        "R": 0.2,
        "v_0": 1.34,
        "v_max": 3.0,
        "dt": 0.05,
        "spawn_rate": 2.0,
        "min_spawn_distance": 1.0,
        "walkway_width": 10.0,
        "walkway_length": 50.0,
    }
    
    def _time_to_collision(r_ab: np.ndarray, v_ab: np.ndarray, epsilon: float = 0.01, t_max: float = 10.0) -> float:
        v_rel_mag = float(np.linalg.norm(v_ab))
        if v_rel_mag < epsilon:
            return t_max
        approach_dist = -float(np.dot(r_ab, v_ab)) / v_rel_mag
        if approach_dist <= 0:
            return t_max
        ttc = approach_dist / v_rel_mag
        return float(min(ttc, t_max))
    
    def _distance_to_segment(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> Tuple[float, np.ndarray]:
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
# PURE LLM POLICY (OUTPUTS VELOCITY VECTOR)
# ========================================================================

PURE_LLM_SYSTEM_PROMPT = (
    "Task Description: You control a pedestrian in a crowded walkway simulation. Your objective is to navigate through the passage and reach the other side as quickly as possible. CRITICAL RULES: DO NOT move sideways along walls - your goal is to CROSS the passage. Move DIRECTLY toward your destination - avoid unnecessary lateral movement. In bottleneck scenarios, aim for the DOORWAY OPENING, not the walls. Only adjust your path to avoid collisions with other pedestrians, not to avoid the passage itself.\n\n"
    "Action Rules: You must output a velocity vector [vx, vy] that represents your desired movement direction and speed. The velocity vector should point DIRECTLY toward your destination (minimize sideways movement), have a reasonable speed (0.5 to 3.0 m/s for normal walking), only adjust direction to avoid nearby pedestrians (NOT to avoid walls or the passage), and prioritize forward progress over lateral movement. Action space: velocity [vx, vy] in m/s, speed magnitude (0.5 to 3.0 m/s).\n\n"
    "Response Format: Return ONLY a JSON object with the velocity vector: {\"velocity\": [vx, vy], \"speed\": v, \"reasoning\": \"brief explanation\"}"
)

class PureLLMPolicy:
    """纯LLM策略，输出速度向量"""
    
    def __init__(self, model: str = "gpt-4o-2024-08-06"):
        env_model = os.getenv("SFM_LLM_MODEL")
        self.model = env_model if env_model else model
        api_key = self._get_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        
        try:
            import openai
            openai.api_key = api_key
        except ImportError:
            raise RuntimeError("OpenAI package not available")
    
    def _get_api_key(self) -> Optional[str]:
        """获取API key"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                key_file = 'OPENAI_API_KEY.txt'
                if os.path.exists(key_file):
                    with open(key_file, 'r') as f:
                        api_key = f.read().strip()
            except Exception:
                pass
        return api_key
    
    def _get_base_url(self) -> Optional[str]:
        """获取Base URL（参考NaSch的实现）"""
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            base_url = os.getenv("OPENAI_API_BASE")
        if not base_url:
            # 检查配置文件
            try:
                base_url_file = 'OPENAI_BASE_URL.txt'
                if os.path.exists(base_url_file):
                    with open(base_url_file, 'r') as f:
                        base_url = f.read().strip()
            except Exception:
                pass
        return base_url
    
    def _call_openai(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """调用OpenAI API"""
        try:
            import openai
            
            # 获取Base URL
            base_url = self._get_base_url()
            
            # 简化观察（只保留关键信息）
            obs_trim = {
                "self_state": observation.get("self_state", {}),
                "local_neighbors": observation.get("local_neighbors", [])[:5],  # 只保留5个最近的
                "environment": observation.get("environment", {}),
                "bottleneck_info": observation.get("bottleneck_info", {}),  # 包含bottleneck信息
            }
            
            # 根据场景构建不同的prompt
            scenario = observation.get("environment", {}).get("scenario", "bidirectional")
            bottleneck_info = observation.get("bottleneck_info", {})
            
            # Build user prompt in unified format
            self_state = obs_trim.get("self_state", {})
            local_neighbors = obs_trim.get("local_neighbors", [])
            environment = obs_trim.get("environment", {})
            
            # Agent Information
            attributes = f"""Position: {self_state.get('position', [0, 0])}
Velocity: {self_state.get('velocity', [0, 0])}
Speed: {self_state.get('speed', 0):.2f}
Goal direction: {self_state.get('goal_direction', [0, 0])}
Destination: {self_state.get('destination', [0, 0])}
Nearby pedestrians: {len(local_neighbors)}
Nearby pedestrians details: {json.dumps(local_neighbors[:5], separators=(',', ':'))}"""
            
            # Environment Settings
            env_configs = f"""Scenario: {scenario}
Walkway width: {environment.get('walkway_width', 10.0):.1f}
Walkway length: {environment.get('walkway_length', 50.0):.1f}
Density: {environment.get('density', 0.0):.3f}"""
            
            if scenario == "bottleneck" and bottleneck_info:
                env_configs += f"""
Bottleneck info: {json.dumps(bottleneck_info, separators=(',', ':'))}"""
            
            user_prompt = f"""Agent Information: {attributes}

Environment Settings: {env_configs}"""
            
            max_retries = 3
            backoff = 1.0
            for attempt in range(max_retries + 1):
                result = [None]
                exception = [None]
                
                def api_call():
                    try:
                        # 创建客户端（支持自定义base_url）
                        client_kwargs = {"api_key": self._get_api_key()}
                        if base_url:
                            client_kwargs["base_url"] = base_url
                        
                        client = openai.OpenAI(**client_kwargs, timeout=(5.0, 30.0))
                        
                        response = client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": PURE_LLM_SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.2,
                            max_tokens=500,
                        )
                        content = response.choices[0].message.content or "{}"
                        
                        # 提取JSON
                        start = content.find("{")
                        end = content.rfind("}")
                        if start == -1 or end == -1:
                            raise ValueError("No JSON found in response")
                        
                        parsed = json.loads(content[start:end+1])
                        result[0] = parsed
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=api_call)
                thread.daemon = True
                thread.start()
                thread.join(timeout=30.0)
                
                if thread.is_alive():
                    exception[0] = TimeoutError("LLM API call timed out")
                elif exception[0]:
                    if attempt < max_retries:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise exception[0]
                else:
                    return result[0]
            
            raise RuntimeError("LLM call failed after retries")
        except Exception as e:
            raise
    
    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """单个决策"""
        return self._call_openai(observation)
    
    def decide_batch(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量决策（每个agent单独调用）"""
        results = []
        for obs in observations:
            try:
                action = self._call_openai(obs)
                results.append(action)
            except Exception as e:
                raise RuntimeError(
                    f"LLM API call failed for agent: {e}. "
                    f"Pure LLM mode requires all agents to have LLM decisions. "
                    f"No fallback allowed."
                ) from e
        return results

# ========================================================================
# Synchronous LLM decision processing functions
# ========================================================================

def request_single_llm_decision(engine: 'PureLLMEngine', pedestrian_id: int) -> tuple:
    """
    Request LLM decision for a single pedestrian (for concurrent processing)
    
    Returns:
        (pedestrian_id, velocity_dict) or (pedestrian_id, {"__error__": error_message})
    """
    pedestrian = next((p for p in engine.pedestrians if p.id == pedestrian_id and p.active), None)
    if pedestrian is None:
        return (pedestrian_id, {"__error__": "Pedestrian not found"})
    
    try:
        observation = pedestrian.build_observation(engine)
        action = engine.llm_policy.decide(observation)
        
        if "velocity" in action:
            vel = np.array(action["velocity"], dtype=float)
            speed = np.linalg.norm(vel)
            v_max = PARAMETERS["v_max"]
            if speed > v_max and speed > 1e-8:
                vel = (vel / speed) * v_max
            return (pedestrian_id, {"velocity": vel})
        else:
            goal_vec = np.array(observation["self_state"]["goal_direction"])
            speed = PARAMETERS["v_0"] * action.get("speed_modulation", {}).get("factor", 1.0)
            vel = goal_vec * speed
            return (pedestrian_id, {"velocity": vel})
    except Exception as e:
        return (pedestrian_id, {"__error__": str(e)})


def request_llm_decision_batch(engine: 'PureLLMEngine', pedestrian_ids: List[int], 
                               batch_num: int, total_batches: int, 
                               max_workers: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
    """
    Request LLM decisions for a batch of pedestrians (truly concurrent processing)
    
    Args:
        engine: PureLLMEngine instance
        pedestrian_ids: List of pedestrian IDs to process
        batch_num: Current batch number
        total_batches: Total number of batches
        max_workers: Maximum concurrent threads per batch
    """
    results = {}
    
    if engine.llm_policy is None:
        engine.llm_policy = PureLLMPolicy()
    
    if max_workers is None:
        max_workers = len(pedestrian_ids)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pedestrian = {
            executor.submit(request_single_llm_decision, engine, pid): pid
            for pid in pedestrian_ids
        }
        
        for future in as_completed(future_to_pedestrian):
            pid = future_to_pedestrian[future]
            try:
                result_pid, result = future.result()
                results[result_pid] = result
                
                if isinstance(result, dict) and "__error__" in result:
                    engine.llm_errors_count += 1
                    if engine.debug:
                        print(f"[BATCH {batch_num}/{total_batches}] Error for pedestrian {pid}: {result['__error__']}")
                else:
                    engine.llm_decisions_count += 1
            except Exception as e:
                results[pid] = {"__error__": str(e)}
                engine.llm_errors_count += 1
                if engine.debug:
                    print(f"[BATCH {batch_num}/{total_batches}] Exception for pedestrian {pid}: {e}")
    
    return results


def wait_for_all_llm_decisions(engine: 'PureLLMEngine', batch_size: int = 20, 
                                max_workers_per_batch: Optional[int] = None) -> Dict[int, np.ndarray]:
    """
    Wait for all pedestrians' LLM decisions to complete (batched concurrent, synchronous processing)
    
    Args:
        engine: PureLLMEngine instance
        batch_size: Number of concurrent requests per batch
        max_workers_per_batch: Maximum concurrent threads per batch
    
    Returns:
        Dict[int, np.ndarray]: {pedestrian_id: velocity_vector}
    """
    all_pedestrian_ids = [p.id for p in engine.pedestrians if p.active]
    total_pedestrians = len(all_pedestrian_ids)
    if total_pedestrians == 0:
        return {}
    
    total_batches = (total_pedestrians + batch_size - 1) // batch_size
    all_results = {}
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_pedestrians)
        batch_ids = all_pedestrian_ids[start_idx:end_idx]
        
        if engine.debug:
            print(f"[BATCH {batch_idx + 1}/{total_batches}] Processing {len(batch_ids)} pedestrians...")
        
        batch_results = request_llm_decision_batch(
            engine, batch_ids, batch_idx + 1, total_batches, 
            max_workers=max_workers_per_batch
        )
        all_results.update(batch_results)
    
    velocity_map = {}
    for pid, result in all_results.items():
        if isinstance(result, dict) and "__error__" in result:
            raise RuntimeError(
                f"LLM API call failed for pedestrian {pid}: {result['__error__']}. "
                f"Pure LLM mode requires all agents to have LLM decisions. "
                f"No fallback allowed."
            )
        elif "velocity" in result:
            velocity_map[pid] = result["velocity"]
        else:
            raise RuntimeError(f"Invalid LLM result for pedestrian {pid}: {result}")
    
    return velocity_map


# ========================================================================
# PURE LLM PEDESTRIAN (NO PHYSICS)
# ========================================================================

class PureLLMPedestrian:
    """Pure LLM-driven pedestrian without physics forces"""
    
    def __init__(self, position: np.ndarray, destination: np.ndarray, 
                 scenario: str = "bidirectional", pedestrian_id: int = 0):
        self.position = position.copy()
        self.destination = destination.copy()
        self.scenario = scenario
        self.active = True
        self.id = pedestrian_id
        
        self.radius = PARAMETERS["R"] * random.uniform(0.9, 1.1)
        
        goal_vec = destination - position
        goal_dist = np.linalg.norm(goal_vec)
        if goal_dist > 1e-8:
            goal_dir = goal_vec / goal_dist
            initial_speed = PARAMETERS.get("v_0", 1.2) * random.uniform(0.8, 1.2)
            self.velocity = goal_dir * initial_speed
        else:
            self.velocity = np.zeros(2)
        self.speed = np.linalg.norm(self.velocity)
        
        self.last_llm_decision_time: float = 0.0
        
        # Track initial spawn direction for visualization
        self.spawn_direction: Optional[str] = None
    
    def update_position(self, dt: float, engine: 'PureLLMEngine'):
        """Update position using velocity vector (simple Euler integration) with wall collision detection"""
        if not self.active:
            return
        
        speed = np.linalg.norm(self.velocity)
        speed = np.linalg.norm(self.velocity)
        v_max = PARAMETERS["v_max"]
        if speed > v_max and speed > 1e-8:
            self.velocity = (self.velocity / speed) * v_max
            speed = v_max
        
        self.speed = speed
        
        # 计算新位置
        new_position = self.position + self.velocity * dt
        
        hit_wall = False
        for wall in engine.walls:
            wall_start = np.array(wall.start)
            wall_end = np.array(wall.end)
            distance, nearest_point = _distance_to_segment(new_position, wall_start, wall_end)
            
            if distance < self.radius:
                hit_wall = True
                wall_vec = wall_end - wall_start
                wall_normal = np.array([-wall_vec[1], wall_vec[0]])
                wall_normal_len = np.linalg.norm(wall_normal)
                if wall_normal_len > 1e-8:
                    wall_normal = wall_normal / wall_normal_len
                    if np.dot(new_position - nearest_point, wall_normal) < 0:
                        wall_normal = -wall_normal
                    
                    velocity_normal = np.dot(self.velocity, wall_normal) * wall_normal
                    self.velocity = self.velocity - 2 * velocity_normal
                    
                    push_distance = self.radius - distance + 0.01
                    new_position = new_position + wall_normal * push_distance
                    break
        
        # 更新位置
        self.position = new_position
    
    def check_boundaries(self, width: float, length: float):
        """检查是否到达边界或目的地"""
        # 检查是否到达目的地
        distance_to_dest = np.linalg.norm(self.position - self.destination)
        if distance_to_dest < 0.5:
            self.active = False
            return
        
        # 检查是否离开模拟区域
        if (self.position[0] < -1.0 or self.position[0] > length + 1.0 or
            self.position[1] < -1.0 or self.position[1] > width + 1.0):
            self.active = False
    
    def build_observation(self, engine: 'PureLLMEngine') -> Dict[str, Any]:
        """构建观察（简化版，专注于位置和邻居）"""
        pos = self.position.copy()
        vel = self.velocity.copy()
        speed = float(np.linalg.norm(vel))
        goal_vec = self.destination - self.position
        goal_dist = float(np.linalg.norm(goal_vec))
        goal_dir = goal_vec / goal_dist if goal_dist > 1e-8 else np.zeros(2)
        
        # 邻居信息（简化）
        local_neighbors = []
        for other in engine.pedestrians:
            if other is self or not other.active:
                continue
            r = other.position - pos
            d = float(np.linalg.norm(r))
            if d < 1e-6 or d > 5.0:  # 只考虑5米内的邻居
                continue
            local_neighbors.append({
                "relative_position": [float(r[0]), float(r[1])],
                "distance": d,
                "relative_velocity": [float((other.velocity - vel)[0]), float((other.velocity - vel)[1])],
            })
        # 只保留最近的7个
        local_neighbors.sort(key=lambda x: x["distance"])
        local_neighbors = local_neighbors[:7]
        
        # 环境信息
        nearest_wall_distance = 1e9
        for w in engine.walls:
            d, _ = _distance_to_segment(pos, np.array(w.start), np.array(w.end))
            if d < nearest_wall_distance:
                nearest_wall_distance = d
        
        # 瓶颈场景特殊处理
        bottleneck_info = {}
        if self.scenario == "bottleneck":
            door_center = np.array([engine.length * 0.5, engine.width * 0.5])
            door_vec = door_center - pos
            doorway_distance = float(np.linalg.norm(door_vec))
            bottleneck_info = {
                "doorway_distance": doorway_distance,
                "doorway_center": [float(door_center[0]), float(door_center[1])],
            }
        
        return {
            "self_state": {
                "position": [float(pos[0]), float(pos[1])],
                "velocity": [float(vel[0]), float(vel[1])],
                "speed": speed,
                "destination": [float(self.destination[0]), float(self.destination[1])],
                "goal_direction": [float(goal_dir[0]), float(goal_dir[1])],
                "goal_distance": goal_dist,
            },
            "local_neighbors": local_neighbors,
            "environment": {
                "nearest_wall_distance": float(nearest_wall_distance),
                "scenario": self.scenario,
            },
            "bottleneck_info": bottleneck_info,
        }

# ========================================================================
# PURE LLM ENGINE
# ========================================================================

class PureLLMEngine:
    """纯LLM驱动的模拟引擎"""
    
    def __init__(self, scenario: str = "bidirectional", 
                 width: float = 10.0, length: float = 50.0,
                 decision_interval: Optional[float] = None,
                 seed: Optional[int] = None,
                 batch_size: int = 20,
                 max_workers_per_batch: Optional[int] = None):
        """Initialize pure LLM simulation engine"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.scenario = scenario
        self.width = width
        self.length = length
        if decision_interval is None:
            self.decision_interval = PARAMETERS["dt"]
        else:
            self.decision_interval = decision_interval
        self.dt = PARAMETERS["dt"]
        PARAMETERS["walkway_width"] = width
        PARAMETERS["walkway_length"] = length
        
        # 模拟状态
        self.pedestrians: List[PureLLMPedestrian] = []
        self.walls: List[Wall] = []
        self.time = 0.0
        self.step_count = 0
        self.last_decision_time = 0.0
        
        # 生成
        self.last_spawn_time = 0.0
        self.spawn_counter = 0
        
        # LLM处理
        self.llm_policy: Optional[PureLLMPolicy] = None
        self.llm_request_queue = queue.Queue()
        self.llm_response_queue = queue.Queue()
        self.llm_worker_thread = None
        self.pending_llm_request = None
        self.synchronous_mode = True
        self.debug = False
        
        self.llm_decisions_count = 0
        self.llm_errors_count = 0
        self.llm_decision_round = 0
        
        self.batch_size = batch_size
        self.max_workers_per_batch = max_workers_per_batch
        
        self._lane_counts: List[float] = []
        self._bottleneck_metrics: List[Dict[str, float]] = []
        
        self.create_walls()
        self.spawn_initial_pedestrians(30)
    
    def create_walls(self):
        """创建墙壁边界"""
        if self.scenario == "bidirectional":
            self.walls = [
                Wall((0, 0), (self.length, 0), "bottom_wall"),
                Wall((0, self.width), (self.length, self.width), "top_wall")
            ]
        elif self.scenario == "bottleneck":
            bottleneck_position = self.length * 0.5
            doorway_width = 1.2
            walkway_center = self.width / 2
            door_top = walkway_center + doorway_width / 2
            door_bottom = walkway_center - doorway_width / 2
            
            self.walls = [
                Wall((0, 0), (bottleneck_position, 0), "bottom_wall_left"),
                Wall((bottleneck_position, 0), (self.length, 0), "bottom_wall_right"),
                Wall((0, self.width), (bottleneck_position, self.width), "top_wall_left"),
                Wall((bottleneck_position, self.width), (self.length, self.width), "top_wall_right"),
                Wall((bottleneck_position, 0), (bottleneck_position, door_bottom), "left_door_wall"),
                Wall((bottleneck_position, door_top), (bottleneck_position, self.width), "right_door_wall")
            ]
    
    def spawn_initial_pedestrians(self, count: int):
        """生成初始行人"""
        for i in range(count):
            self.spawn_pedestrian()
            self.time += 0.5  # 分散生成时间
    
    def spawn_pedestrian(self):
        """Spawn a new pedestrian"""
        spawn_side = "left" if random.random() < 0.5 else "right"
        
        if spawn_side == "left":
            spawn_x = -0.5
            destination_x = self.length + 0.5
        else:
            spawn_x = self.length + 0.5
            destination_x = -0.5
        
        if self.scenario == "bottleneck":
            spawn_y = random.uniform(0.5, self.width - 0.5)
            doorway_center_y = self.width / 2
            destination_y = doorway_center_y
        else:
            spawn_y = random.uniform(0.5, self.width - 0.5)
            destination_y = spawn_y
        
        spawn_pos = np.array([spawn_x, spawn_y])
        min_distance = PARAMETERS["min_spawn_distance"]
        
        too_close = any(
            np.linalg.norm(spawn_pos - p.position) < min_distance 
            for p in self.pedestrians if p.active
        )
        
        if not too_close:
            destination = np.array([destination_x, destination_y])
            pedestrian = PureLLMPedestrian(spawn_pos, destination, self.scenario, int(self.spawn_counter))
            pedestrian.spawn_direction = spawn_side
            self.pedestrians.append(pedestrian)
            self.spawn_counter += 1
    
    def _check_collision(self, p1: PureLLMPedestrian, p2: PureLLMPedestrian) -> bool:
        """检查两个行人是否碰撞"""
        distance = np.linalg.norm(p1.position - p2.position)
        min_distance = p1.radius + p2.radius
        return distance < min_distance
    
    def _is_position_free(self, position: np.ndarray, radius: float, exclude_pedestrian: Optional[PureLLMPedestrian] = None) -> bool:
        """检查位置是否空闲（不与其他人重叠，不在墙壁内）"""
        # 检查边界
        if position[0] < radius or position[0] > self.length - radius:
            return False
        if position[1] < radius or position[1] > self.width - radius:
            return False
        
        # 检查墙壁
        for wall in self.walls:
            d, _ = _distance_to_segment(position, np.array(wall.start), np.array(wall.end))
            if d < radius:
                return False
        
        # 检查与其他行人的重叠
        for other in self.pedestrians:
            if other is exclude_pedestrian or not other.active:
                continue
            distance = np.linalg.norm(position - other.position)
            if distance < radius + other.radius:
                return False
        
        return True
    
    def _find_free_position(self, pedestrian: PureLLMPedestrian, max_attempts: int = 100) -> Optional[np.ndarray]:
        """查找空闲位置（三层策略）"""
        current_pos = pedestrian.position.copy()
        radius = pedestrian.radius
        
        # 方法1: 网格搜索
        offsets = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        for offset in offsets:
            for angle_deg in angles:
                angle_rad = math.radians(angle_deg)
                candidate = current_pos + offset * np.array([math.cos(angle_rad), math.sin(angle_rad)])
                if self._is_position_free(candidate, radius, pedestrian):
                    return candidate
        
        # 方法2: 随机采样
        for _ in range(max_attempts):
            offset = random.uniform(0.2, 1.5)
            angle = random.uniform(0, 2 * math.pi)
            candidate = current_pos + offset * np.array([math.cos(angle), math.sin(angle)])
            if self._is_position_free(candidate, radius, pedestrian):
                return candidate
        
        # 方法3: 最小位移分离
        # 找到最近的冲突行人，向反方向移动
        min_distance = 1e9
        closest_other = None
        for other in self.pedestrians:
            if other is pedestrian or not other.active:
                continue
            distance = np.linalg.norm(pedestrian.position - other.position)
            if distance < min_distance:
                min_distance = distance
                closest_other = other
        
        if closest_other is not None:
            direction = pedestrian.position - closest_other.position
            if np.linalg.norm(direction) > 1e-8:
                direction = direction / np.linalg.norm(direction)
                required_distance = pedestrian.radius + closest_other.radius + 0.1
                candidate = closest_other.position + direction * required_distance
                if self._is_position_free(candidate, radius, pedestrian):
                    return candidate
        
        return None
    
    def _resolve_collisions(self):
        """解决所有碰撞（顺序处理）"""
        # 检测所有碰撞对
        collisions = []
        for i, p1 in enumerate(self.pedestrians):
            if not p1.active:
                continue
            for j, p2 in enumerate(self.pedestrians[i+1:], i+1):
                if not p2.active:
                    continue
                if self._check_collision(p1, p2):
                    collisions.append((p1, p2))
        
        # 按ID顺序处理（优先级）
        processed = set()
        for p1, p2 in collisions:
            if p1.id in processed and p2.id in processed:
                continue
            
            # 优先级：ID小的先处理
            if p1.id not in processed:
                free_pos = self._find_free_position(p1)
                if free_pos is not None:
                    p1.position = free_pos
                    processed.add(p1.id)
            
            if p2.id not in processed:
                free_pos = self._find_free_position(p2)
                if free_pos is not None:
                    p2.position = free_pos
                    processed.add(p2.id)
    
    def _llm_worker_thread(self):
        """后台线程处理LLM API调用"""
        while True:
            try:
                request = self.llm_request_queue.get(timeout=1.0)
                if request is None:  # 关闭信号
                    break
                
                agents, observations, agent_obs_map = request
                
                try:
                    # 创建policy（如果还没有）
                    if self.llm_policy is None:
                        self.llm_policy = PureLLMPolicy()
                    
                    # 批量调用LLM（每个agent单独调用）
                    actions = self.llm_policy.decide_batch(observations)
                    
                    # 解析速度向量
                    agent_velocity_map = {}
                    for (agent_id, obs), action in zip(agent_obs_map.items(), actions):
                        # 新格式：LLM返回速度向量
                        if "velocity" in action:
                            vel = np.array(action["velocity"], dtype=float)
                            # 速度限制
                            speed = np.linalg.norm(vel)
                            v_max = PARAMETERS["v_max"]
                            if speed > v_max and speed > 1e-8:
                                vel = (vel / speed) * v_max
                            agent_velocity_map[agent_id] = vel
                        else:
                            # 兼容旧格式：从direction_bias和speed_modulation计算速度
                            # 这里简化处理，使用默认速度
                            goal_vec = np.array(obs["self_state"]["goal_direction"])
                            speed = PARAMETERS["v_0"] * action.get("speed_modulation", {}).get("factor", 1.0)
                            vel = goal_vec * speed
                            agent_velocity_map[agent_id] = vel
                    
                    self.llm_response_queue.put(('success', agent_velocity_map))
                    self.llm_decisions_count += len(actions)
                    
                except Exception as e:
                    print(f"[LLM worker] Error: {e}")
                    self.llm_response_queue.put(('error', str(e)))
                    self.llm_errors_count += 1
                
                self.llm_request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[LLM worker] Fatal error: {e}")
                continue
    
    def _start_llm_worker(self):
        """启动LLM工作线程"""
        if self.llm_worker_thread is None or not self.llm_worker_thread.is_alive():
            self.llm_worker_thread = threading.Thread(target=self._llm_worker_thread, daemon=True)
            self.llm_worker_thread.start()
    
    def step(self):
        """Execute one simulation step (synchronous mode: wait for all LLM decisions)"""
        self.step_count += 1
        self.time += self.dt
        
        if self.synchronous_mode and (self.time - self.last_decision_time) >= self.decision_interval:
            velocity_map = wait_for_all_llm_decisions(
                self, 
                batch_size=self.batch_size,
                max_workers_per_batch=self.max_workers_per_batch
            )
            
            for pedestrian_id, velocity in velocity_map.items():
                pedestrian = next((p for p in self.pedestrians if p.id == pedestrian_id and p.active), None)
                if pedestrian:
                    pedestrian.velocity = velocity.copy()
                    pedestrian.last_llm_decision_time = self.time
            
            self.last_decision_time = self.time
            self.llm_decision_round += 1
            
            if self.debug:
                print(f"[LLM] Decision round {self.llm_decision_round}: {len(velocity_map)} pedestrians")
        
        for pedestrian in self.pedestrians:
            if pedestrian.active:
                pedestrian.update_position(self.dt, self)
                pedestrian.check_boundaries(self.width, self.length)
        
        # 解决碰撞
        self._resolve_collisions()
        
        # 移除inactive的行人
        self.pedestrians = [p for p in self.pedestrians if p.active]
        
        # 生成新行人
        time_since_spawn = self.time - self.last_spawn_time
        if time_since_spawn >= 1.0 / PARAMETERS["spawn_rate"]:
            self.spawn_pedestrian()
            self.last_spawn_time = self.time
    
    def get_active_pedestrians(self) -> List[PureLLMPedestrian]:
        """获取所有active的行人"""
        return [p for p in self.pedestrians if p.active]
    
    def _estimate_bottleneck_metrics(self) -> Optional[Dict[str, float]]:
        """估算瓶颈指标（仅bottleneck场景）"""
        if self.scenario != "bottleneck":
            return None
        
        active = self.get_active_pedestrians()
        if not active:
            return None
        
        bottleneck_x = self.length * 0.5  # 瓶颈位置
        doorway_width = 1.2
        walkway_center = self.width / 2
        door_top = walkway_center + doorway_width / 2
        door_bottom = walkway_center - doorway_width / 2
        
        # 定义瓶颈区域：门道前后各2米
        bottleneck_zone_width = 4.0  # 门道前后各2米
        bottleneck_left = bottleneck_x - bottleneck_zone_width / 2
        bottleneck_right = bottleneck_x + bottleneck_zone_width / 2
        
        # 统计瓶颈区域内的行人
        bottleneck_pedestrians = []
        queue_pedestrians = []  # 瓶颈前等待的行人
        
        for p in active:
            x, y = p.position[0], p.position[1]
            # 在瓶颈区域内
            if bottleneck_left <= x <= bottleneck_right:
                # 检查是否在门道内（y方向）
                if door_bottom <= y <= door_top:
                    bottleneck_pedestrians.append(p)
                # 在瓶颈区域但不在门道内（排队）
                elif (bottleneck_left <= x < bottleneck_x) or (bottleneck_x < x <= bottleneck_right):
                    queue_pedestrians.append(p)
        
        # 计算瓶颈密度（门道内的密度）
        bottleneck_area = doorway_width * bottleneck_zone_width
        bottleneck_density = len(bottleneck_pedestrians) / bottleneck_area if bottleneck_area > 0 else 0.0
        
        # 计算排队长度（瓶颈前等待的行人数量）
        queue_length = len(queue_pedestrians)
        
        # 计算瓶颈通过率（通过门道的速度）
        # 统计正在通过门道的行人速度
        passing_speeds = []
        for p in bottleneck_pedestrians:
            # 计算朝向门道的速度分量
            px = p.position[0]
            vx = p.velocity[0]
            speed = np.linalg.norm(p.velocity)
            # 如果从左侧来（px < bottleneck_x），vx应该为正
            # 如果从右侧来（px > bottleneck_x），vx应该为负
            if (px < bottleneck_x and vx > 0) or (px > bottleneck_x and vx < 0):
                passing_speeds.append(speed)
        
        # 通过率 = 平均速度 × 门道内行人数量（行人/秒）
        bottleneck_throughput = np.mean(passing_speeds) * len(bottleneck_pedestrians) if passing_speeds else 0.0
        
        return {
            "bottleneck_density": float(bottleneck_density),
            "queue_length": float(queue_length),
            "bottleneck_throughput": float(bottleneck_throughput),
        }
    
    def _estimate_lane_count(self) -> Optional[float]:
        """估算车道数量（仅bidirectional场景）"""
        if self.scenario != "bidirectional":
            return None
        active = self.get_active_pedestrians()
        if not active:
            return None
        
        # 按方向分离（使用x方向速度或目的地方向）
        left_to_right = []
        right_to_left = []
        for p in active:
            vx = p.velocity[0]
            # 如果速度接近0，使用目的地方向
            if abs(vx) < 1e-3:
                dir_sign = 1.0 if (p.destination[0] - p.position[0]) >= 0 else -1.0
            else:
                dir_sign = 1.0 if vx >= 0 else -1.0
            if dir_sign >= 0:
                left_to_right.append(p.position[1])
            else:
                right_to_left.append(p.position[1])
        
        def count_bands(y_vals: List[float], bandwidth: float = 0.8) -> int:
            """计算y方向的带数（车道数）"""
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
        # 两个方向的平均车道数
        return 0.5 * (bands_lr + bands_rl)
    
    def update_lane_metrics(self):
        """更新车道指标"""
        lane_estimate = self._estimate_lane_count()
        if lane_estimate is not None:
            self._lane_counts.append(lane_estimate)
    
    def update_bottleneck_metrics(self):
        """更新瓶颈指标"""
        bottleneck_metrics = self._estimate_bottleneck_metrics()
        if bottleneck_metrics is not None:
            self._bottleneck_metrics.append(bottleneck_metrics)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取模拟指标"""
        active_pedestrians = self.get_active_pedestrians()
        
        # 更新场景特定指标
        if self.scenario == "bidirectional":
            self.update_lane_metrics()
        elif self.scenario == "bottleneck":
            self.update_bottleneck_metrics()
        
        if not active_pedestrians:
            lane_mean = float(np.mean(self._lane_counts)) if self._lane_counts else None
            bottleneck_density = None
            queue_length = None
            bottleneck_throughput = None
            
            if self._bottleneck_metrics:
                bottleneck_density = float(np.mean([m["bottleneck_density"] for m in self._bottleneck_metrics]))
                queue_length = float(np.mean([m["queue_length"] for m in self._bottleneck_metrics]))
                bottleneck_throughput = float(np.mean([m["bottleneck_throughput"] for m in self._bottleneck_metrics]))
            
            return {
                "total_pedestrians": len(self.pedestrians),
                "active_pedestrians": 0,
                "average_speed": 0.0,
                "density": 0.0,
                "flow_rate": 0.0,
                "lane_mean": lane_mean,
                "bottleneck_density": bottleneck_density,
                "queue_length": queue_length,
                "bottleneck_throughput": bottleneck_throughput,
            }
        
        avg_speed = np.mean([p.speed for p in active_pedestrians])
        density = len(active_pedestrians) / (self.width * self.length)
        flow_rate = avg_speed * density
        
        # 计算场景特定指标
        lane_mean = float(np.mean(self._lane_counts)) if self._lane_counts else None
        
        bottleneck_density = None
        queue_length = None
        bottleneck_throughput = None
        if self._bottleneck_metrics:
            bottleneck_density = float(np.mean([m["bottleneck_density"] for m in self._bottleneck_metrics]))
            queue_length = float(np.mean([m["queue_length"] for m in self._bottleneck_metrics]))
            bottleneck_throughput = float(np.mean([m["bottleneck_throughput"] for m in self._bottleneck_metrics]))
        
        return {
            "total_pedestrians": len(self.pedestrians),
            "active_pedestrians": len(active_pedestrians),
            "average_speed": avg_speed,
            "density": density,
            "flow_rate": flow_rate,
            "lane_mean": lane_mean,
            "bottleneck_density": bottleneck_density,
            "queue_length": queue_length,
            "bottleneck_throughput": bottleneck_throughput,
            "llm_decisions": self.llm_decisions_count,
            "llm_errors": self.llm_errors_count,
        }
    
    def export_metrics_csv(self, path: str):
        """导出指标到CSV文件（追加模式）"""
        import csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        metrics = self.get_metrics()
        lane_metrics = {}
        if self.scenario == "bidirectional":
            lane_metrics = {
                "lane_count_mean": metrics.get("lane_mean"),
            }
        elif self.scenario == "bottleneck":
            lane_metrics = {
                "bottleneck_density": metrics.get("bottleneck_density"),
                "queue_length": metrics.get("queue_length"),
                "bottleneck_throughput": metrics.get("bottleneck_throughput"),
            }
        
        fields = {
            "scenario": self.scenario,
            "width": float(self.width),
            "length": float(self.length),
            "time": float(self.time),
            "step_count": int(self.step_count),
            "total_pedestrians": metrics["total_pedestrians"],
            "active_pedestrians": metrics["active_pedestrians"],
            "average_speed": metrics["average_speed"],
            "density": metrics["density"],
            "flow_rate": metrics["flow_rate"],
            "llm_decisions": metrics["llm_decisions"],
            "llm_errors": metrics["llm_errors"],
            **lane_metrics,
        }
        
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fields.keys()))
            if write_header:
                w.writeheader()
            w.writerow(fields)

