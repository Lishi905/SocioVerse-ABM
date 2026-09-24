# -*- coding: utf-8 -*-
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
Schelling Model - LCM-LBF-VE变体
LLM Context Modeling + LLM Behavior Function + Venue Enhancement
Level 1邻居信息 + Level 2场所信息
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from socioverse.behavior_engine.LLM_agent import LLM_Agent
from socioverse.behavior_engine.LLM_based import parse_attributes
from simulations.Schelling_model import SchellingBase, SchellingVisualization, setup_logger, save_step_log
from simulations.Schelling_model.venue import (
    Venue, get_accessible_venues, get_venue_encounters, create_venues, manhattan_distance
)


class SchellingLCM_LBF_VE:
    """Schelling LCM-LBF-VE变体 - 完整上下文 + 场所信息 + LLM推理"""
    
    def __init__(self, grid_size, steps, density, minority_pc, homophily, seed, output_pdf,
                 model="gpt-4o-2024-08-06", log_file=None, enable_step_log=False,
                 venue_config=None):
        """
        初始化 Schelling LCM-LBF-VE 模型
        
        Args:
            grid_size: 网格大小 (width, height)
            steps: 仿真步数
            density: 网格占用率 (0-1)
            minority_pc: 少数群体比例 (0-1)
            homophily: 同质性阈值 (期望的相似邻居数量)
            seed: 随机种子
            output_pdf: 输出路径
            model: LLM模型名称 (default: gpt-4o-2024-08-06)
            log_file: 日志文件路径
            enable_step_log: 是否启用每步的JSON日志记录
            venue_config: 场所配置字典，包含：
                - "mode": "single_center", "multi_center", "mixed"
                - "num_exclusive_per_group": 每个群体的专属场所数量
                - "num_public": 公共场所数量
                - "catchment_radius": 影响范围
        """
        self.width, self.height = grid_size
        self.steps = steps
        self.density = density
        self.minority_pc = minority_pc
        self.homophily = homophily
        self.seed = seed
        self.output_pdf = output_pdf
        self.model = model
        self.enable_step_log = enable_step_log
        self.llm_step_log = {}
        
        # 初始化基础模型
        self.base = SchellingBase(self.width, self.height, self.density, self.minority_pc, self.seed)
        self.base.homophily = self.homophily
        
        # 为每个代理分配冒险精神（Adventurousness）参数（0-1随机分布）
        # 这是Silver论文中引入的异质性参数
        random.seed(self.seed)
        self.agent_adventurousness = {}  # {agent_id: adventurousness}
        
        # 创建场所
        if venue_config is None:
            venue_config = {
                "mode": "mixed",
                "num_exclusive_per_group": 1,
                "num_public": 2,
                "catchment_radius": max(self.width, self.height) // 4,
                "exclusivity": 1.0  # 默认完全排他
            }
        self.venues = create_venues(venue_config, self.width, self.height, self.seed)
        
        # 配置日志
        mode_name = "LCM-LBF-VE"
        if log_file is None:
            log_file = f"./logs/schelling_{mode_name.lower()}_seed{seed}.log"
        self.logger = setup_logger(log_file, f"Schelling_{mode_name}", seed)
        
        # 配置 JSON step log 目录
        if self.enable_step_log:
            self.step_log_dir = os.path.join(self.output_pdf, "logs")
            os.makedirs(self.step_log_dir, exist_ok=True)
            self.logger.info(f"Step logs will be saved to: {self.step_log_dir}")
            self.logger.info(f"Created {len(self.venues)} venues: {[v.venue_type for v in self.venues]}")
        else:
            self.step_log_dir = None
        
        self.llm_agent = None
        self.history = []
        self.snapshots = []
        self.mode_name = mode_name
        
        # OCC 优化参数
        self.max_workers = None  # None 表示使用默认值（CPU核心数）
        self.use_occ = True  # 是否启用 OCC 优化
        
        # 代理视野参数（用于计算可移动的空位范围）
        # 如果为None，则代理可以移动到网格中的任何空位
        # 如果为整数，表示代理只能移动到曼哈顿距离 <= vision_range 的空位
        self.agent_vision_range = None  # None 表示无限制，或设置为整数（如 3, 5 等）
    
    def _generate_proposal_for_agent(self, agent, grid_snapshot, empty_cells_snapshot):
        """
        为单个代理生成移动提议（OCC 提议阶段）
        
        Args:
            agent: 代理字典
            grid_snapshot: 网格快照（代理位置字典 {pos: agent_id}）
            empty_cells_snapshot: 空位快照（集合）
        
        Returns:
            proposal: 提议字典，包含 agent_id, action, target_pos, decision_info
        """
        # 创建独立的 LLM agent 实例（避免线程冲突）
        llm_agent = LLM_Agent(model=self.model, scenario="Schelling_LCM_LBF_VE")
        
        # 计算邻居信息
        similar, total = self.base.count_similar_neighbors(agent['pos'], agent['type'])
        neighbors_info = self.base._get_neighbors_detail(agent['pos'], agent['type'])
        
        # 获取代理的冒险精神
        agent_adventurousness = self.agent_adventurousness.get(agent['id'], 0.5)
        
        # 获取可访问的场所信息
        accessible_venues = get_accessible_venues(
            agent['pos'], agent['type'], agent_adventurousness,
            self.venues, self.width, self.height
        )
        
        # 计算在场所中遇到的其他代理人
        venue_encounters = get_venue_encounters(
            agent['pos'], agent['type'], agent_adventurousness,
            accessible_venues, self.base.agents,
            self.width, self.height
        )
        
        # 计算合并的满意度
        combined_same = similar + venue_encounters["venue_same_group_count"]
        combined_total = total + venue_encounters["venue_total_count"]
        combined_satisfaction_ratio = combined_same / combined_total if combined_total > 0 else 0.0
        is_satisfied = combined_same >= self.homophily if combined_total > 0 else False
        
        # # 如果已经满意，直接返回 stay
        # if is_satisfied:
        #     return {
        #         'agent_id': agent['id'],
        #         'action': 'stay',
        #         'target_pos': None,
        #         'decision': {'decision': 'stay'},
        #         'prompt': None,
        #         'reasoning': 'Already satisfied'
        #     }
        
        # 构造 LLM 输入
        attributes = {
            "agent_pos": agent['pos'],
            "agent_type": agent['type'],
            "similar_neighbors": similar,
            "total_neighbors": total,
            "neighbors_info": neighbors_info,
            "homophily_threshold": self.homophily,
            "grid_size": (self.width, self.height),
            "empty_cells_count": len(empty_cells_snapshot),
            "sample_empty_cells": list(empty_cells_snapshot)[:min(5, len(empty_cells_snapshot))],
            "accessible_venues": accessible_venues,
            "venue_encounters": venue_encounters,
            "combined_same_group": combined_same,
            "combined_total": combined_total,
            "combined_satisfaction_ratio": round(combined_satisfaction_ratio, 3),
            "is_satisfied": is_satisfied,
            "adventurousness": round(agent_adventurousness, 3)
        }
        
        llm_agent.update_attributes(attributes)
        
        # 获取 prompt
        prompt = parse_attributes(llm_agent.scenario, attributes)
        
        # 调用 LLM 生成决策
        decision = llm_agent.take_actions()
        
        if isinstance(decision, dict):
            action = decision.get('decision', 'stay').lower()
            target_pos = decision.get('target_position')
            
            # 验证目标位置
            if action == 'move' and target_pos:
                if isinstance(target_pos, (list, tuple)) and len(target_pos) == 2:
                    target_pos = tuple(target_pos)
                    # 验证目标位置是否在空位快照中
                    if target_pos not in empty_cells_snapshot:
                        # 无效位置，记录错误但不移动
                        decision['validation_error'] = f'目标位置 {target_pos} 不在空位列表中'
                        action = 'stay'
                        target_pos = None
                else:
                    # 目标位置格式无效
                    decision['validation_error'] = f'目标位置格式无效: {target_pos}'
                    action = 'stay'
                    target_pos = None
            elif action == 'move' and not target_pos:
                # LLM 决定 move 但未指定位置，记录错误但不移动
                decision['validation_error'] = 'LLM决定移动但未指定目标位置'
                action = 'stay'
                target_pos = None
            else:
                target_pos = None
        else:
            action = 'stay'
            target_pos = None
            decision = {'decision': 'stay'}
        
        return {
            'agent_id': agent['id'],
            'action': action,
            'target_pos': target_pos,
            'decision': decision,
            'prompt': prompt,
            'reasoning': decision.get('reasoning', '') if isinstance(decision, dict) else ''
        }
    
    def run_simulation(self):
        """运行LCM-LBF-VE仿真（使用OCC优化）"""
        print(f"[{self.mode_name}模式] 开始运行 {self.steps} 步仿真...")
        print(f"[说明] LLM基于完整上下文和场所信息进行综合推理决策")
        print(f"[场所] 共 {len(self.venues)} 个场所")
        if self.use_occ:
            print(f"[优化] 启用 OCC (乐观并发控制) 并行优化")
        
        self.base.init_grid_model()
        self.llm_agent = LLM_Agent(model=self.model, scenario="Schelling_LCM_LBF_VE")
        
        # 为每个代理分配冒险精神（如果还没有分配）
        if not self.agent_adventurousness:
            for agent in self.base.agents:
                # 随机分配冒险精神（0-1均匀分布）
                self.agent_adventurousness[agent['id']] = random.random()
        
        # 初始状态统计（使用合并的满意度计算）
        happy_count = self._update_happiness_with_venues()
        
        self.history.append({
            'iteration': 0,
            'happy_count': happy_count,
            'total_agents': len(self.base.agents),
            'happiness_ratio': happy_count / len(self.base.agents)
        })
        
        # 保存初始状态快照
        self.snapshots.append({
            'step': 0,
            'happy_count': happy_count,
            'agents': [{'id': a['id'], 'type': a['type'], 'pos': a['pos'], 'happy': a['happy']}
                      for a in self.base.agents]
        })
        
        # 保存初始状态的 step log
        if self.enable_step_log:
            save_step_log(0, happy_count, self.base.agents, self.base, self.mode_name, self.model,
                         self.step_log_dir, self.logger)
        
        # 迭代
        print(f"\n开始仿真迭代...")
        pbar = tqdm(total=self.steps, desc=f"{self.mode_name}仿真进度",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                   mininterval=0.1, maxinterval=1.0)
        
        for step in range(self.steps):
            # 在开始处理这一步时刷新显示
            pbar.update(0)
            
            random.shuffle(self.base.agents)
            step_llm_calls = {}
            step_llm_prompts = {}
            
            if self.use_occ:
                # ========== OCC 优化模式：提议-验证-提交 ==========
                
                # 阶段1：创建网格快照（Read Phase）
                grid_snapshot = {agent['pos']: agent['id'] for agent in self.base.agents}
                # 当前被占用的位置集合
                occupied_positions = {agent['pos'] for agent in self.base.agents}
                
                # 阶段2：并行生成提议（Propose Phase）
                proposals = []
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # 为每个代理计算其专属的空位集合（基于代理位置和视野范围）
                    # 每个代理的空位 = 视野范围内的位置 - 其他代理占用的位置 + 代理自己的位置
                    future_to_agent = {}
                    for agent in self.base.agents:
                        # 使用基础类的辅助函数计算代理视野范围内的空位
                        agent_empty_cells = self.base._get_empty_cells_in_vision(
                            agent['pos'],
                            occupied_positions,
                            self.agent_vision_range
                        )
                        
                        future = executor.submit(
                            self._generate_proposal_for_agent, 
                            agent, 
                            grid_snapshot, 
                            agent_empty_cells
                        )
                        future_to_agent[future] = agent
                    
                    # 收集所有提议（错误直接抛出，不捕获）
                    for future in as_completed(future_to_agent):
                        proposal = future.result()  # 如果出错会直接抛出异常
                        proposals.append(proposal)
                        
                        # 保存 LLM 调用信息
                        if proposal['decision']:
                            step_llm_calls[proposal['agent_id']] = proposal['decision']
                        if proposal['prompt']:
                            step_llm_prompts[proposal['agent_id']] = proposal['prompt']
                
                # 阶段3：验证和解决冲突（Validate Phase）
                # 使用并行+winner方式：收集所有提议，对于冲突的目标位置随机选择winner
                
                # 计算所有可能的空位：
                # 1. 当前未被占用的位置
                # 2. 所有决定移动的代理的当前位置（如果代理移动，其位置会变成空位）
                potential_empty_cells = set()
                # 首先添加所有未被占用的位置
                for x in range(self.width):
                    for y in range(self.height):
                        pos = (x, y)
                        if pos not in occupied_positions:
                            potential_empty_cells.add(pos)
                # 然后添加所有决定移动的代理的当前位置
                for proposal in proposals:
                    if proposal['action'] == 'move':
                        agent_id = proposal['agent_id']
                        agent = next(a for a in self.base.agents if a['id'] == agent_id)
                        potential_empty_cells.add(agent['pos'])
                
                # 收集所有移动提议的目标位置
                target_positions = {}  # {target_pos: [agent_ids]}
                
                for proposal in proposals:
                    agent_id = proposal['agent_id']
                    if proposal['action'] == 'move' and proposal['target_pos']:
                        target_pos = proposal['target_pos']
                        # 验证目标位置是否在可能的空位集合中
                        if target_pos in potential_empty_cells:
                            if target_pos not in target_positions:
                                target_positions[target_pos] = []
                            target_positions[target_pos].append(agent_id)
                
                # 解决冲突：如果多个代理选择同一位置，随机选择一个winner
                resolved_moves = {}  # {agent_id: target_pos}
                used_positions = set()  # 追踪已分配的位置
                
                for target_pos, agent_ids in target_positions.items():
                    if len(agent_ids) == 1:
                        # 无冲突
                        resolved_moves[agent_ids[0]] = target_pos
                        used_positions.add(target_pos)
                    else:
                        # 有冲突，随机选择一个winner
                        winner = random.choice(agent_ids)
                        resolved_moves[winner] = target_pos
                        used_positions.add(target_pos)
                        # 其他代理回滚（不移动），在下一轮重试
                
                # 阶段4：批量提交更新（Commit Phase）
                moves_count = 0
                
                for agent in self.base.agents:
                    agent_id = agent['id']
                    if agent_id in resolved_moves:
                        # 执行移动
                        agent['pos'] = resolved_moves[agent_id]
                        moves_count += 1
                    # 如果不在 resolved_moves 中，说明是 stay 或冲突输家，保持原位置
            
            self.llm_step_log[step + 1] = step_llm_calls
            
            # 重新计算幸福度（使用合并的满意度计算）
            happy_count = self._update_happiness_with_venues()
            
            # 更新进度条（完成一步）
            pbar.update(1)
            pbar.set_postfix({'happy': f'{happy_count}/{len(self.base.agents)}'})
            
            self.history.append({
                'iteration': step + 1,
                'happy_count': happy_count,
                'total_agents': len(self.base.agents),
                'happiness_ratio': happy_count / len(self.base.agents)
            })
            
            # 保存每一步的快照（用于后续四分位点选择）
            self.snapshots.append({
                'step': step + 1,
                'happy_count': happy_count,
                'agents': [{'id': a['id'], 'type': a['type'], 'pos': a['pos'], 'happy': a['happy']}
                          for a in self.base.agents]
            })
            
            if self.enable_step_log:
                # 将step_llm_calls转换为agent_llm_responses格式
                agent_llm_responses = step_llm_calls if step_llm_calls else None
                save_step_log(step + 1, happy_count, self.base.agents, self.base, self.mode_name, self.model,
                             self.step_log_dir, self.logger, agent_llm_responses=agent_llm_responses,
                             agent_llm_prompts=step_llm_prompts)
            
            if happy_count == len(self.base.agents):
                pbar.set_postfix({'status': 'Converged!'})
                print(f"\n[收敛] 所有智能体在第 {step+1} 步达到满意状态")
                break
        
        pbar.close()
        print(f"\n仿真完成！最终幸福度: {happy_count}/{len(self.base.agents)}")
        return self.history
    
    def visualization(self):
        """可视化结果"""
        print(f"\n{'='*60}")
        print(f"开始运行 {self.mode_name} 模式仿真...")
        print(f"{'='*60}\n")
        
        history = self.run_simulation()
        
        print(f"\n{'='*60}")
        print(f"开始生成可视化...")
        print(f"{'='*60}\n")
        
        # 创建可视化对象
        viz = SchellingVisualization(
            self.width, self.height, self.density, self.minority_pc,
            self.homophily, self.mode_name, self.model
        )
        
        # 执行可视化（传递场所信息以在网格上标记）
        viz.visualize(
            history, self.snapshots, self.output_pdf, self.seed,
            len(self.base.agents), venues=self.venues
        )
        
        print(f"\n✅ {self.mode_name} 模式完成！")
    
    def _update_happiness_with_venues(self):
        """
        更新所有智能体的幸福度（基于Silver论文的合并满意度计算）
        
        满意度 = (居住地同族邻居 + 场所同族人数) / (居住地总邻居 + 场所总人数)
        
        Returns:
            happy_count: 幸福智能体数量
        """
        happy_count = 0
        for agent in self.base.agents:
            # 计算居住地邻居
            similar, total = self.base.count_similar_neighbors(agent['pos'], agent['type'])
            
            # 获取代理的冒险精神
            agent_adventurousness = self.agent_adventurousness.get(agent['id'], 0.5)
            
            # 获取可访问的场所
            accessible_venues = get_accessible_venues(
                agent['pos'], agent['type'], agent_adventurousness,
                self.venues, self.width, self.height
            )
            
            # 计算在场所中遇到的其他代理人
            venue_encounters = get_venue_encounters(
                agent['pos'], agent['type'], agent_adventurousness,
                accessible_venues, self.base.agents,
                self.width, self.height
            )
            
            # 合并计算满意度
            combined_same = similar + venue_encounters["venue_same_group_count"]
            combined_total = total + venue_encounters["venue_total_count"]
            
            # 判断是否满意（基于阈值）
            agent['happy'] = combined_same >= self.homophily if combined_total > 0 else False
            if agent['happy']:
                happy_count += 1
        
        return happy_count

