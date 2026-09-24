# -*- coding: utf-8 -*-
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
Schelling Model - ABM-VE变体
基于规则的ABM实现 + Venue Enhancement（场所增强）
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

from simulations.Schelling_model import SchellingBase, SchellingVisualization, setup_logger, save_step_log
from simulations.Schelling_model.venue import (
    Venue, get_accessible_venues, get_venue_encounters, create_venues, 
    create_venues_from_positions, manhattan_distance
)


class SchellingABM_VE:
    """Schelling ABM-VE变体 - 基于规则的方法 + 场所信息"""
    
    def __init__(self, grid_size, steps, density, minority_pc, homophily, seed, output_pdf,
                 log_file=None, enable_step_log=False, venue_config=None):
        """
        初始化 Schelling ABM-VE 模型
        
        Args:
            grid_size: 网格大小 (width, height)
            steps: 仿真步数
            density: 网格占用率 (0-1)
            minority_pc: 少数群体比例 (0-1)
            homophily: 同质性阈值 (期望的相似邻居数量)
            seed: 随机种子
            output_pdf: 输出路径
            log_file: 日志文件路径
            enable_step_log: 是否启用每步的JSON日志记录
            venue_config: 场所配置字典，支持两种模式：
                模式1（自动生成）：
                    - "mode": "single_center", "multi_center", "mixed"
                    - "num_exclusive_per_group": 每个群体的专属场所数量
                    - "num_public": 公共场所数量
                    - "catchment_radius": 影响范围
                    - "exclusivity": 专属场所的排他性（默认1.0）
                模式2（手动指定位置）：
                    - "venue_positions": 字典，格式为 {
                        None: [(x1, y1), (x2, y2), ...],  # 公共场所
                        0: [(x1, y1), ...],                # 群体0的专属场所
                        1: [(x1, y1), ...],                # 群体1的专属场所
                      }
                    - "catchment_radius": 所有场所的统一辐射距离
                    - "exclusivity": 专属场所的排他性（默认1.0）
        """
        self.width, self.height = grid_size
        self.steps = steps
        self.density = density
        self.minority_pc = minority_pc
        self.homophily = homophily
        self.seed = seed
        self.output_pdf = output_pdf
        self.enable_step_log = enable_step_log
        
        # 初始化基础模型（注意：SchellingBase.__init__ 会设置 random.seed）
        self.base = SchellingBase(self.width, self.height, self.density, self.minority_pc, self.seed)
        self.base.homophily = self.homophily
        
        # 创建场所（注意：create_venues/create_venues_from_positions 内部会设置 random.seed）
        # 这会在创建场所时消耗一些随机数
        if venue_config is None:
            venue_config = {
                "mode": "mixed",
                "num_exclusive_per_group": 1,
                "num_public": 2,
                "catchment_radius": max(self.width, self.height) // 8,
                "exclusivity": 1.0  # 默认完全排他
            }
        
        # 检查是否使用手动指定位置的模式
        if "venue_positions" in venue_config:
            # 模式2：手动指定位置
            venue_positions = venue_config.get("venue_positions", {})
            catchment_radius = venue_config.get("catchment_radius", min(self.width, self.height) // 4)
            exclusivity = venue_config.get("exclusivity", 1.0)
            self.venues = create_venues_from_positions(
                venue_positions, 
                catchment_radius=catchment_radius,
                exclusivity=exclusivity,
                seed=self.seed
            )
        else:
            # 模式1：自动生成
            self.venues = create_venues(venue_config, self.width, self.height, self.seed)
        
        # 为每个代理分配冒险精神（Adventurousness）参数（0-1随机分布）
        # 注意：在创建venues之后分配，确保与abm_variant的初始化顺序一致
        self.agent_adventurousness = {}  # {agent_id: adventurousness}
        
        # 配置日志
        mode_name = "ABM-VE"
        if log_file is None:
            log_file = f"./logs/schelling_{mode_name.lower()}_{self.width}x{self.height}_{self.density}_seed{seed}.log"
        self.logger = setup_logger(log_file, f"Schelling_{mode_name}", seed)
        
        # 配置 JSON step log 目录
        if self.enable_step_log:
            self.step_log_dir = os.path.join(self.output_pdf, "logs")
            os.makedirs(self.step_log_dir, exist_ok=True)
            self.logger.info(f"Step logs will be saved to: {self.step_log_dir}")
            self.logger.info(f"Created {len(self.venues)} venues: {[v.venue_type for v in self.venues]}")
        else:
            self.step_log_dir = None
        
        self.history = []
        self.snapshots = []
        self.mode_name = mode_name
        
        # OCC 优化参数
        self.max_workers = None  # None 表示使用默认值（CPU核心数）
        self.use_occ = True  # 是否启用 OCC 优化
        
        # 代理视野参数（用于计算可移动的空位范围）
        self.agent_vision_range = None  # None 表示无限制，或设置为整数（如 3, 5 等）
    
    def _generate_proposal_for_agent(self, agent, grid_snapshot, empty_cells_snapshot):
        """
        为单个代理生成移动提议（OCC 提议阶段）- ABM-VE版本
        考虑场所信息的满意度计算
        
        Args:
            agent: 代理字典
            grid_snapshot: 网格快照（代理位置字典 {pos: agent_id}）
            empty_cells_snapshot: 空位快照（列表）
        
        Returns:
            proposal: 提议字典，包含 agent_id, action, target_pos
        """
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
        
        # 合并计算满意度（居住地 + 场所）
        combined_same = similar + venue_encounters["venue_same_group_count"]
        combined_total = total + venue_encounters["venue_total_count"]
        
        # 判断是否满足阈值
        if combined_total > 0:
            is_satisfied = combined_same / combined_total >= self.homophily / 8
        else:
            is_satisfied = False  # 没有邻居或场所接触，无法满足阈值
        
        return combined_same, combined_total, is_satisfied
    
    def _generate_proposal_for_agent(self, agent, grid_snapshot, empty_cells_snapshot):
        """
        为单个代理生成移动提议（OCC 提议阶段）- ABM-VE版本
        考虑场所信息的满意度计算
        
        移动规则（基于满意度判断）：
        1. 如果 Agent 当前已经满意（考虑居住地邻居 + 场所接触），则 stay 在原地
        2. 如果 Agent 当前不满意，则扫描所有可用的空位
        3. 对每个空位，计算移动到该位置后的满意度（考虑居住地邻居 + 场所接触）
        4. 只有当某个空位能让 Agent 变得"满意"（满足阈值）时，才会选择移动到该位置
        5. 如果找到多个满意的位置，随机选择一个
        6. 如果所有空位都不能让 Agent 满意，则 stay 在原地
        
        注意：不再使用简单的随机选择，而是基于满意度判断来决定是否移动
        
        Args:
            agent: 代理字典
            grid_snapshot: 网格快照（代理位置字典 {pos: agent_id}）
            empty_cells_snapshot: 空位快照（列表）
        
        Returns:
            proposal: 提议字典，包含 agent_id, action, target_pos
        """
        # 步骤1：计算当前位置的满意度（考虑居住地邻居 + 场所接触）
        current_similar, current_total = self.base.count_similar_neighbors(agent['pos'], agent['type'])
        agent_adventurousness = self.agent_adventurousness.get(agent['id'], 0.5)
        current_accessible_venues = get_accessible_venues(
            agent['pos'], agent['type'], agent_adventurousness,
            self.venues, self.width, self.height
        )
        current_venue_encounters = get_venue_encounters(
            agent['pos'], agent['type'], agent_adventurousness,
            current_accessible_venues, self.base.agents,
            self.width, self.height
        )
        current_combined_same = current_similar + current_venue_encounters["venue_same_group_count"]
        current_combined_total = current_total + current_venue_encounters["venue_total_count"]
        
        # 判断当前是否满意（阈值：combined_same/combined_total >= homophily/8）
        current_is_satisfied = False
        if current_combined_total > 0:
            current_is_satisfied = current_combined_same / current_combined_total >= self.homophily / 8
        
        # 步骤2：如果当前已经满意，保持原位
        if current_is_satisfied:
            return {
                'agent_id': agent['id'],
                'action': 'stay',
                'target_pos': None
            }
        
        # 步骤3：如果当前不满意，扫描所有空位，寻找能让它变得"满意"的位置
        if empty_cells_snapshot:
            satisfying_positions = []
            for target_pos in empty_cells_snapshot:
                # 计算移动到该位置后的满意度（考虑居住地邻居 + 场所接触）
                _, _, is_satisfied = self._calculate_satisfaction_at_position(
                    agent, target_pos, grid_snapshot
                )
                # 只有当该位置能让 Agent 满意时，才加入候选列表
                if is_satisfied:
                    satisfying_positions.append(target_pos)
            
            # 步骤4：如果找到满足阈值的位置，随机选择一个
            if satisfying_positions:
                target_pos = random.choice(satisfying_positions)
                return {
                    'agent_id': agent['id'],
                    'action': 'move',
                    'target_pos': target_pos
                }
        
        # 步骤5：如果所有空位都不能满足要求，被迫留在原地
        return {
            'agent_id': agent['id'],
            'action': 'stay',
            'target_pos': None
        }
    
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
            # 如果 combined_total 为 0，说明 agent 既没有居住地邻居，也没有在场所中遇到任何人
            # 这种情况下无法满足同质性要求，应该是不满意的
            if combined_total > 0:
                agent['happy'] = combined_same/combined_total >= self.homophily/8
            else:
                agent['happy'] = False
            
            if agent['happy']:
                happy_count += 1
        
        return happy_count
    
    def _calculate_satisfaction_stats(self):
        """
        计算满意度相关指标（用于step_log记录）
        
        Returns:
            stats_dict: 包含满意度相关指标的字典
        """
        residential_similarities = []
        venue_similarities = []
        combined_similarities = []
        accessible_venue_counts = []
        venue_same_group_counts = []
        venue_total_counts = []
        
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
            
            # 居住地相似度
            if total > 0:
                residential_similarities.append(similar / total)
            
            # 场所相似度
            venue_same = venue_encounters["venue_same_group_count"]
            venue_total = venue_encounters["venue_total_count"]
            if venue_total > 0:
                venue_similarities.append(venue_same / venue_total)
            
            # 合并相似度
            combined_same = similar + venue_same
            combined_total = total + venue_total
            if combined_total > 0:
                combined_similarities.append(combined_same / combined_total)
            
            # 可访问场所数量
            accessible_venue_counts.append(len(accessible_venues))
            
            # 场所同族人数和总人数
            venue_same_group_counts.append(venue_same)
            venue_total_counts.append(venue_total)
        
        # 计算平均值
        import numpy as np
        stats = {
            "avg_residential_similarity": round(np.mean(residential_similarities) if residential_similarities else 0, 3),
            "avg_venue_similarity": round(np.mean(venue_similarities) if venue_similarities else 0, 3),
            "avg_combined_similarity": round(np.mean(combined_similarities) if combined_similarities else 0, 3),
            "avg_accessible_venue_count": round(np.mean(accessible_venue_counts) if accessible_venue_counts else 0, 2),
            "avg_venue_same_group_count": round(np.mean(venue_same_group_counts) if venue_same_group_counts else 0, 2),
            "avg_venue_total_count": round(np.mean(venue_total_counts) if venue_total_counts else 0, 2),
        }
        
        return stats
    
    def _calculate_agent_satisfaction_fields(self):
        """
        计算每个agent的满意度相关指标（用于step_log记录）
        
        Returns:
            agent_fields_dict: 字典 {agent_id: {field: value}}，包含每个agent的满意度指标
        """
        agent_fields = {}
        
        for agent in self.base.agents:
            agent_id = agent['id']
            
            # 计算居住地邻居
            similar, total = self.base.count_similar_neighbors(agent['pos'], agent['type'])
            
            # 获取代理的冒险精神
            agent_adventurousness = self.agent_adventurousness.get(agent_id, 0.5)
            
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
            
            # 计算各种相似度
            residential_similarity = (similar / total) if total > 0 else 0.0
            venue_same = venue_encounters["venue_same_group_count"]
            venue_total = venue_encounters["venue_total_count"]
            venue_similarity = (venue_same / venue_total) if venue_total > 0 else 0.0
            combined_same = similar + venue_same
            combined_total = total + venue_total
            combined_similarity = (combined_same / combined_total) if combined_total > 0 else 0.0
            
            # 构建agent的满意度指标字典
            agent_fields[agent_id] = {
                "residential_similarity": round(residential_similarity, 3),
                "venue_similarity": round(venue_similarity, 3),
                "combined_similarity": round(combined_similarity, 3),
                "accessible_venue_count": len(accessible_venues),
                "venue_same_group_count": venue_same,
                "venue_total_count": venue_total,
                "residential_same_count": similar,
                "residential_total_count": total
            }
        
        return agent_fields
    
    def run_simulation(self):
        """运行ABM-VE仿真（使用OCC优化）"""
        print(f"[ABM-VE模式] 开始运行 {self.steps} 步仿真...")
        print(f"[说明] 基于规则的ABM + 场所增强（Venue Enhancement）")
        if self.use_occ:
            print(f"[优化] 启用 OCC (乐观并发控制) 并行优化")
        
        self.base.init_grid_model()
        
        # 为每个代理分配冒险精神（在网格初始化之后）
        for agent in self.base.agents:
            self.agent_adventurousness[agent['id']] = random.random()
        
        # 初始状态统计（使用合并满意度计算）
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
            # 计算每个agent的满意度指标
            agent_satisfaction_fields = self._calculate_agent_satisfaction_fields()
            save_step_log(0, happy_count, self.base.agents, self.base, self.mode_name, "abm-ve",
                         self.step_log_dir, self.logger, 
                         agent_additional_fields=agent_satisfaction_fields)
        
        # 迭代
        print(f"\n开始仿真迭代...")
        pbar = tqdm(range(self.steps), desc="ABM-VE仿真进度",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for step in pbar:
            # 随机打乱智能体顺序
            random.shuffle(self.base.agents)
            
            if self.use_occ:
                # ========== OCC 优化模式：提议-验证-提交 ==========
                
                # 阶段1：创建网格快照（Read Phase）
                grid_snapshot = {agent['pos']: agent['id'] for agent in self.base.agents}
                occupied_positions = {agent['pos'] for agent in self.base.agents}
                
                # 阶段2：并行生成提议（Propose Phase）
                proposals = []
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # 为每个代理计算其专属的空位集合（基于代理位置和视野范围）
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
                    
                    # 收集所有提议
                    for future in as_completed(future_to_agent):
                        try:
                            proposal = future.result()
                            proposals.append(proposal)
                        except Exception as e:
                            agent = future_to_agent[future]
                            self.logger.warning(f"Agent {agent['id']} 提议生成失败: {e}")
                
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
                for agent in self.base.agents:
                    agent_id = agent['id']
                    if agent_id in resolved_moves:
                        agent['pos'] = resolved_moves[agent_id]
            
            # 重新计算幸福度（使用合并的满意度计算）
            happy_count = self._update_happiness_with_venues()
            
            # 更新进度条信息
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
            
            # 保存 step log (ABM-VE)
            if self.enable_step_log:
                # 计算每个agent的满意度指标
                agent_satisfaction_fields = self._calculate_agent_satisfaction_fields()
                save_step_log(step + 1, happy_count, self.base.agents, self.base, self.mode_name, "abm-ve",
                             self.step_log_dir, self.logger,
                             agent_additional_fields=agent_satisfaction_fields)
            
            # 如果所有智能体都满意，提前结束
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
            self.homophily, self.mode_name, "abm-ve"
        )
        
        # 执行可视化（传递场所信息以在网格上标记）
        viz.visualize(
            history, self.snapshots, self.output_pdf, self.seed,
            len(self.base.agents), venues=self.venues
        )
        
        print(f"\n✅ {self.mode_name} 模式完成！")

