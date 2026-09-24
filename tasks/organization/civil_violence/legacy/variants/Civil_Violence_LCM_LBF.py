# -*- coding: utf-8 -*-
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
Civil Violence 模型 - LCM-LBF (LLM Context Modeling + LLM Behavior Function)
- 完整的上下文信息（位置、状态、邻居详情、满意度指标等）
- LLM充分发挥推理能力（考虑多维度因素）
- 展示LLM在丰富上下文下的综合决策能力
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
from tqdm import tqdm
from enum import Enum
import json
import logging
import os

from socioverse.behavior_engine.LLM_agent import LLM_Agent


class CitizenState(Enum):
    """市民状态枚举"""
    QUIET = 0      # 安静状态
    ACTIVE = 1     # 激进状态
    ARRESTED = 2   # 被捕状态


class CivilViolence_LCM_LBF:
    def __init__(self, grid_size, steps, citizen_count, cop_count, legitimacy, 
                 citizen_vision, cop_vision, max_jail_time, seed, output_pdf,
                 model="gpt-4o-2024-08-06", log_file=None, enable_step_log=False):
        """
        初始化 Civil Violence LCM-LBF 模型
        
        Args:
            grid_size: 网格大小 (width, height)
            steps: 仿真步数
            citizen_count: 市民数量
            cop_count: 警察数量
            legitimacy: 政府合法性 (0-1)
            citizen_vision: 市民视野范围
            cop_vision: 警察视野范围
            max_jail_time: 最大监禁时间
            seed: 随机种子
            output_pdf: 输出路径
            model: LLM模型名称
            log_file: 日志文件路径
            enable_step_log: 是否启用每步的JSON日志记录
        """
        self.width, self.height = grid_size
        self.steps = steps
        self.citizen_count = citizen_count
        self.cop_count = cop_count
        self.legitimacy = legitimacy
        self.citizen_vision = citizen_vision
        self.cop_vision = cop_vision
        self.max_jail_time = max_jail_time
        self.seed = seed
        self.output_pdf = output_pdf
        self.model = model
        self.enable_step_log = enable_step_log
        
        # 配置日志
        if log_file is None:
            log_file = f"./logs/civil_lcm_lbf_seed{seed}.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        self.logger = logging.getLogger(f"CivilViolence_LCM_LBF_{seed}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(name)s:%(levelname)s:%(message)s', 
                                     datefmt='%d-%m-%Y %H:%M:%S')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # 配置 JSON step log 目录
        if self.enable_step_log:
            self.step_log_dir = os.path.join(
                self.output_pdf, 
                f"step_logs_LCM_LBF_{self.width}x{self.height}_seed{seed}"
            )
            os.makedirs(self.step_log_dir, exist_ok=True)
            self.logger.info(f"Step logs will be saved to: {self.step_log_dir}")
        else:
            self.step_log_dir = None
        
        # 模型常数
        self.grid_capacity = self.width * self.height
        self.arrest_prob_constant = 2.3
        self.threshold = 0.1  # 激进阈值
        
        self.llm_agent = None
        self.citizens = []
        self.cops = []
        self.history = []
        self.snapshots = []  # 保存关键步骤的智能体状态快照
    
    def init_grid_model(self):
        """初始化网格和智能体"""
        self.citizens = []
        self.cops = []
        
        all_positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        random.shuffle(all_positions)
        
        agent_id = 0
        
        # 放置市民
        for i in range(min(self.citizen_count, len(all_positions))):
            x, y = all_positions[i]
            citizen = {
                'id': agent_id,
                'type': 'citizen',
                'pos': (x, y),
                'position_history': [(x, y)],  # 记录位置历史用于绘制轨迹
                'state': CitizenState.QUIET,
                'hardship': random.random(),
                'risk_aversion': random.random(),
                'grievance': 0.0,
                'arrest_probability': 0.0,
                'jail_sentence': 0,
                'vision': self.citizen_vision
            }
            citizen['grievance'] = citizen['hardship'] * (1 - self.legitimacy)
            self.citizens.append(citizen)
            agent_id += 1
        
        # 放置警察
        for i in range(min(self.cop_count, len(all_positions) - len(self.citizens))):
            x, y = all_positions[len(self.citizens) + i]
            cop = {
                'id': agent_id,
                'type': 'cop',
                'pos': (x, y),
                'position_history': [(x, y)],  # 记录位置历史用于绘制轨迹
                'vision': self.cop_vision
            }
            self.cops.append(cop)
            agent_id += 1
        
        return self.citizens, self.cops
    
    def get_neighbors_in_vision(self, pos, vision):
        """获取视野范围内的邻居"""
        x, y = pos
        neighbors = []
        
        for dx in range(-vision, vision + 1):
            for dy in range(-vision, vision + 1):
                if dx == 0 and dy == 0:
                    continue
                
                nx = (x + dx) % self.width
                ny = (y + dy) % self.height
                
                for citizen in self.citizens:
                    if citizen['pos'] == (nx, ny):
                        neighbors.append(citizen)
                
                for cop in self.cops:
                    if cop['pos'] == (nx, ny):
                        neighbors.append(cop)
        
        return neighbors
    
    def update_citizen_arrest_probability(self, citizen):
        """更新市民的被捕概率估计"""
        neighbors = self.get_neighbors_in_vision(citizen['pos'], citizen['vision'])
        
        cops_in_vision = 0
        actives_in_vision = 1  # 包括自己
        
        for neighbor in neighbors:
            if neighbor['type'] == 'cop':
                cops_in_vision += 1
            elif neighbor['type'] == 'citizen' and neighbor['state'] == CitizenState.ACTIVE:
                actives_in_vision += 1
        
        import math
        citizen['arrest_probability'] = 1 - math.exp(
            -1 * self.arrest_prob_constant * (cops_in_vision / actives_in_vision)
        )
        
        return cops_in_vision, actives_in_vision - 1
    
    def run_simulation(self):
        """运行LCM-LBF仿真"""
        print(f"[LCM-LBF模式] 开始运行 {self.steps} 步仿真...")
        print(f"[说明] LLM基于完整上下文进行综合推理决策")
        
        self.init_grid_model()
        self.llm_agent = LLM_Agent(model=self.model, scenario="Civil_Violence_LCM_LBF")
        
        # 初始状态统计
        self._record_history(0, [])
        
        # 迭代
        pbar = tqdm(range(self.steps), desc="LCM-LBF仿真进度", position=0, leave=True)
        for step in pbar:
            # 1. 市民决策 (LLM完整上下文决策)
            random.shuffle(self.citizens)
            
            step_records = {
                c['id']: {
                    'citizen_id': c['id'],
                    'step': step,
                    'initial_state': c['state'].name,
                    'initial_pos': c['pos'],
                    'initial_jail_sentence': c['jail_sentence'],
                    'decision_source': 'jailed' if c['jail_sentence'] > 0 else 'LLM',
                    'llm_attributes': None,
                    'llm_response': None,
                    'decision': None,
                    'reasoning': None,
                    'movement': {'from': c['pos'], 'to': c['pos'], 'moved': False},
                    'notes': ''
                } for c in self.citizens
            }
            
            # 获取需要处理的活跃市民
            active_citizens = [c for c in self.citizens if c['jail_sentence'] == 0]
            pbar.set_description(f"LCM-LBF Step {step+1}/{self.steps} (处理 {len(active_citizens)} 个市民)")
            
            # 使用嵌套进度条显示LLM调用进度（仅在市民数量较多时显示）
            use_nested = len(active_citizens) >= 5
            if use_nested:
                citizen_pbar = tqdm(enumerate(active_citizens), total=len(active_citizens), 
                                   desc="  调用LLM", position=1, leave=False, ncols=80)
            else:
                citizen_pbar = enumerate(active_citizens)
            
            for idx, citizen in citizen_pbar:
                if use_nested:
                    citizen_pbar.set_description(f"  市民 {citizen['id']+1}/{len(active_citizens)}")
                
                cops_nearby, actives_nearby = self.update_citizen_arrest_probability(citizen)
                
                # LCM-LBF: 提供完整的上下文信息
                attributes = {
                    "citizen_pos": citizen['pos'],
                    "hardship": round(citizen['hardship'], 2),
                    "risk_aversion": round(citizen['risk_aversion'], 2),
                    "grievance": round(citizen['grievance'], 2),
                    "arrest_probability": round(citizen['arrest_probability'], 2),
                    "government_legitimacy": self.legitimacy,
                    "threshold": self.threshold,
                    "cops_nearby": cops_nearby,
                    "actives_nearby": actives_nearby,
                    "current_state": citizen['state'].name,
                    "max_jail_time": self.max_jail_time
                }
                
                record = step_records.get(citizen['id'])
                if record:
                    record['llm_attributes'] = attributes
                
                self.llm_agent.update_attributes(attributes)
                
                try:
                    decision = self.llm_agent.take_actions()
                    
                    if isinstance(decision, dict):
                        action = decision.get('decision', 'quiet').lower()
                        
                        if action == 'protest' or action == 'active':
                            citizen['state'] = CitizenState.ACTIVE
                        else:
                            citizen['state'] = CitizenState.QUIET
                        
                        if record:
                            record['llm_response'] = decision
                            record['decision'] = action
                            record['reasoning'] = decision.get('reasoning')
                    else:
                        if record:
                            record['llm_response'] = decision
                            record['notes'] = (record['notes'] + f' unexpected_llm_output:{type(decision).__name__}').strip()
                
                except Exception as e:
                    self.logger.warning(f"Citizen {citizen['id']} LLM决策失败: {e}")
                    if record:
                        record['decision_source'] = 'fallback_rule'
                        record['error'] = str(e)
                    # 回退到规则决策
                    net_risk = citizen['grievance'] - (citizen['risk_aversion'] * citizen['arrest_probability'])
                    if net_risk > self.threshold:
                        citizen['state'] = CitizenState.ACTIVE
                    else:
                        citizen['state'] = CitizenState.QUIET
                    if record:
                        record['decision'] = citizen['state'].name.lower()
            
            # 处理监禁中的市民
            for citizen in self.citizens:
                if citizen['jail_sentence'] > 0:
                    citizen['jail_sentence'] -= 1
                    if citizen['jail_sentence'] == 0:
                        citizen['state'] = CitizenState.QUIET
                    record = step_records.get(citizen['id'])
                    if record:
                        if citizen['jail_sentence'] == 0:
                            record['notes'] = (record['notes'] + ' released_this_step').strip()
                        else:
                            record['notes'] = (record['notes'] + ' serving_sentence').strip()
            
            # 2. 警察执法 (保持ABM规则)
            random.shuffle(self.cops)
            
            for cop in self.cops:
                neighbors = self.get_neighbors_in_vision(cop['pos'], cop['vision'])
                active_neighbors = [n for n in neighbors 
                                  if n['type'] == 'citizen' and n['state'] == CitizenState.ACTIVE]
                
                if active_neighbors:
                    arrestee = random.choice(active_neighbors)
                    arrestee['state'] = CitizenState.ARRESTED
                    arrestee['jail_sentence'] = random.randint(1, self.max_jail_time)
            
            # 3. 移动
            self._move_agents()
            
            # 更新动作日志中的最终状态/位置
            for citizen in self.citizens:
                record = step_records.get(citizen['id'])
                if not record:
                    continue
                record['final_state'] = citizen['state'].name
                record['final_pos'] = citizen['pos']
                record['final_jail_sentence'] = citizen['jail_sentence']
                record['movement']['to'] = citizen['pos']
                record['movement']['moved'] = record['movement']['from'] != citizen['pos']
            
            # 4. 记录历史
            self._record_history(step + 1, list(step_records.values()))
        
        return self.history
    
    def _update_agent_position(self, agent, new_pos):
        """更新智能体位置并记录历史"""
        agent['pos'] = new_pos
        if 'position_history' in agent:
            agent['position_history'].append(new_pos)
        else:
            agent['position_history'] = [new_pos]
    
    def _move_agents(self):
        """智能体随机移动到相邻空位"""
        occupied_positions = {agent['pos'] for agent in self.citizens + self.cops}
        empty_cells = [(x, y) for x in range(self.width) for y in range(self.height) 
                       if (x, y) not in occupied_positions]
        
        if not empty_cells:
            return
        
        # 市民移动
        for citizen in self.citizens:
            if citizen['jail_sentence'] > 0:
                continue
            
            x, y = citizen['pos']
            adjacent_empty = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx) % self.width
                    ny = (y + dy) % self.height
                    if (nx, ny) in empty_cells:
                        adjacent_empty.append((nx, ny))
            
            if adjacent_empty and random.random() < 0.1:
                new_pos = random.choice(adjacent_empty)
                empty_cells.remove(new_pos)
                empty_cells.append(citizen['pos'])
                self._update_agent_position(citizen, new_pos)
        
        # 警察移动
        for cop in self.cops:
            x, y = cop['pos']
            adjacent_empty = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx) % self.width
                    ny = (y + dy) % self.height
                    if (nx, ny) in empty_cells:
                        adjacent_empty.append((nx, ny))
            
            if adjacent_empty and random.random() < 0.1:
                new_pos = random.choice(adjacent_empty)
                empty_cells.remove(new_pos)
                empty_cells.append(cop['pos'])
                self._update_agent_position(cop, new_pos)
    
    def _record_history(self, step, step_actions=None):
        """记录历史数据"""
        quiet_count = sum(1 for c in self.citizens if c['state'] == CitizenState.QUIET and c['jail_sentence'] == 0)
        active_count = sum(1 for c in self.citizens if c['state'] == CitizenState.ACTIVE)
        jailed_count = sum(1 for c in self.citizens if c['jail_sentence'] > 0)
        
        self.history.append({
            'iteration': step,
            'quiet': quiet_count,
            'active': active_count,
            'jailed': jailed_count,
            'total_citizens': len(self.citizens),
            'active_ratio': active_count / len(self.citizens) if self.citizens else 0
        })
        
        # 保存快照（用于四分位点可视化）
        self.snapshots.append({
            'step': step,
            'active_count': active_count,
            'citizens': [{
                'id': c['id'], 
                'pos': c['pos'], 
                'state': c['state'].name, 
                'jail_sentence': c['jail_sentence'],
                'position_history': c.get('position_history', [c['pos']])  # 保存位置历史
            } for c in self.citizens],
            'cops': [{
                'id': cop['id'], 
                'pos': cop['pos'],
                'position_history': cop.get('position_history', [cop['pos']])  # 保存位置历史
            } for cop in self.cops]
        })
        
        if self.enable_step_log:
            self._save_step_log(step, quiet_count, active_count, jailed_count, step_actions or [])
    
    def _save_step_log(self, step, quiet_count, active_count, jailed_count, step_actions):
        """保存每一步的详细状态到 JSON 文件"""
        step_log = {
            "step": step,
            "mode": "LCM-LBF",
            "statistics": {
                "quiet_count": quiet_count,
                "active_count": active_count,
                "jailed_count": jailed_count,
                "total_citizens": len(self.citizens),
                "active_ratio": active_count / len(self.citizens) if self.citizens else 0
            },
            "citizen_actions": step_actions,
            "model_params": {
                "legitimacy": self.legitimacy,
                "threshold": self.threshold,
                "grid_size": [self.width, self.height],
                "model": self.model
            }
        }
        
        json_path = os.path.join(self.step_log_dir, f"step_{step:03d}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(step_log, f, indent=2, ensure_ascii=False)
    
    def _select_snapshots_by_quartiles(self):
        """基于active citizens数量的四分位点选择快照"""
        if len(self.snapshots) < 2:
            return self.snapshots
        
        active_counts = [s['active_count'] for s in self.snapshots]
        min_active = min(active_counts)
        max_active = max(active_counts)
        
        q1_target = min_active + (max_active - min_active) * 0.25
        q2_target = min_active + (max_active - min_active) * 0.50
        
        selected = []
        selected.append(self.snapshots[0])
        
        q1_idx = min(range(1, len(self.snapshots)-1), 
                    key=lambda i: abs(self.snapshots[i]['active_count'] - q1_target))
        if q1_idx not in [s['step'] for s in selected]:
            selected.append(self.snapshots[q1_idx])
        
        q2_idx = min(range(1, len(self.snapshots)-1), 
                    key=lambda i: abs(self.snapshots[i]['active_count'] - q2_target))
        if q2_idx not in [s['step'] for s in selected]:
            selected.append(self.snapshots[q2_idx])
        
        if self.snapshots[-1] not in selected:
            selected.append(self.snapshots[-1])
        
        while len(selected) < 4 and len(self.snapshots) > len(selected):
            for snap in self.snapshots:
                if snap not in selected:
                    selected.append(snap)
                    break
        
        return sorted(selected, key=lambda x: x['step'])[:4]
    
    def _create_grid_from_snapshot(self, snapshot):
        """从快照创建网格数组
        返回值: (citizen_grid, cop_grid)
        citizen_grid: 0=QUIET, 1=ACTIVE, 2=ARRESTED, -1=empty
        cop_grid: True=有警察, False=无警察
        """
        citizen_grid = np.full((self.width, self.height), -1)
        cop_grid = np.zeros((self.width, self.height), dtype=bool)
        
        for citizen in snapshot['citizens']:
            x, y = citizen['pos']
            if citizen['state'] == 'QUIET':
                citizen_grid[x, y] = 0
            elif citizen['state'] == 'ACTIVE':
                citizen_grid[x, y] = 1
            elif citizen['state'] == 'ARRESTED' or citizen['jail_sentence'] > 0:
                citizen_grid[x, y] = 2
        
        for cop in snapshot['cops']:
            x, y = cop['pos']
            cop_grid[x, y] = True
        
        return citizen_grid, cop_grid
    
    def visualization(self):
        """可视化结果"""
        history = self.run_simulation()
        
        iterations = [h['iteration'] for h in history]
        quiet_counts = [h['quiet'] for h in history]
        active_counts = [h['active'] for h in history]
        jailed_counts = [h['jailed'] for h in history]
        active_ratios = [h['active_ratio'] for h in history]
        total_citizens = len(self.citizens)
        
        # 基于active citizens数量的四分位点选择快照
        selected_snapshots = self._select_snapshots_by_quartiles()
        
        if len(selected_snapshots) < 4:
            print(f"[Warning] Insufficient snapshots: {len(selected_snapshots)}/4")
        
        # 创建图表：2行4列
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
        
        # 准备颜色映射（市民）- 移除白色，使用透明背景
        colors = ['lightblue', 'red', 'orange']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        # 第一行: 绘制4个时间点的网格状态
        stage_names = ['Initial', 'Q1 (25%)', 'Q2 (50%)', 'Final']
        
        for i, (snapshot, stage_name) in enumerate(zip(selected_snapshots[:4], stage_names[:4])):
            ax = fig.add_subplot(gs[0, i])
            citizen_grid, cop_grid = self._create_grid_from_snapshot(snapshot)
            
            # 创建masked array，隐藏空位置（-1值）
            masked_grid = np.ma.masked_where(citizen_grid == -1, citizen_grid)
            
            # 绘制背景（浅灰色表示空位置）
            ax.imshow(np.ones((self.width, self.height)), cmap='gray', vmin=0, vmax=1, 
                     alpha=0.1, origin='lower')
            
            # 绘制市民（只显示非空位置）
            im = ax.imshow(masked_grid, cmap=cmap, norm=norm, origin='lower', alpha=0.8)
            
            # 绘制移动轨迹（从初始位置到当前切片位置）
            initial_snapshot = self.snapshots[0]
            for citizen in snapshot['citizens']:
                citizen_id = citizen['id']
                # 找到初始位置
                initial_citizen = next((c for c in initial_snapshot['citizens'] if c['id'] == citizen_id), None)
                if initial_citizen:
                    initial_pos = initial_citizen['pos']
                    current_pos = citizen['pos']
                    # 获取从初始到当前的位置历史
                    position_history = citizen.get('position_history', [initial_pos, current_pos])
                    if len(position_history) > 1:
                        # 绘制轨迹线
                        xs = [p[1] for p in position_history]  # 注意：imshow使用(y,x)，但scatter使用(x,y)
                        ys = [p[0] for p in position_history]
                        # 根据状态选择轨迹颜色
                        if citizen['state'] == 'ACTIVE':
                            ax.plot(xs, ys, color='darkred', linewidth=0.5, alpha=0.3, zorder=1)
                        elif citizen['state'] == 'ARRESTED' or citizen['jail_sentence'] > 0:
                            ax.plot(xs, ys, color='darkorange', linewidth=0.5, alpha=0.3, zorder=1)
                        else:
                            ax.plot(xs, ys, color='darkblue', linewidth=0.5, alpha=0.3, zorder=1)
            
            # 绘制警察轨迹
            for cop in snapshot['cops']:
                cop_id = cop['id']
                initial_cop = next((c for c in initial_snapshot['cops'] if c['id'] == cop_id), None)
                if initial_cop:
                    position_history = cop.get('position_history', [initial_cop['pos'], cop['pos']])
                    if len(position_history) > 1:
                        xs = [p[1] for p in position_history]
                        ys = [p[0] for p in position_history]
                        ax.plot(xs, ys, color='gray', linewidth=0.5, alpha=0.2, linestyle='--', zorder=1)
            
            # 绘制警察（黑色方块）
            cop_positions = np.where(cop_grid)
            if len(cop_positions[0]) > 0:
                ax.scatter(cop_positions[1], cop_positions[0], c='black', s=100, 
                          marker='s', edgecolors='gray', linewidth=1, label='Cops', zorder=5)
            
            active_pct = snapshot['active_count'] / total_citizens * 100 if total_citizens > 0 else 0
            ax.set_title(f'{stage_name}\n(Step {snapshot["step"]}, Active: {active_pct:.1f}%)', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('X', fontsize=9)
            ax.set_ylabel('Y', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.set_xlim(-0.5, self.height - 0.5)
            ax.set_ylim(-0.5, self.width - 0.5)
            
            if i == 0:
                quiet_patch = mpatches.Patch(color='lightblue', label='Quiet')
                active_patch = mpatches.Patch(color='red', label='Active')
                jailed_patch = mpatches.Patch(color='orange', label='Jailed')
                cop_patch = mpatches.Patch(color='black', label='Cops')
                ax.legend(handles=[quiet_patch, active_patch, jailed_patch, cop_patch], 
                         loc='upper right', fontsize=8)
        
        # 第二行: 市民状态演化 + 激进比例
        ax_curve = fig.add_subplot(gs[1, :])
        quiet_line, = ax_curve.plot(iterations, quiet_counts, label='Quiet Citizens', 
                                    linewidth=2, color='#3498db')
        active_line, = ax_curve.plot(iterations, active_counts, label='Active Citizens', 
                                     linewidth=2.5, marker='o', markersize=2, color='#e74c3c')
        jailed_line, = ax_curve.plot(iterations, jailed_counts, label='Jailed Citizens', 
                                     linewidth=2, color='#f1c40f')
        
        ratio_axis = ax_curve.twinx()
        ratio_percent = [ratio * 100 for ratio in active_ratios]
        ratio_line, = ratio_axis.plot(iterations, ratio_percent, label='Active Ratio (%)',
                                      linewidth=2, linestyle='--', color='#8e44ad')
        ratio_axis.set_ylabel('Active Ratio (%)', color='#8e44ad')
        ratio_axis.tick_params(axis='y', colors='#8e44ad')
        
        # 标记快照点
        for snapshot in selected_snapshots:
            step = snapshot['step']
            if step < len(active_counts):
                ax_curve.axvline(x=step, color='red', linestyle=':', alpha=0.3)
                ax_curve.plot(step, active_counts[step], 'ro', markersize=6)
        
        ax_curve.set_title(f'Citizen State Evolution (LCM-LBF)', 
                          fontsize=14, fontweight='bold')
        ax_curve.set_xlabel('Step', fontsize=11)
        ax_curve.set_ylabel('Number of Citizens', fontsize=11)
        ax_curve.grid(True, alpha=0.3)
        
        legend_lines = [quiet_line, active_line, jailed_line, ratio_line]
        legend_labels = [line.get_label() for line in legend_lines]
        ax_curve.legend(legend_lines, legend_labels, fontsize=10, loc='upper right')
        
        # 添加统计信息
        final_ratio = active_ratios[-1]
        stats_text = f'Mode: LCM-LBF\n'
        stats_text += f'Grid Size: {self.width}×{self.height}\n'
        stats_text += f'Legitimacy: {self.legitimacy:.2f}\n'
        stats_text += f'Citizens: {total_citizens}, Cops: {len(self.cops)}\n'
        stats_text += f'Initial Active: {active_counts[0]} ({active_ratios[0]:.1%})\n'
        stats_text += f'Initial Quiet: {quiet_counts[0]}, Initial Jailed: {jailed_counts[0]}\n'
        stats_text += f'Final Active: {active_counts[-1]} ({final_ratio:.1%})\n'
        stats_text += f'Final Quiet: {quiet_counts[-1]}, Final Jailed: {jailed_counts[-1]}\n'
        stats_text += f'Peak Active: {max(active_counts)} at step {active_counts.index(max(active_counts))}'
        
        ax_curve.text(0.02, 0.98, stats_text, transform=ax_curve.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='plum', 
                              alpha=0.9, edgecolor='black'))
        
        plt.tight_layout()
        
        filename = f"{self.output_pdf}Civil_LCM_LBF_{self.width}x{self.height}_seed{self.seed}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[保存] 结果保存至: {filename}")
        
        print(f"\n{'='*60}")
        print(f"[统计] Civil Violence LCM-LBF")
        print(f"{'='*60}")
        print(f"网格: {self.width}×{self.height}, 合法性: {self.legitimacy:.2f}")
        print(f"最终激进: {active_counts[-1]}/{total_citizens} ({active_ratios[-1]:.1%})")
        print(f"峰值激进: {max(active_counts)} at step {active_counts.index(max(active_counts))}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Civil Violence LCM-LBF Model")
    
    parser.add_argument("--width", type=int, default=40, help="网格宽度")
    parser.add_argument("--height", type=int, default=40, help="网格高度")
    parser.add_argument("--steps", type=int, default=100, help="仿真步数")
    parser.add_argument("--citizen_count", type=int, default=50, help="市民数量")
    parser.add_argument("--cop_count", type=int, default=10, help="警察数量")
    parser.add_argument("--legitimacy", type=float, default=0.6, help="政府合法性")
    parser.add_argument("--citizen_vision", type=int, default=2, help="市民视野")
    parser.add_argument("--cop_vision", type=int, default=5, help="警察视野")
    parser.add_argument("--max_jail_time", type=int, default=30, help="最大监禁时间")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out_pdf", type=str, default="./res_figs_civil/", help="输出路径")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06", help="LLM模型")
    parser.add_argument("--log_file", type=str, default=None, help="日志文件路径")
    parser.add_argument("--enable_step_log", action="store_true", help="启用步骤日志")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.out_pdf, exist_ok=True)
    
    model = CivilViolence_LCM_LBF(
        grid_size=(args.width, args.height),
        steps=args.steps,
        citizen_count=args.citizen_count,
        cop_count=args.cop_count,
        legitimacy=args.legitimacy,
        citizen_vision=args.citizen_vision,
        cop_vision=args.cop_vision,
        max_jail_time=args.max_jail_time,
        seed=args.seed,
        output_pdf=args.out_pdf,
        model=args.model,
        log_file=args.log_file,
        enable_step_log=args.enable_step_log
    )
    
    model.visualization()

