# -*- coding: utf-8 -*-
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
Schelling Model 基础功能模块
提供网格初始化、邻居计算、位置管理等核心功能
"""

import random
import mesa


class SchellingBase:
    """Schelling模型基础类,提供核心功能"""
    
    def __init__(self, width, height, density, minority_pc, seed):
        """
        初始化基础参数
        
        Args:
            width: 网格宽度
            height: 网格高度
            density: 网格占用率 (0-1)
            minority_pc: 少数群体比例 (0-1)
            seed: 随机种子
        """
        self.width = width
        self.height = height
        self.density = density
        self.minority_pc = minority_pc
        self.seed = seed
        
        # # 设置随机种子
        # random.seed(seed)
        
        # 模型组件
        self.grid = None
        self.agents = []
    
    def init_grid_model(self):
        """初始化网格和智能体"""
        # 设置随机种子，确保每次初始化都使用完整的随机数序列，网格初始化仍然一致
        random.seed(self.seed)
        
        # 创建网格
        self.grid = mesa.space.SingleGrid(self.width, self.height, torus=True)
        
        # 初始化智能体
        agent_id = 0
        for x in range(self.width):
            for y in range(self.height):
                if random.random() < self.density:
                    # 决定智能体类型
                    if random.random() < self.minority_pc:
                        agent_type = 1  # 少数群体
                    else:
                        agent_type = 0  # 多数群体
                    
                    agent = {
                        'id': agent_id,
                        'type': agent_type,
                        'pos': (x, y),
                        'happy': False
                    }
                    self.agents.append(agent)
                    agent_id += 1
        
        return self.grid, self.agents
    
    def count_similar_neighbors(self, agent_pos, agent_type):
        """
        计算相似邻居数量
        
        Args:
            agent_pos: 智能体位置 (x, y)
            agent_type: 智能体类型 (0 或 1)
        
        Returns:
            (similar_count, total_count): 相似邻居数量和总邻居数量
        """
        x, y = agent_pos
        similar_count = 0
        total_count = 0
        
        # Moore 邻域 (8个邻居)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                # 处理环形拓扑
                nx = (x + dx) % self.width
                ny = (y + dy) % self.height
                
                # 查找该位置的智能体
                neighbor = self._get_agent_at_pos((nx, ny))
                if neighbor:
                    total_count += 1
                    if neighbor['type'] == agent_type:
                        similar_count += 1
        
        return similar_count, total_count
    
    def _get_agent_at_pos(self, pos):
        """
        获取指定位置的智能体
        
        Args:
            pos: 位置 (x, y)
        
        Returns:
            agent字典或None
        """
        for agent in self.agents:
            if agent['pos'] == pos:
                return agent
        return None
    
    def _get_empty_cells(self):
        """
        获取所有空闲位置
        
        Returns:
            空闲位置列表 [(x1, y1), (x2, y2), ...]
        """
        occupied_positions = {agent['pos'] for agent in self.agents}
        empty_cells = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in occupied_positions:
                    empty_cells.append((x, y))
        return empty_cells
    
    def _get_empty_cells_in_vision(self, agent_pos, occupied_positions, vision_range=None):
        """
        获取代理视野范围内的空位列表（基于代理位置和视野范围）
        
        Args:
            agent_pos: 代理位置 (x, y)
            occupied_positions: 被占用的位置集合
            vision_range: 视野范围（曼哈顿距离），None 表示无限制
        
        Returns:
            empty_cells: 代理视野范围内的空位列表（包含代理自己的位置，随机顺序）
        """
        empty_cells = []
        
        if vision_range is None:
            # 无视野限制：考虑整个网格
            for x in range(self.width):
                for y in range(self.height):
                    pos = (x, y)
                    # 如果位置未被占用，或者是代理自己的位置，则可用
                    if pos not in occupied_positions or pos == agent_pos:
                        empty_cells.append(pos)
        else:
            # 有视野限制：只考虑曼哈顿距离 <= vision_range 的位置
            for x in range(self.width):
                for y in range(self.height):
                    pos = (x, y)
                    # 计算曼哈顿距离（考虑环形拓扑）
                    dx = min(abs(x - agent_pos[0]), self.width - abs(x - agent_pos[0]))
                    dy = min(abs(y - agent_pos[1]), self.height - abs(y - agent_pos[1]))
                    distance = dx + dy
                    
                    if distance <= vision_range:
                        # 如果位置未被占用，或者是代理自己的位置，则可用
                        if pos not in occupied_positions or pos == agent_pos:
                            empty_cells.append(pos)
        
        # 随机打乱顺序
        random.shuffle(empty_cells)
        return empty_cells
    
    def _get_neighbors_detail(self, pos, agent_type):
        """
        获取邻居详细信息
        
        Args:
            pos: 智能体位置 (x, y)
            agent_type: 智能体类型 (用于兼容性,实际未使用)
        
        Returns:
            邻居信息列表,每个元素包含 {'id', 'pos', 'type'}
        """
        x, y = pos
        neighbors_info = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx = (x + dx) % self.width
                ny = (y + dy) % self.height
                
                neighbor = self._get_agent_at_pos((nx, ny))
                if neighbor:
                    info = {
                        "id": neighbor["id"],
                        "pos": neighbor["pos"],
                        "type": neighbor["type"]
                    }
                    neighbors_info.append(info)
        
        return neighbors_info
    
    def update_happiness(self, homophily):
        """
        更新所有智能体的幸福度
        
        Args:
            homophily: 同质性阈值
        
        Returns:
            happy_count: 幸福智能体数量
        """
        happy_count = 0
        for agent in self.agents:
            similar, total = self.count_similar_neighbors(agent['pos'], agent['type'])
            agent['happy'] = similar >= homophily
            if agent['happy']:
                happy_count += 1
        return happy_count

