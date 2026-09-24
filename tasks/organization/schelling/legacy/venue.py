# -*- coding: utf-8 -*-
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
Schelling Model - 场所（Venue）模块
提供场所类和相关工具函数
"""

import random
from typing import List, Dict, Tuple, Optional


class Venue:
    """场所类 - 基于Silver等(2021)的建模"""
    
    def __init__(self, venue_id: int, venue_type: str, location: Tuple[int, int],
                 belonging_group: Optional[int] = None, exclusivity: float = 0.0,
                 obligatoriness: float = 1.0, catchment_radius: int = 3):
        """
        初始化场所
        
        Args:
            venue_id: 场所ID
            venue_type: 场所类型（如"Church", "Mosque", "Cafe", "Park"等）
            location: 场所位置 (x, y)
            belonging_group: 归属群体（None表示公共场所）
            exclusivity: 排他性 (0-1)，定义对异族群体的开放程度
                - 0.0: 完全开放（公共场所）
                - 1.0: 完全排他（仅归属群体可访问）
            obligatoriness: 强制性 (0-1)，代理人前往该场所的必要程度
                - 1.0: 只要条件允许就会去（论文中通常设为1）
            catchment_radius: 影响范围（曼哈顿距离），即代理人的最大出行距离
        """
        self.venue_id = venue_id
        self.venue_type = venue_type
        self.location = location
        self.belonging_group = belonging_group
        self.exclusivity = exclusivity  # 0-1浮点数
        self.obligatoriness = obligatoriness  # 0-1浮点数
        self.catchment_radius = catchment_radius
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "venue_id": self.venue_id,
            "type": self.venue_type,
            "location": list(self.location),
            "belonging_group": self.belonging_group,
            "exclusivity": self.exclusivity,
            "obligatoriness": self.obligatoriness,
            "catchment_radius": self.catchment_radius
        }


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    计算曼哈顿距离
    
    Args:
        pos1: 位置1 (x, y)
        pos2: 位置2 (x, y)
    
    Returns:
        曼哈顿距离
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_accessible_venues(agent_pos: Tuple[int, int], agent_type: int, agent_adventurousness: float,
                          venues: List[Venue], grid_width: int, grid_height: int) -> List[Dict]:
    """
    收集代理可访问的场所信息（基于Silver等2021的逻辑）
    
    访问条件：
    1. 距离条件：场所 within catchment_radius
    2. 资格条件：
       - 同族场所：直接访问
       - 异族场所：需要 agent_adventurousness > venue.exclusivity
    
    Args:
        agent_pos: 代理位置 (x, y)
        agent_type: 代理类型
        agent_adventurousness: 代理的冒险精神 (0-1)
        venues: 场所列表
        grid_width: 网格宽度（用于环形拓扑）
        grid_height: 网格高度（用于环形拓扑）
    
    Returns:
        可访问场所信息列表，每个元素包含场所信息和距离
    """
    accessible = []
    
    for venue in venues:
        # 计算距离（考虑环形拓扑）
        dx = min(abs(agent_pos[0] - venue.location[0]),
                 grid_width - abs(agent_pos[0] - venue.location[0]))
        dy = min(abs(agent_pos[1] - venue.location[1]),
                 grid_height - abs(agent_pos[1] - venue.location[1]))
        distance = dx + dy
        
        # 检查是否在影响范围内
        if distance <= venue.catchment_radius:
            # 检查访问权限（基于Silver论文的逻辑）
            can_access = False
            
            if venue.belonging_group is None:
                # 公共场所（exclusivity = 0），所有代理都可访问
                can_access = True
            elif venue.belonging_group == agent_type:
                # 同族场所：直接访问
                can_access = True
            else:
                # 异族场所：需要冒险精神 > 排他性
                if agent_adventurousness > venue.exclusivity:
                    can_access = True
            
            if can_access:
                venue_info = venue.to_dict()
                venue_info["distance"] = distance
                accessible.append(venue_info)
    
    return accessible


def get_venue_encounters(agent_pos: Tuple[int, int], agent_type: int, agent_adventurousness: float,
                         accessible_venues: List[Dict], all_agents: List[Dict],
                         grid_width: int, grid_height: int) -> Dict:
    """
    计算代理在可访问场所中遇到的其他代理人
    
    基于Silver论文：满意度 = (居住地同族邻居 + 场所辐射范围内同族人数) / (居住地总邻居 + 场所辐射范围内总人数)
    
    Args:
        agent_pos: 代理位置 (x, y)
        agent_type: 代理类型
        agent_adventurousness: 代理的冒险精神（用于确定访问哪些场所）
        accessible_venues: 可访问场所列表
        all_agents: 所有代理列表
        grid_width: 网格宽度
        grid_height: 网格高度
    
    Returns:
        字典，包含：
        - venue_same_group_count: 在场所中遇到的同族人数
        - venue_total_count: 在场所中遇到的总人数
        - venue_details: 每个场所中遇到的人员详情
    """
    venue_same_group_count = 0
    venue_total_count = 0
    venue_details = []
    
    for venue_info in accessible_venues:
        venue_location = tuple(venue_info["location"])
        venue_same = 0
        venue_total = 0
        venue_encounters = []
        
        # 遍历所有代理，找出也在访问该场所的代理
        for other_agent in all_agents:
            # 注意：这里不跳过自己，因为我们需要计算所有在场所中的人
            # 但实际应用中，可以排除自己来计算"遇到的其他代理人"
            
            other_pos = other_agent['pos']
            other_type = other_agent['type']
            
            # 计算其他代理到该场所的距离
            dx = min(abs(other_pos[0] - venue_location[0]),
                     grid_width - abs(other_pos[0] - venue_location[0]))
            dy = min(abs(other_pos[1] - venue_location[1]),
                     grid_height - abs(other_pos[1] - venue_location[1]))
            distance = dx + dy
            
            # 检查其他代理是否也在访问该场所
            # 简化假设：如果其他代理也在该场所的catchment_radius内，且满足访问条件，则会在场所中遇到
            if distance <= venue_info["catchment_radius"]:
                # 检查访问权限（简化：同族或公共场所）
                can_other_access = False
                if venue_info["belonging_group"] is None:
                    can_other_access = True
                elif venue_info["belonging_group"] == other_type:
                    can_other_access = True
                # 注意：这里简化了，实际应该检查other_agent的adventurousness
                # 但为了简化，我们假设所有代理都倾向于访问公共场所和同族场所
                
                if can_other_access:
                    venue_total += 1
                    if other_type == agent_type:
                        venue_same += 1
                    venue_encounters.append({
                        "agent_id": other_agent['id'],
                        "agent_type": other_type,
                        "distance_to_venue": distance
                    })
        
        venue_same_group_count += venue_same
        venue_total_count += venue_total
        venue_details.append({
            "venue_id": venue_info["venue_id"],
            "venue_type": venue_info["type"],
            "same_group_count": venue_same,
            "total_count": venue_total,
            "encounters": venue_encounters
        })
    
    return {
        "venue_same_group_count": venue_same_group_count,
        "venue_total_count": venue_total_count,
        "venue_details": venue_details
    }


def calculate_venue_satisfaction_factors(agent_pos: Tuple[int, int], agent_type: int,
                                         accessible_venues: List[Dict]) -> Dict:
    """
    计算场所满意度因子
    
    Args:
        agent_pos: 代理位置 (x, y)
        agent_type: 代理类型
        accessible_venues: 可访问场所列表
    
    Returns:
        场所满意度因子字典
    """
    own_group_venue_accessible = False
    other_group_venues_count = 0
    public_venues_count = 0
    nearest_venue_distance = float('inf')
    venue_diversity_score = 0.0
    
    for venue in accessible_venues:
        # 检查专属场所
        if venue["belonging_group"] == agent_type:
            own_group_venue_accessible = True
        
        # 统计其他群体专属场所
        if venue["belonging_group"] is not None and venue["belonging_group"] != agent_type:
            other_group_venues_count += 1
        
        # 统计公共场所
        if venue["belonging_group"] is None:
            public_venues_count += 1
        
        # 更新最近距离
        if venue["distance"] < nearest_venue_distance:
            nearest_venue_distance = venue["distance"]
    
    # 计算多样性得分（基于可访问场所的多样性）
    if accessible_venues:
        unique_groups = set()
        for venue in accessible_venues:
            if venue["belonging_group"] is not None:
                unique_groups.add(venue["belonging_group"])
            else:
                unique_groups.add("public")  # 公共场所作为特殊类型
        venue_diversity_score = len(unique_groups) / max(len(accessible_venues), 1)
    
    return {
        "own_group_venue_accessible": own_group_venue_accessible,
        "other_group_venues_count": other_group_venues_count,
        "public_venues_count": public_venues_count,
        "nearest_venue_distance": nearest_venue_distance if nearest_venue_distance != float('inf') else None,
        "venue_diversity_score": round(venue_diversity_score, 2)
    }


def create_venues(venue_config: Dict, grid_width: int, grid_height: int, seed: int) -> List[Venue]:
    """
    根据配置创建场所（基于Silver等2021的建模）
    
    Args:
        venue_config: 场所配置字典，包含：
            - "mode": "single_center", "multi_center", "mixed"
            - "num_exclusive_per_group": 每个群体的专属场所数量
            - "num_public": 公共场所数量
            - "catchment_radius": 影响范围
            - "exclusivity": 专属场所的排他性（默认1.0，完全排他）
        grid_width: 网格宽度
        grid_height: 网格高度
        seed: 随机种子
    
    Returns:
        场所列表
    """
    random.seed(seed)
    venues = []
    venue_id = 0
    
    mode = venue_config.get("mode", "mixed")
    num_exclusive_per_group = venue_config.get("num_exclusive_per_group", 1)
    num_public = venue_config.get("num_public", 2)
    catchment_radius = venue_config.get("catchment_radius", max(grid_width, grid_height) // 10)
    exclusivity = venue_config.get("exclusivity", 1.0)  # 默认完全排他
    
    # 确定群体类型（假设有2个群体：0和1）
    groups = [0, 1]  # 可以根据需要扩展
    
    if mode == "single_center":
        # 单中心模式：在网格中心放置一个大型公共场所
        center_x, center_y = grid_width // 2, grid_height // 2
        venues.append(Venue(
            venue_id=venue_id,
            venue_type="Community Center",
            location=(center_x, center_y),
            belonging_group=None,
            exclusivity=0.0,  # 公共场所，完全开放
            obligatoriness=1.0,
            catchment_radius=min(grid_width, grid_height) // 3
        ))
        venue_id += 1
    
    elif mode == "multi_center":
        # 多中心模式：每个群体2-3个专属场所，随机分布
        for group in groups:
            for _ in range(num_exclusive_per_group):
                x = random.randint(0, grid_width - 1)
                y = random.randint(0, grid_height - 1)
                
                # 根据群体类型选择场所类型
                venue_type = "Church" if group == 0 else "Mosque"
                
                venues.append(Venue(
                    venue_id=venue_id,
                    venue_type=venue_type,
                    location=(x, y),
                    belonging_group=group,
                    exclusivity=exclusivity,  # 使用配置的排他性
                    obligatoriness=1.0,
                    catchment_radius=catchment_radius
                ))
                venue_id += 1
    
    elif mode == "mixed":
        # 混合模式：专属场所 + 公共场所，混合分布
        # 创建专属场所
        for group in groups:
            for _ in range(num_exclusive_per_group):
                x = random.randint(0, grid_width - 1)
                y = random.randint(0, grid_height - 1)
                
                venue_type = "Church" if group == 0 else "Mosque"
                
                venues.append(Venue(
                    venue_id=venue_id,
                    venue_type=venue_type,
                    location=(x, y),
                    belonging_group=group,
                    exclusivity=exclusivity,  # 使用配置的排他性
                    obligatoriness=1.0,
                    catchment_radius=catchment_radius
                ))
                venue_id += 1
        
        # 创建公共场所
        public_venue_types = ["Cafe", "Shopping Mall", "Park", "Community Center"]
        for _ in range(num_public):
            x = random.randint(0, grid_width - 1)
            y = random.randint(0, grid_height - 1)
            venue_type = random.choice(public_venue_types)
            
            venues.append(Venue(
                venue_id=venue_id,
                venue_type=venue_type,
                location=(x, y),
                belonging_group=None,
                exclusivity=0.0,  # 公共场所，完全开放
                obligatoriness=1.0,
                catchment_radius=catchment_radius
            ))
            venue_id += 1
    
    return venues


def create_venues_from_positions(venue_positions: Dict, catchment_radius: int = 5, 
                                  exclusivity: float = 1.0, seed: int = None) -> List[Venue]:
    """
    根据指定的位置创建场所
    
    Args:
        venue_positions: 场所位置字典，格式为：
            {
                None: [(x1, y1), (x2, y2), ...],  # 公共场所，None作为键
                0: [(x1, y1), (x2, y2), ...],     # 群体0的专属场所
                1: [(x1, y1), (x2, y2), ...],     # 群体1的专属场所
            }
            键表示归属群体（None表示公共场所），值是一个位置元组列表
        catchment_radius: 所有场所的统一辐射距离（曼哈顿距离）
        exclusivity: 专属场所的排他性（默认1.0，完全排他）
        seed: 随机种子（用于选择venue_type，如果为None则不设置）
    
    Returns:
        场所列表
    """
    # 设置随机种子（如果提供了seed），确保venue类型选择的一致性
    if seed is not None:
        random.seed(seed)
    
    venues = []
    venue_id = 0
    
    # 公共场所类型列表
    public_venue_types = ["Cafe", "Shopping Mall", "Park", "Community Center"]
    # 群体0的场所类型
    group0_venue_types = ["Church", "Community Center", "School"]
    # 群体1的场所类型
    group1_venue_types = ["Mosque", "Community Center", "School"]
    
    # 遍历每个群体类型
    for belonging_group, positions in venue_positions.items():
        if not isinstance(positions, list):
            continue
        
        for pos in positions:
            if not isinstance(pos, (tuple, list)) or len(pos) != 2:
                continue
            
            x, y = int(pos[0]), int(pos[1])
            
            # 根据归属群体选择场所类型
            if belonging_group is None:
                # 公共场所
                venue_type = random.choice(public_venue_types) if seed is not None else public_venue_types[0]
                venue_exclusivity = 0.0  # 公共场所完全开放
            elif belonging_group == 0:
                # 群体0的专属场所
                venue_type = random.choice(group0_venue_types) if seed is not None else group0_venue_types[0]
                venue_exclusivity = exclusivity
            elif belonging_group == 1:
                # 群体1的专属场所
                venue_type = random.choice(group1_venue_types) if seed is not None else group1_venue_types[0]
                venue_exclusivity = exclusivity
            else:
                # 未知群体，使用默认类型
                venue_type = "Community Center"
                venue_exclusivity = exclusivity
            
            venues.append(Venue(
                venue_id=venue_id,
                venue_type=venue_type,
                location=(x, y),
                belonging_group=belonging_group,
                exclusivity=venue_exclusivity,
                obligatoriness=1.0,
                catchment_radius=catchment_radius
            ))
            venue_id += 1
    
    return venues

