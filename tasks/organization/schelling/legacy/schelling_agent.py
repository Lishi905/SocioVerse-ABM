import mesa
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class SchellingAgent(mesa.Agent):
    """
    A Schelling segregation agent with support for both LLM and rule-based decision making.
    
    The agent can use:
    1. Rule-based decision: Move if similar neighbors < homophily threshold
    2. LLM-based decision: Let LLM decide where to move using complete reasoning
    """
    
    def __init__(self, unique_id, model, agent_type, homophily, use_llm=False, llm_decision_maker=None):
        super().__init__(unique_id, model)
        self.type = agent_type
        self.homophily = homophily
        self.use_llm = use_llm
        self.llm_decision_maker = llm_decision_maker
        self.decision_method = "llm" if use_llm else "rule_based"
    
    @staticmethod
    def prepare_batch_requests(agents: List['SchellingAgent'], grid, grid_size, homophily) -> List[Dict[str, Any]]:
        """
        Prepare batch decision requests for all LLM agents.
        
        Args:
            agents: List of agents to process
            grid: The grid object
            grid_size: Tuple of (width, height)
            homophily: Homophily threshold
        
        Returns:
            List of request dicts for batch processing
        """
        batch_requests = []
        
        for agent in agents:
            if agent.use_llm and agent.llm_decision_maker:
                # Get neighborhood info
                neighbors = grid.get_neighbors(agent.pos, moore=True, include_center=False)
                similar = sum(1 for n in neighbors if n.type == agent.type)
                neighbor_types = [n.type for n in neighbors]
                
                # Collect all empty cells (do this once for efficiency)
                if not hasattr(agent.model, '_cached_empty_cells') or agent.model._cache_step != agent.model.steps:
                    agent.model._cached_empty_cells = []
                    for x in range(grid.width):
                        for y in range(grid.height):
                            if grid.is_cell_empty((x, y)):
                                agent.model._cached_empty_cells.append((x, y))
                    agent.model._cache_step = agent.model.steps
                
                empty_cells = agent.model._cached_empty_cells
                
                request = {
                    "agent_id": agent.unique_id,
                    "agent_pos": agent.pos,
                    "agent_type": agent.type,
                    "neighbors_info": {
                        "similar_neighbors": similar,
                        "total_neighbors": len(neighbors),
                        "neighbor_types": neighbor_types
                    },
                    "empty_cells": empty_cells,
                    "grid_size": grid_size,
                    "homophily_threshold": homophily,
                    "kwargs": {"temperature": 0.7}
                }
                
                batch_requests.append(request)

        return batch_requests
    
    @staticmethod
    def apply_batch_decisions(agents: List['SchellingAgent'], batch_results: List[Dict[str, Any]], grid):
        """
        Apply decisions from batch processing to agents.
        
        Args:
            agents: List of agents that were processed
            batch_results: List of decisions from make_batch_decisions_async
            grid: The grid object
        """
        for agent, decision in zip(agents, batch_results):
            agent._apply_decision(decision)
    
    def _apply_decision(self, decision: Dict[str, Any]):
        """
        Execute a decision (from either batch or individual processing).
        
        Args:
            decision: Decision dict from LLM
        """
        try:
            # Get empty cells for validation
            empty_cells = []
            for x in range(self.model.grid.width):
                for y in range(self.model.grid.height):
                    if self.model.grid.is_cell_empty((x, y)):
                        empty_cells.append((x, y))
            
            if decision["action"] == "move" and decision["new_pos"]:
                if decision["new_pos"] in empty_cells:
                    self.model.grid.move_agent(self, decision["new_pos"])
                    logger.debug(f"Agent {self.unique_id} moved to {decision['new_pos']} "
                                f"(confidence: {decision['confidence']:.2f})")
        except Exception as e:
            logger.error(f"Error applying decision for agent {self.unique_id}: {e}")
            # Fallback to rule-based
            similar = sum(1 for n in self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False
            ) if n.type == self.type)
            
            if similar < self.model.homophily:
                if empty_cells:
                    new_position = self.random.choice(empty_cells)
                    self.model.grid.move_agent(self, new_position)

    def _decide_rule_based(self, similar: int) -> bool:
        """
        Traditional rule-based decision: move if similar neighbors < homophily threshold.
        
        Args:
            similar: Number of similar neighbors
            
        Returns:
            True if should move, False otherwise
        """
        return similar < self.model.homophily

    def _decide_llm_based(self, similar: int, neighbors: list) -> bool:
        """
        LLM-based decision making.
        
        Args:
            similar: Number of similar neighbors
            neighbors: List of neighbor agents
            
        Returns:
            True if should move, False otherwise
        """
        if not self.llm_decision_maker:
            # Fallback to rule-based if LLM is not available
            return self._decide_rule_based(similar)
        
        # Count empty cells
        empty_cells = 0
        for x in range(self.model.grid.width):
            for y in range(self.model.grid.height):
                if self.model.grid.is_cell_empty((x, y)):
                    empty_cells += 1
        
        # Use LLM to make decision
        try:
            decision = self.llm_decision_maker.make_decision(
                agent_type=self.type,
                similar_neighbors=similar,
                total_neighbors=len(neighbors),
                homophily_threshold=self.model.homophily,
                empty_cells_available=empty_cells
            )
            return decision
        except Exception as e:
            # Fallback to rule-based if LLM fails
            return self._decide_rule_based(similar)

    def step(self):
        """
        Complete step method that can use either LLM or rule-based decision making.
        
        LLM-based step:
        1. Analyzes current neighborhood
        2. Uses LLM to decide: stay or move?
        3. If move, LLM decides specific target location
        4. Considers multiple factors (happiness, costs, uncertainty, stability)
        
        Rule-based step (original):
        1. Counts similar neighbors
        2. If similar < homophily, moves to random empty cell
        3. Otherwise stays
        """
        # Get neighborhood information
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        similar = sum(1 for neighbor in neighbors if neighbor.type == self.type)
        
        # Get available empty cells
        empty_cells = []
        for x in range(self.model.grid.width):
            for y in range(self.model.grid.height):
                if self.model.grid.is_cell_empty((x, y)):
                    empty_cells.append((x, y))
        
        # Make decision based on method
        if self.use_llm and self.llm_decision_maker:
            self._step_llm_based(similar, neighbors, empty_cells)
        else:
            self._step_rule_based(similar, empty_cells)
    
    def _step_rule_based(self, similar: int, empty_cells: list):
        """
        Traditional Schelling rule-based step.
        
        If unhappy (similar neighbors < homophily threshold):
            - Move to a random empty cell
        Otherwise:
            - Stay in place
        
        Note: Happiness is calculated after all moves in step_happiness()
        """
        if similar < self.model.homophily:
            # Unhappy - try to move
            if empty_cells:
                new_position = self.random.choice(empty_cells)
                self.model.grid.move_agent(self, new_position)
    
    def _step_llm_based(self, similar: int, neighbors: list, empty_cells: list):
        """
        LLM-based complete step decision.
        
        The LLM decides:
        1. Whether to move or stay (considering all factors, not just similarity count)
        2. Where to move (specific target position with reasoning)
        
        LLM considers:
        - Current neighborhood satisfaction
        - Movement costs
        - Uncertainty about new locations  
        - Community stability impacts
        - Long-term welfare
        """
        try:
            # Prepare neighbor information
            neighbor_types = [n.type for n in neighbors]
            neighbors_info = {
                "similar_neighbors": similar,
                "total_neighbors": len(neighbors),
                "neighbor_types": neighbor_types
            }
            
            # Make complete step decision via LLM
            decision = self.llm_decision_maker.make_step_decision(
                agent_pos=self.pos,
                agent_type=self.type,
                neighbors_info=neighbors_info,
                empty_cells=empty_cells,
                grid_size=(self.model.grid.width, self.model.grid.height),
                homophily_threshold=self.model.homophily,
                temperature=0.7
            )
            
            # Execute decision
            if decision["action"] == "move" and decision["new_pos"]:
                # Move to LLM-chosen location
                if decision["new_pos"] in empty_cells:
                    self.model.grid.move_agent(self, decision["new_pos"])
                    logger.debug(f"Agent {self.unique_id} moved to {decision['new_pos']} "
                                f"(confidence: {decision['confidence']:.2f})")
            
            # Store decision info for analysis (optional)
            if hasattr(self, 'decision_history'):
                self.decision_history.append({
                    'step': self.model.steps,
                    'action': decision['action'],
                    'reasoning': decision['reasoning'],
                    'confidence': decision['confidence']
                })
        
        except Exception as e:
            logger.error(f"LLM step failed for agent {self.unique_id}: {e}")
            # Fallback to rule-based decision
            if similar < self.model.homophily:
                if empty_cells:
                    new_position = self.random.choice(empty_cells)
                    self.model.grid.move_agent(self, new_position)

    def step_happiness(self):
        """
        Calculate and update this agent's happiness status.
        Happy if the number of similar neighbors >= homophily threshold.
        """
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        similar = sum(1 for neighbor in neighbors if neighbor.type == self.type)
        self.happy = similar >= self.model.homophily