#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minority Game Agent Implementation with Mesa Framework

This module defines the agent and strategy classes for the Minority Game.
"""

import numpy as np
from mesa import Agent


class Strategy:
    """
    A strategy maps each possible history pattern (of length M) to an action (0 or 1).
    Strategy table size is 2^M.
    
    Attributes:
        M: Memory length
        table: Numpy array of size 2^M, mapping history index to action (0 or 1)
        virtual_score: Accumulated virtual score for this strategy
    """
    
    def __init__(self, M, strategy_table=None, rng=None):
        """
        Initialize a strategy.
        
        Args:
            M: Memory length
            strategy_table: Optional pre-defined strategy table
            rng: Random number generator (Mesa's random.Random or numpy RNG)
        """
        self.M = M
        self.table_size = 2 ** M
        
        if strategy_table is not None:
            self.table = strategy_table.copy()
        else:
            # Randomly initialize strategy table
            if rng is None:
                # Use numpy RNG if no RNG provided
                rng = np.random.default_rng()
                self.table = rng.integers(0, 2, size=self.table_size)
            elif hasattr(rng, 'integers'):
                # NumPy RNG
                self.table = rng.integers(0, 2, size=self.table_size)
            else:
                # Mesa's random.Random (Python standard library)
                self.table = np.array([rng.randint(0, 1) for _ in range(self.table_size)])
        
        self.virtual_score = 0.0
    
    def get_action(self, history_index):
        """
        Get action (0 or 1) for a given history pattern index.
        
        Args:
            history_index: Integer representing the history pattern (0 to 2^M - 1)
            
        Returns:
            Action: 0 (side A) or 1 (side B)
        """
        return int(self.table[history_index])
    
    def copy(self):
        """Create a deep copy of this strategy."""
        new_strategy = Strategy(self.M, self.table)
        new_strategy.virtual_score = self.virtual_score
        return new_strategy


class MinorityGameAgent(Agent):
    """
    An agent in the Minority Game.
    
    Each agent:
    - Has S strategies
    - Tracks virtual scores for each strategy
    - Chooses actions based on the best-performing strategy
    - Updates strategy scores based on counterfactual analysis
    
    Attributes:
        unique_id: Unique identifier
        model: Reference to the model
        M: Memory length (can be different for heterogeneous agents)
        S: Number of strategies
        strategies: List of Strategy objects
        current_strategy_idx: Index of currently active strategy
        real_score: Total wins accumulated
        total_switches: Number of times the agent switched strategies
    """
    
    def __init__(self, unique_id, model, M, S):
        """
        Create a new Minority Game agent.
        
        Args:
            unique_id: Unique identifier for this agent (stored manually)
            model: Reference to the MinorityGameModel
            M: Memory length for this agent
            S: Number of strategies
        """
        super().__init__(model)
        self.unique_id = unique_id  # Store manually for compatibility
        self.M = M
        self.S = S
        
        # Create S random strategies
        self.strategies = []
        for _ in range(S):
            strategy = Strategy(M, rng=self.random)
            self.strategies.append(strategy)
        
        # Initialize state
        self.current_strategy_idx = 0
        self.real_score = 0.0
        self.total_switches = 0
        self.last_action = None
        
    def choose_action(self, history_index):
        """
        Choose action based on the best strategy (highest virtual score).
        If multiple strategies are tied for the best, randomly select among them.
        
        Args:
            history_index: Current history pattern index
            
        Returns:
            Action (0 or 1)
        """
        # Find strategies with maximum virtual score
        max_score = max(s.virtual_score for s in self.strategies)
        best_indices = [i for i, s in enumerate(self.strategies) 
                       if s.virtual_score == max_score]
        
        # Check if we should switch (threshold = 1)
        current_score = self.strategies[self.current_strategy_idx].virtual_score
        if max_score >= current_score + 1:
            self.total_switches += 1
        
        # Choose among best strategies (random tie-breaking)
        if len(best_indices) > 1:
            # Mesa's random.choice works with lists
            self.current_strategy_idx = self.random.choice(best_indices)
        else:
            self.current_strategy_idx = best_indices[0]
        
        # Get action from chosen strategy
        action = self.strategies[self.current_strategy_idx].get_action(history_index)
        self.last_action = action
        
        return action
    
    def update_strategies(self, history_index, winning_side):
        """
        Update virtual scores for all strategies using counterfactual reasoning.
        Each strategy's virtual score increases by 1 if it would have chosen
        the winning side.
        
        Args:
            history_index: History pattern index for this round
            winning_side: The side that won (0 or 1)
        """
        for strategy in self.strategies:
            if strategy.get_action(history_index) == winning_side:
                strategy.virtual_score += 1.0
    
    def update_real_score(self, is_winner):
        """
        Update agent's real score if they won this round.
        
        Args:
            is_winner: Boolean indicating if this agent won
        """
        if is_winner:
            self.real_score += 1.0
    
    def step(self):
        """
        Mesa agent step method.
        In Minority Game, the actual logic is coordinated by the model,
        so individual agent steps are handled there.
        """
        pass

