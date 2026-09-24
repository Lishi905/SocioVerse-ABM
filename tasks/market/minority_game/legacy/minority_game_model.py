#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minority Game Model Implementation with Mesa Framework

This module defines the main model class for the Minority Game.
"""

import numpy as np
import mesa
from mesa import Model
from minority_game_agent import MinorityGameAgent


class MinorityGameModel(Model):
    """
    The Minority Game Model.
    
    In each round:
    1. N agents choose between A (0) or B (1) based on their best strategy
    2. The minority side wins
    3. Agents update their strategy scores using counterfactual reasoning
    4. Agents may switch to better-performing strategies
    
    Attributes:
        N: Total number of agents
        M: Default memory length (or max M if heterogeneous)
        S: Number of strategies per agent
        history: List of winning sides from past rounds
        step_data: List recording statistics for each step
    """
    
    def __init__(self, N, M, S, seed=None, M_mix=None):
        """
        Initialize the Minority Game model.
        
        Args:
            N: Number of agents
            M: Memory length (default, or max M if M_mix provided)
            S: Number of strategies per agent
            seed: Random seed for reproducibility
            M_mix: Dictionary {M_value: count} for heterogeneous memory agents
        """
        super().__init__(seed=seed)
        
        self.N = N
        self.M = M
        self.S = S
        self.M_mix = M_mix
        
        # Create agents (Mesa 3.x will automatically add them to self.agents)
        if M_mix is None:
            # Homogeneous: all agents have same M
            self.max_M = M
            for i in range(N):
                MinorityGameAgent(i, self, M, S)
        else:
            # Heterogeneous: different M values
            self.max_M = max(M_mix.keys())
            agent_id = 0
            for M_val in sorted(M_mix.keys()):
                count = M_mix[M_val]
                for _ in range(count):
                    MinorityGameAgent(agent_id, self, M_val, S)
                    agent_id += 1
        
        # Initialize history with random winners
        # History stores winning sides (0 or 1) for past rounds
        # Mesa's random uses Python's random module
        self.history = [self.random.randint(0, 1) for _ in range(self.max_M)]
        
        # Data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "A_count": lambda m: m.current_A_count,
                "B_count": lambda m: m.current_B_count,
                "minority_side": lambda m: m.current_minority_side,
                "minority_size": lambda m: m.current_minority_size,
                "variance": lambda m: m.get_variance()
            },
            agent_reporters={
                "M": lambda a: a.M,
                "real_score": lambda a: a.real_score,
                "switches": lambda a: a.total_switches,
                "last_action": lambda a: a.last_action
            }
        )
        
        # Current round statistics
        self.current_A_count = 0
        self.current_B_count = 0
        self.current_minority_side = 0
        self.current_minority_size = 0
        
        # Track variance over time
        self.A_count_history = []
        
        # Step counter
        self.step_count = 0
        
    def history_to_index(self, M):
        """
        Convert the last M steps of history to an integer index (0 to 2^M - 1).
        
        Args:
            M: Memory length
            
        Returns:
            Integer index representing the history pattern
        """
        recent_history = self.history[-M:]
        index = 0
        for i, bit in enumerate(recent_history):
            index += int(bit) * (2 ** (M - 1 - i))
        return index
    
    def step(self):
        """
        Execute one step of the Minority Game:
        1. Each agent chooses an action
        2. Determine the minority side (winner)
        3. Update all agents' strategies and scores
        4. Update history
        5. Collect data
        """
        # Step 1: Each agent chooses action
        actions = []
        for agent in self.agents:
            history_idx = self.history_to_index(agent.M)
            action = agent.choose_action(history_idx)
            actions.append(action)
        
        # Step 2: Count A (0) and B (1)
        actions_array = np.array(actions)
        count_A = int(np.sum(actions_array == 0))
        count_B = int(np.sum(actions_array == 1))
        
        # Step 3: Determine winner (minority side)
        if count_A < count_B:
            winning_side = 0
            minority_size = count_A
        elif count_B < count_A:
            winning_side = 1
            minority_size = count_B
        else:
            # Tie (rare with odd N) - randomly break
            winning_side = self.random.randint(0, 1)
            minority_size = count_A
        
        # Step 4: Update agents
        for i, agent in enumerate(self.agents):
            # Update virtual scores for all strategies (counterfactual)
            history_idx = self.history_to_index(agent.M)
            agent.update_strategies(history_idx, winning_side)
            
            # Update real score if won
            is_winner = (actions[i] == winning_side)
            agent.update_real_score(is_winner)
        
        # Step 5: Update history
        self.history.append(winning_side)
        
        # Step 6: Store current statistics
        self.current_A_count = count_A
        self.current_B_count = count_B
        self.current_minority_side = winning_side
        self.current_minority_size = minority_size
        self.A_count_history.append(count_A)
        
        # Step 7: Collect data
        self.datacollector.collect(self)
        
        # Step 8: Increment step counter
        self.step_count += 1
        
    def get_variance(self):
        """Calculate variance of A_count over all steps so far."""
        if len(self.A_count_history) == 0:
            return 0.0
        return float(np.var(self.A_count_history))
    
    def get_step_data(self, burn_in=0):
        """
        Get step-by-step data after burn-in period.
        
        Args:
            burn_in: Number of initial steps to discard
            
        Returns:
            Dictionary with step data
        """
        model_data = self.datacollector.get_model_vars_dataframe()
        if burn_in > 0:
            model_data = model_data.iloc[burn_in:]
        return model_data
    
    def get_agent_stats(self, burn_in=0):
        """
        Get statistics for each agent after burn-in period.
        
        Args:
            burn_in: Number of initial steps to discard
            
        Returns:
            List of dictionaries containing agent statistics
        """
        agent_data = self.datacollector.get_agent_vars_dataframe()
        
        # Get final step data for each agent
        total_steps = self.step_count
        if total_steps == 0:
            return []
        
        agent_stats = []
        for agent in self.agents:
            # Calculate wins after burn-in
            # We approximate by scaling the total score
            effective_steps = max(1, total_steps - burn_in)
            wins = agent.real_score * (effective_steps / total_steps) if total_steps > 0 else 0
            win_rate = agent.real_score / total_steps if total_steps > 0 else 0
            
            stats = {
                'AgentID': agent.unique_id,
                'M': agent.M,
                'wins': agent.real_score,  # Total wins
                'moves': total_steps,
                'win_rate': win_rate,
                'switches': agent.total_switches,
                'switch_freq': agent.total_switches / total_steps if total_steps > 0 else 0
            }
            agent_stats.append(stats)
        
        return agent_stats

