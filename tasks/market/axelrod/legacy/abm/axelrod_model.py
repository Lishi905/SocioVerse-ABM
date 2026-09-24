#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Axelrod Tournament Model Implementation with Mesa Framework

Implements a round-robin tournament where all strategies play against
each other (including self-play).
"""

from mesa import Model
from axelrod_agent import create_strategy, ALL_STRATEGIES
from collections import defaultdict


class AxelrodModel(Model):
    """
    Axelrod Tournament Model.
    
    Runs a round-robin tournament where each strategy plays against
    all strategies (including itself).
    """
    
    # Payoff matrix (classic Prisoner's Dilemma)
    # T > R > P > S and 2R > T + S
    PAYOFFS = {
        'T': 5,  # Temptation (I defect, opponent cooperates)
        'R': 3,  # Reward (mutual cooperation)
        'P': 1,  # Punishment (mutual defection)
        'S': 0   # Sucker (I cooperate, opponent defects)
    }
    
    def __init__(self, n_rounds=10000, n_replicates=1, seed=20251017):
        """
        Initialize Axelrod tournament.
        
        Args:
            n_rounds: Number of rounds per match
            n_replicates: Number of times to replicate the tournament
            seed: Random seed
        """
        super().__init__(seed=seed)
        
        self.n_rounds = n_rounds
        self.n_replicates = n_replicates
        self.base_seed = seed
        
        # Create strategies
        self.strategies = []
        for strategy_name in ALL_STRATEGIES:
            strategy = create_strategy(strategy_name, self)
            self.strategies.append(strategy)
        
        # Results storage
        self.match_results = []  # List of match results
        self.strategy_scores = defaultdict(lambda: {'total_score': 0, 'rounds': 0})
        
    def get_payoff(self, my_action, opponent_action):
        """
        Get payoff for my action given opponent's action.
        
        Args:
            my_action: 'C' or 'D'
            opponent_action: 'C' or 'D'
            
        Returns:
            Payoff value
        """
        if my_action == 'C' and opponent_action == 'C':
            return self.PAYOFFS['R']  # Mutual cooperation
        elif my_action == 'C' and opponent_action == 'D':
            return self.PAYOFFS['S']  # Sucker
        elif my_action == 'D' and opponent_action == 'C':
            return self.PAYOFFS['T']  # Temptation
        else:  # Both defect
            return self.PAYOFFS['P']  # Punishment
    
    def play_match(self, strategy1, strategy2, replicate_seed):
        """
        Play a single match between two strategies.
        
        Args:
            strategy1: First strategy
            strategy2: Second strategy
            replicate_seed: Random seed for this replicate
            
        Returns:
            Tuple of (strategy1_total_score, strategy2_total_score)
        """
        # Reset both strategies
        strategy1.reset()
        strategy2.reset()
        
        # Initialize scores for this match
        score1 = 0
        score2 = 0
        
        # Play n_rounds
        for round_num in range(self.n_rounds):
            # Both strategies choose actions simultaneously
            action1 = strategy1.choose_action(round_num)
            action2 = strategy2.choose_action(round_num)
            
            # Calculate payoffs
            payoff1 = self.get_payoff(action1, action2)
            payoff2 = self.get_payoff(action2, action1)
            
            # Update scores
            score1 += payoff1
            score2 += payoff2
            
            # Update histories (for TFT and other adaptive strategies)
            strategy1.update_history(action2)
            strategy2.update_history(action1)
        
        return score1, score2
    
    def run_tournament(self):
        """
        Run the full round-robin tournament.
        
        Each strategy plays against all strategies (including itself)
        for n_replicates times.
        """
        print(f"Running Axelrod Tournament...")
        print(f"  Strategies: {len(self.strategies)}")
        print(f"  Rounds per match: {self.n_rounds}")
        print(f"  Replicates: {self.n_replicates}")
        print(f"  Total matches: {len(self.strategies) ** 2 * self.n_replicates}")
        print()
        
        # For each replicate
        for replicate in range(self.n_replicates):
            replicate_seed = self.base_seed + replicate
            
            # Generate all pairings (including self-play)
            match_count = 0
            for i, strategy1 in enumerate(self.strategies):
                for j, strategy2 in enumerate(self.strategies):
                    # Play match
                    score1, score2 = self.play_match(strategy1, strategy2, replicate_seed)
                    
                    # Record results
                    self.match_results.append({
                        'replicate': replicate,
                        'strategy1': strategy1.strategy_name,
                        'strategy2': strategy2.strategy_name,
                        'score1': score1,
                        'score2': score2,
                        'avg_score1': score1 / self.n_rounds,
                        'avg_score2': score2 / self.n_rounds
                    })
                    
                    # Update strategy totals
                    self.strategy_scores[strategy1.strategy_name]['total_score'] += score1
                    self.strategy_scores[strategy1.strategy_name]['rounds'] += self.n_rounds
                    
                    match_count += 1
            
            if self.n_replicates > 1:
                print(f"  Completed replicate {replicate + 1}/{self.n_replicates}")
        
        print(f"\nTournament complete! Played {len(self.match_results)} matches.")
    
    def get_rankings(self):
        """
        Get strategy rankings by average score per round.
        
        Returns:
            List of tuples (strategy_name, avg_score_per_round) sorted by score descending
        """
        rankings = []
        for strategy_name, scores in self.strategy_scores.items():
            avg_score = scores['total_score'] / scores['rounds'] if scores['rounds'] > 0 else 0
            rankings.append((strategy_name, avg_score))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def print_rankings(self):
        """Print the final rankings."""
        rankings = self.get_rankings()
        
        print("\n" + "=" * 60)
        print("FINAL RANKINGS (by average score per round)")
        print("=" * 60)
        
        for rank, (strategy_name, avg_score) in enumerate(rankings, 1):
            print(f"{rank:2d}. {strategy_name:15s}  {avg_score:.4f}")
        
        print("=" * 60)
    
    def get_detailed_stats(self):
        """
        Get detailed statistics for each strategy.
        
        Returns:
            Dictionary with detailed stats per strategy
        """
        stats = {}
        
        for strategy_name in ALL_STRATEGIES:
            # Get all matches where this strategy played
            matches = [m for m in self.match_results if m['strategy1'] == strategy_name]
            
            if not matches:
                continue
            
            total_score = sum(m['score1'] for m in matches)
            total_rounds = len(matches) * self.n_rounds
            avg_score = total_score / total_rounds if total_rounds > 0 else 0
            
            stats[strategy_name] = {
                'total_score': total_score,
                'total_rounds': total_rounds,
                'avg_score_per_round': avg_score,
                'matches_played': len(matches)
            }
        
        return stats

