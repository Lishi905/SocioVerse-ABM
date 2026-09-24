#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Axelrod Tournament Agent Implementation using Axelrod Library

This module wraps all 218+ strategies from the Axelrod library.
The Axelrod library is a comprehensive Python library for studying 
the Iterated Prisoner's Dilemma with 200+ strategies.

Installation:
    pip install axelrod

Reference:
    - Axelrod Library: https://github.com/Axelrod-Python/Axelrod
    - Paper: Knight et al. (2016) "An open reproducible framework for the 
             study of the iterated prisoner's dilemma"
"""

try:
    import axelrod as axl
    AXELROD_AVAILABLE = True
except ImportError:
    AXELROD_AVAILABLE = False
    print("Warning: Axelrod library not found. Please install it with: pip install axelrod")

SELECTED_STRATEGY_NAMES = [
    "Alexei",
    "EugineNier",
    "First by Tideman and Chieruzzi",
    "Michaelos",
    "MEM2",
    "Evolved FSM 16",
    "Stalker",
    "First by Stein and Rapoport",
    "DoubleCrosser",
    "Limited Retaliate 2",
    "EvolvedLookerUp1_1_1",
    "Limited Retaliate 3",
    "Evolved FSM 6",
    "Punisher",
    "Spiteful Tit For Tat",
    "Evolved FSM 16 Noise 05",
    "BackStabber",
    "Retaliate 3",
    "Retaliate",
    "Second by RichardHufford",
    "EvolvedLookerUp2_2_2",
    "Tricky Level Punisher",
    "Adaptive Pavlov 2006",
    "CAPRI",
    "Forgiver",
    "Limited Retaliate",
    "Retaliate 2",
    "Forgetful Grudger",
    "Evolved ANN 5",
    "AON2",
    "Grudger",
    "SpitefulCC",
    "Evolved HMM 5",
    "TF3",
    "Evolved ANN",
    "PSO Gambler Mem1",
    "Adaptive Pavlov 2011",
    "Omega TFT",
    "Inverse Punisher",
    "PSO Gambler 2_2_2 Noise 05",
    "Fool Me Once",
    "Adaptive",
    "Winner12",
    "Tit For Tat",
    "Evolved FSM 4",
    "PSO Gambler 2_2_2",
    "AdaptorBrief",
    "Second by GraaskampKatzen",
    "ZD-Mischief",
    "First by Graaskamp"
]


def get_all_strategies():
    """
    Get all available strategies from the Axelrod library.
    
    Returns:
        list: List of strategy classes (~204 strategies)
    """
    if not AXELROD_AVAILABLE:
        raise ImportError("Axelrod library is required but not installed.")
    
    # axl.all_strategies is already a list of strategy classes
    # that obey Axelrod's original tournament rules
    from axelrod import all_strategies
    return all_strategies


def get_short_run_strategies():
    """
    Get strategies that have short run times (suitable for large tournaments).
    
    Returns:
        list: List of strategy classes with short run times (~180 strategies)
    """
    if not AXELROD_AVAILABLE:
        raise ImportError("Axelrod library is required but not installed.")
    
    from axelrod import short_run_time_strategies
    return short_run_time_strategies


def get_selected_strategies():
    """
    Get the curated 50-strategy subset aligned with the LLM tournament experiments.
    
    Returns:
        list: List of strategy classes matching SELECTED_STRATEGY_NAMES
    """
    if not AXELROD_AVAILABLE:
        raise ImportError("Axelrod library is required but not installed.")

    from axelrod import short_run_time_strategies

    name_to_class = {}
    for strategy_class in short_run_time_strategies:
        try:
            strategy_name = strategy_class().name
        except Exception:
            continue
        if strategy_name not in name_to_class:
            name_to_class[strategy_name] = strategy_class

    selected = []
    missing = []

    for display_name in SELECTED_STRATEGY_NAMES:
        strategy_class = name_to_class.get(display_name)
        if strategy_class is None:
            missing.append(display_name)
            continue
        selected.append(strategy_class)

    if missing:
        raise ValueError(
            "Selected strategies missing from Axelrod short-run set: "
            + ", ".join(missing)
        )

    return selected


def get_demo_strategies():
    """
    Get a small set of demo strategies for testing.
    
    Returns:
        list: List of 5 basic strategy classes
    """
    if not AXELROD_AVAILABLE:
        raise ImportError("Axelrod library is required but not installed.")
    
    from axelrod import demo_strategies
    return demo_strategies


def get_axelrod_first_strategies():
    """
    Get the 15 strategies from Axelrod's first tournament (1980).
    
    Returns:
        list: List of strategy classes from the original tournament
    """
    if not AXELROD_AVAILABLE:
        raise ImportError("Axelrod library is required but not installed.")
    
    from axelrod import axelrod_first_strategies
    return axelrod_first_strategies


def get_basic_strategies():
    """
    Get basic deterministic strategies.
    
    Returns:
        list: List of basic strategy classes
    """
    if not AXELROD_AVAILABLE:
        raise ImportError("Axelrod library is required but not installed.")
    
    from axelrod import basic_strategies
    return basic_strategies


# List all available strategies by name
ALL_STRATEGY_NAMES = None
SHORT_RUN_STRATEGY_NAMES = None
DEMO_STRATEGY_NAMES = None

if AXELROD_AVAILABLE:
    ALL_STRATEGY_NAMES = [s.__name__ for s in get_all_strategies()]
    SHORT_RUN_STRATEGY_NAMES = [s.__name__ for s in get_short_run_strategies()]
    DEMO_STRATEGY_NAMES = [s.__name__ for s in get_demo_strategies()]


def create_strategy(strategy_name):
    """
    Create a strategy instance by name.
    
    Args:
        strategy_name: Name of strategy class
        
    Returns:
        Strategy instance (axelrod.Player)
    """
    if not AXELROD_AVAILABLE:
        raise ImportError("Axelrod library is required but not installed.")
    
    # Get all strategies (includes meta strategies)
    from axelrod import all_strategies
    
    # Find strategy by name
    for strategy_class in all_strategies:
        if strategy_class.__name__ == strategy_name:
            return strategy_class()
    
    raise ValueError(f"Unknown strategy: {strategy_name}")


if __name__ == '__main__':
    # Print available strategies
    if AXELROD_AVAILABLE:
        print(f"Total strategies available: {len(get_all_strategies())}")
        print(f"Short run time strategies: {len(get_short_run_strategies())}")
        print(f"Demo strategies: {len(get_demo_strategies())}")
        print(f"\nAll strategy names ({len(ALL_STRATEGY_NAMES)}):")
        for i, name in enumerate(sorted(ALL_STRATEGY_NAMES), 1):
            print(f"  {i:3d}. {name}")
    else:
        print("Axelrod library not available. Please install it.")
