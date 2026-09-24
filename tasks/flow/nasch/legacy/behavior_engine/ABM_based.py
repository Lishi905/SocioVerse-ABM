"""
This component stores the behavior functions powered by traditional ABMs for NaSch traffic model.
"""
import random

def parse_attributes(scenario, attributes):
    """
    Parse attributes and return action for NaSch traffic model.
    
    Args:
        scenario: "nasch" for traffic model
        attributes: dict containing:
            - current_speed: int, current vehicle speed
            - gap_ahead: int, distance to next vehicle
            - max_speed: int, maximum allowed speed
            - randomization_prob: float, probability of random slowdown
    
    Returns:
        dict: {"speed": new_speed} where new_speed is in [0, max_speed]
    """
    if scenario == "nasch":
        current_speed = attributes["current_speed"]
        gap_ahead = attributes["gap_ahead"]
        max_speed = attributes["max_speed"]
        p = attributes["randomization_prob"]
        
        # Step 1: Acceleration (NaSch rule 1)
        new_speed = min(current_speed + 1, max_speed)
        
        # Step 2: Deceleration (NaSch rule 2) - avoid collision
        new_speed = min(new_speed, gap_ahead)
        
        # Step 3: Randomization (NaSch rule 3) - random slowdown
        if random.random() < p and new_speed > 0:
            new_speed = max(new_speed - 1, 0)
        
        # Step 4: Movement (NaSch rule 4) - handled by model
        # Return the speed decision
        return {"speed": new_speed}
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
