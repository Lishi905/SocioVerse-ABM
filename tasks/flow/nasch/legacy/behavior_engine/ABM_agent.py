"""
This component transform the ABM's output into actions under specific scenario.
"""
from .ABM_based import parse_attributes

class ABM_Agent():
    def __init__(self, model, scenario) -> None:
        self.model = model
        self.scenario = scenario
        self.attributes = None
    
    def update_attributes(self, attributes):
        self.attributes = attributes
        
    def take_actions(self):
        if self.attributes is None:
            raise ValueError("No attributes specified.")
        abm_output = parse_attributes(self.scenario, self.attributes)
        return abm_output
