"""
This component transform the LLM's output into actions under specific scenario.
"""
from .LLM_based import generate, generate_and_parser, get_embedding, parse_attributes


class LLM_Agent():
    def __init__(self, model, scenario, attributes=None, mem_model="") -> None:
        self.model = model
        self.scenario = scenario
        self.mem_model = mem_model
        self.memory = []
        self.attributes = attributes

    def update_attributes(self, attributes):
        self.attributes = attributes
        
    def take_actions(self):
        inputs = parse_attributes(self.scenario, self.attributes)
        llm_output = generate_and_parser(self.model, inputs)
        return llm_output
    
    def generate(self, input):
        llm_output = generate(self.model, input)
        return llm_output
    
    def update_memory(self, input):
        if self.mem_model != "":
            embedding = get_embedding(self.mem_model, input)
            self.memory.extend(embedding)
        else:
            raise ValueError("No memory model specified.")
