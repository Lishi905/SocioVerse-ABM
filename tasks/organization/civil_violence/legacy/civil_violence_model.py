# Portions of this file are adapted from the Mesa Epstein civil violence example (epstein_civil_violence/model.py),
# https://github.com/projectmesa/mesa (and mesa-examples).
# Copyright Core Mesa Team and contributors. Licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Modified by the
# SocioVerse2 Authors. See NOTICE in the repository root.

import mesa
from mesa import Model, DataCollector
from mesa.space import MultiGrid
import numpy as np
from agents import Citizen, Cop, CitizenState


class CivilViolenceModel(Model):
    """
    Model class for the Epstein civil violence model.
    """

    def __init__(self, height=40, width=40, citizen_density=0.7, cop_density=0.04,
                 citizen_vision=1, cop_vision=2, legitimacy=0.8, threshold=0.05,
                 max_jail_time=30, arrest_prob_constant=2.3, movement=True):
        """
        Create a new civil violence model.

        Args:
            height: Grid height
            width: Grid width
            citizen_density: Density of citizens
            cop_density: Density of cops
            citizen_vision: Vision radius for citizens
            cop_vision: Vision radius for cops
            legitimacy: Government legitimacy (0-1)
            threshold: Threshold for citizen activation
            max_jail_time: Maximum jail sentence
            arrest_prob_constant: Constant for arrest probability calculation
            movement: Whether agents can move
        """
        super().__init__()
        self.height = height
        self.width = width
        self.citizen_density = citizen_density
        self.cop_density = cop_density
        self.citizen_vision = citizen_vision
        self.cop_vision = cop_vision
        self.legitimacy = legitimacy
        self.threshold = threshold
        self.max_jail_time = max_jail_time
        self.arrest_prob_constant = arrest_prob_constant
        self.movement = movement

        # Mesa 3.x uses agents collection instead of schedule
        self.grid = MultiGrid(width, height, torus=True)

        self.datacollector = DataCollector(
            model_reporters={
                "Quiescent": lambda m: self.count_type(m, "Quiescent"),
                "Active": lambda m: self.count_type(m, "Active"),
                "Jailed": lambda m: self.count_type(m, "Jailed"),
                "Total_Citizens": lambda m: self.count_type(m, "Citizen"),
                "Total_Cops": lambda m: self.count_type(m, "Cop")
            }
        )

        unique_id = 0

        # Create citizens
        if self.citizen_density + self.cop_density > 1:
            raise ValueError("Combined density cannot exceed 1")

        for x in range(self.width):
            for y in range(self.height):
                if self.random.random() < self.citizen_density:
                    # Create citizen with random attributes
                    hardship = self.random.random()
                    risk_aversion = self.random.random()

                    citizen = Citizen(unique_id, self, self.legitimacy, self.threshold,
                                    self.citizen_vision, self.arrest_prob_constant,
                                    hardship, risk_aversion)

                    self.grid.place_agent(citizen, (x, y))
                    unique_id += 1

                elif self.random.random() < self.cop_density:
                    # Create cop
                    cop = Cop(unique_id, self, self.cop_vision, self.max_jail_time)
                    self.grid.place_agent(cop, (x, y))
                    unique_id += 1

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        Run one step of the model.
        """
        # Get all agents from the grid
        agents = [agent for cell in self.grid.coord_iter() for agent in cell[0]]
        
        # All agents act (which includes moving)
        for agent in agents:
            if hasattr(agent, 'step'):
                agent.step()

        # Re-add jailed citizens whose time is up
        jailed_citizens = [agent for agent in agents
                          if isinstance(agent, Citizen) and agent.jail_sentence == 0
                          and agent.state.value == 3]

        for citizen in jailed_citizens:
            citizen.state = CitizenState.QUIET
            # Find an empty cell to place the citizen
            empty_cells = []
            for x in range(self.width):
                for y in range(self.height):
                    if self.grid.is_cell_empty((x, y)):
                        empty_cells.append((x, y))

            if empty_cells:
                new_pos = self.random.choice(empty_cells)
                self.grid.place_agent(citizen, new_pos)

        self.datacollector.collect(self)

    @staticmethod
    def count_type(model, agent_type):
        """
        Count agents by type.
        """
        # Get all agents from the grid
        agents = [agent for cell in model.grid.coord_iter() for agent in cell[0]]
        
        if agent_type == "Citizen":
            return len([agent for agent in agents
                       if isinstance(agent, Citizen)])
        elif agent_type == "Cop":
            return len([agent for agent in agents
                       if isinstance(agent, Cop)])
        elif agent_type == "Quiescent":
            return len([agent for agent in agents
                       if isinstance(agent, Citizen) and agent.state.value == 2])
        elif agent_type == "Active":
            return len([agent for agent in agents
                       if isinstance(agent, Citizen) and agent.state.value == 1])
        elif agent_type == "Jailed":
            return len([agent for agent in agents
                       if isinstance(agent, Citizen) and agent.jail_sentence > 0])
        else:
            return 0