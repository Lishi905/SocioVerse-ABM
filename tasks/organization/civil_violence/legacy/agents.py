# Portions of this file are adapted from the Mesa Epstein civil violence example (epstein_civil_violence/agents.py),
# https://github.com/projectmesa/mesa (and mesa-examples).
# Copyright Core Mesa Team and contributors. Licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Modified by the
# SocioVerse2 Authors. See NOTICE in the repository root.

from mesa import Agent
import math
from enum import Enum


class CitizenState(Enum):
    """Citizen states in the civil violence model."""
    ACTIVE = 1
    QUIET = 2
    ARRESTED = 3


class EpsteinAgent(Agent):
    """Base agent class with common functionality for neighbors and movement."""
    
    def __init__(self, unique_id, model):
        """Initialize the base agent."""
        super().__init__(unique_id, model)
        self.neighbors = []
        self.empty_neighbors = []
    
    def update_neighbors(self):
        """
        Look around and see who my neighbors are.
        """
        # Get neighborhood coordinates within vision
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, 
            moore=False,          # Moore neighborhood (8 directions if True)
            include_center=False, 
            radius=self.vision
        )

        # Get neighbor agents
        self.neighbors = self.model.grid.get_cell_list_contents(neighborhood)
        
        # Find empty neighbor coordinates
        self.empty_neighbors = [pos for pos in neighborhood 
                              if self.model.grid.is_cell_empty(pos)]

    def move(self):
        """Move to a random empty neighboring cell if movement is enabled."""
        if self.model.movement and self.empty_neighbors:
            new_pos = self.random.choice(self.empty_neighbors)
            self.model.grid.move_agent(self, new_pos)


class Citizen(EpsteinAgent):
    """
    A citizen agent for the civil violence model.
    
    Summary of rule: If grievance - risk > threshold, rebel.
    
    Attributes:
        hardship: Agent's perceived hardship (physical or economic privation). 
                  Exogenous, drawn from U(0,1).
        regime_legitimacy: Agent's perception of regime legitimacy, equal across agents.
        risk_aversion: Exogenous, drawn from U(0,1).
        threshold: if (grievance - (risk_aversion * arrest_probability)) > threshold, 
                   go/remain Active
        vision: number of cells in each direction that agent can inspect
        state: Can be QUIET, ACTIVE, or ARRESTED
        grievance: deterministic function of hardship and regime_legitimacy
        arrest_probability: agent's assessment of arrest probability, given rebellion
    """

    def __init__(self, unique_id, model, regime_legitimacy, threshold, vision, 
                 arrest_prob_constant, hardship=None, risk_aversion=None):
        """
        Create a new Citizen.
        
        Args:
            unique_id: Unique identifier for this agent
            model: The model instance
            regime_legitimacy: Agent's perception of regime legitimacy, equal across agents
            threshold: if (grievance - (risk_aversion * arrest_probability)) > threshold, 
                       go/remain Active
            vision: number of cells in each direction that agent can inspect
            arrest_prob_constant: constant for arrest probability calculation
            hardship: Agent's perceived hardship. If None, drawn from U(0,1)
            risk_aversion: Risk aversion level. If None, drawn from U(0,1)
        """
        super().__init__(unique_id, model)
        
        # Exogenous parameters
        self.hardship = hardship if hardship is not None else self.random.random()
        self.risk_aversion = risk_aversion if risk_aversion is not None else self.random.random()
        self.regime_legitimacy = regime_legitimacy
        self.threshold = threshold
        self.vision = vision
        self.arrest_prob_constant = arrest_prob_constant
        
        # State variables
        self.state = CitizenState.QUIET
        self.jail_sentence = 0
        
        # Calculated attributes
        self.grievance = self.hardship * (1 - self.regime_legitimacy)
        self.arrest_probability = None
        
        # Neighbor tracking
        self.neighborhood = []
        self.neighbors = []
        self.empty_neighbors = []

    def step(self):
        """
        Decide whether to activate, then move if applicable.
        """
        if self.jail_sentence:
            self.jail_sentence -= 1
            return  # no other changes or movements if agent is in jail.
        
        self.update_neighbors()
        self.update_estimated_arrest_probability()

        net_risk = self.risk_aversion * self.arrest_probability
        if (self.grievance - net_risk) > self.threshold:
            self.state = CitizenState.ACTIVE
        else:
            self.state = CitizenState.QUIET

        self.move()

    def update_estimated_arrest_probability(self):
        """
        Based on the ratio of cops to actives in my neighborhood, estimate the
        p(Arrest | I go active).
        """
        cops_in_vision = 0
        actives_in_vision = 1  # citizen counts herself
        
        for neighbor in self.neighbors:
            if isinstance(neighbor, Cop):
                cops_in_vision += 1
            elif isinstance(neighbor, Citizen) and neighbor.state == CitizenState.ACTIVE:
                actives_in_vision += 1

        # There is a body of literature on this equation
        # The round is not in the PNAS paper but without it, it's impossible to replicate
        # the dynamics shown there.
        self.arrest_probability = 1 - math.exp(
            -1 * self.arrest_prob_constant * round(cops_in_vision / actives_in_vision)
        )



class Cop(EpsteinAgent):
    """
    A cop agent for the civil violence model.
    
    Summary of rule: Inspect local vision and arrest a random active agent.
    
    Attributes:
        unique_id: unique int
        pos: Grid coordinates
        vision: number of cells in each direction that cop is able to inspect
        max_jail_term: maximum jail sentence that can be imposed
    """

    def __init__(self, unique_id, model, vision, max_jail_term):
        """
        Create a new Cop.
        
        Args:
            unique_id: Unique identifier for this agent
            model: The model instance
            vision: number of cells in each direction that agent can inspect
            max_jail_term: maximum jail sentence that can be imposed
        """
        super().__init__(unique_id, model)
        self.vision = vision
        self.max_jail_term = max_jail_term

    def step(self):
        """
        Inspect local vision and arrest a random active agent. Move if applicable.
        """
        self.update_neighbors()
        active_neighbors = []
        
        for agent in self.neighbors:
            if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE:
                active_neighbors.append(agent)
        
        if active_neighbors:
            arrestee = self.random.choice(active_neighbors)
            arrestee.jail_sentence = self.random.randint(0, self.max_jail_term)
            arrestee.state = CitizenState.ARRESTED

        self.move()