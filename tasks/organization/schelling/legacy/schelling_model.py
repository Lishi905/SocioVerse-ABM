# Portions of this file are adapted from the Mesa Schelling segregation example (schelling/model.py),
# https://github.com/projectmesa/mesa (and mesa-examples).
# Copyright Core Mesa Team and contributors. Licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Modified by the
# SocioVerse2 Authors. See NOTICE in the repository root.

import mesa
import asyncio
import logging
from schelling_agent import SchellingAgent
from typing import Optional

logger = logging.getLogger(__name__)


class SchellingModel(mesa.Model):
    """
    Model class for the Schelling segregation model with LLM integration.
    Supports both synchronous and asynchronous decision making.
    """

    def __init__(self, height=200, width=200, density=0.8, minority_pc=0.2, homophily=5,
                 use_llm: bool = False, llm_decision_maker=None, llm_only_percentage: float = 1.0):
        """
        Create a new Schelling model.

        Args:
            height: Grid height
            width: Grid width
            density: What fraction of grid cells are occupied
            minority_pc: What fraction of agents are in the minority group
            homophily: How many similar neighbors agents want
            use_llm: Whether to use LLM for agent decisions
            llm_decision_maker: LLMDecisionMaker instance (required if use_llm=True)
            llm_only_percentage: Percentage of agents to use LLM (0.0-1.0)
        """
        super().__init__()
        self.height = height
        self.width = width
        self.density = density
        self.minority_pc = minority_pc
        self.homophily = homophily
        self.use_llm = use_llm
        self.llm_decision_maker = llm_decision_maker
        self.llm_only_percentage = llm_only_percentage

        self.grid = mesa.space.SingleGrid(width, height, torus=True)

        # Initialize schedule and agents list
        self.schedule = mesa.time.RandomActivation(self)
        self.agents = set()  # Keep track of all agents for compatibility

        self.happy = 0
        self.llm_agents_count = 0
        self.steps = 0  # Track the number of steps
        self.datacollector = mesa.DataCollector(
            model_reporters={"Happy": lambda m: m.happy, "LLMAgents": lambda m: m.llm_agents_count},
            agent_reporters={"x": lambda a: a.pos[0] if a.pos else None, "y": lambda a: a.pos[1] if a.pos else None}
        )

        # Set up agents
        agent_id = 0
        for x in range(self.width):
            for y in range(self.height):
                if self.random.random() < self.density:
                    if self.random.random() < self.minority_pc:
                        agent_type = 1
                    else:
                        agent_type = 0

                    # Decide if this agent should use LLM
                    agent_use_llm = False
                    if self.use_llm and self.random.random() < self.llm_only_percentage:
                        agent_use_llm = True
                        self.llm_agents_count += 1

                    agent = SchellingAgent(
                        agent_id,
                        self,
                        agent_type,
                        homophily,
                        use_llm=agent_use_llm,
                        llm_decision_maker=self.llm_decision_maker
                    )
                    agent_id += 1
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
                    self.agents.add(agent)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """Execute one step of the model with synchronous agent updates."""
        # Increment step counter
        self.steps += 1

        # Collect data
        self.datacollector.collect(self)

        # Reset happy count
        self.happy = 0

        # Shuffle agents to randomize evaluation order
        agents_copy = list(self.agents)
        self.random.shuffle(agents_copy)

        # Move dissatisfied agents sequentially
        for agent in agents_copy:
            agent.step()
        
        # Calculate happiness after all moves are done
        for agent in self.agents:
            agent.step_happiness()
            if agent.happy:
                self.happy += 1

    def step_with_concurrency(self):
        """Execute one step using async/concurrent LLM calls for better performance."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, use run_in_executor
                loop.run_until_complete(self.step_async())
            else:
                asyncio.run(self.step_async())
        except RuntimeError:
            # No event loop, create one
            asyncio.run(self.step_async())

    async def step_async(self):
        """Asynchronously execute one step with concurrent LLM agent processing."""
        # Increment step counter
        self.steps += 1

        # Collect data
        self.datacollector.collect(self)

        # Reset happy count
        self.happy = 0
        
        # Get all LLM agents
        llm_agents = [a for a in self.agents if a.use_llm]
        rule_based_agents = [a for a in self.agents if not a.use_llm]
        
        # If we have LLM agents, process them concurrently
        if llm_agents:
            logger.info(f"Processing {len(llm_agents)} LLM agents concurrently...")
            
            # Prepare batch requests
            batch_requests = SchellingAgent.prepare_batch_requests(
                llm_agents, self.grid, (self.width, self.height), self.homophily
            )
            
            try:
                # Make async batch decisions
                decisions = await self.llm_decision_maker.make_batch_decisions_async(batch_requests)
                
                # Apply decisions
                SchellingAgent.apply_batch_decisions(llm_agents, decisions, self.grid)
                logger.info(f"Applied decisions for {len(llm_agents)} LLM agents")
            except Exception as e:
                logger.error(f"Error in async LLM step: {e}")
                # Fall back to synchronous processing
                for agent in llm_agents:
                    agent.step()
        
        # Process rule-based agents sequentially
        self.random.shuffle(rule_based_agents)
        for agent in rule_based_agents:
            agent.step()
        
        # Calculate happiness
        for agent in self.agents:
            agent.step_happiness()
            if agent.happy:
                self.happy += 1

