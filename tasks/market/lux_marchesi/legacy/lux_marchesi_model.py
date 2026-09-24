# -*- coding: utf-8 -*-
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
Lux-Marchesi Financial Market Model
"""

import numpy as np
import math
import random
import argparse
import json
import os
import sys
import queue
import threading
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any, Set
from random import choices
from tqdm import tqdm

# Ensure socioverse is importable (needed for worker thread)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)



class LuxMarchesiAgent:
    """
    Individual agent in the Lux-Marchesi model.
    
    Each agent has a type: 'fundamentalist', 'noise_optimist', or 'noise_pessimist'.
    Agents independently reconsider their type based on transition probabilities.
    
    In LLM mode, each agent can independently call LLM to decide whether to switch type.
    Decisions are cached for efficiency.
    """
    
    # Memory optimization - include LLM-related fields
    __slots__ = ['_id', '_type', 'llm_agent', 'cached_decision', 
                 'decision_expiry_step', 'last_llm_step', 'cached_switch_tendency']
    
    TYPES = ('fundamentalist', 'noise_optimist', 'noise_pessimist')
    
    # Valid LLM decisions
    VALID_DECISIONS = ('stay', 'switch_to_fundamentalist', 'switch_to_optimist', 'switch_to_pessimist')
    
    def __init__(self, agent_id: int, agent_type: str = 'fundamentalist'):
        self._id = agent_id
        self._type = agent_type if agent_type in self.TYPES else 'fundamentalist'
        # LLM-related state
        self.llm_agent = None  # Lazy initialization
        self.cached_decision = None  # Cached LLM decision: 'stay' or 'switch_to_X'
        self.cached_switch_tendency = 0.0  # Cached switch probability from LLM
        self.decision_expiry_step = -1  # Step at which cached decision expires
        self.last_llm_step = -1  # Last step when LLM was consulted
    
    @property
    def id(self) -> int:
        return self._id
    
    @property
    def type(self) -> str:
        return self._type
    
    @type.setter
    def type(self, value: str):
        if value in self.TYPES:
            self._type = value
    
    def reconsider(self, transition_probs: Dict[str, float], dt: float) -> bool:
        """
        Agent independently decides whether to switch type.
        
        Args:
            transition_probs: Dict with keys like 'pi_plus_minus', 'pi_f_plus', etc.
            dt: Time step size
            
        Returns:
            True if agent switched type, False otherwise
        """
        old_type = self._type
        
        if self._type == 'fundamentalist':
            # f -> optimist, f -> pessimist, or stay
            p_f_plus = dt * transition_probs['pi_f_plus']
            p_f_minus = dt * transition_probs['pi_f_minus']
            p_f_f = max(0, 1 - p_f_plus - p_f_minus)
            
            weights = [p_f_f, p_f_plus, p_f_minus]
            new_type = choices(self.TYPES, weights)[0]
            
        elif self._type == 'noise_optimist':
            # + -> f, + -> -, or stay
            p_plus_f = dt * transition_probs['pi_plus_f']
            p_plus_minus = dt * transition_probs['pi_plus_minus']
            p_plus_plus = max(0, 1 - p_plus_f - p_plus_minus)
            
            weights = [p_plus_f, p_plus_plus, p_plus_minus]
            new_type = choices(self.TYPES, weights)[0]
            
        elif self._type == 'noise_pessimist':
            # - -> f, - -> +, or stay
            p_minus_f = dt * transition_probs['pi_minus_f']
            p_minus_plus = dt * transition_probs['pi_minus_plus']
            p_minus_minus = max(0, 1 - p_minus_f - p_minus_plus)
            
            weights = [p_minus_f, p_minus_plus, p_minus_minus]
            new_type = choices(self.TYPES, weights)[0]
        else:  # derive from initial fractions
            new_type = self._type
        
        self._type = new_type
        return old_type != new_type
    
    # =========================================================================
    # LLM Per-Agent Decision Methods
    # =========================================================================
    
    def apply_llm_decision(self, decision: str, cache_duration: int, current_step: int) -> bool:
        """
        Apply and cache an LLM decision.
        
        Args:
            decision: One of 'stay', 'switch_to_fundamentalist', 'switch_to_optimist', 'switch_to_pessimist'
            cache_duration: Number of steps to cache this decision
            current_step: Current simulation step
            
        Returns:
            True if agent switched type, False otherwise
        """
        old_type = self._type
        self.cached_decision = decision
        self.decision_expiry_step = current_step + cache_duration
        self.last_llm_step = current_step
        
        # Apply the decision
        if decision == 'stay':
            return False
        elif decision == 'switch_to_fundamentalist' and self._type != 'fundamentalist':
            self._type = 'fundamentalist'
        elif decision == 'switch_to_optimist' and self._type != 'noise_optimist':
            self._type = 'noise_optimist'
        elif decision == 'switch_to_pessimist' and self._type != 'noise_pessimist':
            self._type = 'noise_pessimist'
        
        return old_type != self._type
    
    def has_valid_cached_decision(self, current_step: int) -> bool:
        """
        Check if the cached decision is still valid.
        
        Args:
            current_step: Current simulation step
            
        Returns:
            True if cached decision is valid, False otherwise
        
        Note:
            Using `<` instead of `<=` to avoid extending cache validity
            across decision interval boundaries when cache_duration == interval.
        """
        return (self.cached_decision is not None and 
                current_step < self.decision_expiry_step)
    
    def get_cached_decision(self) -> Optional[str]:
        """Get the cached decision if available."""
        return self.cached_decision
    
    def build_observation(self, model: 'LuxMarchesiModel') -> Dict[str, Any]:
        """
        Build observation dict for LLM decision making.
        
        Args:
            model: The parent LuxMarchesiModel instance
            
        Returns:
            Dict containing agent state and market context for LLM
        """
        p = model.params
        n_f, n_plus, n_minus = model._count_agents()
        nc = n_plus + n_minus
        
        # Compute market metrics
        trend = model.dp_dt / model.price if model.price > 0 else 0
        mispricing_ratio = abs(model.price - model.pf) / model.pf if model.pf > 0 else 0
        opinion_index = (n_plus - n_minus) / nc if nc > 0 else 0
        
        # Compute expected returns (matching ABM formulas)
        # Chartist return: dividend + capital gains from trend
        chartist_return = p.r + trend / p.v2 if p.v2 > 0 else p.r
        # Fundamentalist return: alternative return + discounted arbitrage profit
        fundamentalist_return = p.R + p.s * mispricing_ratio
        
        return {
            # Agent state
            "agent_id": self._id,
            "agent_type": self._type,
            
            # Market state
            "price": model.price,
            "pf": model.pf,
            "trend": trend,
            "mispricing_ratio": mispricing_ratio,
            
            # Population
            "N": p.N,
            "n_fundamentalists": n_f,
            "n_optimists": n_plus,
            "n_pessimists": n_minus,
            "n_chartists": nc,
            "opinion_index": opinion_index,
            
            # Returns
            "chartist_return": chartist_return,
            "fundamentalist_return": fundamentalist_return,
            
            # Model parameters (needed for prompt to explain dynamics)
            "alpha1": p.alpha1,
            "alpha2": p.alpha2,
            "v1": p.v1,
            "v2": p.v2,
            "dividend_yield": p.r,
            "alternative_return": p.R,
            "dt": p.dt,
        }


# =============================================================================
# Model Parameters
# =============================================================================

@dataclass
class LuxMarchesiParams:
    """
    Parameters for the Lux-Marchesi model.
    
    Default values follow Samanidou et al. (2007) "Agent-based Models of Financial Markets"
    and Lux & Marchesi (1999) "Scaling and criticality in a stochastic multi-agent model".
    """
    # Population
    # NOTE: For full compatibility with `Lux-Marchesi-ABM/` you may optionally
    # specify explicit initial populations (num_*). If provided, N will be
    # overridden to their sum (matching the legacy implementation).
    N: int = 500                    # Total number of agents
    num_fundamentalists: Optional[int] = None
    num_noise_optimist: Optional[int] = None
    num_noise_pessimist: Optional[int] = None
    
    # Time settings
    dt: float = 0.01                # Time step size
    T: int = 50000                  # Total number of time steps
    
    # Opinion switching parameters (Module 1: optimist <-> pessimist)
    # Following Samanidou et al. (2007) Table 1 / Lux-Marchesi standard
    v1: float = 2.0                 # Re-evaluation frequency
    alpha1: float = 0.6             # Herding effect weight (majority influence)
    alpha2: float = 1.5             # Trend-following weight (price momentum)
    
    # Strategy switching parameters (Module 2: chartist <-> fundamentalist)
    v2: float = 0.6                 # Re-evaluation frequency
    alpha3: float = 1.0             # Sensitivity to profit differential
    r: float = 0.0004               # Dividend yield
    R: float = 0.0004               # Return on alternative assets
    s: float = 0.75                 # Discount factor for fundamentalist returns
    
    # Price adjustment parameters (Module 3: Poisson-tick mechanism)
    beta: float = 4.0               # Price adjustment speed
    tc: float = 0.001               # Chartist trade volume per agent (legacy name: t_c)
    gamma: float = 0.01             # Fundamentalist sensitivity to mispricing
    # NOTE: `price_tick` is used as the legacy tick size `delta_p` in Lux-Marchesi-ABM.
    # Kept for backward compatibility with earlier revisions of this folder.
    price_tick: float = 0.001       # Price tick size / delta_p (relative)
    # Price update scheme:
    # - "walrasian": continuous log-price update (mean-field limit of the tick model)
    #       d log(p) ≈ Δp/p · (π↑ - π↓) · dt = price_tick · β · (ED + μ) · dt
    #   This mode avoids discrete-tick lumpiness and is usually best for
    #   reproducing stylized facts (fat tails / volatility clustering / tail fits)
    #   with finite simulation lengths.
    # - "poisson": compound Poisson ticks with intensity π*dt (allows 0/1/2.. ticks per step)
    # - "conditional_tick": always move ±1 tick each step (legacy-like; good for bubble/crash visuals)
    # - "bernoulli": 3-state {up,down,none} with probabilities π*dt (paper-like when dt is small)
    price_update_mode: str = "walrasian"
    
    # Noise parameters
    price_change_mu: float = 0.0    # Mean of price noise (excess demand)
    price_change_sigma: float = 0.05  # Std of price noise
    value_change_mu: float = 0.0    # Mean of fundamental value noise
    value_change_sigma: float = 0.005  # Std of fundamental value noise
    
    # Population bounds (prevent absorbing states)
    min_population_frac: float = 0.01  # Minimum fraction for any agent type
    
    # Initial conditions
    initial_price: float = 100.0
    initial_pf: float = 100.0
    initial_fundamentalist_frac: float = 1/3
    initial_optimist_frac: float = 1/3
    initial_pessimist_frac: float = 1/3
    
    # History settings
    # Paper (Section 9): dp/dt uses the average price change over the last 10 time increments.
    trend_lookback_steps: int = 10
    
    # ABM update scheme (paper alignment)
    # - "synchronous": all agents reconsider against the same frozen counts (default, legacy)
    # - "sequential": each agent's switch immediately updates running counts (closer to
    #   asynchronous Poisson updating; reduces artificial synchronicity)
    abm_update_scheme: str = "synchronous"
    
    # Burst sub-stepping (paper Section 9: Δt=0.01 normally, Δt=0.002 during bursts)
    # When instantaneous price-change activity is high, split the step for finer resolution.
    burst_substep_enabled: bool = False
    burst_substep_threshold: float = 0.1  # If (π↑+π↓)*dt > threshold, trigger sub-stepping
    burst_substep_factor: int = 5         # Split step into this many substeps (dt_sub = dt/factor)
    
    # Random seed
    seed: Optional[int] = None

    def __post_init__(self):
        """Lightweight validation + legacy parameter harmonization."""
        explicit = (self.num_fundamentalists, self.num_noise_optimist, self.num_noise_pessimist)
        if any(v is not None for v in explicit):
            if not all(isinstance(v, int) for v in explicit):
                raise ValueError("If any of num_fundamentalists/num_noise_optimist/num_noise_pessimist is set, all three must be set as ints.")
            n_f, n_o, n_p = explicit
            if n_f < 0 or n_o < 0 or n_p < 0:
                raise ValueError("Initial populations must be non-negative.")
            total = n_f + n_o + n_p
            if total <= 0:
                raise ValueError("Total population must be > 0.")
            # Match legacy implementation: N is derived, not independent.
            self.N = total
            # Keep fractions consistent for metadata/debugging (not used if explicit counts provided).
            self.initial_fundamentalist_frac = n_f / total
            self.initial_optimist_frac = n_o / total
            self.initial_pessimist_frac = n_p / total

        if self.price_update_mode not in {"walrasian", "poisson", "conditional_tick", "bernoulli"}:
            raise ValueError("price_update_mode must be one of: 'walrasian', 'poisson', 'conditional_tick', 'bernoulli'.")
        
        if self.abm_update_scheme not in {"synchronous", "sequential"}:
            raise ValueError("abm_update_scheme must be one of: 'synchronous', 'sequential'.")
        
        if self.burst_substep_factor < 1:
            raise ValueError("burst_substep_factor must be >= 1.")


class LuxMarchesiModel:
    """
    Lux-Marchesi Financial Market Model with explicit Agent objects.
    
    Implements the four core modules:
    1. Opinion switching (optimist <-> pessimist among chartists)
    2. Strategy switching (chartist <-> fundamentalist)
    3. Price dynamics based on excess demand
    4. Fundamental value evolution (random walk)
    """
    
    def __init__(self, params: LuxMarchesiParams, mode: str = "abm", 
                 llm_model: str = "gpt-4o-2024-08-06"):
        """
        Initialize the model.
        
        Args:
            params: Model parameters
            mode: "abm" for rule-based or "llm" for LLM-based decisions
            llm_model: LLM model name for LLM mode
        """
        self.params = params
        self.mode = mode
        self.llm_model = llm_model

        # Randomness:
        # - Use a dedicated NumPy Generator for the "main" stream (fundamental + μ noise),
        #   so that adding extra random draws elsewhere (e.g., Poisson ticks) does not
        #   inadvertently change the fundamental-value path.
        # - Use a second Generator for auxiliary draws (e.g., Poisson tick counts).
        if params.seed is not None:
            self._np_rng_main = np.random.default_rng(params.seed)
            self._np_rng_aux = np.random.default_rng(params.seed + 1)
            random.seed(params.seed)
        else:
            self._np_rng_main = np.random.default_rng()
            self._np_rng_aux = np.random.default_rng()
        
        # Initialize agents and state
        self._init_agents()
        self._init_state()
        
        # =====================================================================
        # LLM Per-Agent Scheduling Parameters
        # =====================================================================
        self.llm_decision_tick_interval = 50   # Trigger LLM decisions every N steps
        self.llm_cache_duration = 100          # Cache decisions for N steps
        self.max_llm_agents_per_tick = 20      # Max agents to process per round
        self.min_seconds_between_requests = 0.5  # Rate limiting
        self._last_llm_request_time = 0.0
        self._llm_agent_index = 0              # Round-robin index for agent selection
        
        # Synchronous mode: wait for LLM responses before continuing simulation
        # Set to True for accurate per-agent-per-step LLM control (slower)
        # Set to False for async mode where responses may arrive in future steps
        self.llm_sync_mode = True              # Default: sync mode for accurate control
        self.llm_sync_timeout = 120.0          # Max seconds to wait per batch
        
        # Concurrency control for API calls (防止大并发触发 429/timeout)
        self.llm_concurrent_batch_size = 10    # Max concurrent API calls within a batch
        self.llm_batch_delay = 0.5             # Seconds to wait between sub-batches
        
        # LLM async processing system
        self.llm_request_queue: queue.Queue = queue.Queue(maxsize=200)
        self.llm_response_queue: queue.Queue = queue.Queue()
        self.llm_worker_thread: Optional[threading.Thread] = None
        self.llm_pending_requests: Set[int] = set()  # Agent IDs with pending requests
        
        # LLM control statistics (with error classification)
        self.llm_control_stats = {
            "total_decisions": 0,
            "llm_decisions": 0,
            "cache_hits": 0,
            "errors": 0,
            "api_errors": 0,        # 429/timeout/connection errors
            "parse_errors": 0,      # JSON parsing failures
            "validation_errors": 0, # Invalid decision values
            "requests_sent": 0,
            "responses_received": 0,
            "switches_blocked_by_min_pop": 0,  # Switches prevented by population constraint
        }
        
        # LLM usage tracking (for recording API calls and token usage)
        self.llm_usage_records: List[Dict[str, Any]] = []  # Individual call records
        self.llm_usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "num_requests": 0,
            "num_successful": 0,
            "num_failed": 0,
        }
        
        # Debug flag
        self.llm_debug = False
        
        # LLM API parameters (can be configured before run())
        self.llm_temperature = 0.0      # Lower temperature = more deterministic
        self.llm_use_json_mode = True   # Use OpenAI JSON response format
        self.llm_max_retries = 2        # Retry failed API calls
        
        # Initialize LLM mode (parameters will be configured by caller before run())
        if mode == "llm":
            # Note: Don't print default params here - they will be overridden before run()
            # Start background worker thread
            self._start_llm_worker()
        
        # Statistics tracking
        self.stats = {
            "opinion_switches": 0,
            "strategy_switches": 0,
            "price_up_events": 0,
            "price_down_events": 0,
            "llm_calls": 0,
            "llm_decisions": 0,
            "llm_cache_hits": 0,
            "llm_errors": 0,
            "switches_blocked_by_min_pop": 0,  # Switches prevented by min_population_frac constraint
            "abm_fallback_switches": 0,  # Switches made by ABM fallback logic (agents without LLM decisions)
        }
    
    def _init_agents(self):
        """Initialize the list of Agent objects."""
        p = self.params
        
        # Calculate initial counts
        if p.num_fundamentalists is not None:
            # Legacy-compatible explicit initialization (matches Lux-Marchesi-ABM)
            n_fundamentalists = int(p.num_fundamentalists)
            n_optimists = int(p.num_noise_optimist)
            n_pessimists = int(p.num_noise_pessimist)
        else:
            n_fundamentalists = int(p.initial_fundamentalist_frac * p.N)
            n_optimists = int(p.initial_optimist_frac * p.N)
            n_pessimists = p.N - n_fundamentalists - n_optimists
        
        # Create agent list (ORDER MATTERS for exact legacy reproducibility).
        # Lux-Marchesi-ABM creates N default fundamentalists, then assigns
        # the first block to optimists and the second block to pessimists:
        #   [optimists][pessimists][fundamentalists]
        self.agents: List[LuxMarchesiAgent] = [LuxMarchesiAgent(i, 'fundamentalist') for i in range(p.N)]

        for i in range(n_optimists):
            self.agents[i].type = 'noise_optimist'
        for i in range(n_pessimists):
            self.agents[i + n_optimists].type = 'noise_pessimist'

        # Cache populations for efficiency (kept consistent during simulation)
        self.n_f = n_fundamentalists
        self.n_plus = n_optimists
        self.n_minus = n_pessimists
    
    def _init_state(self):
        """Initialize model state variables."""
        p = self.params
        
        # Price and fundamental value
        self.price = p.initial_price
        self.pf = p.initial_pf
        
        # Price history for trend estimation (legacy dp/dt uses this)
        self.price_history = [self.price]
        
        # Current time step
        self.current_step = 0
        
        # Cached transition probabilities (updated each step)
        self.transition_probs = {}
        
        # dp/dt estimation
        self.dp_dt = 0.0
        
        # Price tick mechanism bookkeeping (legacy compatibility)
        self.pi_price_up = 0.0
        self.pi_price_down = 0.0
        self.price_change = 0.0
        self.value_change = 0.0

        # Full history for analysis (includes legacy fields for easier comparison)
        self.history = {
            "price": [],
            "pf": [],
            "nc": [],
            "nf": [],
            "n_plus": [],
            "n_minus": [],
            "dp_dt": [],
            "excess_demand": [],
            "pi_plus_minus": [],
            "pi_minus_plus": [],
            "pi_plus_f": [],
            "pi_f_plus": [],
            "pi_minus_f": [],
            "pi_f_minus": [],
            "pi_price_up": [],
            "pi_price_down": [],
            "price_change": [],
            "value_change": [],
            "burst_substeps": [],  # Track when burst sub-stepping was used (0 or n_substeps)
        }
        
        # Track burst sub-stepping usage (separate from per-step flag)
        self._current_burst_substeps = 0

        # Record initial state (t=0), matching Lux-Marchesi-ABM which stores step-0 snapshot
        self._record_history(excess_demand=0.0, transition_probs={
            "pi_plus_minus": 0.0,
            "pi_minus_plus": 0.0,
            "pi_plus_f": 0.0,
            "pi_f_plus": 0.0,
            "pi_minus_f": 0.0,
            "pi_f_minus": 0.0,
        })

    def _record_history(self, excess_demand: float, transition_probs: Dict[str, float]):
        """Append current state to history arrays."""
        self.history["price"].append(self.price)
        self.history["pf"].append(self.pf)
        self.history["nf"].append(self.n_f)
        self.history["n_plus"].append(self.n_plus)
        self.history["n_minus"].append(self.n_minus)
        self.history["nc"].append(self.n_plus + self.n_minus)
        self.history["dp_dt"].append(self.dp_dt)
        self.history["excess_demand"].append(excess_demand)

        self.history["pi_plus_minus"].append(float(transition_probs.get("pi_plus_minus", 0.0)))
        self.history["pi_minus_plus"].append(float(transition_probs.get("pi_minus_plus", 0.0)))
        self.history["pi_plus_f"].append(float(transition_probs.get("pi_plus_f", 0.0)))
        self.history["pi_f_plus"].append(float(transition_probs.get("pi_f_plus", 0.0)))
        self.history["pi_minus_f"].append(float(transition_probs.get("pi_minus_f", 0.0)))
        self.history["pi_f_minus"].append(float(transition_probs.get("pi_f_minus", 0.0)))

        self.history["pi_price_up"].append(float(self.pi_price_up))
        self.history["pi_price_down"].append(float(self.pi_price_down))
        self.history["price_change"].append(float(self.price_change))
        self.history["value_change"].append(float(self.value_change))
        self.history["burst_substeps"].append(int(self._current_burst_substeps))

    # =========================================================================
    # LLM Per-Agent Async Processing
    # =========================================================================
    
    def _start_llm_worker(self):
        """Start the background LLM worker thread."""
        if self.llm_worker_thread is not None and self.llm_worker_thread.is_alive():
            return
        
        self.llm_worker_thread = threading.Thread(
            target=self._llm_worker_loop, 
            daemon=True,
            name="LuxMarchesi-LLM-Worker"
        )
        self.llm_worker_thread.start()
        if self.llm_debug:
            print("[LLM Worker] Background thread started")
    
    def _llm_worker_loop(self):
        """Background thread for processing LLM API calls with async concurrency."""
        import asyncio
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_worker_main())
        finally:
            loop.close()
    
    async def _async_worker_main(self):
        """Main async worker loop - collects batches and processes them with concurrency control.
        
        Key optimization: Uses semaphore + sub-batching to prevent overwhelming the API
        with too many concurrent requests, which would trigger 429/timeout errors.
        """
        import asyncio
        from openai import AsyncOpenAI
        from socioverse.behavior_engine.LLM_based import parse_attributes
        
        # Initialize async OpenAI client
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        async_client = AsyncOpenAI(api_key=openai_key, base_url=openai_base)
        
        # Semaphore for controlling concurrent API calls
        semaphore = asyncio.Semaphore(self.llm_concurrent_batch_size)
        
        if self.llm_debug:
            print(f"[LLM Worker] Async worker started with concurrency limit={self.llm_concurrent_batch_size}")
        
        while True:
            try:
                # Collect all available requests into a batch
                batch = []
                try:
                    # First request - blocking wait
                    request = self.llm_request_queue.get(timeout=1.0)
                    if request is None:  # Shutdown signal
                        if self.llm_debug:
                            print("[LLM Worker] Shutdown signal received")
                        break
                    batch.append(request)
                    
                    # Collect more requests if available (non-blocking)
                    while not self.llm_request_queue.empty():
                        try:
                            request = self.llm_request_queue.get_nowait()
                            if request is None:
                                break
                            batch.append(request)
                        except queue.Empty:
                            break
                        
                except queue.Empty:
                    continue
                
                if not batch:
                    continue
                
                batch_size = len(batch)
                concurrent_limit = self.llm_concurrent_batch_size
                
                if self.llm_debug:
                    print(f"[LLM Worker] Processing batch of {batch_size} requests "
                          f"(concurrent limit={concurrent_limit})...")
                
                start_time = time.time()
                
                # Split batch into sub-batches for controlled concurrency
                # This prevents overwhelming the API with too many simultaneous requests
                if batch_size <= concurrent_limit:
                    # Small batch: process all at once
                    tasks = [
                        self._process_single_request_with_semaphore(
                            async_client, semaphore, agent_id, observation
                        )
                        for agent_id, observation in batch
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # Large batch: process in sub-batches with delays between them
                    results = []
                    num_sub_batches = (batch_size + concurrent_limit - 1) // concurrent_limit
                    
                    for i in range(0, batch_size, concurrent_limit):
                        sub_batch = batch[i:i + concurrent_limit]
                        sub_batch_idx = i // concurrent_limit + 1
                        
                        if self.llm_debug and num_sub_batches > 1:
                            print(f"[LLM Worker] Sub-batch {sub_batch_idx}/{num_sub_batches} "
                                  f"({len(sub_batch)} requests)")
                        
                        tasks = [
                            self._process_single_request_with_semaphore(
                                async_client, semaphore, agent_id, observation
                            )
                            for agent_id, observation in sub_batch
                        ]
                        sub_results = await asyncio.gather(*tasks, return_exceptions=True)
                        results.extend(sub_results)
                        
                        # Add delay between sub-batches to avoid rate limiting
                        if i + concurrent_limit < batch_size and self.llm_batch_delay > 0:
                            await asyncio.sleep(self.llm_batch_delay)
                
                elapsed = time.time() - start_time
                if self.llm_debug:
                    print(f"[LLM Worker] Batch completed in {elapsed:.2f}s "
                          f"(avg {elapsed/batch_size:.2f}s/agent)")
                
                # Put results in response queue
                from datetime import datetime
                for (agent_id, observation), result in zip(batch, results):
                    if isinstance(result, Exception):
                        self.llm_response_queue.put((agent_id, None, str(result), None))
                        self.llm_control_stats["errors"] += 1
                        # Classify error type
                        error_str = str(result).lower()
                        if "429" in error_str or "rate" in error_str or "timeout" in error_str:
                            self.llm_control_stats["api_errors"] += 1
                        self.llm_usage_total["num_failed"] += 1
                        # Record failed request
                        error_record = {
                            "timestamp": datetime.now().isoformat(),
                            "agent_id": agent_id,
                            "step": self.current_step,
                            "agent_type": observation.get("agent_type", "unknown"),
                            "observation": observation,
                            "error": str(result),
                            "success": False,
                        }
                        self.llm_usage_records.append(error_record)
                        if self.llm_debug:
                            print(f"[LLM Worker ERROR] Agent {agent_id}: {result}")
                    else:
                        decision_data, response_text, usage_record = result
                        # decision_data is now a dict with 'decision' and 'switch_tendency'
                        self.llm_response_queue.put((agent_id, decision_data, None, usage_record))
                        self.llm_control_stats["responses_received"] += 1
                        
                        # Update total usage statistics
                        if usage_record and "usage" in usage_record:
                            usage = usage_record["usage"]
                            self.llm_usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                            self.llm_usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                            self.llm_usage_total["total_tokens"] += usage.get("total_tokens", 0)
                        self.llm_usage_total["num_requests"] += 1
                        self.llm_usage_total["num_successful"] += 1
                        
                        # Record successful request
                        usage_record["success"] = True
                        self.llm_usage_records.append(usage_record)
                        
                        if self.llm_debug:
                            decision = decision_data.get("decision", "unknown")
                            tendency = decision_data.get("switch_tendency", 0)
                            print(f"[LLM Worker] Agent {agent_id}: {decision} (tendency={tendency:.2f})")
                    
                    self.llm_request_queue.task_done()
                    
            except Exception as e:
                if self.llm_debug:
                    print(f"[LLM Worker ERROR] Unexpected error: {e}")
                continue
    
    async def _process_single_request_with_semaphore(
        self, async_client, semaphore, agent_id: int, observation: Dict
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Wrapper that uses semaphore to control concurrency.
        
        This ensures we don't exceed the concurrent request limit even when
        using asyncio.gather on a large batch.
        """
        import asyncio
        async with semaphore:
            return await self._process_single_request_async(async_client, agent_id, observation)
    
    async def _process_single_request_async(self, async_client, agent_id: int, observation: Dict) -> Tuple[str, str, Dict[str, Any]]:
        """Process a single LLM request asynchronously.
        
        Returns:
            Tuple of (decision_data, response_text, usage_record)
            decision_data is a dict with 'decision' and 'switch_tendency' keys
            usage_record follows OpenAI response format with additional metadata
        """
        import json
        import re
        from datetime import datetime
        from socioverse.behavior_engine.LLM_based import parse_attributes
        
        # Build prompt
        prompt = parse_attributes("Lux_Marchesi_agent", observation)
        sys_prompt = "You are a helpful assistant simulating bounded rational behavior in a financial market."
        
        # Record request timestamp
        request_timestamp = datetime.now().isoformat()
        
        # Call OpenAI API asynchronously with lower temperature for more consistent outputs
        response = await async_client.chat.completions.create(
            model=self.llm_model,
            max_tokens=512,
            temperature=self.llm_temperature,  # Use configurable temperature (default 0.0)
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Record response timestamp
        response_timestamp = datetime.now().isoformat()
        
        response_text = response.choices[0].message.content
        
        # Build usage record following OpenAI response format
        usage_record = {
            "id": response.id if hasattr(response, 'id') else None,
            "object": response.object if hasattr(response, 'object') else "chat.completion",
            "created": response.created if hasattr(response, 'created') else None,
            "model": response.model if hasattr(response, 'model') else self.llm_model,
            "request_timestamp": request_timestamp,
            "response_timestamp": response_timestamp,
            "agent_id": agent_id,
            "step": self.current_step,
            "agent_type": observation.get("agent_type", "unknown"),
            "observation": observation,  # Include full observation for debugging
            "prompt": prompt,
            "response": response_text,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            } if hasattr(response, 'usage') and response.usage else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "choices": [
                {
                    "index": choice.index if hasattr(choice, 'index') else i,
                    "message": {
                        "role": choice.message.role if hasattr(choice.message, 'role') else "assistant",
                        "content": choice.message.content if hasattr(choice.message, 'content') else "",
                    },
                    "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else None,
                }
                for i, choice in enumerate(response.choices)
            ] if hasattr(response, 'choices') else [],
        }
        
        # Parse JSON from response
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if not matches:
            usage_record["parse_error"] = f"No JSON found in response: {response_text[:100]}"
            self.llm_control_stats["parse_errors"] += 1
            raise ValueError(f"No JSON found in response: {response_text[:100]}")
        
        result = json.loads(matches[-1])
        decision = result.get("decision", "stay")
        
        # Parse switch_tendency (new probabilistic field)
        # Default to 0.5 if not provided (uncertain)
        switch_tendency = result.get("switch_tendency", 0.5)
        try:
            switch_tendency = float(switch_tendency)
            switch_tendency = max(0.0, min(1.0, switch_tendency))  # Clamp to [0, 1]
        except (TypeError, ValueError):
            switch_tendency = 0.5
        
        # Validate decision
        if decision not in LuxMarchesiAgent.VALID_DECISIONS:
            usage_record["validation_warning"] = f"Invalid decision '{decision}', defaulting to 'stay'"
            self.llm_control_stats["validation_errors"] += 1
            decision = "stay"
            switch_tendency = 0.0
        
        # Build decision data dict
        decision_data = {
            "decision": decision,
            "switch_tendency": switch_tendency,
            "reasoning": result.get("reasoning", ""),
        }
        
        usage_record["parsed_decision"] = decision
        usage_record["switch_tendency"] = switch_tendency
        
        return decision_data, response_text, usage_record
    
    def _request_llm_decisions(self):
        """
        Select a batch of agents and request LLM decisions.
        Uses round-robin selection to ensure all agents get LLM coverage over time.
        """
        current_time = time.time()
        
        # Rate limiting
        if current_time - self._last_llm_request_time < self.min_seconds_between_requests:
            return
        
        # Check queue capacity - ensure there's room for at least some requests
        queue_size = self.llm_request_queue.qsize()
        available_slots = self.llm_request_queue.maxsize - queue_size
        if available_slots <= 0:
            if self.llm_debug:
                print(f"[LLM] Queue full ({queue_size}/{self.llm_request_queue.maxsize}), skipping request")
            return
        
        # Select agents for this batch using round-robin
        agents_to_process = []
        n_agents = len(self.agents)
        
        # Limit batch size by both max_agents_per_tick and available queue slots
        batch_size = min(self.max_llm_agents_per_tick, n_agents, available_slots)
        
        for _ in range(batch_size):
            agent = self.agents[self._llm_agent_index]
            
            # Skip if already has pending request
            if agent.id not in self.llm_pending_requests:
                # Skip if has valid cached decision
                if not agent.has_valid_cached_decision(self.current_step):
                    agents_to_process.append(agent)
            
            # Advance round-robin index
            self._llm_agent_index = (self._llm_agent_index + 1) % n_agents
        
        # Enqueue requests
        for agent in agents_to_process:
            try:
                observation = agent.build_observation(self)
                self.llm_request_queue.put_nowait((agent.id, observation))
                self.llm_pending_requests.add(agent.id)
                self.llm_control_stats["requests_sent"] += 1
                
                if self.llm_debug:
                    print(f"[LLM] Enqueued request for agent {agent.id}")
                    
            except queue.Full:
                if self.llm_debug:
                    print(f"[LLM] Queue full, stopping at agent {agent.id}")
                break
        
        self._last_llm_request_time = current_time
    
    def _process_llm_responses(self) -> int:
        """
        Process all pending LLM responses from the queue.
        
        Key optimization (matching ABM behavior):
        - Uses probabilistic sampling based on switch_tendency * dt
        - Applies min_population_frac constraint to prevent absorbing states
        - This "soft switching" prevents the winner-take-all phase collapse
        
        The ABM uses π * dt as probability, where π = v * (n/N) * exp(U).
        We use switch_tendency * dt as an approximation, where switch_tendency
        captures the LLM's assessment of utility differential U.
        
        Returns:
            Number of type switches that occurred
        """
        switches = 0
        p = self.params
        min_pop = int(p.min_population_frac * p.N)
        
        while not self.llm_response_queue.empty():
            try:
                agent_id, decision_data, error, usage_record = self.llm_response_queue.get_nowait()
                
                # Remove from pending set
                self.llm_pending_requests.discard(agent_id)
                
                if error is not None:
                    # LLM call failed - agent keeps current type
                    self.stats["llm_errors"] += 1
                    if self.llm_debug:
                        print(f"[LLM] Agent {agent_id} error: {error}")
                    continue
                
                # Extract decision and switch_tendency from decision_data
                decision = decision_data.get("decision", "stay") if isinstance(decision_data, dict) else decision_data
                switch_tendency = decision_data.get("switch_tendency", 0.5) if isinstance(decision_data, dict) else 0.5
                
                agent = self.agents[agent_id]
                old_type = agent.type
                
                # Probabilistic switching (matching ABM's soft switching behavior)
                # ABM formula: p_switch = dt * π, where π = v * (n/N) * exp(U)
                # 
                # KEY INSIGHT: ABM samples EVERY STEP, LLM samples at decision intervals.
                # To match ABM behavior over a time window:
                #   - ABM: window_steps * dt * π expected switches
                #   - LLM: should have same expected switches
                #
                # If cache_duration > 0: Decision is cached and sampled every step
                #   → Use per-step probability: p = dt * v * (n/N) * exp_U
                # If cache_duration == 0: Decision is sampled ONCE per LLM call
                #   → Use accumulated probability for the interval: p = interval * dt * v * (n/N) * exp_U
                #
                # LLM's switch_tendency (0-1) maps to exp(U) effect:
                #   - tendency=0: no desire to switch → exp(U) ≈ 0.1
                #   - tendency=0.5: moderate → exp(U) ≈ 1
                #   - tendency=1.0: strong → exp(U) ≈ 10
                should_switch = False
                if decision != 'stay':
                    n_f, n_plus, n_minus = self._count_agents()
                    
                    # Get population fraction (source group)
                    if old_type == 'fundamentalist':
                        pop_fraction = n_f / p.N
                    elif old_type == 'noise_optimist':
                        pop_fraction = n_plus / p.N
                    else:
                        pop_fraction = n_minus / p.N
                    
                    # Map switch_tendency [0, 1] to exp(U) equivalent [0.1, 10]
                    # Using exponential mapping: exp(U) = 0.1 * 100^tendency = 0.1 to 10
                    exp_U_equivalent = 0.1 * math.pow(100, switch_tendency)
                    
                    # Determine which v to use based on decision type
                    if decision in ['switch_to_optimist', 'switch_to_pessimist'] and old_type in ['noise_optimist', 'noise_pessimist']:
                        v_rate = p.v1  # Opinion switching
                    else:
                        v_rate = p.v2  # Strategy switching
                    
                    # Base per-step probability (ABM-consistent)
                    p_per_step = p.dt * v_rate * pop_fraction * exp_U_equivalent
                    
                    # If cache_duration > 0, we'll sample every step → use p_per_step
                    # If cache_duration == 0, sample ONCE → accumulate over decision interval
                    if self.llm_cache_duration > 0:
                        effective_probability = p_per_step
                    else:
                        # One-shot probability: accumulate expected switches over interval
                        # P(at least one switch in N steps) = 1 - (1-p)^N ≈ N*p for small p
                        n_steps = self.llm_decision_tick_interval
                        effective_probability = min(1.0, n_steps * p_per_step)
                    
                    # Cap probability at 1.0
                    effective_probability = min(effective_probability, 1.0)
                    
                    # Sample switch decision
                    should_switch = random.random() < effective_probability
                    
                    if self.llm_debug and switch_tendency > 0.3:
                        print(f"[LLM Prob] Agent {agent_id}: tendency={switch_tendency:.2f}, "
                              f"exp_U={exp_U_equivalent:.2f}, p_step={p_per_step:.4f}, "
                              f"eff_prob={effective_probability:.4f}, should_switch={should_switch}")
                
                # Check min_pop constraint before applying decision (matches ABM behavior)
                can_switch = True
                if should_switch:
                    # Check if switching would violate min_pop constraint
                    if old_type == 'fundamentalist' and n_f <= min_pop:
                        can_switch = False
                    elif old_type == 'noise_optimist' and n_plus <= min_pop:
                        can_switch = False
                    elif old_type == 'noise_pessimist' and n_minus <= min_pop:
                        can_switch = False
                    
                    if not can_switch:
                        self.stats["switches_blocked_by_min_pop"] += 1
                        self.llm_control_stats["switches_blocked_by_min_pop"] += 1
                        if self.llm_debug:
                            print(f"[LLM] Agent {agent_id}: switch blocked by min_pop constraint "
                                  f"(type={old_type}, n_f={n_f}, n+={n_plus}, n-={n_minus}, min={min_pop})")
                
                # Cache the decision (including switch_tendency for later use)
                agent.cached_decision = decision
                agent.cached_switch_tendency = switch_tendency if hasattr(agent, 'cached_switch_tendency') else switch_tendency
                agent.decision_expiry_step = self.current_step + self.llm_cache_duration
                agent.last_llm_step = self.current_step
                
                if should_switch and can_switch:
                    # Apply the switch
                    switched = agent.apply_llm_decision(
                        decision, 
                        self.llm_cache_duration, 
                        self.current_step
                    )
                    
                    if switched:
                        switches += 1
                        if self.llm_debug:
                            print(f"[LLM] Agent {agent_id}: {old_type} -> {agent.type} (probabilistic switch)")
                
                self.stats["llm_decisions"] += 1
                self.llm_control_stats["llm_decisions"] += 1
                
                self.llm_response_queue.task_done()
                
            except queue.Empty:
                break
        
        return switches
    
    def _count_agents(self) -> Tuple[int, int, int]:
        """Count agents by type. O(N) operation."""
        n_f = n_plus = n_minus = 0
        for agent in self.agents:
            if agent.type == 'fundamentalist':
                n_f += 1
            elif agent.type == 'noise_optimist':
                n_plus += 1
            else:  # noise_pessimist
                n_minus += 1
        return n_f, n_plus, n_minus
    
    def _compute_dp_dt(self, iteration: int) -> float:
        """
        Compute dp/dt as average price change over the last `trend_lookback_steps` increments.

        Paper (Section 9): "For the time derivative, dp/dt, the average of the price
        changes during the interval [t − 10Δt, t] has been used."

        Formula:
            dp/dt = (p_t - p_{t-k}) / (k * dt)   where k = trend_lookback_steps

        For early iterations (t < k), we use all available history:
            dp/dt = (p_t - p_0) / (t * dt)

        Args:
            iteration: 1-based iteration index for the current step.
        """
        if iteration <= 0:
            return 0.0

        p = self.params
        k = p.trend_lookback_steps  # Default: 10 (paper)

        if iteration < k:
            # Early phase: use all available history
            return (self.price - self.price_history[0]) / (iteration * p.dt)

        # Normal phase: lookback k steps
        idx = iteration - k
        if idx < 0 or idx >= len(self.price_history):
            idx = max(0, len(self.price_history) - 1)
        return (self.price - self.price_history[idx]) / (k * p.dt)
    
    def _compute_U1(self, n_plus: int, n_minus: int) -> float:
        """
        Compute U1 utility differential for opinion switching.
        
        Following Samanidou et al. (2007) Eq. 17:
        U₁ = α₁x + (α₂/v₁)(dp/dt)/p
        
        where x = (n₊ - n₋)/n_c is the opinion index.
        
        Interpretation:
        - α₁x: Majority effect (herding towards dominant opinion)
        - (α₂/v₁)(dp/dt)/p: Trend following (follow price momentum)
        
        When U1 > 0: Optimism dominates (more optimists or rising prices)
        When U1 < 0: Pessimism dominates (more pessimists or falling prices)
        
        Transition rates (Eq. 17):
        - π_{+-} = v₁(n_c/N)exp(-U₁) : optimist → pessimist
        - π_{-+} = v₁(n_c/N)exp(U₁)  : pessimist → optimist
        
        This creates HERDING: when optimism dominates (U₁>0), pessimists are
        more likely to switch to optimism and optimists are less likely to switch
        to pessimism.
        """
        p = self.params
        nc = n_plus + n_minus
        
        if nc == 0:
            return 0.0
        
        x = (n_plus - n_minus) / nc
        # (α₂/v₁)(dp/dt)/p: trend following term
        trend_effect = p.alpha2 * self.dp_dt / (p.v1 * self.price)
        return p.alpha1 * x + trend_effect
    
    def _compute_U21(self) -> float:
        """
        Compute U21 utility differential for optimist strategy switching.

        Samanidou et al. (2007) / Lux & Marchesi (1999) profit differential
        (cf. Eq. (20) in the review):
            U2,1 = α3 * [ ( (r + (1/v2) * dp/dt) / p ) - R - s * |(pf - p)/p| ]

        Notes:
        - `r` is treated as a dividend *yield*, so nominal dividend is `r * pf`.
          (The review sets r/pf = R, i.e. r_yield ≈ R.)
        - The dp/dt term is scaled by 1/v2 to reflect average price change over
          the strategy revision interval.
        """
        p = self.params
        if self.price <= 0 or self.pf <= 0:
            return 0.0
        # Nominal dividend (review uses r/pf = R; we keep r configurable as yield)
        r_nominal = p.r * self.pf
        chartist_revenue_per_asset = (r_nominal + (self.dp_dt / max(p.v2, 1e-12))) / self.price
        fundamentalist_excess_profit = p.s * abs((self.pf - self.price) / self.price)
        return p.alpha3 * (chartist_revenue_per_asset - p.R - fundamentalist_excess_profit)
    
    def _compute_U22(self) -> float:
        """
        Compute U22 utility differential for pessimist strategy switching.

        Samanidou et al. (2007) / Lux & Marchesi (1999) profit differential
        (cf. Eq. (21) in the review), from the perspective of pessimistic chartists:
            U2,2 = α3 * [ R - ( (r + (1/v2) * dp/dt) / p ) - s * |(pf - p)/p| ]
        """
        p = self.params
        if self.price <= 0 or self.pf <= 0:
            return 0.0
        r_nominal = p.r * self.pf
        chartist_revenue_per_asset = (r_nominal + (self.dp_dt / max(p.v2, 1e-12))) / self.price
        fundamentalist_excess_profit = p.s * abs((self.pf - self.price) / self.price)
        return p.alpha3 * (p.R - chartist_revenue_per_asset - fundamentalist_excess_profit)
    
    def _compute_transition_probs(self, n_f: int, n_plus: int, n_minus: int) -> Dict[str, float]:
        """
        Compute all transition probabilities.
        
        Args:
            n_f: Number of fundamentalists
            n_plus: Number of optimists
            n_minus: Number of pessimists
            
        Returns:
            Dictionary of transition rates
        """
        p = self.params
        N = p.N
        nc = n_plus + n_minus
        
        # Compute utility differentials
        U1 = self._compute_U1(n_plus, n_minus)
        U21 = self._compute_U21()
        U22 = self._compute_U22()
        
        # Opinion switching rates (optimist <-> pessimist)
        # Samanidou et al. (2007) Eq. (17): probabilities in dt are π*dt with
        #   π_{+-} = v1 * (nc/N) * exp(-U1)   (+ -> -)
        #   π_{-+} = v1 * (nc/N) * exp(U1)    (- -> +)
        pi_plus_minus = p.v1 * nc / N * math.exp(-U1)    # + -> -
        pi_minus_plus = p.v1 * nc / N * math.exp(U1)     # - -> +
        
        # Strategy switching rates (chartist <-> fundamentalist)
        #
        # KEY FIX: Use SOURCE GROUP scaling instead of destination group scaling.
        #
        # The original Samanidou et al. (2007) Eq. (18)-(19) uses destination group
        # scaling (e.g., π_{+f} ~ n_+/N for f→+), which creates a positive feedback
        # loop: when chartists shrink, f→chartist rate shrinks further, while
        # chartist→f rate stays high (scaled by large n_f), causing chartist collapse.
        #
        # With SOURCE GROUP scaling, the transition rate for an agent in group X
        # is proportional to n_X/N, providing natural stabilization:
        # - When chartists shrink, their outflow rate also shrinks
        # - When fundamentalists shrink, their outflow rate also shrinks
        #
        # This preserves the utility-based switching direction (exp(±U)) while
        # preventing the "rich-get-richer" collapse of chartists.
        #
        # Source->target naming convention:
        #   pi_plus_f  : + -> f (source = +, so use n_plus)
        #   pi_f_plus  : f -> + (source = f, so use n_f)
        #   pi_minus_f : - -> f (source = -, so use n_minus)
        #   pi_f_minus : f -> - (source = f, so use n_f)
        pi_plus_f = p.v2 * (n_plus / N) * math.exp(-U21)       # + -> f (source = +)
        pi_f_plus = p.v2 * (n_f / N) * math.exp(U21)           # f -> + (source = f)
        pi_minus_f = p.v2 * (n_minus / N) * math.exp(-U22)     # - -> f (source = -)
        pi_f_minus = p.v2 * (n_f / N) * math.exp(U22)          # f -> - (source = f)
        
        return {
            'pi_plus_minus': pi_plus_minus,
            'pi_minus_plus': pi_minus_plus,
            'pi_plus_f': pi_plus_f,
            'pi_f_plus': pi_f_plus,
            'pi_minus_f': pi_minus_f,
            'pi_f_minus': pi_f_minus,
        }
    
    def _compute_excess_demand(self, n_plus: int, n_minus: int, n_f: int) -> float:
        """
        Compute excess demand (Samanidou et al. 2007, Eq. (22)).

            ED_c = (n_plus - n_minus) * t_c
            ED_f = n_f * gamma * (pf - p) / p
            ED = ED_c + ED_f
        """
        p = self.params
        # Chartist excess demand is SIGNED: optimists buy, pessimists sell
        ed_c = (n_plus - n_minus) * p.tc
        # Fundamentalist excess demand: stabilizing force (mean reversion)
        ed_f = n_f * p.gamma * (self.pf - self.price) / self.price
        return ed_c + ed_f
    
    def _update_agents(self, transition_probs: Dict[str, float], dt_override: Optional[float] = None) -> Tuple[int, int, int, int]:
        """
        Update all agents' types via O(N) iteration.
        
        Supports two update schemes controlled by params.abm_update_scheme:
        - "synchronous" (default): all agents sample against frozen counts (legacy behavior)
        - "sequential": each switch immediately updates running counts (closer to
          asynchronous Poisson updating; reduces artificial synchronicity)
        
        Args:
            transition_probs: Dictionary of transition rates (used in synchronous mode
                              or as initial rates in sequential mode)
            dt_override: Optional override for dt (used in burst sub-stepping)
            
        Returns:
            (n_f, n_plus, n_minus, num_switches)
        """
        p = self.params
        dt = dt_override if dt_override is not None else p.dt
        min_pop = p.min_population_frac * p.N
        
        # Use cached populations (kept consistent with agent list)
        n_f, n_plus, n_minus = self.n_f, self.n_plus, self.n_minus
        num_switches = 0
        
        if p.abm_update_scheme == "sequential":
            # SEQUENTIAL MODE: Update counts immediately after each switch.
            # This approximates asynchronous updating and reduces artificial synchronicity.
            # We also recompute transition probabilities after each switch for better accuracy.
            for agent in self.agents:
                old_type = agent.type
                
                # Check min_pop constraint
                can_switch = True
                if old_type == 'fundamentalist' and n_f <= min_pop:
                    can_switch = False
                elif old_type == 'noise_optimist' and n_plus <= min_pop:
                    can_switch = False
                elif old_type == 'noise_pessimist' and n_minus <= min_pop:
                    can_switch = False
                
                if can_switch:
                    # Recompute transition probabilities based on current (updated) counts
                    current_probs = self._compute_transition_probs(n_f, n_plus, n_minus)
                    switched = agent.reconsider(current_probs, dt)
                    
                    if switched:
                        num_switches += 1
                        new_type = agent.type
                        
                        # Immediately update running counts
                        if old_type == 'fundamentalist':
                            n_f -= 1
                        elif old_type == 'noise_optimist':
                            n_plus -= 1
                        else:
                            n_minus -= 1
                        
                        if new_type == 'fundamentalist':
                            n_f += 1
                        elif new_type == 'noise_optimist':
                            n_plus += 1
                        else:
                            n_minus += 1
        else:
            # SYNCHRONOUS MODE (default, legacy): All agents sample against frozen counts.
            delta_f = delta_plus = delta_minus = 0
            
            for agent in self.agents:
                old_type = agent.type
                
                can_switch = True
                if old_type == 'fundamentalist' and n_f <= min_pop:
                    can_switch = False
                elif old_type == 'noise_optimist' and n_plus <= min_pop:
                    can_switch = False
                elif old_type == 'noise_pessimist' and n_minus <= min_pop:
                    can_switch = False
                
                if can_switch:
                    switched = agent.reconsider(transition_probs, dt)
                    
                    if switched:
                        num_switches += 1
                        new_type = agent.type
                        
                        if old_type == 'fundamentalist':
                            delta_f -= 1
                        elif old_type == 'noise_optimist':
                            delta_plus -= 1
                        else:
                            delta_minus -= 1
                        
                        if new_type == 'fundamentalist':
                            delta_f += 1
                        elif new_type == 'noise_optimist':
                            delta_plus += 1
                        else:
                            delta_minus += 1
            
            # Apply accumulated deltas
            n_f += delta_f
            n_plus += delta_plus
            n_minus += delta_minus
        
        return n_f, n_plus, n_minus, num_switches
    
    def _update_agents_llm(self) -> Tuple[int, int, int, int]:
        """
        Update agent populations using per-agent LLM decision making.
        
        Each agent independently decides whether to switch type based on
        market conditions. Supports both sync and async modes.
        
        Architecture:
        1. Process any already-available LLM responses
        2. Trigger new LLM requests if decision interval reached
        3. (Sync mode) Wait for all pending requests to complete
        4. Apply cached decisions with PROBABILISTIC SAMPLING (matching ABM)
        
        Key optimization (matching ABM behavior):
        - Cached decisions are applied probabilistically each step
        - This prevents "one decision → repeated deterministic switches"
        - Matches ABM's "each step has small probability of switching"
        
        Returns:
            (n_f, n_plus, n_minus, num_switches)
        """
        p = self.params
        min_pop = int(p.min_population_frac * p.N)
        
        # Step 1: Process any already-available LLM responses
        response_switches = self._process_llm_responses()
        
        # Step 2: Trigger new LLM requests if interval reached
        if self.current_step % self.llm_decision_tick_interval == 0:
            self._request_llm_decisions()
        
        # Step 3 (Sync mode): Wait for all pending requests to complete
        if self.llm_sync_mode and len(self.llm_pending_requests) > 0:
            start_wait = time.time()
            pending_count = len(self.llm_pending_requests)
            
            if self.llm_debug:
                print(f"[LLM Sync] Step {self.current_step}: Waiting for {pending_count} pending requests...")
            
            while len(self.llm_pending_requests) > 0:
                # Check timeout
                elapsed = time.time() - start_wait
                if elapsed > self.llm_sync_timeout:
                    if self.llm_debug:
                        print(f"[LLM Sync] Timeout after {self.llm_sync_timeout:.1f}s, {len(self.llm_pending_requests)} requests still pending")
                    break
                
                # Process any responses that arrived
                new_switches = self._process_llm_responses()
                response_switches += new_switches
                
                # Small sleep to avoid busy-waiting (but still responsive)
                if len(self.llm_pending_requests) > 0:
                    time.sleep(0.1)
            
            if self.llm_debug:
                elapsed = time.time() - start_wait
                completed = pending_count - len(self.llm_pending_requests)
                print(f"[LLM Sync] Completed {completed}/{pending_count} in {elapsed:.2f}s, {response_switches} switches")
        
        # Step 4: Apply cached decisions with PROBABILISTIC SAMPLING
        # AND use ABM fallback for agents without LLM decisions
        #
        # Key insight: In pure LLM mode, agents without decisions would be "frozen".
        # To match ABM behavior, we use ABM transition probabilities as fallback
        # for agents without LLM coverage.
        cache_switches = 0
        abm_fallback_switches = 0
        n_f, n_plus, n_minus = self._count_agents()
        
        # Compute ABM transition probabilities for fallback
        transition_probs = self._compute_transition_probs(n_f, n_plus, n_minus)
        
        for agent in self.agents:
            # Skip if agent has pending LLM request
            if agent.id in self.llm_pending_requests:
                continue
            
            # Check for valid cached decision
            if agent.has_valid_cached_decision(self.current_step):
                decision = agent.get_cached_decision()
                switch_tendency = getattr(agent, 'cached_switch_tendency', 0.0)
                
                # Only process non-stay decisions
                if decision and decision != 'stay':
                    old_type = agent.type
                    
                    # Probabilistic application of cached decision (matching ABM)
                    # ABM formula: p = dt * v * (n/N) * exp(U)
                    # Map switch_tendency [0, 1] to exp(U) equivalent [0.1, 10]
                    if old_type == 'fundamentalist':
                        pop_fraction = n_f / p.N
                    elif old_type == 'noise_optimist':
                        pop_fraction = n_plus / p.N
                    else:
                        pop_fraction = n_minus / p.N
                    
                    # exp(U) mapping: 0.1 * 100^tendency → [0.1, 10]
                    exp_U_equivalent = 0.1 * math.pow(100, switch_tendency)
                    
                    # Determine v rate based on decision type
                    if decision in ['switch_to_optimist', 'switch_to_pessimist'] and old_type in ['noise_optimist', 'noise_pessimist']:
                        v_rate = p.v1  # Opinion switching
                    else:
                        v_rate = p.v2  # Strategy switching
                    
                    effective_probability = p.dt * v_rate * pop_fraction * exp_U_equivalent
                    effective_probability = min(effective_probability, 1.0)
                    
                    # Sample whether to switch this step
                    should_switch = random.random() < effective_probability
                    
                    if should_switch:
                        # Check population constraints before switching
                        can_switch = True
                        if old_type == 'fundamentalist' and n_f <= min_pop:
                            can_switch = False
                        elif old_type == 'noise_optimist' and n_plus <= min_pop:
                            can_switch = False
                        elif old_type == 'noise_pessimist' and n_minus <= min_pop:
                            can_switch = False
                        
                        if can_switch:
                            # Apply the cached decision
                            switched = False
                            if decision == 'switch_to_fundamentalist' and old_type != 'fundamentalist':
                                agent.type = 'fundamentalist'
                                switched = True
                            elif decision == 'switch_to_optimist' and old_type != 'noise_optimist':
                                agent.type = 'noise_optimist'
                                switched = True
                            elif decision == 'switch_to_pessimist' and old_type != 'noise_pessimist':
                                agent.type = 'noise_pessimist'
                                switched = True
                            
                            if switched:
                                cache_switches += 1
                                # Update local population counts for subsequent agents
                                if old_type == 'fundamentalist':
                                    n_f -= 1
                                elif old_type == 'noise_optimist':
                                    n_plus -= 1
                                else:
                                    n_minus -= 1
                                
                                if agent.type == 'fundamentalist':
                                    n_f += 1
                                elif agent.type == 'noise_optimist':
                                    n_plus += 1
                                else:
                                    n_minus += 1
                                
                                # Clear cached decision after successful switch
                                agent.cached_decision = None
                                agent.cached_switch_tendency = 0.0
                                
                                if self.llm_debug:
                                    print(f"[LLM Cache] Agent {agent.id}: {old_type} -> {agent.type} "
                                          f"(prob={effective_probability:.4f})")
                        else:
                            self.stats["switches_blocked_by_min_pop"] += 1
                            self.llm_control_stats["switches_blocked_by_min_pop"] += 1
                
                self.stats["llm_cache_hits"] += 1
                self.llm_control_stats["cache_hits"] += 1
            else:
                # ABM FALLBACK: Agent has no LLM decision, use ABM transition probabilities
                # This ensures agents without LLM coverage still behave like ABM
                old_type = agent.type
                
                # Check min_pop constraint before allowing switch
                can_switch = True
                if old_type == 'fundamentalist' and n_f <= min_pop:
                    can_switch = False
                elif old_type == 'noise_optimist' and n_plus <= min_pop:
                    can_switch = False
                elif old_type == 'noise_pessimist' and n_minus <= min_pop:
                    can_switch = False
                
                if can_switch:
                    # Use ABM reconsider logic
                    switched = agent.reconsider(transition_probs, p.dt)
                    if switched:
                        abm_fallback_switches += 1
                        new_type = agent.type
                        
                        # Update local population counts
                        if old_type == 'fundamentalist':
                            n_f -= 1
                        elif old_type == 'noise_optimist':
                            n_plus -= 1
                        else:
                            n_minus -= 1
                        
                        if new_type == 'fundamentalist':
                            n_f += 1
                        elif new_type == 'noise_optimist':
                            n_plus += 1
                        else:
                            n_minus += 1
        
        # Count final populations (recount for accuracy)
        n_f, n_plus, n_minus = self._count_agents()
        total_switches = response_switches + cache_switches + abm_fallback_switches
        
        # Update cached counts
        self.n_f = n_f
        self.n_plus = n_plus
        self.n_minus = n_minus
        
        # Update stats
        self.stats["llm_calls"] = self.llm_control_stats["requests_sent"]
        self.stats["abm_fallback_switches"] += abm_fallback_switches
        
        return n_f, n_plus, n_minus, total_switches
    
    def _update_price(self, ed: float, dt_override: Optional[float] = None, 
                      sigma_override: Optional[float] = None):
        """
        Update price using the Poisson tick mechanism from the review (Eq. (23)).

        During a small time increment dt, the price changes in discrete ticks
        of size Δp = ± price_tick * p with Poisson rates:
            π_up   = max(0, β (ED + μ))
            π_down = -min(0, β (ED + μ))
        where μ is Gaussian noise.

        Implementation note:
        - We sample the number of up/down ticks in this dt via Poisson(π*dt).
          This avoids the degenerate "one tick every step" dynamics which makes
          1-step returns essentially two-point distributed (excess kurtosis ≈ -2).
        
        Args:
            ed: Excess demand
            dt_override: Optional override for dt (used in burst sub-stepping)
            sigma_override: Optional override for price_change_sigma (scaled for substeps)
        """
        p = self.params
        dt = dt_override if dt_override is not None else p.dt
        sigma = sigma_override if sigma_override is not None else p.price_change_sigma

        # Noise in excess demand (Eq. (24) uses μ)
        mu = self._np_rng_main.normal(p.price_change_mu, sigma)
        a = p.beta * (ed + mu)

        # Keep paper-style π↑, π↓ in history for diagnostics even if we use a continuous update.
        self.pi_price_up = max(0.0, a)
        self.pi_price_down = (-1.0) * min(0.0, a)

        old_price = self.price
        mode = (p.price_update_mode or "walrasian").lower()

        if mode == "walrasian":
            # Mean-field / Walrasian adjustment limit (Eq. (25)):
            # The tick model moves the price by ±price_tick·p. Its mean-field limit is:
            #     d log(p) ≈ price_tick · (π↑ - π↓) · dt
            #           = price_tick · β · (ED + μ) · dt
            # Using an exponential update guarantees positivity.
            if not (0.0 < p.price_tick < 1.0):
                raise ValueError("price_tick must be in (0, 1) for walrasian log-price update.")
            self.price = max(self.price * math.exp((p.price_tick * a) * dt), 1e-10)
            self.price_change = self.price - old_price

        elif mode == "conditional_tick":
            if not (0.0 < p.price_tick < 1.0):
                raise ValueError("price_tick must be in (0, 1) for multiplicative ticks.")
            # Legacy-like: always move one tick (direction depends on relative weights).
            if self.pi_price_up == 0.0 and self.pi_price_down == 0.0:
                self.price_change = 0.0
                return
            sign = choices([1, -1], weights=[self.pi_price_up, self.pi_price_down])[0]
            self.price_change = sign * p.price_tick * self.price
            self.price = max(self.price + self.price_change, 1e-10)

        elif mode == "bernoulli":
            if not (0.0 < p.price_tick < 1.0):
                raise ValueError("price_tick must be in (0, 1) for multiplicative ticks.")
            # Paper-like: 3-state up/down/none with probabilities π*dt.
            prob_up = self.pi_price_up * dt
            prob_down = self.pi_price_down * dt
            total = prob_up + prob_down
            if total > 1.0:
                # dt too large for these rates; renormalize to avoid negative stay-probability.
                prob_up /= total
                prob_down /= total
                prob_stay = 0.0
            else:
                prob_stay = 1.0 - total

            move = choices([0, 1, -1], weights=[prob_stay, prob_up, prob_down])[0]
            if move == 0:
                self.price_change = 0.0
                return
            self.price_change = move * p.price_tick * self.price
            self.price = max(self.price + self.price_change, 1e-10)

        else:
            if not (0.0 < p.price_tick < 1.0):
                raise ValueError("price_tick must be in (0, 1) for multiplicative ticks.")
            # Default: compound Poisson ticks with intensity π*dt (allows multi-tick moves).
            lam_up = self.pi_price_up * dt
            lam_down = self.pi_price_down * dt
            k_up = int(self._np_rng_aux.poisson(lam_up)) if lam_up > 0 else 0
            k_down = int(self._np_rng_aux.poisson(lam_down)) if lam_down > 0 else 0

            if k_up == 0 and k_down == 0:
                self.price_change = 0.0
                return

            # Apply multiplicative ticks. Order doesn't matter under constant relative tick size.
            new_price = self.price
            if k_up:
                new_price *= (1.0 + p.price_tick) ** k_up
            if k_down:
                new_price *= (1.0 - p.price_tick) ** k_down

            self.price = max(new_price, 1e-10)
            self.price_change = self.price - old_price

        if self.price_change > 0:
            self.stats["price_up_events"] += 1
        elif self.price_change < 0:
            self.stats["price_down_events"] += 1
    
    def _update_fundamental(self, sigma_override: Optional[float] = None):
        """
        Update fundamental value via geometric random walk (legacy-compatible).
        
        Args:
            sigma_override: Optional override for value_change_sigma (scaled for substeps)
        """
        p = self.params
        sigma = sigma_override if sigma_override is not None else p.value_change_sigma
        old_pf = self.pf
        eps = self._np_rng_main.normal(p.value_change_mu, sigma)
        self.pf *= math.exp(eps)
        self.pf = max(self.pf, 1e-10)
        # Legacy stores value_change as (previous - current)
        self.value_change = old_pf - self.pf
    
    def step(self):
        """
        Execute one time step of the simulation (aligned with Lux-Marchesi-ABM).
        
        Supports optional burst sub-stepping (paper Section 9): when price activity
        is high, the step is split into finer substeps for more accurate dynamics.
        """
        p = self.params
        
        # Legacy uses 1-based iteration counter for dp/dt
        iteration = self.current_step + 1
        
        # Check if burst sub-stepping should be used
        use_substeps = False
        n_substeps = 1
        
        if p.burst_substep_enabled and p.price_update_mode in ("poisson", "bernoulli"):
            # IMPROVED: Use expected event intensity including noise scale.
            # The actual intensity is π = β·(ED + μ), where μ ~ N(0, σ).
            # We estimate the expected total intensity as:
            #   E[π_up + π_down] ≈ β·|ED| + β·E[|μ|] = β·(|ED| + σ·√(2/π))
            # Then expected events per step = intensity × dt.
            #
            # Alternative: use previous step's actual π values (self.pi_price_up, self.pi_price_down)
            # which already include realized noise. This provides a lagged but accurate measure.
            
            if self.current_step > 0:
                # Use previous step's actual intensity (includes noise realization)
                pi_total_prev = self.pi_price_up + self.pi_price_down
                expected_events = pi_total_prev * p.dt
            else:
                # First step: estimate from ED + expected noise magnitude
                ed_estimate = self._compute_excess_demand(self.n_plus, self.n_minus, self.n_f)
                # E[|μ|] for normal(0,σ) is σ·√(2/π) ≈ 0.798·σ
                expected_abs_mu = p.price_change_sigma * 0.7979  # sqrt(2/π)
                pi_estimate = p.beta * (abs(ed_estimate) + expected_abs_mu)
                expected_events = pi_estimate * p.dt
            
            if expected_events > p.burst_substep_threshold:
                use_substeps = True
                n_substeps = p.burst_substep_factor
                self.stats["burst_substeps_triggered"] = self.stats.get("burst_substeps_triggered", 0) + 1
        
        # Track for history recording
        self._current_burst_substeps = n_substeps if use_substeps else 0
        
        if use_substeps:
            # BURST SUB-STEPPING MODE
            # Paper uses Δt=0.01 normally, Δt=0.002 during bursts (factor=5)
            dt_sub = p.dt / n_substeps
            
            # Scale noise std by sqrt(1/n_substeps) for consistent variance over the full step
            # (variance of sum of n independent draws with σ/√n each = σ²)
            value_sigma_sub = p.value_change_sigma / math.sqrt(n_substeps)
            price_sigma_sub = p.price_change_sigma / math.sqrt(n_substeps)
            
            # Accumulate price changes over substeps for history
            total_price_change = 0.0
            total_value_change = 0.0
            total_switches = 0
            
            # Track final ED and transition probs for history (use last substep's values)
            final_ed = 0.0
            final_transition_probs = {}
            
            for substep in range(n_substeps):
                # 1) Fundamental value update
                old_pf = self.pf
                self._update_fundamental(sigma_override=value_sigma_sub)
                total_value_change += (old_pf - self.pf)
                
                # 2) Price update with scaled dt and sigma
                ed = self._compute_excess_demand(self.n_plus, self.n_minus, self.n_f)
                self._update_price(ed, dt_override=dt_sub, sigma_override=price_sigma_sub)
                total_price_change += self.price_change
                final_ed = ed
                
                # 3) Compute dp/dt (use full iteration for lookback consistency)
                self.dp_dt = self._compute_dp_dt(iteration=iteration)
                
                # 4) Transition probabilities
                transition_probs = self._compute_transition_probs(self.n_f, self.n_plus, self.n_minus)
                final_transition_probs = transition_probs
                
                # 5) Agent updates with scaled dt
                if self.mode == "llm":
                    n_f_new, n_plus_new, n_minus_new, num_sw = self._update_agents_llm()
                else:
                    n_f_new, n_plus_new, n_minus_new, num_sw = self._update_agents(
                        transition_probs, dt_override=dt_sub
                    )
                self.n_f, self.n_plus, self.n_minus = n_f_new, n_plus_new, n_minus_new
                total_switches += num_sw
            
            # Store accumulated changes for history
            self.price_change = total_price_change
            self.value_change = total_value_change
            self.transition_probs = final_transition_probs
            
            # Record history with accumulated values
            self._record_history(excess_demand=final_ed, transition_probs=final_transition_probs)
        else:
            # STANDARD SINGLE-STEP MODE (legacy behavior)
            
            # 1) Fundamental value update (happens BEFORE price update in legacy code)
            self._update_fundamental()

            # 2) Price update uses excess demand computed from PRE-switch populations
            ed = self._compute_excess_demand(self.n_plus, self.n_minus, self.n_f)
            self._update_price(ed)

            # 3) Compute dp/dt AFTER price update using legacy lookback convention
            self.dp_dt = self._compute_dp_dt(iteration=iteration)

            # 4) Transition probabilities computed AFTER dp/dt (but still using PRE-switch populations)
            transition_probs = self._compute_transition_probs(self.n_f, self.n_plus, self.n_minus)
            self.transition_probs = transition_probs

            # 5) Agents reconsider AFTER price update (legacy ordering)
            if self.mode == "llm":
                n_f_new, n_plus_new, n_minus_new, num_switches = self._update_agents_llm()
            else:
                n_f_new, n_plus_new, n_minus_new, num_switches = self._update_agents(transition_probs)
            
            # Update cached populations
            self.n_f, self.n_plus, self.n_minus = n_f_new, n_plus_new, n_minus_new

            # 6) Record history snapshot (legacy stores post-switch populations together with π, dp_dt, price/value changes)
            self._record_history(excess_demand=ed, transition_probs=transition_probs)

        # 7) Append price history AFTER dp/dt is computed (legacy semantics)
        self.price_history.append(self.price)
        
        self.current_step += 1
    
    def run(self, progress_bar: bool = True) -> Dict[str, np.ndarray]:
        """
        Run the full simulation.
        
        Args:
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary of simulation history arrays
        """
        iterator = range(self.params.T)
        if progress_bar:
            iterator = tqdm(iterator, desc=f"Lux-Marchesi ({self.mode.upper()})")
        
        for _ in iterator:
            self.step()
        
        # Convert history to numpy arrays
        result = {key: np.array(val) for key, val in self.history.items()}
        return result
    
    def get_returns(self) -> np.ndarray:
        """Compute log returns from price history."""
        prices = np.array(self.history["price"])
        prices = prices[prices > 0]
        
        if len(prices) < 2:
            return np.array([])
        
        return np.diff(np.log(prices))
    
    def get_state(self) -> Dict[str, Any]:
        """Get current model state as a dictionary."""
        state = {
            "step": self.current_step,
            "price": self.price,
            "pf": self.pf,
            "nc": self.n_plus + self.n_minus,
            "nf": self.n_f,
            "n_plus": self.n_plus,
            "n_minus": self.n_minus,
            "mode": self.mode,
            "stats": self.stats.copy(),
        }
        
        # Add LLM-specific statistics if in LLM mode
        if self.mode == "llm":
            state["llm_control_stats"] = self.llm_control_stats.copy()
            state["llm_params"] = {
                "decision_tick_interval": self.llm_decision_tick_interval,
                "cache_duration": self.llm_cache_duration,
                "max_agents_per_tick": self.max_llm_agents_per_tick,
            }
            # Compute LLM control rate
            total = self.llm_control_stats["llm_decisions"] + self.llm_control_stats["cache_hits"]
            if total > 0:
                state["llm_control_rate"] = self.llm_control_stats["llm_decisions"] / total
            else:
                state["llm_control_rate"] = 0.0
        
        return state
    
    def save_results(self, output_dir: str):
        """Save simulation results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save history as numpy arrays
        history_file = os.path.join(output_dir, "history.npz")
        np.savez(history_file, **{key: np.array(val) for key, val in self.history.items()})
        
        # Save parameters and stats as JSON
        metadata = {
            "params": asdict(self.params),
            "mode": self.mode,
            "llm_model": self.llm_model if self.mode == "llm" else None,
            "stats": self.stats,
            "final_state": self.get_state(),
        }
        
        # Add LLM-specific metadata
        if self.mode == "llm":
            metadata["llm_control_stats"] = self.llm_control_stats
            metadata["llm_params"] = {
                "decision_tick_interval": self.llm_decision_tick_interval,
                "cache_duration": self.llm_cache_duration,
                "max_agents_per_tick": self.max_llm_agents_per_tick,
                "min_seconds_between_requests": self.min_seconds_between_requests,
                "concurrent_batch_size": self.llm_concurrent_batch_size,
                "batch_delay": self.llm_batch_delay,
            }
            metadata["llm_usage_total"] = self.llm_usage_total
        
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save LLM usage records if in LLM mode
        if self.mode == "llm":
            self._save_llm_usage(output_dir)
        
        print(f"[Save] Results saved to {output_dir}")
        
        # Print LLM statistics summary
        if self.mode == "llm":
            print(f"[LLM Stats] Requests sent: {self.llm_control_stats['requests_sent']}")
            print(f"[LLM Stats] Responses received: {self.llm_control_stats['responses_received']}")
            print(f"[LLM Stats] LLM decisions: {self.llm_control_stats['llm_decisions']}")
            print(f"[LLM Stats] Cache hits: {self.llm_control_stats['cache_hits']}")
            print(f"[LLM Stats] Errors: {self.llm_control_stats['errors']}")
            print(f"[LLM Stats] Total tokens used: {self.llm_usage_total['total_tokens']}")
    
    def _save_llm_usage(self, output_dir: str):
        """Save LLM usage records to files.
        
        Saves:
        - llm_usage.jsonl: Individual LLM call records (JSON Lines format)
        - llm_usage_total.json: Aggregated usage statistics
        """
        # Save individual records to JSONL (one JSON object per line)
        usage_jsonl_file = os.path.join(output_dir, "llm_usage.jsonl")
        with open(usage_jsonl_file, "w") as f:
            for record in self.llm_usage_records:
                f.write(json.dumps(record, default=str) + "\n")
        
        # Save total usage statistics
        usage_total_file = os.path.join(output_dir, "llm_usage_total.json")
        total_stats = {
            "model": self.llm_model,
            "usage": self.llm_usage_total,
            "summary": {
                "total_requests": self.llm_usage_total["num_requests"],
                "successful_requests": self.llm_usage_total["num_successful"],
                "failed_requests": self.llm_usage_total["num_failed"],
                "total_prompt_tokens": self.llm_usage_total["prompt_tokens"],
                "total_completion_tokens": self.llm_usage_total["completion_tokens"],
                "total_tokens": self.llm_usage_total["total_tokens"],
                "avg_prompt_tokens_per_request": (
                    self.llm_usage_total["prompt_tokens"] / self.llm_usage_total["num_successful"]
                    if self.llm_usage_total["num_successful"] > 0 else 0
                ),
                "avg_completion_tokens_per_request": (
                    self.llm_usage_total["completion_tokens"] / self.llm_usage_total["num_successful"]
                    if self.llm_usage_total["num_successful"] > 0 else 0
                ),
                "avg_total_tokens_per_request": (
                    self.llm_usage_total["total_tokens"] / self.llm_usage_total["num_successful"]
                    if self.llm_usage_total["num_successful"] > 0 else 0
                ),
            },
            "llm_params": {
                "decision_tick_interval": self.llm_decision_tick_interval,
                "cache_duration": self.llm_cache_duration,
                "max_agents_per_tick": self.max_llm_agents_per_tick,
                "min_seconds_between_requests": self.min_seconds_between_requests,
                "concurrent_batch_size": self.llm_concurrent_batch_size,
                "batch_delay": self.llm_batch_delay,
            },
            "simulation_params": {
                "N": self.params.N,
                "T": self.params.T,
                "dt": self.params.dt,
            },
        }
        with open(usage_total_file, "w") as f:
            json.dump(total_stats, f, indent=2, default=str)
        
        print(f"[LLM Usage] Saved {len(self.llm_usage_records)} records to {usage_jsonl_file}")
        print(f"[LLM Usage] Saved total usage to {usage_total_file}")


# =============================================================================
# CLI 
# =============================================================================

def main():
    """Main entry point for running the simulation."""
    parser = argparse.ArgumentParser(description="Lux-Marchesi Financial Market Model")
    
    # Legacy config (Lux-Marchesi-ABM) convenience
    parser.add_argument("--config", type=str, default=None,
                       help="Path to Lux-Marchesi-ABM style configuration.json (optional)")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="abm", choices=["abm", "llm"],
                       help="Simulation mode: abm (rule-based) or llm (LLM-based)")
    
    # Population parameters
    parser.add_argument("--N", type=int, default=500, help="Total number of agents")
    parser.add_argument("--num_fundamentalists", type=int, default=None, help="Legacy initial number of fundamentalists")
    parser.add_argument("--num_noise_optimist", type=int, default=None, help="Legacy initial number of noise optimists")
    parser.add_argument("--num_noise_pessimist", type=int, default=None, help="Legacy initial number of noise pessimists")
    
    # Time parameters
    parser.add_argument("--T", type=int, default=50000, help="Number of time steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("--time_of_simulation", type=float, default=None,
                       help="Legacy total simulation time (overrides --T via T=int(time_of_simulation/dt))")
    parser.add_argument("--trend_lookback_steps", type=int, default=10,
                       help="dp/dt lookback in steps (paper: 10)")
    
    # Opinion switching parameters
    parser.add_argument("--v1", type=float, default=2.0, help="Opinion re-evaluation frequency")
    parser.add_argument("--alpha1", type=float, default=0.6, help="Herding effect weight")
    parser.add_argument("--alpha2", type=float, default=1.5, help="Trend-following weight")
    
    # Strategy switching parameters
    parser.add_argument("--v2", type=float, default=0.6, help="Strategy re-evaluation frequency")
    parser.add_argument("--alpha3", type=float, default=1.0, help="Profit sensitivity")
    parser.add_argument("--r", type=float, default=0.0004, help="Dividend yield")
    parser.add_argument("--R", type=float, default=0.0004, help="Alternative asset return")
    parser.add_argument("--s", type=float, default=0.75, help="Fundamental return discount factor")
    
    # Price parameters
    parser.add_argument("--beta", type=float, default=4.0, help="Price adjustment speed")
    parser.add_argument("--tc", type=float, default=0.001, help="Chartist trade volume")
    parser.add_argument("--t_c", type=float, default=None, help="Legacy alias for --tc")
    parser.add_argument("--gamma", type=float, default=0.01, help="Fundamentalist sensitivity")
    parser.add_argument("--price_tick", type=float, default=0.001, help="Price tick size (legacy: delta_p)")
    parser.add_argument("--delta_p", type=float, default=None, help="Legacy alias for --price_tick")
    parser.add_argument(
        "--price_update_mode",
        type=str,
        default="walrasian",
        choices=["walrasian", "poisson", "conditional_tick", "bernoulli"],
        help="Price update scheme (see LuxMarchesiParams.price_update_mode)",
    )
    
    # Noise parameters
    parser.add_argument("--price_noise_sigma", type=float, default=0.05, help="Price noise std")
    parser.add_argument("--value_noise_sigma", type=float, default=0.005, help="Fundamental value noise std")
    
    # Initial conditions
    parser.add_argument("--initial_price", type=float, default=100.0, help="Initial price")
    parser.add_argument("--initial_pf", type=float, default=100.0, help="Initial fundamental value")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results/Lux_Marchesi",
                       help="Output directory for results")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06",
                       help="LLM model name for LLM mode")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number for output")
    parser.add_argument("--no_progress_bar", action="store_true",
                       help="Disable tqdm progress bar during simulation")
    
    # LLM per-agent scheduling parameters
    parser.add_argument("--llm_decision_interval", type=int, default=50,
                       help="LLM decision tick interval (steps between LLM calls)")
    parser.add_argument("--llm_cache_duration", type=int, default=100,
                       help="How many steps to cache LLM decisions")
    parser.add_argument("--llm_max_agents", type=int, default=20,
                       help="Maximum agents to process per LLM decision round")
    parser.add_argument("--llm_concurrent_batch_size", type=int, default=10,
                       help="Max concurrent API calls within a batch (controls rate limiting)")
    parser.add_argument("--llm_batch_delay", type=float, default=0.5,
                       help="Seconds to wait between sub-batches (prevents 429 errors)")
    parser.add_argument("--llm_debug", action="store_true",
                       help="Enable LLM debug output")
    
    # Paper-alignment dynamics parameters
    parser.add_argument("--abm_update_scheme", type=str, default="synchronous",
                       choices=["synchronous", "sequential"],
                       help="ABM update scheme: synchronous (legacy) or sequential (asynchronous approximation)")
    parser.add_argument("--burst_substep_enabled", action="store_true",
                       help="Enable burst sub-stepping for high-activity periods")
    parser.add_argument("--burst_substep_threshold", type=float, default=0.1,
                       help="Activity threshold for triggering burst sub-stepping")
    parser.add_argument("--burst_substep_factor", type=int, default=5,
                       help="Number of substeps when burst sub-stepping is triggered")
    
    args = parser.parse_args()
    
    def _flag_provided(flag: str) -> bool:
        argv = sys.argv[1:]
        return any(a == flag or a.startswith(flag + "=") for a in argv)

    # Determine aliases
    tc = args.t_c if args.t_c is not None else args.tc
    price_tick = args.delta_p if args.delta_p is not None else args.price_tick

    # Determine T from time_of_simulation if provided
    T = int(args.time_of_simulation / args.dt) if args.time_of_simulation is not None else args.T

    # Create parameters (optionally from legacy config)
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = json.load(f)

        # Base legacy values
        cfg_dt = float(cfg.get("delta_t", args.dt))
        cfg_time = float(cfg.get("time_of_simulation", 0.0))
        cfg_T = int(cfg_time / cfg_dt) if cfg_time > 0 else args.T

        # Apply overrides only if explicitly provided on CLI
        dt = args.dt if _flag_provided("--dt") else cfg_dt
        if args.time_of_simulation is not None:
            cfg_time = float(args.time_of_simulation)
        T = args.T if _flag_provided("--T") else int(cfg_time / dt) if cfg_time > 0 else cfg_T

        tc = tc if (_flag_provided("--tc") or _flag_provided("--t_c")) else float(cfg.get("t_c", cfg.get("tc", 0.001)))
        price_tick = price_tick if (_flag_provided("--price_tick") or _flag_provided("--delta_p")) else float(cfg.get("delta_p", 0.001))

        params = LuxMarchesiParams(
            # Explicit populations (legacy)
            num_fundamentalists=int(cfg.get("num_fundamentalists", 0)),
            num_noise_optimist=int(cfg.get("num_noise_optimist", 0)),
            num_noise_pessimist=int(cfg.get("num_noise_pessimist", 0)),

            # Time
            T=T,
            dt=dt,
            trend_lookback_steps=args.trend_lookback_steps,

            # Core parameters
            v1=float(cfg.get("v1", args.v1)) if not _flag_provided("--v1") else args.v1,
            alpha1=float(cfg.get("alpha1", args.alpha1)) if not _flag_provided("--alpha1") else args.alpha1,
            alpha2=float(cfg.get("alpha2", args.alpha2)) if not _flag_provided("--alpha2") else args.alpha2,
            v2=float(cfg.get("v2", args.v2)) if not _flag_provided("--v2") else args.v2,
            alpha3=float(cfg.get("alpha3", args.alpha3)) if not _flag_provided("--alpha3") else args.alpha3,
            R=float(cfg.get("R", args.R)) if not _flag_provided("--R") else args.R,
            s=float(cfg.get("s", args.s)) if not _flag_provided("--s") else args.s,
            beta=float(cfg.get("beta", args.beta)) if not _flag_provided("--beta") else args.beta,
            tc=tc,
            gamma=float(cfg.get("gamma", args.gamma)) if not _flag_provided("--gamma") else args.gamma,
            price_tick=price_tick,
            price_update_mode=str(cfg.get("price_update_mode", args.price_update_mode))
            if not _flag_provided("--price_update_mode") else args.price_update_mode,

            # Noise
            price_change_mu=float(cfg.get("price_change_mu", 0.0)),
            price_change_sigma=float(cfg.get("price_change_sigma", args.price_noise_sigma)) if not _flag_provided("--price_noise_sigma") else args.price_noise_sigma,
            value_change_mu=float(cfg.get("value_change_mu", 0.0)),
            value_change_sigma=float(cfg.get("value_change_sigma", args.value_noise_sigma)) if not _flag_provided("--value_noise_sigma") else args.value_noise_sigma,

            # Population bounds
            min_population_frac=float(cfg.get("minimal_agent_population_frac", 0.01)),

            # Initial conditions
            initial_price=float(cfg.get("market_price", args.initial_price)) if not _flag_provided("--initial_price") else args.initial_price,
            initial_pf=float(cfg.get("market_value", args.initial_pf)) if not _flag_provided("--initial_pf") else args.initial_pf,

            # Seed
            seed=args.seed,

            # Keep `r` for API completeness (unused in legacy ABM formulas)
            r=args.r,

            # Paper-alignment dynamics (CLI overrides config)
            abm_update_scheme=args.abm_update_scheme if _flag_provided("--abm_update_scheme") else str(cfg.get("abm_update_scheme", "synchronous")),
            burst_substep_enabled=args.burst_substep_enabled if _flag_provided("--burst_substep_enabled") else bool(cfg.get("burst_substep_enabled", False)),
            burst_substep_threshold=args.burst_substep_threshold if _flag_provided("--burst_substep_threshold") else float(cfg.get("burst_substep_threshold", 0.1)),
            burst_substep_factor=args.burst_substep_factor if _flag_provided("--burst_substep_factor") else int(cfg.get("burst_substep_factor", 5)),
        )
    else:
        params = LuxMarchesiParams(
            N=args.N,
            num_fundamentalists=args.num_fundamentalists,
            num_noise_optimist=args.num_noise_optimist,
            num_noise_pessimist=args.num_noise_pessimist,
            T=T,
            dt=args.dt,
            trend_lookback_steps=args.trend_lookback_steps,
            v1=args.v1,
            alpha1=args.alpha1,
            alpha2=args.alpha2,
            v2=args.v2,
            alpha3=args.alpha3,
            r=args.r,
            R=args.R,
            s=args.s,
            beta=args.beta,
            tc=tc,
            gamma=args.gamma,
            price_tick=price_tick,
            price_update_mode=args.price_update_mode,
            price_change_sigma=args.price_noise_sigma,
            value_change_sigma=args.value_noise_sigma,
            initial_price=args.initial_price,
            initial_pf=args.initial_pf,
            seed=args.seed,
            # Paper-alignment dynamics
            abm_update_scheme=args.abm_update_scheme,
            burst_substep_enabled=args.burst_substep_enabled,
            burst_substep_threshold=args.burst_substep_threshold,
            burst_substep_factor=args.burst_substep_factor,
        )
    
    # Create output directory
    if args.mode == "llm":
        output_dir = os.path.join(args.output_dir, args.model, f"iteration_{args.iteration}")
    else:
        output_dir = os.path.join(args.output_dir, "abm", f"iteration_{args.iteration}")
    
    print("=" * 60)
    print("Lux-Marchesi Financial Market Model")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Agents: {params.N}")
    print(f"Time steps: {params.T}")
    print(f"Random seed: {args.seed}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Create and run model
    model = LuxMarchesiModel(params, mode=args.mode, llm_model=args.model)
    
    # Apply LLM parameters if in LLM mode
    if args.mode == "llm":
        model.llm_decision_tick_interval = args.llm_decision_interval
        model.llm_cache_duration = args.llm_cache_duration
        model.max_llm_agents_per_tick = args.llm_max_agents
        model.llm_concurrent_batch_size = args.llm_concurrent_batch_size
        model.llm_batch_delay = args.llm_batch_delay
        model.llm_debug = args.llm_debug
        print(f"[LLM] Decision interval: {args.llm_decision_interval} steps")
        print(f"[LLM] Cache duration: {args.llm_cache_duration} steps")
        print(f"[LLM] Max agents per tick: {args.llm_max_agents}")
        print(f"[LLM] Concurrent batch size: {args.llm_concurrent_batch_size}")
        print(f"[LLM] Batch delay: {args.llm_batch_delay}s")
    
    history = model.run(progress_bar=not args.no_progress_bar)
    
    # Save results
    model.save_results(output_dir)
    
    # Print summary statistics
    returns = model.get_returns()
    if len(returns) > 0:
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        print(f"Final price: {model.price:.4f}")
        print(f"Final fundamental value: {model.pf:.4f}")
        print(f"Price/Fundamental ratio: {model.price / model.pf:.4f}")
        
        n_f, n_plus, n_minus = model.n_f, model.n_plus, model.n_minus
        nc = n_plus + n_minus
        print(f"Final chartists: {nc} ({nc/params.N:.1%})")
        print(f"Final fundamentalists: {n_f} ({n_f/params.N:.1%})")
        print(f"Final optimists: {n_plus} ({n_plus/nc:.1%} of chartists)" if nc > 0 else "")
        
        print(f"\nReturn statistics:")
        print(f"  Mean: {returns.mean():.6f}")
        print(f"  Std: {returns.std():.6f}")
        if returns.std() > 0:
            skew = ((returns - returns.mean())**3).mean() / returns.std()**3
            kurt = ((returns - returns.mean())**4).mean() / returns.std()**4
            print(f"  Skewness: {skew:.4f}")
            print(f"  Kurtosis: {kurt:.4f}")
        
        print(f"\nModel statistics:")
        for key, val in model.stats.items():
            print(f"  {key}: {val}")
    
    print("=" * 60)
    print("Simulation completed successfully!")
    
    return model, history


if __name__ == "__main__":
    main()
