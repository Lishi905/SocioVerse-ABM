> **Legacy snapshot, not maintained.** This folder keeps the original scripts verbatim for provenance and as the reference behaviour for the refactored task in [`../`](../). It is not imported by the task at runtime and is not part of the test suite. The scripts need their own environment: Mesa 2.x (`mesa.time`) and the `openai>=1` client. They do not run against the current kernel.
>
> Original author: Jianing Shi ([@Brishian427](https://github.com/Brishian427)).
>
> Edits for the public release: the LLM client reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the environment (default: the official OpenAI endpoint) instead of a third-party relay, and the missing-key message in `nasch_mesa.py` names `OPENAI_API_KEY`.

# NaSch Traffic Model with LLM Integration

## Overview

This implementation transforms the traditional Nagel-Schreckenberg (NaSch) traffic model into an agent-based architecture following the Sugarscape design pattern. The key innovation is the ability to replace only the decision-making "brain" of vehicles while keeping the action space (speed values 0-5) unchanged.

## Architecture

### Key Design Principles

1. **Action Space Unchanged**: Vehicles always output speed in [0,1,2,3,4,5]
2. **Minimal Observation Space**: Only what agents can actually observe (current_speed, gap_ahead)
3. **Pluggable Behavior Engines**: ABM (rule-based) and LLM (AI-driven) modes
4. **Lazy Initialization**: Behavior engines created only when needed
5. **Reusable Components**: Leverages Sugarscape's behavior engine architecture

### File Structure

```
NaSch/
├── nasch_mesa.py             # Core Mesa model and Vehicle agent
├── nasch_abm.py              # Traditional ABM simulation
├── nasch_llm.py              # LLM-driven simulation
├── behavior_engine/          # Pluggable behavior engines (core module)
│   ├── __init__.py          # Module initialization
│   ├── ABM_agent.py         # ABM agent wrapper
│   ├── LLM_agent.py         # LLM agent wrapper
│   ├── ABM_based.py         # NaSch 4-step rules implementation
│   └── LLM_based.py         # LLM interaction logic
└── README.md                # This file

Note: Analysis scripts, test scripts, documentation, results, checkpoints, experiments, 
      and utility scripts have been moved to `NaSch_Analysis_Results/` at the project 
      root level (alongside `Codebase/`).
```

## Components

### Vehicle Agent (`nasch_mesa.py`)

The `Vehicle` class represents individual vehicles with pluggable behavior engines:

```python
class Vehicle(Agent):
    def __init__(self, unique_id, model, position, speed=0, max_speed=5, mode="abm"):
        # Behavior engines (lazy initialization)
        self.abm_agent = None
        self.llm_agent = None
    
    def step(self):
        # Build observation (minimal: current_speed, gap_ahead)
        observation = self._build_observation()
        
        # Add model parameters (not observations)
        observation["max_speed"] = self.max_speed
        observation["randomization_prob"] = self.model.p
        
        # Pluggable decision making
        if self.mode == "abm":
            # Traditional NaSch rules
        elif self.mode == "llm":
            # LLM-driven decisions
```

### Behavior Engines

#### ABM Engine (`ABM_based.py`)
Implements the traditional NaSch 4-step rules:
1. **Acceleration**: `speed = min(speed + 1, max_speed)`
2. **Deceleration**: `speed = min(speed, gap_ahead)`
3. **Randomization**: `if random() < p: speed = max(speed - 1, 0)`
4. **Movement**: `position = (position + speed) % L`

#### LLM Engine (`LLM_based.py`)
Generates intelligent prompts for LLM decision making:
- Provides context about traffic situation
- Asks for speed decision considering safety and efficiency
- Maintains same action space constraints

## Usage

#### **ABM Mode (Traditional Rules)**
```bash
python nasch_abm.py
```
- Fast, deterministic simulation
- Follows exact NaSch rules
- No API key required
- Generates `nasch_abm_spacetime.png`

#### **LLM Mode (AI-Driven)**
```bash
python nasch_llm.py
```
- Intelligent, adaptive simulation
- Requires OpenAI API key
- Generates `nasch_llm_spacetime.png`

### Programmatic Usage

```python
from nasch_mesa import NaSchModel

# ABM mode (traditional rules)
abm_model = NaSchModel(L=200, rho=0.25, vmax=5, p=0.2, mode="abm")
for _ in range(100):
    abm_model.step()

# LLM mode (AI-driven)
llm_model = NaSchModel(L=200, rho=0.25, vmax=5, p=0.2, mode="llm")
for _ in range(100):
    llm_model.step()
```

## Dependencies

- `mesa`: Agent-based modeling framework
- `openai`: LLM API integration
- `numpy`: Numerical computations
- `matplotlib`: Visualization

## Setup

1. Install dependencies:
```bash
pip install mesa openai numpy matplotlib
```

2. For LLM mode, set your OpenAI API key (and, optionally, an OpenAI-compatible base URL):
```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # optional
```

## Key Features

### 1. Minimal Observation Space
Agents only observe what they can actually see:
- `current_speed`: Their own speed
- `gap_ahead`: Distance to next vehicle

### 2. Unchanged Action Space
Both ABM and LLM modes output the same action space:
- Speed values in [0, 1, 2, 3, 4, 5]
- No changes to the fundamental NaSch physics

### 3. Pluggable Behavior
Easy switching between decision-making approaches:
- `mode="abm"`: Traditional rule-based behavior
- `mode="llm"`: AI-driven intelligent behavior

### 4. Reusable Architecture
Leverages Sugarscape's proven behavior engine pattern:
- `ABM_agent.py` and `LLM_agent.py` copied directly
- `ABM_based.py` and `LLM_based.py` adapted for NaSch

## Comparison: ABM vs LLM

| Aspect | ABM Mode | LLM Mode |
|--------|----------|----------|
| **Decision Speed** | Fast (O(1)) | Slow (API calls) |
| **Predictability** | High | Variable |
| **Adaptability** | None | High |
| **Realism** | Rule-based | Context-aware |
| **Resource Usage** | Low | High (API costs) |

## Research Applications

This implementation enables research into:
- **Behavioral Differences**: How do rule-based vs AI-driven agents behave?
- **Traffic Flow**: Do intelligent agents improve traffic efficiency?
- **Emergent Behaviors**: What new patterns emerge with LLM decision making?
- **Hybrid Systems**: Can we combine ABM and LLM approaches?

## Future Extensions

- **Mixed Populations**: Some vehicles ABM, others LLM
- **Adaptive Switching**: Vehicles switch modes based on conditions
- **Multi-Agent LLM**: LLM considers multiple vehicles simultaneously
- **Learning**: LLM agents that learn from experience

## References

- Nagel, K., & Schreckenberg, M. (1992). A cellular automaton model for freeway traffic.
- Epstein, J. M., & Axtell, R. (1996). Growing artificial societies: social science from the bottom up.
- Mesa Documentation: https://mesa.readthedocs.io/