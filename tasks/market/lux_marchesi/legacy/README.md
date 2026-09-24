> **Legacy snapshot, not maintained.** This folder keeps the original scripts verbatim for provenance and as the reference behaviour for the refactored task in [`../`](../). It is not imported by the task at runtime and is not part of the test suite. The scripts need their own environment: `numpy`, `scipy`, `matplotlib` and the `openai>=1` client; no Mesa. The LLM path also imports the pre-refactor `socioverse.behavior_engine` kernel, which is not shipped here. They do not run against the current kernel.
>
> Original author: Zijian Ling ([@Georgelingzj](https://github.com/Georgelingzj)).

# Lux-Marchesi Financial Market Model

A Python implementation of the Lux-Marchesi agent-based financial market model, supporting both traditional ABM (rule-based) and LLM (reasoning-based) decision modes.

## Model Overview

The Lux-Marchesi model simulates a financial market with heterogeneous traders who can switch between different trading strategies. It reproduces several empirical "stylized facts" of financial markets:

- **Fat tails** in return distributions
- **Volatility clustering** (persistence of large/small returns)
- **Absence of linear autocorrelation** in returns
- **Mean reversion** of prices toward fundamental value

## Installation

The model uses the `socioverse` framework for behavior engine integration. Ensure you have the project dependencies installed:

```bash
cd /path/to/SocioVerse-ABM
```

### Install Dependencies

Copy the dependencies from the requirements.txt file into the SocioVerse-ABM directory.

```txt
aiohttp
anthropic
dashscope
faiss-cpu
FlagEmbedding
matplotlib
mesa
ndlib
networkx
numpy
openai
pandas
Pillow
requests
scikit-learn
scipy
setuptools
streamlit
tavily-python
tqdm
transformers

```

then run the following command to install the dependencies:

```bash
pip install -r requirements.txt
```


For LLM mode:
- openai
- Environment variable `OPENAI_API_KEY` must be set

## Usage

### Basic ABM Simulation

```bash
cd raw_simulations/market_model/Lux-Marchesi

# Run ABM mode (rule-based)
python lux_marchesi_model.py --mode abm --T 50000 --N 500 --seed 42

# Run with llm mode
python run_experiment.py --mode llm --T 100 --N 50 \
    --llm_decision_interval 1 \
    --llm_cache_duration 1 \
    --llm_max_agents 500

# Run with LLM mode in root directory
python raw_simulations/market_model/Lux_Marchesi/run_experiment.py \
    --mode llm \
    --N 10 \
    --T 20 \
    --llm_decision_interval 1 \
    --llm_cache_duration 1 \
    --llm_max_agents 500 \
    --llm_rate_limit 0 \
    --llm_debug

# Run with custom parameters
python lux_marchesi_model.py --mode abm --T 100000 --N 1000 --alpha1 1.5 --beta 2e-4
```

### LLM Simulation (Per-Agent Mode)

The LLM mode uses per-agent decision making similar to NaSch/Boids/Schelling models:
- Each agent independently calls LLM to decide whether to switch trader type
- Decisions are cached for efficiency (default: 100 steps)
- Background thread handles async API calls
- Rate limiting and batch processing for cost control

```bash
# Run LLM mode with default parameters (requires OPENAI_API_KEY)
python lux_marchesi_model.py --mode llm --T 10000 --N 500 --model gpt-4o-2024-08-06

# Smaller test run
python lux_marchesi_model.py --mode llm --T 100 --N 50 --llm_debug
```

#### LLM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--llm_decision_interval` | 50 | Steps between LLM decision rounds |
| `--llm_cache_duration` | 100 | How many steps to cache each decision |
| `--llm_max_agents` | 20 | Max agents to process per round |
| `--llm_debug` | False | Enable verbose LLM debug output |

#### Cost Estimation

With default parameters (T=50000, interval=50, batch=20):
- LLM calls ≈ 50000/50 × 20 = **20,000 calls** (vs. 25 million with naive per-step)

### Run from Project Root

#### ABM Mode (Rule-based)

```bash
# Quick test with default parameters
python raw_simulations/market_model/Lux_Marchesi/run_experiment.py \
    --mode abm --T 5000 --seed 42

# Paper-aligned: Stylized facts (fat tails, volatility clustering)
python raw_simulations/market_model/Lux_Marchesi/run_experiment.py \
    --mode abm --preset paper_stylized --T 100000 --seed 42

# Paper-aligned: Fig.16-style intermittency (returns + chartist co-movement)
python raw_simulations/market_model/Lux_Marchesi/run_experiment.py \
    --mode abm --preset paper_fig16 --T 100000 --seed 42

# Custom config file
python raw_simulations/market_model/Lux_Marchesi/run_experiment.py \
    --mode abm --T 50000 \
    --config raw_simulations/market_model/Lux_Marchesi/configuration_paper.json
```

#### LLM Mode (LLM-augmented agents)

```bash
# LLM mode with optimized parameters
# - Sequential update scheme for paper-aligned dynamics
# - Lower batch size to prevent rate limit errors
# - Decision interval=50 for cost efficiency (~20k LLM calls for T=5000)
python raw_simulations/market_model/Lux_Marchesi/run_experiment.py \
    --mode llm --T 5000 --seed 42 \
    --price_update_mode walrasian \
    --abm_update_scheme sequential \
    --llm_decision_interval 50 \
    --llm_cache_duration 50 \
    --llm_max_agents 50 \
    --llm_concurrent_batch_size 10 \
    --llm_batch_delay 0.5
```

#### Comparison Mode (ABM vs LLM)

```bash
# Run ABM and LLM with identical parameters for direct comparison
python raw_simulations/market_model/Lux_Marchesi/run_experiment.py \
    --compare --T 1000 --seed 42 \
    --llm_decision_interval 10 \
    --llm_cache_duration 10 \
    --llm_max_agents 30 \
    --llm_concurrent_batch_size 5 \
    --llm_batch_delay 1.0
```

## Paper-Aligned Simulations

The implementation includes preset configurations for reproducing results from the original Lux-Marchesi papers (especially the intermittent behavior shown in Fig.16 and the stylized facts like power-law tails with exponent ~3).

###  Paper-Aligned Runs

```bash
# Fig.16-style intermittency (returns + chartist fraction co-move)
python run_experiment.py --preset paper_fig16 --seed 42

# Stylized facts reproduction (fat tails, volatility clustering, power-law)
python run_experiment.py --preset paper_stylized --seed 42

# Using the paper-aligned config file (ABM mode)
python raw_simulations/market_model/Lux_Marchesi/run_experiment.py \
    --mode abm --seed 42 \
    --config raw_simulations/market_model/Lux_Marchesi/configuration_paper.json

# Using the paper-aligned config file (LLM mode)
python raw_simulations/market_model/Lux_Marchesi/run_experiment.py \
    --mode llm --seed 42 \
    --config raw_simulations/market_model/Lux_Marchesi/configuration_paper.json \
    --llm_decision_interval 200 \
    --llm_cache_duration 0 \
    --llm_max_agents 100 \
    --llm_concurrent_batch_size 50 \
    --llm_batch_delay 0.5
```
