> **Legacy snapshot, not maintained.** This folder keeps the original scripts verbatim for provenance and as the reference behaviour for the refactored task in [`../`](../). It is not imported by the task at runtime and is not part of the test suite. The scripts need their own environment: `numpy`, `matplotlib`, the `openai>=1` client and `dashscope` for the Qwen runs; no Mesa. The code uses the `openai>=1` API, although the original instructions below say `openai==0.28`. They do not run against the current kernel.
>
> Original author: Jianing Shi ([@Brishian427](https://github.com/Brishian427)).
>
> Edits for the public release: a sentence about historical key files that are not part of this repository was removed from the setup notes below.

# Boids Simplified Implementation

A consolidated, clean implementation of the Boids flocking simulation with separate traditional and LLM demos, following architectural simplicity while preserving Boids' asynchronous LLM advantages.

## Architecture

```
Boids/
├── boids_core.py                    # Core engine + embedded OpenAI policy + flocking algorithms
├── boids_traditional.py             # Traditional rule-based behavior demo
├── boids_llm.py                     # LLM-based behavior demo (strict mode, no fallback)
├── evaluate_boids_llm.py            # LLM evaluation script (batch mode)
├── evaluate_boids_llm_per_step.py   # Per-step LLM evaluation (concurrent)
├── evaluate_boids_traditional.py    # Traditional ABM evaluation
├── run_all_models_per_step.py       # Multi-model pipeline runner
└── README.md                        # This file

Note: Analysis scripts, test scripts, and historical results have been moved to 
      `Boids_Analysis_Results/` at the project root level (alongside `Codebase/`).
```

Following architectural simplicity:
- **1 core file** with embedded OpenAI policy and flocking algorithms
- **2 demo files** for different modes (traditional vs LLM)
- **No complex directory nesting** or import path manipulation
- **True error propagation** for proper ABM-LLM comparison
- **Mode separation** - traditional and LLM demos are completely separate

## Files

### `boids_core.py` (~450 lines)
- **Boid** dataclass with position, velocity, behavior parameters
- **BoidsEngine** class with:
  - Flocking algorithms (separation, alignment, cohesion)
  - Async LLM integration with threading
  - Embedded **OpenAIPolicy** class (no separate imports needed)
  - Scenario support (open_field, predator_prey)
- **Constants** and schemas embedded directly
- **True error handling** - no silent fallbacks

### `boids_traditional.py`
Traditional rule-based behavior demo featuring:
- Traditional flocking demo with matplotlib visualization
- Interactive controls (sliders, pause/reset buttons)
- Manual start/stop controls with pause/resume functionality
- Refresh button for manual interface updates
- 40 boids in traditional mode
- Simple import: `from boids_core import BoidsEngine`

### `boids_llm.py`
LLM-based behavior demo featuring:
- LLM-enhanced demo with OpenAI integration
- Behavior info display with error monitoring
- Manual start/stop controls with pause/resume functionality
- Refresh button for manual interface updates
- Scenario switching and LLM interval controls
- Enhanced error tracking for debugging
- Strict LLM mode with no fallback mechanisms
- Simple import: `from boids_core import BoidsEngine`

## Quick Start

### Prerequisites
```bash
pip install numpy matplotlib openai==0.28
```

**Important**: Use OpenAI library version 0.28 for compatibility with the existing code.

### Set up API Keys
Set environment variables for API keys:
```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key-here"

# Qwen (via DashScope)
export DASHSCOPE_API_KEY="your-dashscope-api-key-here"

# DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-api-key-here"
```

**Note:** API keys should be stored as environment variables.

### Running the Demos

**Traditional Rule-Based Demo:**
```bash
cd Boids
python boids_traditional.py
```

**LLM-Based Demo:**
```bash
cd Boids
python boids_llm.py
```

### Running Evaluations

**Traditional ABM Evaluation:**
```bash
python evaluate_boids_traditional.py --steps 100 --seed 42
```

**LLM Evaluation (Batch Mode):**
```bash
python evaluate_boids_llm.py --model qwen --steps 100 --seed 42
```

**LLM Evaluation (Per-Step, Concurrent):**
```bash
python evaluate_boids_llm_per_step.py --model qwen --steps 100 --seed 42 --batch-size 45
```

**Run All Models:**
```bash
python run_all_models_per_step.py --steps 100
```

### Error Testing
```bash
export OPENAI_API_KEY="invalid"
python boids_llm.py
# Expected: Console shows LLM errors, UI shows error count
```

## Key Improvements

### Simplification
- **15+ files → 3 files** (80% reduction)
- **5-level nesting → flat structure**
- **Complex imports → simple imports**
- **Dynamic loading → embedded policy**

### Error Handling
- **No silent fallbacks** - LLM errors surface properly
- **Error monitoring** in UI for debugging
- **Traditional fallback count** displayed
- **Enables proper ABM vs LLM comparison**

### Performance
- **Async LLM processing** (Boids advantage over Sugarscape)
- **Background threading** with queues
- **Batched LLM requests** (every N steps)
- **Real-time visualization** maintained

## Comparison

| Aspect | Original Boids | Simplified Boids | Sugarscape |
|--------|---------------|------------------|------------|
| Files | 15+ files | 3 files ✅ | 5 files |
| Nesting | 5 levels | 0 levels ✅ | 2 levels |
| Import complexity | High | Simple ✅ | Simple |
| LLM approach | Async ✅ | Async ✅ | Synchronous |
| Error handling | Silent fallbacks | True errors ✅ | Not specified |
| Maintenance | Complex | Simple ✅ | Simple |

**Result:** Best of both worlds - Sugarscape's simplicity + Boids' performance + proper error handling
