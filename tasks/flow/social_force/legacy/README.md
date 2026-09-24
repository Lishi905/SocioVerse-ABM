> **Legacy snapshot, not maintained.** This folder keeps the original scripts verbatim for provenance and as the reference behaviour for the refactored task in [`../`](../). It is not imported by the task at runtime and is not part of the test suite. The scripts need their own environment: `numpy`, `matplotlib` and the `openai>=1` client; no Mesa. The code uses the `openai>=1` API, although the original instructions below say `openai==0.28`. They do not run against the current kernel.
>
> Original author: Jianing Shi ([@Brishian427](https://github.com/Brishian427)).

# Social Force Model - Simplified Implementation

A consolidated, high-performance implementation of the Social Force Model for pedestrian dynamics simulation with full LLM integration and comprehensive analysis tools.

## Overview

This implementation consolidates the Social Force Model from scattered legacy files into a streamlined architecture with separate traditional and LLM demos. It features performance optimizations, comprehensive LLM integration, and redesigned interfaces with manual control and refresh capabilities.

## Architecture

```
Social_Force_Model/
├── social_force_core.py              # Core engine + embedded OpenAI policy + performance optimizations
├── social_force_traditional_demo.py   # Traditional rule-based behavior demo
├── social_force_llm_demo.py          # LLM-based behavior demo (strict mode, no fallback)
├── social_force_llm_pure.py         # Pure LLM implementation
├── social_force_llm_pure_demo.py   # Pure LLM demo
└── README.md                         # This file

Note: Analysis scripts, test scripts, documentation, results, and utility scripts have been 
      moved to `SFM_Analysis_Results/` at the project root level (alongside `Codebase/`).
```

## Quick Start

### Prerequisites
```bash
pip install numpy matplotlib openai==0.28
```

**Important**: Use OpenAI library version 0.28 for compatibility with the existing code.

### Set up OpenAI API Key
Either set environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create `OPENAI_API_KEY.txt` in the Social_Force_Model directory with your API key.

### Running the Demos

**Traditional Rule-Based Demo:**
```bash
cd Social_Force_Model
python social_force_traditional_demo.py
```

**LLM-Based Demo:**
```bash
cd Social_Force_Model
python social_force_llm_demo.py
```

### Running Scripts

**From root directory:**
```bash
cd Social_Force_Model
python scripts/run_sfm_properly.py
python scripts/run_sfm_gpt5.py
```

**Or from scripts directory:**
```bash
cd Social_Force_Model/scripts
python run_sfm_properly.py
```

### Using Utility Scripts

**From root directory:**
```bash
cd Social_Force_Model
python utils/calculate_llm_cost.py
python utils/monitor_simulations.py
```

## Files Description

### Core Files

**`social_force_core.py`**
The core simulation engine containing:
- **Embedded OpenAIPolicy**: Complete LLM integration with timeout protection
- **Performance Optimizations**: Request throttling, policy instance reuse, 3-second timeouts
- **Complex Observation Building**: Bottleneck detection, crowd metrics, visual FOV, doorway navigation
- **All Three Scenarios**: Bidirectional, bottleneck, and open space configurations
- **Batch Processing**: LLM processes multiple pedestrians per API call
- **Error Transparency**: Strict mode with true error propagation (no silent fallbacks)

**`social_force_traditional_demo.py`**
Traditional rule-based behavior demo featuring:
- Single-panel matplotlib visualization
- Interactive sliders for key parameters (spawn rate, desired speed, max speed)
- Radio buttons for scenario selection (bidirectional, bottleneck)
- Manual start/stop controls with pause/resume functionality
- Refresh button for manual interface updates
- Clean, intuitive interface for testing traditional behavior

**`social_force_llm_demo.py`**
LLM-based behavior demo featuring:
- Single-panel matplotlib visualization with LLM status display
- Interactive sliders for key parameters (spawn rate, desired speed, max speed)
- Radio buttons for scenario selection (bidirectional, bottleneck)
- Manual start/stop controls with pause/resume functionality
- Refresh button for manual interface updates
- Strict LLM mode with no fallback mechanisms
- Real-time LLM agent count and status monitoring

### Documentation (`docs/`)

- **START_HERE.md**: Getting started guide
- **SIMULATION_SPECIFICATIONS.md**: Complete simulation parameters and API rate limits
- **LLM_ANALYSIS_REPORT.md**: Detailed analysis of LLM usage, caching, and checkpoint mechanisms
- **LLM_SIMULATION_LOGIC.md**: Complete explanation of LLM simulation workflow
- **LLM_VS_ABM_COMPARISON.md**: Comparison between LLM and ABM modes
- **HOW_LLM_WORKS_PLAIN_ENGLISH.md**: Plain English explanation of LLM integration
- **API_KEYS_DOCUMENTATION.md**: API key setup instructions
- Benchmark and results documentation files

### Scripts (`scripts/`)

- **run_sfm_properly.py**: Standard simulation run script
- **run_sfm_gpt5.py**: GPT-5 model run script
- **run_sfm_deepseek_r1.py**: DeepSeek R1 model run script
- **run_sfm_qwen_thinking.py**: Qwen thinking mode run script
- **run_deepseek_benchmark.py**: DeepSeek benchmark script
- **run_qwen_benchmark.py**: Qwen benchmark script
- **run_all_reasoning_parallel_sfm.py**: Parallel reasoning models run script

### Utilities (`utils/`)

- **evaluate_sfm.py**: Evaluation and analysis script
- **monitor_simulations.py**: Real-time simulation monitoring
- **calculate_llm_cost.py**: LLM API cost calculation
- **calculate_100_steps_cost.py**: Cost calculation for 100 steps
- **calculate_100_steps_all_agents_cost.py**: Cost calculation for all agents per step
- **check_progress.py**: Check simulation progress
- **check_reasoning_progress.py**: Check reasoning model progress
- **check_running_processes.py**: Check running simulation processes
- **check_file_timestamps.py**: Check file modification times
- **quick_report.py**: Quick results report
- **report_all_reasoning_models.py**: Report all reasoning model results
- **report_reasoning_results.py**: Report reasoning results
- **debug_bottleneck.py**: Debug bottleneck scenario

### Tests (`tests/`)

- **test_simulation_run.py**: Simulation run test

## Scenarios

### Bidirectional
Simple rectangular walkway with pedestrians moving in both directions. Good for basic flocking behavior testing.

### Bottleneck
Narrow doorway (1.2m) in the middle of the corridor. Tests navigation through constrained spaces and congestion dynamics.

### Open Space
~~Enclosed rectangular area with boundary walls. Useful for studying crowd behavior in open areas.~~ **Removed** - Only bidirectional and bottleneck scenarios are available in the current implementation.

## Modes

### ABM (Agent-Based Model)
Traditional rule-based behavior using social forces:
- Desired velocity towards destination
- Repulsive forces from other pedestrians
- Repulsive forces from walls
- Relaxation towards desired speed

### LLM (Large Language Model)
AI-driven behavior using OpenAI GPT models:
- Complex observation building with bottleneck detection
- Crowd metrics and visual field of view
- Tactical decision making for navigation
- Batch processing for efficiency
- **Strict Mode**: No fallback to traditional behavior on API errors
- **Error Transparency**: Simulation stops on LLM failures for true error visibility

## Performance Optimizations

### Request Throttling
- Bounded queue (maxsize=100) prevents memory overflow
- Automatic throttling when queue is full
- Real-time monitoring of throttled requests

### Policy Instance Reuse
- Single OpenAIPolicy instance created once and reused
- Eliminates redundant initialization overhead
- Significant performance improvement for batch processing

### Timeout Protection
- 10-second timeout wrapper around API calls (increased from 3 seconds)
- Prevents indefinite hangs from API issues
- Fast error detection and recovery

### Performance Monitoring
Real-time statistics tracking:
- ✅ Requests sent
- ✅ Responses received  
- ❌ Errors encountered
- ⚠️ Throttled requests
- 🔧 Policy creation status
- 📊 Pending request count

## Interactive Controls

### Manual Control System
Both demos feature a **manual start/stop system** - simulations do not start automatically:

### Buttons
- **START/STOP**: Main control to begin/end simulation (green/red)
- **PAUSE/RESUME**: Pause/resume running simulation (red/green)
- **RESET**: Stop and clear simulation (blue)
- **REFRESH**: Manually refresh interface and update all visual elements (yellow)

### Sliders
- **Spawn Rate**: Control pedestrian generation rate (0.1-2.0 peds/s)
- **Desired Speed**: Set target walking speed (0.5-3.0 m/s)
- **Max Speed**: Adjust maximum achievable speed (1.0-5.0 m/s)

### Radio Buttons
- **Scenario**: Switch between bidirectional and bottleneck scenarios
- **Mode Separation**: Traditional and LLM demos are completely separate

### Status Display
- **Traditional Demo**: Shows "Traditional Mode Ready/Running/Paused"
- **LLM Demo**: Shows "LLM Mode Ready/Running/Paused" with agent count

## LLM Integration Details

### Observation Building
Each pedestrian builds a comprehensive observation including:
- Self state (position, velocity, speed, goal direction)
- Local neighbors within 200° field of view
- Environment context (wall distances, doorway metrics)
- Crowd metrics (density, flow direction)
- Scenario-specific context (bottleneck detection, queue status)

### Action Application
LLM actions are applied with bounded modulation:
- Speed modulation (0.3-1.0x factor)
- Direction bias (angle offset ±90°, strength 0.0-1.0)
- Social parameters (personal space, assertiveness) - ⚠️ **Not yet implemented**
- Tactical intent (normal, yield, push, follow) - ⚠️ **Not yet implemented**

### Batch Processing
- Multiple pedestrians processed per API call
- Individual timeout protection for each request
- **Strict Error Handling**: No graceful degradation - simulation stops on API failures

## Cost Analysis

### Current Mechanism (10 minutes)
- **Total Requests**: 120 (每5秒1轮，每轮4个agent)
- **Total Cost**: ~$1.68

### If Every Step Called LLM (10 minutes)
- **Total Requests**: 720,000 (每步每个agent)
- **Total Cost**: ~$2,520
- **Savings**: 99.9% with caching mechanism

See `utils/calculate_llm_cost.py` for detailed cost calculations.

## Troubleshooting

### Common Issues

**"OPENAI_API_KEY not set"**
- Ensure API key is set in environment or `OPENAI_API_KEY.txt` file in the Social_Force_Model directory
- Check file permissions and encoding

**"LLM API call timed out"**
- Network connectivity issues
- API rate limiting
- Check OpenAI service status
- **Note**: Timeout increased to 10 seconds for better reliability

**"Queue full, skipping request"**
- Normal throttling behavior when LLM requests exceed capacity
- Adjust spawn rate or LLM decision interval if needed

**"ModuleNotFoundError: No module named 'openai'"**
```bash
pip install openai==0.28
```

**"API Removed in V1" Error**
- This occurs when using OpenAI library v1.x+ with v0.x code
- Solution: Downgrade to compatible version:
```bash
pip install openai==0.28
```

### Performance Issues

**High pending request count**
- Check API key validity
- Verify network connectivity
- Consider reducing spawn rate

**Slow visualization**
- Reduce number of pedestrians
- Increase time step (dt) parameter
- Close other resource-intensive applications

### Debug Mode
Enable detailed logging:
```bash
export SFM_LLM_DEBUG=1
export SFM_LLM_STRICT=1
```

## Technical Details

### Force Calculations
- **Desired Force**: Drives pedestrians towards their destination
- **Social Force**: Repulsive force between pedestrians (exponential decay)
- **Wall Force**: Repulsive force from boundaries
- **Relaxation**: Damping towards desired velocity

### Integration Method
Velocity Verlet integration for stable, energy-conserving dynamics:
```
position(t+dt) = position(t) + velocity(t)*dt + 0.5*acceleration(t)*dt²
velocity(t+dt) = velocity(t) + acceleration(t)*dt
```

### Parameters
- **V_0**: Pedestrian repulsion strength (2.1 m²/s²)
- **sigma**: Interaction range (0.3 m)
- **U_0**: Wall repulsion strength (10.0 m²/s²)
- **tau**: Relaxation time (0.5 s)
- **v_0**: Desired speed (1.34 m/s)

## Comparison with Original

### Improvements
- **Performance**: 10x reduction in pending requests through throttling
- **Reliability**: 10-second timeout prevents infinite hangs
- **Monitoring**: Comprehensive real-time statistics
- **Architecture**: Clean separated demo structure vs mixed legacy files
- **Error Handling**: True error propagation instead of silent fallbacks
- **Interface**: Manual start/stop controls with refresh functionality
- **Mode Separation**: Strict separation between traditional and LLM behavior
- **API Compatibility**: Fixed OpenAI library version compatibility issues
- **Folder Organization**: Well-organized structure with docs/, scripts/, utils/, tests/

### Preserved Features
- All complex observation building logic
- Batch processing for LLM requests
- Bidirectional and bottleneck scenario configurations
- Rich metrics and analysis capabilities
- Interactive visualization controls

## License

This implementation is part of the LLM-ABM integration task group and follows the same licensing terms as the parent project.

## Contributing

When making changes:
1. Follow the established architecture patterns
2. Maintain performance optimizations
3. Update documentation for new features
4. Test both ABM and LLM modes
5. Verify all three scenarios work correctly
6. Keep files organized in appropriate directories
