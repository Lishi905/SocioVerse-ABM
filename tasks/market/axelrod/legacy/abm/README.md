> **Legacy snapshot, not maintained.** This folder keeps the original scripts verbatim for provenance and as the reference behaviour for the refactored task in [`../../`](../../). It is not imported by the task at runtime and is not part of the test suite. The scripts need their own environment: the Axelrod library (`axelrod`) and Mesa; no LLM client. They do not run against the current kernel.
>
> Original author: Chenyu Li ([@if111111111111111111111](https://github.com/if111111111111111111111)).

# Axelrod Tournament - 200+ Strategies

This implementation uses the [Axelrod Python Library](https://github.com/Axelrod-Python/Axelrod) to run comprehensive tournaments with over 200 strategies for the Iterated Prisoner's Dilemma (IPD).

## Overview

The Axelrod library is a comprehensive research tool containing 204+ strategies for playing the Iterated Prisoner's Dilemma, including:
- All strategies from Axelrod's original 1980 tournament
- All strategies from Axelrod's second 1980 tournament  
- Modern strategies discovered since then
- Meta-strategies that adapt based on opponent behavior
- Evolved strategies using machine learning

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- `axelrod>=4.11.0` - Main library with all strategies
- `matplotlib>=3.5.0` - For plotting results
- `numpy>=1.21.0` - Numerical operations
- `pandas>=1.3.0` - Data analysis
- `mesa>=2.0.0` - Agent-based modeling framework
- `tqdm>=4.62.0` - Progress bars

## Quick Start

### Recommended: Run 224 Short-Run Strategies Tournament

```bash
python axelrod_tournament.py --strategy_set short --turns 200 --repetitions 1 --top_n 50 --seed 42
```

This will run a comprehensive tournament with 224 strategies (takes 5-15 minutes).

### Run Demo Tournament (5 strategies)

```bash
python axelrod_tournament.py --strategy_set demo --turns 200
```

### Run Axelrod's First Tournament (15 strategies)

```bash
python axelrod_tournament.py --strategy_set first --turns 200 --top_n 15
```

### Run All Strategies (~244 strategies)

```bash
python axelrod_tournament.py --strategy_set all --turns 200 --repetitions 10 --top_n 50
```

## Unified Top-50 Workflow

To synchronise with the LLM tournament, run the lightweight batch script:

```bash
python run_experiments.py --turns 10 --seed 123 --output-dir outputs
```

This will:

- execute a round-robin tournament on the **full Axelrod roster** (~240 strategies) with 10 turns per match by default,
- export a timestamped CSV of rankings,
- generate a horizontal bar chart of the average scores (top-N configurable),
- write `abm_top_strategies_latest.json` containing metadata for **all** strategies (unless you pass `--top-n`), which the LLM experiment consumes, and
- include self-play matches by default for parity with the canonical Axelrod setup (use `--exclude-self-play` if you need to match the older “no self-play” workflow).

Use `--strategy-set selected` to revert to the curated 50-strategy roster, or `--strategy-set short` to limit the run to the short-run subset. Additional options:

```bash
python run_experiments.py \
  --turns 10 \
  --seed 123 \
  --strategy-set all \
  --top-n 0 \
  --exclude-self-play \
  --output-dir outputs
```

The JSON contains each strategy’s name, Axelrod class, docstring, and average score.

The curated 50-strategy subset is still available via `--strategy-set selected`, while `--strategy-set short` and the default `--strategy-set all` expose the wider Axelrod rosters without any code changes. Use `--top-n` to trim the exported JSON/plot if needed (set to 0 or omit to include everyone).

## Top-N Subset Replay (ABM/LLM Parity)

Once `abm_top_strategies_latest.json` exists, you can benchmark any Top‑N slice without rerunning the entire 200+ strategy tournament. This is the exact roster the LLM tournament consumes.

1. **Refresh or reuse the ranking artifact**  
   ```powershell
   cd SocioVerse-ABM/raw_simulations/organization_model/axelrod_experiment/abm
   python run_experiments.py --turns 10 --top-n 50 --output-dir outputs_abm_top50
   ```  
   This command emits timestamped CSV/JSON files *and* overwrites `abm_top_strategies_latest.json`. Pass `--ranking-path` to the subset script if you want to target another artifact.

2. **Run the ABM Top‑N replay**  
   ```powershell
   python run_top_subset_experiment.py --top-n 30 --plot
   ```  
   Defaults: 10 turns, seed 123, self-play enabled, one repetition. Results land in `abm/outputs_top_subset/`:

   | File | Contents |
   | --- | --- |
   | `abm_top<k>_rankings_<ts>.csv` | average score per match-up |
   | `abm_top<k>_summary_<ts>.json` | roster metadata plus aggregate stats |
   | `abm_top<k>_bar_<ts>.png` | optional bar chart when `--plot` is set |

   Use `--processes <n>` to control Axelrod’s multiprocessing, and `--ranking-path` / `--top-n` to choose different slices. The companion LLM workflow is documented in `llm_simulation/README.md`.

## Command Line Options

```bash
python axelrod_tournament.py [OPTIONS]
```

### Options:

- `--turns N` - Number of turns per match (default: 200)
- `--repetitions N` - Number of times to repeat tournament (default: 1)
- `--processes N` - Number of parallel processes (default: auto)
- `--seed N` - Random seed for reproducibility (default: random)
- `--strategy_set SET` - Strategy set to use:
  - `demo` - 5 basic strategies (fast)
  - `first` - 15 strategies from Axelrod's 1980 tournament
  - `basic` - Basic deterministic strategies
  - `short` - ~180 short run time strategies (recommended)
  - `all` - All 204+ strategies (slow, may take hours)
- `--top_n N` - Number of top strategies to show in plots (default: 50)
- `--no_plots` - Skip generating plots
- `--output_dir DIR` - Output directory for results (default: current dir)

## Examples

### Quick Test with Demo Strategies

```bash
python axelrod_tournament.py --strategy_set demo --turns 100 --top_n 5
```

### Comprehensive Tournament with Visualization

```bash
python axelrod_tournament.py --strategy_set short --turns 200 --repetitions 5 --top_n 30 --seed 42
```

### Full Tournament (Warning: Very Slow!)

```bash
python axelrod_tournament.py --strategy_set all --turns 200 --repetitions 10 --processes 8 --top_n 50
```

## Output Files

The tournament generates several output files:

1. **CSV Results**: `tournament_results_{set}_{timestamp}.csv`
   - Complete rankings with scores and statistics
   - Columns: Rank, Strategy, Total_Score, Avg_Score_Per_Turn, Std_Score, Median_Score

2. **Rankings Plot**: `rankings_{set}_{timestamp}.png`
   - Horizontal bar chart showing top N strategies
   - Average score per turn

3. **Payoff Matrix**: `payoff_matrix_{set}_{timestamp}.png`
   - Heatmap showing head-to-head results
   - Shows which strategies perform well against which opponents

4. **Wins Distribution**: `wins_{set}_{timestamp}.png`
   - Bar chart showing number of wins for top strategies

## Strategy Categories

### 1. Demo Strategies (5 strategies)
- **Cooperator**: Always cooperates
- **Defector**: Always defects
- **TitForTat**: Cooperates first, then copies opponent
- **Grudger**: Cooperates until opponent defects, then always defects
- **Random**: Cooperates randomly (50%)

### 2. Axelrod's First Tournament (15 strategies)
Includes the 14 submissions plus Random from the famous 1980 tournament:
- TitForTat (winner!)
- Tideman and Chieruzzi
- Nydegger
- Grofman
- Shubik
- Stein and Rapoport
- Grudger
- Davis
- Graaskamp
- Downing
- Feld
- Joss
- Tullock
- Anonymous
- Random

### 3. Short Run Time Strategies (~180 strategies)
All strategies that run efficiently in large tournaments. Filters out:
- Computationally expensive strategies
- Strategies requiring large memory
- Evolved strategies with complex neural networks

### 4. All Strategies (~204 strategies)
Every strategy in the library, including:
- **Reactive**: Copy or respond to opponent moves (TitForTat, Grudger, etc.)
- **Memory-based**: Track opponent history (GoByMajority, SoftGrudger, etc.)
- **Stochastic**: Use randomness (Random, StochasticCooperator, etc.)
- **Meta**: Adapt strategy during game (MetaWinner, MetaHunter, etc.)
- **Mathematical**: Based on game theory (ZD-Extortion, Nash equilibria, etc.)
- **Evolved**: Machine learning (EvolvedANN, EvolvedFSM, etc.)

## Understanding Results

### Payoff Matrix

The classic Prisoner's Dilemma payoff matrix:

```
                Opponent
              C         D
        C  | R=3     S=0  |
Player    |              |
        D  | T=5     P=1  |
```

- **R** (Reward): Mutual cooperation = 3 points
- **T** (Temptation): Defect while opponent cooperates = 5 points
- **S** (Sucker): Cooperate while opponent defects = 0 points
- **P** (Punishment): Mutual defection = 1 point

### Interpreting Rankings

- **Avg Score Per Turn**: Higher is better
- Theoretical maximum: 5.0 (always exploit a cooperator)
- Theoretical minimum: 0.0 (always get exploited)
- Mutual cooperation: 3.0 (usually the best sustainable strategy)
- Mutual defection: 1.0 (worst sustainable outcome)

### Famous Results

From Axelrod's original tournaments:
1. **TitForTat won** despite never defecting first
2. **"Nice" strategies** (never defect first) dominated top ranks
3. **Forgiveness** (return to cooperation after retaliation) is crucial
4. **Clarity** (easy for opponents to understand) helps cooperation
5. **Provocability** (retaliate against defection) prevents exploitation

## Strategy Examples

### Top Performing Strategies (typically)

1. **ZD-GTFT-2**: Zero-Determinant Generous Tit For Tat
2. **TitForTat**: The classic winner
3. **Gradual**: Increasingly severe retaliation
4. **SlowTitForTwoTats2**: Forgives one defection
5. **ContriteTitForTat**: Apologizes for accidental defection

### Interesting Strategies

- **Extortioners** (ZDExtort): Force opponent into unfavorable outcomes
- **Hunters** (DefectorHunter, CooperatorHunter): Target specific opponent types
- **Meta-strategies** (MetaWinner): Switch between sub-strategies
- **Evolved** (EvolvedANN): Neural networks trained via evolution

## Performance Tips

### For Large Tournaments

1. Use `--strategy_set short` instead of `all`
2. Reduce `--turns` to 100-150 for faster results
3. Use `--processes 4` or more for parallelization
4. Start with `--repetitions 1`, increase if needed

### For Reproducibility

Always set `--seed` to a fixed value:
```bash
python axelrod_tournament.py --seed 42 --strategy_set short --turns 200
```

## Project Structure

```
abm/
├── axelrod_agent.py          # Strategy wrapper and utilities
├── axelrod_tournament.py     # Main tournament runner
├── axelrod_model.py          # Legacy Mesa-based model (not used)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## References

1. **Original Papers**:
   - Axelrod, R. (1980). "Effective Choice in the Prisoner's Dilemma"
   - Axelrod, R. (1980). "More Effective Choice in the Prisoner's Dilemma"
   - Axelrod, R. (1984). "The Evolution of Cooperation"

2. **Axelrod Library**:
   - Knight et al. (2016). "An open reproducible framework for the study of the iterated prisoner's dilemma"
   - GitHub: https://github.com/Axelrod-Python/Axelrod
   - Documentation: https://axelrod.readthedocs.io/

3. **Modern Extensions**:
   - Press & Dyson (2012). "Iterated Prisoner's Dilemma contains strategies that dominate any evolutionary opponent"
   - Stewart & Plotkin (2012). "Extortion and cooperation in the Prisoner's Dilemma"

## Troubleshooting

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Out of Memory

```bash
# Use fewer strategies
python axelrod_tournament.py --strategy_set demo

# Or reduce turns
python axelrod_tournament.py --turns 100
```

### Slow Performance

```bash
# Use parallel processing
python axelrod_tournament.py --processes 4

# Use short run time strategies
python axelrod_tournament.py --strategy_set short
```

## License

This implementation uses the Axelrod library which is MIT licensed.

## Contact

For issues or questions about:
- **This implementation**: See main SocioVerse-ABM repository
- **Axelrod library**: https://github.com/Axelrod-Python/Axelrod/issues
