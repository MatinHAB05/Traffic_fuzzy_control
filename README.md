# Computational Intelligence Project — Intelligent Traffic Control with Fuzzy Logic + PSO + ACO

Complete implementation of the Computational Intelligence (CI) course project: A Mamdani fuzzy controller to determine the green light duration of a two-way intersection, optimization of its parameters using **PSO** and **ACO** algorithms, and a performance comparison.

## Project Structure
```text
traffic_fuzzy_control/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── main.py                    # Full project execution (baseline -> PSO -> ACO -> comparison)
├── run_baseline.py            # Runs only the manually designed fuzzy controller
├── run_pso.py                 # Runs only the PSO optimization
├── run_aco.py                 # Runs only the ACO optimization
├── run_comparison.py          # Generates comparison plots/table (requires running the three above)
├── src/
│   ├── config.py               # All settings and hyperparameters in one place
│   ├── fuzzy_system.py         # Mamdani fuzzy controller (membership functions, rules, defuzzification)
│   ├── simulation.py           # Discrete-time simulation of the traffic intersection
│   ├── cost.py                 # Cost function $C = a*W + b*Q + g*S$
│   ├── pso.py                  # PSO implementation
│   ├── aco.py                  # ACO implementation (discretized version for continuous space)
│   └── plotting.py             # Helper functions for plotting
├── results/
│   ├── data/                   # JSON output for each stage (raw numerical results)
│   └── plots/                  # PNG plots (convergence, comparison, stability)
└── report/
└── report.md               # Full report in Persian (modeling, design, results, analysis)

The results and plots available in `results/` are pre-generated and saved (from an actual complete run of the project) so there is no need to re-run it; however, you can run the entire project from scratch if you wish.

## How to Run

### 1. Install Dependencies

bash
pip install -r requirements.txt

Only `numpy` and `matplotlib` are required.

### 2. Full Project Execution (Simplest Method)

bash
python main.py

This command sequentially runs the baseline controller, executes PSO and then ACO on the fuzzy parameters, and finally generates the comparison plots and the final table. All outputs are saved in the `results/` folder. The total execution time with default settings is approximately 30 seconds.

### 3. Step-by-Step Execution (Optional)

bash
python run_baseline.py     # Only baseline
python run_pso.py          # Only PSO (must be run after baseline if you want the comparison to work)
python run_aco.py          # Only ACO
python run_comparison.py   # Final comparison (requires the output of the three above)

## Configurable Settings

All simulation parameters, the cost function, and PSO/ACO hyperparameters are centralized in the `src/config.py` file. This includes vehicle arrival rates, light cycle length, number of particles/ants, algorithm iterations, etc.