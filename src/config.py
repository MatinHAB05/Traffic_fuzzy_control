"""Global configuration constants for the traffic simulation,
fuzzy controller and optimization algorithms."""

# --- Fuzzy domain limits ---
# max queue length used in fuzzy input domain (vehicles)
QUEUE_MAX = 20
CYCLE_TIME = 60         # total signal cycle length (seconds)
GREEN_MIN = 5           # minimum green time (seconds)
GREEN_MAX = 55          # maximum green time (seconds)
SERVICE_RATE = 0.5      # vehicles discharged per second during green

# --- Traffic arrival rates (vehicles / second) ---
ARRIVAL_RATE_1 = 0.25
ARRIVAL_RATE_2 = 0.20

# --- Simulation episode length ---
N_CYCLES = 30           # signal cycles per simulation episode
N_SEEDS = 4             # random seeds averaged per fitness evaluation

# --- Cost function weights: C = ALPHA*W + BETA*Q + GAMMA*S ---
ALPHA = 1.0
BETA = 1.0
GAMMA = 0.05

RANDOM_SEED = 97


# ========== PSO Hyperparameters ==========
PSO_N_PARTICLES = 1000
PSO_ITERATIONS = 50
PSO_W_START = 0.8
PSO_W_END = 0.4

PSO_C1_START = 3.5
PSO_C1_END = 0.5
PSO_C2_START = 0.5
PSO_C2_END = 3.5

PSO_VMAX_FRAC = 0.2                    # velocity limit = fraction of range

# ========== Evaluation ==========
PSO_SEEDS_EARLY = 1                    # seeds in first half of iterations
PSO_SEEDS_LATE = 5                    # seeds in second half
PSO_SEEDS_END_MUL_TERM = 0.9
# ========== Stagnation & Mutation ==========
# start mild mutation after this many stagnant steps
PSO_STAGNATION_LIMIT = 3
PSO_HEAVY_STAGNATION_LIMIT = 5         # reset particle after this many steps

PSO_MUTATION_MIN_RATE = 0.01
PSO_MUTATION_MAX_RATE = 0.4
PSO_MUTATION_EXTRA_RATE = 0.08         # extra rate per stagnant step beyond limit

# probability to jump near gbest instead of random reset
PSO_ELITE_LEARN_PROB = 0.15
# ± fraction of range when jumping near gbest
PSO_ELITE_NOISE_FRAC = 0.05
PSO_RESET_VEL_STD = 0.02               # std of velocity after a reset

# ± fraction of dimension range for mutation
PSO_MUTATION_SHAKE_FRAC = 0.1

# ========== Stagnation tolerance (adaptive) ==========
PSO_TOL_ABSOLUTE = 1e-6
PSO_TOL_RELATIVE = 1e-4                # relative to previous personal best cost

# ========== Parallel ==========
# number of processes (set to None for auto)
PSO_N_WORKERS = 8


# --- ACO hyperparameters (discretized continuous ACO) ---
ACO_N_ANTS = 500
ACO_ITERATIONS = 100
ACO_N_LEVELS = 10
ACO_RHO = 0.3
ACO_Q = 1.0
ACO_ALPHA = 1.0
ACO_TAU0 = 1.0
ACO_ELITE_BONUS = 2.0   # extra pheromone deposit for the best-so-far solution
# --- Pheromone bounds (MMAS) ---
ACO_TAU_MIN = 1e-6
ACO_TAU_MAX = 100.0

# --- Dynamic parameter ranges (linear interpolation between min and max) ---
# alpha: low early (explore) → high late (exploit)
ACO_ALPHA_MIN = 0.2
ACO_ALPHA_MAX = 5.0
# beta: high early (follow heuristic) → low late (follow pheromone)
ACO_BETA_MIN = 0.0
ACO_BETA_MAX = 5.0
# rho: high early (forget bad trails) → low late (converge)
ACO_RHO_MIN = 0.05
ACO_RHO_MAX = 0.9
# epsilon‑greedy: high early (explore) → low late (exploit)
ACO_EPSILON_MIN = 0.05
ACO_EPSILON_MAX = 0.3

# --- Local pheromone update (ACS) ---
# local_rho: high early (diversity) → low late (convergence)
ACO_LOCAL_RHO_MIN = 0.05
ACO_LOCAL_RHO_MAX = 0.2

# --- Stagnation control ---
# iterations without improvement before intervention
ACO_STAGNATION_LIMIT = 15

# --- Pheromone deposition ---
# if True, only top fraction deposit; else all ants deposit
ACO_USE_RANK_DEPOSIT = True
# top 20% of ants deposit (if rank deposit enabled)
ACO_RANK_FRACTION = 0.20

# --- Mutation ---
ACO_MUTATION_PROB = 0.10           # probability per ant to mutate one dimension
ACO_MUTATION_SCALE = 0.05          # fraction of range for Gaussian step

# --- Elite reinforcement ---
# reduces the elite bonus to avoid over‑exploitation
ACO_ELITE_BONUS_MULTIPLIER = 0.5

# ==========================================
# ACOR (Ant Colony Optimization for Continuous) Parameters
# ==========================================

# Core algorithm parameters
ACOR_N_ANTS = 100
ACOR_ITERATIONS = 200
ACOR_ARCHIVE_SIZE = 30
# concentration (lower = more focused on top solutions)
ACOR_Q_INTENSITY = 0.5
ACOR_XI_EVAPORATION = 0.5       # exploration scale (sigma multiplier)
ACOR_RANDOM_SEED = 42

# Stagnation control
ACOR_STAGNATION_LIMIT = 6       # iterations without improvement before intervention
ACOR_ENABLE_ADAPTIVE_XI = True  # if True, increase xi when stagnation detected
ACOR_XI_MAX = 0.95              # maximum allowed xi (for adaptive mode)
ACOR_XI_BASE = 0.5              # base xi (reset after stagnation)

# Archive reset strategy
ACOR_RESET_ON_STAGNATION = True
ACOR_KEEP_BEST_RATIO = 0.5      # fraction of best solutions to keep when resetting

# Minimum sigma (to prevent collapse) – fraction of each dimension’s range
ACOR_MIN_SIGMA_FRACTION = 0.01

# Process pool settings
ACOR_MAX_WORKERS = 8            # number of parallel evaluation workers

# Initialisation method: 'uniform' or 'latin_hypercube'
ACOR_INIT_METHOD = 'latin_hypercube'
