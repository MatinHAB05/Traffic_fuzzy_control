
import numpy as np
from . import config as cfg
from .cost import evaluate, get_bounds, repair


class ACO:
    def __init__(self, n_ants=cfg.ACO_N_ANTS, n_iter=cfg.ACO_ITERATIONS,
                 n_levels=cfg.ACO_N_LEVELS, seed=cfg.RANDOM_SEED):
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.n_levels = n_levels
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = get_bounds()
        self.dim = len(self.lo)

        # Discretized levels
        self.levels = np.array([
            np.linspace(self.lo[d], self.hi[d], n_levels) for d in range(self.dim)
        ])

        # Pheromone initialisation (MMAS style)
        self.tau0 = cfg.ACO_TAU0
        self.tau_min = cfg.ACO_TAU_MIN
        self.tau_max = cfg.ACO_TAU_MAX
        self.tau = np.full((self.dim, n_levels), self.tau0)

        # Heuristic: initial attraction to the centre
        mid = (self.lo + self.hi) / 2.0
        self.eta = np.array([
            1.0 / (1.0 + np.abs(self.levels[d] - mid[d]))
            for d in range(self.dim)
        ])

        # Stagnation control
        self.stagnation_limit = cfg.ACO_STAGNATION_LIMIT
        self.stagnation_counter = 0
        self.best_cost_history = []

        # Deposition parameters
        self.use_rank_deposit = cfg.ACO_USE_RANK_DEPOSIT
        self.rank_fraction = cfg.ACO_RANK_FRACTION

        # Mutation parameters
        self.mutation_prob = cfg.ACO_MUTATION_PROB
        self.mutation_scale = cfg.ACO_MUTATION_SCALE

        # Dynamic parameter ranges (read from config)
        self.alpha_min = cfg.ACO_ALPHA_MIN
        self.alpha_max = cfg.ACO_ALPHA_MAX
        self.beta_min = cfg.ACO_BETA_MIN
        self.beta_max = cfg.ACO_BETA_MAX
        self.rho_min = cfg.ACO_RHO_MIN
        self.rho_max = cfg.ACO_RHO_MAX
        self.epsilon_min = cfg.ACO_EPSILON_MIN
        self.epsilon_max = cfg.ACO_EPSILON_MAX
        self.local_rho_min = cfg.ACO_LOCAL_RHO_MIN
        self.local_rho_max = cfg.ACO_LOCAL_RHO_MAX
        self.elite_bonus_multiplier = cfg.ACO_ELITE_BONUS_MULTIPLIER

    def _get_dynamic_params(self, t):
        """Compute alpha, beta, rho, epsilon based on progress."""
        progress = t / self.n_iter

        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * progress
        beta = self.beta_max - (self.beta_max - self.beta_min) * progress
        rho = self.rho_max - (self.rho_max - self.rho_min) * progress
        epsilon = self.epsilon_max - (self.epsilon_max - self.epsilon_min) * progress

        return alpha, beta, rho, epsilon

    def _update_heuristic(self, best_params):
        """Heuristic points to the current best solution."""
        if best_params is None:
            return
        for d in range(self.dim):
            self.eta[d] = 1.0 / (1.0 + np.abs(self.levels[d] - best_params[d]))
        self.eta = np.clip(self.eta, 1e-6, None)

    def _mutate(self, params):
        """Apply small Gaussian mutation to a random dimension."""
        if self.rng.random() > self.mutation_prob:
            return params
        d = self.rng.integers(self.dim)
        step = self.mutation_scale * (self.hi[d] - self.lo[d])
        new_val = params[d] + self.rng.normal(0, step)
        new_val = np.clip(new_val, self.lo[d], self.hi[d])
        params[d] = new_val
        return params

    def _construct_solution(self, alpha, beta, epsilon, local_rho):
        """
        Build one ant solution with epsilon-greedy exploration and local pheromone update.
        """
        idx = np.zeros(self.dim, dtype=int)
        params = np.zeros(self.dim)

        for d in range(self.dim):
            # Epsilon-greedy
            if self.rng.random() < epsilon:
                i = self.rng.integers(self.n_levels)
            else:
                weights = (self.tau[d] ** alpha) * (self.eta[d] ** beta)
                if np.sum(weights) == 0 or np.isnan(np.sum(weights)):
                    probs = np.ones(self.n_levels) / self.n_levels
                else:
                    probs = weights / np.sum(weights)
                i = self.rng.choice(self.n_levels, p=probs)

            idx[d] = i
            params[d] = self.levels[d, i]

            # ACS local pheromone update
            self.tau[d, i] = (1.0 - local_rho) * self.tau[d, i] + local_rho * self.tau0

        # Apply mutation (continuous)
        params = self._mutate(params)
        return idx, params

    def optimize(self):
        best_params, best_cost = None, np.inf
        best_idx = None
        history = []

        for it in range(self.n_iter):
            alpha, beta, rho, epsilon = self._get_dynamic_params(it)
            self._update_heuristic(best_params)

            all_idx = np.zeros((self.n_ants, self.dim), dtype=int)
            all_cost = np.zeros(self.n_ants)

            # Local rho decreases over time
            local_rho = self.local_rho_max - (self.local_rho_max - self.local_rho_min) * (it / self.n_iter)

            for a in range(self.n_ants):
                idx, params = self._construct_solution(alpha, beta, epsilon, local_rho)
                repaired = repair(params)
                cost = evaluate(repaired)
                all_idx[a] = idx
                all_cost[a] = cost

                if cost < best_cost:
                    best_cost = cost
                    best_params = repaired
                    best_idx = idx

            # Global evaporation
            self.tau *= (1.0 - rho)

            # Pheromone deposition
            if self.use_rank_deposit:
                sorted_indices = np.argsort(all_cost)
                n_rank = max(1, int(self.rank_fraction * self.n_ants))
                for rank_pos in range(n_rank):
                    a = sorted_indices[rank_pos]
                    weight = (n_rank - rank_pos) / n_rank
                    deposit = weight * cfg.ACO_Q / max(all_cost[a], 1e-6)
                    for d in range(self.dim):
                        self.tau[d, all_idx[a, d]] += deposit
            else:
                for a in range(self.n_ants):
                    deposit = cfg.ACO_Q / max(all_cost[a], 1e-6)
                    for d in range(self.dim):
                        self.tau[d, all_idx[a, d]] += deposit

            # Elite reinforcement (reduced bonus)
            if best_idx is not None:
                elite_deposit = self.elite_bonus_multiplier * cfg.ACO_ELITE_BONUS * cfg.ACO_Q / max(best_cost, 1e-6)
                for d in range(self.dim):
                    self.tau[d, best_idx[d]] += elite_deposit

            # Clip pheromone
            self.tau = np.clip(self.tau, self.tau_min, self.tau_max)

            # Stagnation detection
            if it > 0 and best_cost >= self.best_cost_history[-1] - 1e-6:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            # Stronger stagnation recovery
            if self.stagnation_counter >= self.stagnation_limit and best_idx is not None:
                print(f"[ACO] Stagnation at iteration {it+1}. Resetting pheromone structure.")
                # Reset best path to tau0 and reduce all others
                for d in range(self.dim):
                    self.tau[d, best_idx[d]] = self.tau0
                # Reduce overall pheromone to promote exploration
                self.tau *= 0.5
                self.tau = np.clip(self.tau, self.tau_min, self.tau_max)
                # Reset counter
                self.stagnation_counter = 0
                # Re-initialise heuristic to centre to forget the stuck point
                mid = (self.lo + self.hi) / 2.0
                for d in range(self.dim):
                    self.eta[d] = 1.0 / (1.0 + np.abs(self.levels[d] - mid[d]))
                self.eta = np.clip(self.eta, 1e-6, None)

            history.append(best_cost)
            self.best_cost_history.append(best_cost)

            print(
                f"[ACO] iter {it+1}/{self.n_iter} | "
                f"α={alpha:.2f}, β={beta:.2f}, ρ={rho:.2f}, ε={epsilon:.2f} | "
                f"best={best_cost:.4f}"
            )

        return best_params, best_cost, history