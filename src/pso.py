"""
Particle Swarm Optimization for tuning the fuzzy controller parameters.
"""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os

from . import config as cfg
from .cost import evaluate, get_bounds, repair


# ---------- Helper wrapper for pickling ----------
def _evaluate_wrapper(args):
    """Unpicklable wrapper around evaluate() for multiprocessing."""
    x, num_seeds = args
    return evaluate(x, num_seeds=num_seeds)


class PSO:
    def __init__(self, n_particles=cfg.PSO_N_PARTICLES, n_iter=cfg.PSO_ITERATIONS,
                 seed=cfg.RANDOM_SEED, n_workers=cfg.PSO_N_WORKERS, debug_mode=False):
        self.debug_mode = debug_mode
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = get_bounds()
        self.dim = len(self.lo)

        self.v_max = cfg.PSO_VMAX_FRAC * (self.hi - self.lo)

        self.particle_stagnation = np.zeros(self.n_particles, dtype=int)
        self.gbest_idx = 0

        if n_workers is None:
            self.n_workers = os.cpu_count() or 1
        else:
            self.n_workers = n_workers

        if self.debug_mode:
            self.n_particles //= 10
            self.n_iter //= 10

    # ---------- Core helpers ----------
    def _initialize_population(self):
        X = self.rng.uniform(self.lo, self.hi, size=(
            self.n_particles, self.dim))
        X = np.array([repair(x) for x in X])
        V = np.zeros_like(X)
        pbest = X.copy()
        pbest_cost = np.full(self.n_particles, np.inf)
        return X, V, pbest, pbest_cost

    def _evaluate_population(self, executor, X, seeds):
        args_list = [(x, seeds) for x in X]
        return np.array(list(executor.map(_evaluate_wrapper, args_list)))

    def _repair_population(self, X):
        return np.array([repair(np.clip(x, self.lo, self.hi)) for x in X])

    def _update_velocity_and_position(self, X, V, pbest, gbest, it):
        w = cfg.PSO_W_START - \
            (cfg.PSO_W_START - cfg.PSO_W_END) * it / max(1, self.n_iter - 1)
        c1 = cfg.PSO_C1_START - \
            (cfg.PSO_C1_START - cfg.PSO_C1_END) * it / max(1, self.n_iter - 1)
        c2 = cfg.PSO_C2_START + \
            (cfg.PSO_C2_END - cfg.PSO_C2_START) * it / max(1, self.n_iter - 1)

        r1 = self.rng.random((self.n_particles, self.dim))
        r2 = self.rng.random((self.n_particles, self.dim))

        V = (w * V
             + c1 * r1 * (pbest - X)
             + c2 * r2 * (gbest - X))
        V = np.clip(V, -self.v_max, self.v_max)
        X = X + V

        # Elitism: freeze the global best
        X[self.gbest_idx] = gbest.copy()
        V[self.gbest_idx] = np.zeros(self.dim)

        return X, V

    def _apply_mutation_and_reset(self, X, V, pbest_cost, prev_pbest_cost, gbest):
        rng = self.rng
        for i in range(self.n_particles):
            if i == self.gbest_idx:
                continue

            improvement = prev_pbest_cost[i] - pbest_cost[i]
            tol = max(
                cfg.PSO_TOL_ABSOLUTE,
                cfg.PSO_TOL_RELATIVE * abs(prev_pbest_cost[i])
            )

            if improvement < tol:
                self.particle_stagnation[i] += 1
            else:
                self.particle_stagnation[i] = 0

            # ----- Heavy stagnation -> reset / elite learning -----
            if self.particle_stagnation[i] >= cfg.PSO_HEAVY_STAGNATION_LIMIT:
                self.particle_stagnation[i] = 0
                if rng.random() < cfg.PSO_ELITE_LEARN_PROB:
                    noise = rng.uniform(-cfg.PSO_ELITE_NOISE_FRAC,
                                        cfg.PSO_ELITE_NOISE_FRAC,
                                        size=self.dim) * (self.hi - self.lo)
                    X[i] = np.clip(gbest + noise, self.lo, self.hi)
                else:
                    X[i] = rng.uniform(self.lo, self.hi)
                V[i] = rng.normal(0, cfg.PSO_RESET_VEL_STD, self.dim)
                continue

            # ----- Normal stagnation mutation -----
            if self.particle_stagnation[i] >= cfg.PSO_STAGNATION_LIMIT:
                extra = self.particle_stagnation[i] - \
                    cfg.PSO_STAGNATION_LIMIT + 1
                mut_rate = min(
                    cfg.PSO_MUTATION_MAX_RATE,
                    cfg.PSO_MUTATION_MIN_RATE + cfg.PSO_MUTATION_EXTRA_RATE * extra
                )
            else:
                mut_rate = cfg.PSO_MUTATION_MIN_RATE

            if rng.random() < mut_rate:
                dim = rng.integers(0, self.dim)
                dim_range = self.hi[dim] - self.lo[dim]
                shake = rng.uniform(-cfg.PSO_MUTATION_SHAKE_FRAC,
                                    cfg.PSO_MUTATION_SHAKE_FRAC) * dim_range
                X[i, dim] += shake

        return X, V

    def _update_state(self, X, pbest, pbest_cost, gbest, gbest_cost,
                      history, costs):
        improved = costs < pbest_cost
        pbest[improved] = X[improved]
        pbest_cost[improved] = costs[improved]

        best_idx = int(np.argmin(pbest_cost))
        if pbest_cost[best_idx] < gbest_cost:
            gbest_cost = float(pbest_cost[best_idx])
            gbest = pbest[best_idx].copy()
            self.gbest_idx = best_idx

        history.append(gbest_cost)
        return pbest, pbest_cost, gbest, gbest_cost

    # ---------- Main optimisation loop ----------
    def optimize(self):
        X, V, pbest, pbest_cost = self._initialize_population()
        gbest = np.zeros(self.dim)
        gbest_cost = float(np.inf)
        history = []
        prev_pbest_cost = pbest_cost.copy()

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            seeds = cfg.PSO_SEEDS_EARLY
            costs = self._evaluate_population(executor, X, seeds)

            pbest, pbest_cost, gbest, gbest_cost = self._update_state(
                X, pbest, pbest_cost, gbest, gbest_cost, history, costs
            )

            for it in range(self.n_iter):
                X, V = self._update_velocity_and_position(
                    X, V, pbest, gbest, it)

                X, V = self._apply_mutation_and_reset(
                    X, V, pbest_cost, prev_pbest_cost, gbest
                )

                X = self._repair_population(X)

                # seeds = cfg.PSO_SEEDS_EARLY if it < int(
                #     self.n_iter*cfg.PSO_SEEDS_END_MUL_TERM) else cfg.PSO_SEEDS_LATE
                seeds = cfg.PSO_SEEDS_LATE
                costs = self._evaluate_population(executor, X, seeds)

                prev_pbest_cost = pbest_cost.copy()

                pbest, pbest_cost, gbest, gbest_cost = self._update_state(
                    X, pbest, pbest_cost, gbest, gbest_cost, history, costs
                )

                print(
                    f"[PSO] iter {it + 1}/{self.n_iter}  best_cost={gbest_cost:.4f}")

        return gbest, gbest_cost, history
