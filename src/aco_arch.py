
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from . import config as cfg
from .cost import evaluate, get_bounds, repair


class ACO:
    def __init__(
        self,
        n_ants: int = getattr(cfg, "ACOR_N_ANTS", 15),
        n_iter: int = getattr(cfg, "ACOR_ITERATIONS", 30),
        archive_size: int = getattr(cfg, "ACOR_ARCHIVE_SIZE", 30),
        q: float = getattr(cfg, "ACOR_Q_INTENSITY", 0.5),
        xi: float = getattr(cfg, "ACOR_XI_EVAPORATION", 0.5),
        seed: int = getattr(cfg, "ACOR_RANDOM_SEED", 42),
        debug_mode=False
    ):
        self.debug_mode = debug_mode
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.k = archive_size
        self.q = q
        self.xi = xi
        self.base_xi = xi   # for reset after adaptive increase

        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = get_bounds()
        self.dim = len(self.lo)
        self.ranges = self.hi - self.lo

        # ---- Fixed rank‑based weights (ACOR standard) ----
        ranks = np.arange(1, self.k + 1)
        exponent = -((ranks - 1) ** 2) / (2 * (self.q ** 2) * (self.k ** 2))
        self.weights = (
            1.0 / (self.q * self.k * np.sqrt(2 * np.pi))) * np.exp(exponent)
        self.probs = self.weights / np.sum(self.weights)

        # ---- Archive storage ----
        self.archive_solutions = np.zeros((self.k, self.dim))
        self.archive_costs = np.full(self.k, np.inf)

        # ---- Stagnation control ----
        self.stagnation_limit = getattr(cfg, "ACOR_STAGNATION_LIMIT", 6)
        self.stagnation_counter = 0
        self.enable_adaptive_xi = getattr(cfg, "ACOR_ENABLE_ADAPTIVE_XI", True)
        self.xi_max = getattr(cfg, "ACOR_XI_MAX", 0.95)
        self.reset_on_stagnation = getattr(
            cfg, "ACOR_RESET_ON_STAGNATION", True)
        self.keep_best_ratio = getattr(cfg, "ACOR_KEEP_BEST_RATIO", 0.5)
        self.min_sigma_fraction = getattr(cfg, "ACOR_MIN_SIGMA_FRACTION", 0.01)
        self.init_method = getattr(cfg, "ACOR_INIT_METHOD", "latin_hypercube")
        self.max_workers = getattr(cfg, "ACOR_MAX_WORKERS", 4)

        # ---- Create a persistent process pool ----
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)

        if self.debug_mode:
            self.n_ants //= 10
            self.n_iter //= 10

    def _latin_hypercube(self, n: int) -> np.ndarray:
        """Generate n samples in [0,1)^dim using Latin Hypercube."""
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            cut = np.linspace(0, 1, n + 1)
            points[:, d] = self.rng.uniform(cut[:-1], cut[1:])
            self.rng.shuffle(points[:, d])
        return self.lo + points * self.ranges

    def _initialize_archive(self):
        """Initialise archive using the selected method."""
        print("[ACOR] Initialising archive...")
        if self.init_method == "latin_hypercube":
            raw = self._latin_hypercube(self.k)
        else:  # uniform
            raw = self.rng.uniform(self.lo, self.hi, (self.k, self.dim))

        init_solutions = np.array([repair(row) for row in raw])
        # Evaluate in parallel
        new_costs = np.array(list(self.executor.map(evaluate, init_solutions)))
        sorted_idx = np.argsort(new_costs)
        self.archive_solutions = init_solutions[sorted_idx]
        self.archive_costs = new_costs[sorted_idx]

    def _sample_new_solutions(self) -> np.ndarray:
        """Generate new solutions using the archive's Gaussian kernels."""
        new_solutions = np.zeros((self.n_ants, self.dim))

        for a in range(self.n_ants):
            guide_idx = self.rng.choice(self.k, p=self.probs)
            guide = self.archive_solutions[guide_idx]

            ant = np.zeros(self.dim)
            for d in range(self.dim):
                mu = guide[d]
                # Sigma = xi * mean absolute deviation from mu
                abs_dev = np.abs(self.archive_solutions[:, d] - mu)
                sigma = self.xi * np.mean(abs_dev)

                # Ensure minimum sigma (to keep exploring)
                min_sigma = self.min_sigma_fraction * self.ranges[d]
                sigma = max(sigma, min_sigma)

                val = self.rng.normal(mu, sigma)
                ant[d] = np.clip(val, self.lo[d], self.hi[d])

            new_solutions[a] = repair(ant)

        return new_solutions

    def _update_archive(self, new_solutions: np.ndarray, new_costs: np.ndarray):
        """Merge, sort, and keep the best K solutions."""
        combined_sol = np.vstack((self.archive_solutions, new_solutions))
        combined_costs = np.concatenate((self.archive_costs, new_costs))
        sorted_idx = np.argsort(combined_costs)
        self.archive_solutions = combined_sol[sorted_idx[:self.k]]
        self.archive_costs = combined_costs[sorted_idx[:self.k]]

    def _reset_archive_partial(self):
        """Keep the best fraction, replace the rest with random solutions."""
        keep_n = max(1, int(self.keep_best_ratio * self.k))
        best_solutions = self.archive_solutions[:keep_n]
        best_costs = self.archive_costs[:keep_n]

        n_random = self.k - keep_n
        raw = self.rng.uniform(self.lo, self.hi, (n_random, self.dim))
        new_sol = np.array([repair(row) for row in raw])
        new_costs = np.array(list(self.executor.map(evaluate, new_sol)))

        self.archive_solutions = np.vstack([best_solutions, new_sol])
        self.archive_costs = np.concatenate([best_costs, new_costs])
        # Re‑sort
        sorted_idx = np.argsort(self.archive_costs)
        self.archive_solutions = self.archive_solutions[sorted_idx]
        self.archive_costs = self.archive_costs[sorted_idx]

    def optimize(self) -> tuple[np.ndarray, float, list[float]]:
        self._initialize_archive()
        best_cost = self.archive_costs[0]
        best_params = self.archive_solutions[0]
        history = []

        for it in range(self.n_iter):
            # 1. Generate new solutions
            new_solutions = self._sample_new_solutions()

            # 2. Evaluate in parallel
            new_costs = np.array(
                list(self.executor.map(evaluate, new_solutions)))

            # 3. Update archive
            self._update_archive(new_solutions, new_costs)

            # 4. Record best
            current_best = self.archive_costs[0]
            if current_best < best_cost - 1e-6:
                best_cost = current_best
                best_params = self.archive_solutions[0]
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            # 5. Stagnation handling
            if self.stagnation_counter >= self.stagnation_limit:
                print(f"[ACOR] Stagnation at iteration {it+1}. Taking action.")
                if self.enable_adaptive_xi:
                    self.xi = min(self.xi_max, self.xi * 1.5)
                    print(f"  → Increased xi to {self.xi:.3f}")

                if self.reset_on_stagnation:
                    self._reset_archive_partial()
                    print("  → Reset worst part of archive with random solutions.")

                # Reset counter and restore xi to base (if desired)
                self.stagnation_counter = 0
                if self.enable_adaptive_xi:
                    self.xi = self.base_xi  # optional: reset to base after recovery

            history.append(best_cost)
            print(
                f"[ACOR] iter {it+1}/{self.n_iter} | best = {best_cost:.4f} | xi = {self.xi:.3f}")

        self.executor.shutdown()
        return best_params, best_cost, history
