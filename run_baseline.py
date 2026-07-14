"""Run the hand-designed (default) fuzzy controller as a baseline."""
from src.plotting import plot_queue_trace
from src.simulation import TrafficSimulation
from src.cost import evaluate
from src.fuzzy_system import FuzzyController
from src import config as cfg
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


OUT_DATA = "results/data"
OUT_PLOTS = "results/plots"


def main(debug_mode=False):
    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_PLOTS, exist_ok=True)

    params = FuzzyController.default_params()
    controller = FuzzyController(params)
    cost, metrics = evaluate(params, return_metrics=True)

    print("Baseline (hand-designed) fuzzy controller")
    print(
        f"  cost={cost:.4f}  W={metrics['W']:.3f}  Q={metrics['Q']:.3f}  S={metrics['S']:.1f}")

    sim = TrafficSimulation(controller, cfg.ARRIVAL_RATE_1, cfg.ARRIVAL_RATE_2,
                            n_cycles=cfg.N_CYCLES, seed=0)
    result = sim.run()
    plot_queue_trace(result["q1_history"], result["q2_history"],
                     "Baseline Controller - Queue Length per Cycle",
                     f"{OUT_PLOTS}/baseline_queue_trace.png")

    data = {"cost": cost, "params": params.tolist(), **metrics}
    with open(f"{OUT_DATA}/baseline_metrics.json", "w") as f:
        json.dump(data, f, indent=2)

    return data


if __name__ == "__main__":
    main()
