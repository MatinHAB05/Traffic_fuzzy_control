"""Compare the baseline, PSO-optimized and ACO-optimized fuzzy controllers."""
from src.plotting import plot_convergence_comparison, plot_bar_comparison, plot_stability_box
from src.cost import evaluate
import numpy as np
import os
import sys
import json
from src.simulation import TrafficSimulation
from src.fuzzy_system import FuzzyController
from src.plotting import plot_queue_trace
import src.config as cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


OUT_DATA = "results/data"
OUT_PLOTS = "results/plots"


def _load(name):
    path = f"{OUT_DATA}/{name}"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run baseline/pso/aco first.")
    with open(path) as f:
        return json.load(f)


def main(debug_mode=False):
    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_PLOTS, exist_ok=True)

    baseline = _load("baseline_metrics.json")
    pso = _load("pso_result.json")
    aco = _load("aco_result.json")

    labels = ["Baseline", "PSO", "ACO"]
    costs = [baseline["cost"], pso["best_cost"], aco["best_cost"]]

    plot_convergence_comparison(pso["history"], aco["history"],
                                f"{OUT_PLOTS}/convergence_comparison.png")
    plot_bar_comparison(labels, costs, "Cost (C)", "Final Cost Comparison",
                        f"{OUT_PLOTS}/cost_comparison.png")

    # stability: evaluate the best parameters found on unseen random seeds
    test_seeds = list(range(100, 110))
    param_map = {
        "Baseline": baseline["params"],
        "PSO": pso["best_params"],
        "ACO": aco["best_params"],
    }
    stability = {lbl: [evaluate(p, seeds=[s]) for s in test_seeds]
                 for lbl, p in param_map.items()}

    plot_stability_box(stability, "Cost (C)", "Stability Across Unseen Random Seeds",
                       f"{OUT_PLOTS}/stability_comparison.png")

    print("Generating queue trace plots for all controllers...")
    for lbl, params in param_map.items():
        controller = FuzzyController(np.array(params))

        sim = TrafficSimulation(controller, cfg.ARRIVAL_RATE_1, cfg.ARRIVAL_RATE_2,
                                n_cycles=cfg.N_CYCLES, seed=0)
        result = sim.run()

        file_name = lbl.lower().replace(" ", "_")

        plot_queue_trace(result["q1_history"], result["q2_history"],
                         f"{lbl} Controller - Queue Length per Cycle",
                         f"{OUT_PLOTS}/{file_name}_queue_trace.png")
    print("All queue trace plots generated successfully!")
    summary = {
        "labels": labels,
        "costs": {lbl: c for lbl, c in zip(labels, costs)},
        "metrics": {
            "Baseline": {"W": baseline["W"], "Q": baseline["Q"], "S": baseline["S"]},
            "PSO": pso["metrics"],
            "ACO": aco["metrics"],
        },
        "stability_mean": {k: float(np.mean(v)) for k, v in stability.items()},
        "stability_std": {k: float(np.std(v)) for k, v in stability.items()},
    }

    print("Comparison summary")
    for lbl in labels:
        print(f"  {lbl}: cost={summary['costs'][lbl]:.4f}")
    print("Stability (mean +/- std over 10 unseen seeds):")
    for lbl in labels:
        print(
            f"  {lbl}: {summary['stability_mean'][lbl]:.4f} +/- {summary['stability_std'][lbl]:.4f}")

    with open(f"{OUT_DATA}/comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    main()
