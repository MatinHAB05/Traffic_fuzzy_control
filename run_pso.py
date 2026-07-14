"""Run PSO optimization on the fuzzy controller parameters."""
from src.plotting import plot_convergence
from src.cost import evaluate
from src.pso import PSO
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


OUT_DATA = "results/data"
OUT_PLOTS = "results/plots"


def main(debug_mode=False):
    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_PLOTS, exist_ok=True)

    
    optimizer = PSO(debug_mode=debug_mode)
    best_params, best_cost, history = optimizer.optimize()
    _, metrics = evaluate(best_params, return_metrics=True)

    print("PSO optimization finished")
    print(
        f"  best_cost={best_cost:.4f}  W={metrics['W']:.3f}  Q={metrics['Q']:.3f}  S={metrics['S']:.1f}")

    plot_convergence(history, "PSO Convergence",
                     f"{OUT_PLOTS}/pso_convergence.png")

    data = {
        "best_cost": best_cost,
        "metrics": metrics,
        "history": history,
        "best_params": best_params.tolist(),
    }
    with open(f"{OUT_DATA}/pso_result.json", "w") as f:
        json.dump(data, f, indent=2)

    return data


if __name__ == "__main__":
    main()
