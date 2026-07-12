"""Plot helpers for convergence curves and algorithm comparison."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_convergence(history, title, out_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(history, marker="o", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_convergence_comparison(hist_pso, hist_aco, out_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist_pso, label="PSO", marker="o", markersize=3)
    plt.plot(hist_aco, label="ACO", marker="s", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("PSO vs ACO Convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_bar_comparison(labels, values, ylabel, title, out_path):
    plt.figure(figsize=(6, 4.5))
    plt.bar(labels, values, color=["#888888", "#4C72B0", "#55A868"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_queue_trace(q1_hist, q2_hist, title, out_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(q1_hist, label="Queue Road 1")
    plt.plot(q2_hist, label="Queue Road 2")
    plt.xlabel("Cycle")
    plt.ylabel("Queue length (vehicles)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_stability_box(data_dict, ylabel, title, out_path):
    plt.figure(figsize=(6, 4.5))
    plt.boxplot(list(data_dict.values()), labels=list(data_dict.keys()))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
