"""End-to-end pipeline: baseline -> PSO -> ACO -> comparison.
Run with: python main.py
"""
import run_baseline
import os
import sys
import run_pso
import run_aco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# will be added :
# import run_comparison


def main():
    print("=" * 60)
    print("STEP 1/4: Baseline fuzzy controller")
    print("=" * 60)
    run_baseline.main()

    # print("=" * 60)
    # print("STEP 2/4: PSO optimization")
    # print("=" * 60)
    # run_pso.main()

    print("=" * 60)
    print("STEP 3/4: ACO optimization")
    print("=" * 60)
    run_aco.main()

    # print("=" * 60)
    # print("STEP 4/4: Comparison")
    # print("=" * 60)
    # run_comparison.main()

    # print("=" * 60)
    # print("Done. See results/plots and results/data")
    # print("=" * 60)


if __name__ == "__main__":
    main()
