"""End-to-end pipeline: baseline -> PSO -> ACO -> comparison.
Run with: python main.py
"""
import run_baseline
import os
import sys
import run_pso
import run_aco
import run_comparison

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Debug Mode
DEBUG_MODE = False


def SET_DEBUG_MODE():
    global DEBUG_MODE
    DEBUG_MODE = True
    return DEBUG_MODE


def RESET_DEBUG_MODE():
    global DEBUG_MODE
    DEBUG_MODE = False
    return DEBUG_MODE


def main():
    global DEBUG_MODE
    if (len(sys.argv) > 1 and sys.argv[1].upper() == "DEBUG"):
        SET_DEBUG_MODE()
        print(DEBUG_MODE)
        print("-"*40, "DEBUG MODE ENABLED", "-"*40, sep="")

    print("=" * 60)
    print("STEP 1/4: Baseline fuzzy controller")
    print("=" * 60)
    run_baseline.main(debug_mode=DEBUG_MODE)

    print("=" * 60)
    print("STEP 2/4: PSO optimization")
    print("=" * 60)
    run_pso.main(debug_mode=DEBUG_MODE)

    print("=" * 60)
    print("STEP 3/4: ACO optimization")
    print("=" * 60)
    run_aco.main(debug_mode=DEBUG_MODE)

    print("=" * 60)
    print("STEP 4/4: Comparison")
    print("=" * 60)
    run_comparison.main(debug_mode=DEBUG_MODE)

    # print("=" * 60)
    # print("Done. See results/plots and results/data")
    # print("=" * 60)


if __name__ == "__main__":
    main()
