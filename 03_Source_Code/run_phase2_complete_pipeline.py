"""
Master End-to-End Pipeline for Phase II Research Extension
Group 6: Vishnu (23118032) & Bibhas (23118019)

Executes:
1. Stage 1: Two-Stage Hybrid Feature Selection (Filter + Binary-PSO)
2. Stage 2: Automated Hyperparameter Optimization (5-Fold CV) & Testing
3. Comparative Benchmarking vs 3 SOTA Papers & Excel Export
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "Phase2_Feature_Selection"))
sys.path.append(os.path.join(CURRENT_DIR, "Phase2_Model_Optimization"))

from run_hybrid_fs import run_two_stage_hybrid_fs
from train_optimized_xgb import train_and_evaluate_phase2

def main():
    print("=" * 80)
    print("  PHASE II RESEARCH EXTENSION: END-TO-END EXECUTION PIPELINE")
    print("  Group 6: Dhanishetti Vishnu (23118032) & Bibhas Nath (23118019)")
    print("=" * 80)

    start_time = time.time()

    # Step 1: Feature Selection
    print("\n>>> STEP 1: Running Stage 1 Two-Stage Hybrid Feature Selection...")
    selected_features = run_two_stage_hybrid_fs()

    # Step 2: Model Optimization & Evaluation
    print("\n>>> STEP 2: Running Stage 2 Hyperparameter Optimization & Model Evaluation...")
    metrics_df = train_and_evaluate_phase2()

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" [ALL TASKS COMPLETED SUCCESSFULLY in {total_time:.2f} seconds]")
    print(" All Phase II artifacts, results, models, and tables have been generated.")
    print("=" * 80)

if __name__ == "__main__":
    main()
