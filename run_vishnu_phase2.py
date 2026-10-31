"""
Vishnu's Complete Phase II Pipeline Runner
Owner: Dhanishetti Vishnu (Roll No. 23118032)
Role: Stage 1 Lead (Two-Stage Hybrid Feature Selection & Handover Preparation)

Executes:
1. Two-Stage Hybrid Feature Selection (Extra-Trees Filter + Binary PSO).
2. Generates feature rankings and convergence plots.
3. Automatically packages handover files and pre-filtered datasets for Bibhas.
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FS_DIR = os.path.join(CURRENT_DIR, "03_Source_Code", "Phase2_Feature_Selection")
sys.path.append(FS_DIR)

from run_hybrid_fs import run_two_stage_hybrid_fs
from prepare_phase2_handover import prepare_handover

def main():
    print("=" * 80)
    print("  VISHNU'S PHASE II PIPELINE: FEATURE SELECTION & HANDOVER GENERATION")
    print("  Author: Dhanishetti Vishnu (Roll No. 23118032)")
    print("=" * 80)
    
    t0 = time.time()
    
    # 1. Feature Selection
    print("\n>>> [1/2] Executing Stage 1 Two-Stage Hybrid Feature Selection...")
    selected_features = run_two_stage_hybrid_fs()
    
    # 2. Handover Preparation
    print("\n>>> [2/2] Assembling Official Handover Package for Bibhas...")
    prepare_handover()
    
    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f" [VISHNU'S PHASE II WORK COMPLETED IN {total_time:.2f} SECONDS]")
    print(" Handover files are ready in: Vishnu_To_Bibhas/")
    print(" Ready to hand over to Bibhas Nath (Roll No. 23118019)!")
    print("=" * 80)

if __name__ == "__main__":
    main()
