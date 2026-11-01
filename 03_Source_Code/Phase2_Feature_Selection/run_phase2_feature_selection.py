"""
Phase II Research Extension - Stage 1: Two-Stage Hybrid Feature Selection
Author: Dhanishetti Vishnu (Roll No. 23118032)
Collaborator: Bibhas Nath (Roll No. 23118019)
Group 6: IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB
"""

import os
import sys

# Forward to main execution script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SCRIPT = os.path.join(CURRENT_DIR, "..", "..", "run_phase2_feature_selection.py")

if os.path.exists(TARGET_SCRIPT):
    sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")))
    from run_phase2_feature_selection import run_phase2_feature_selection
    if __name__ == "__main__":
        run_phase2_feature_selection()
else:
    # Standalone execution
    import run_hybrid_fs
    run_hybrid_fs.run_two_stage_hybrid_fs()
