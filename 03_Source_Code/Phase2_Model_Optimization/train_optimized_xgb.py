"""
Phase II Model Optimization: train_optimized_xgb
Forwarder to train_phase2_model.train_and_evaluate_phase2
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PHASE2_DIR)

from train_phase2_model import train_and_evaluate_phase2

if __name__ == "__main__":
    train_and_evaluate_phase2()