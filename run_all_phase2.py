"""
Master End-to-End Pipeline for Phase II Research Extension
Group 6: Dhanishetti Vishnu (23118032) & Bibhas Nath (23118019)
Department of Information Technology, National Institute of Technology Raipur

Executes End-to-End:
  1. Stage 1 (Vishnu): Two-Stage Hybrid Feature Selection (Filter + B-PSO) -> Phase2_Selected_Features.csv
  2. Stage 2 (Bibhas): Bayesian Hyperparameter Optimization & Model Training -> Phase2_Trained_Model.joblib
  3. Comparative Benchmarking (Both): Result_Comparison.xlsx
  4. Final Report Generation (Both): PhaseII_Research_Report.pdf
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(CURRENT_DIR, "run_phase2_feature_selection.py")):
    PHASE2_DIR = CURRENT_DIR
else:
    PHASE2_DIR = os.path.join(CURRENT_DIR, "23118032_23118019_Vishnu_Bibhas_PhaseII_ResearchExtension")

sys.path.insert(0, PHASE2_DIR)

from run_phase2_feature_selection import run_phase2_feature_selection
from train_phase2_model import train_and_evaluate_phase2
from generate_phase2_report import generate_report

def main():
    print("=" * 85)
    print("  PHASE II RESEARCH EXTENSION: MASTER END-TO-END PIPELINE")
    print("  Group 6: Dhanishetti Vishnu (Roll 23118032) & Bibhas Nath (Roll 23118019)")
    print("=" * 85)

    start_total = time.time()

    # Step 1: Vishnu's Stage 1 Feature Selection
    print("\n" + "=" * 65)
    print(">>> [STAGE 1/3] VISHNU: TWO-STAGE HYBRID FEATURE SELECTION")
    print("=" * 65)
    selected_features = run_phase2_feature_selection()

    # Step 2: Bibhas's Stage 2 Model Optimization & Testing
    print("\n" + "=" * 65)
    print(">>> [STAGE 2/3] BIBHAS: BAYESIAN MODEL OPTIMIZATION & EVALUATION")
    print("=" * 65)
    metrics_summary = train_and_evaluate_phase2(n_trials=10) # 10 trials for fast end-to-end verification

    # Step 3: Report & Deliverable Compilation
    print("\n" + "=" * 65)
    print(">>> [STAGE 3/3] FINAL RESEARCH REPORT GENERATION")
    print("=" * 65)
    generate_report()

    # Verify deliverables
    required_artifacts = [
        os.path.join(PHASE2_DIR, "Proposed_Methodology.png"),
        os.path.join(PHASE2_DIR, "Proposed_Methodology.drawio"),
        os.path.join(PHASE2_DIR, "08_Comparative_Results", "Result_Comparison.xlsx"),
        os.path.join(PHASE2_DIR, "05_Phase2_Feature_Selection_Results", "Phase2_Selected_Features.csv"),
        os.path.join(PHASE2_DIR, "09_Trained_Models", "Phase2_Trained_Model.joblib"),
        os.path.join(PHASE2_DIR, "10_Research_Extension_Report", "PhaseII_Research_Report.pdf")
    ]

    print("\n" + "=" * 85)
    print("  SUBMISSION DELIVERABLES AUDIT:")
    print("=" * 85)
    all_ok = True
    for artifact in required_artifacts:
        exists = os.path.exists(artifact)
        size_kb = os.path.getsize(artifact) / 1024 if exists else 0
        status = f"[OK] ({size_kb:.1f} KB)" if exists else "[MISSING]"
        if not exists:
            all_ok = False
        print(f"  {status:<15} {os.path.basename(artifact)}")

    total_time = time.time() - start_total
    print("\n" + "=" * 85)
    if all_ok:
        print(f"  [ALL PHASE II DELIVERABLES 100% VERIFIED & READY FOR SUBMISSION in {total_time:.2f}s]")
    else:
        print(f"  [ATTENTION: Some deliverables require check] ({total_time:.2f}s)")
    print("=" * 85)

if __name__ == "__main__":
    main()
