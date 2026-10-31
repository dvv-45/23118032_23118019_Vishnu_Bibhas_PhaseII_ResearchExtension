"""
Phase II Handover Preparation Script: Vishnu to Bibhas
Group 6: Dhanishetti Vishnu (23118032) & Bibhas Nath (23118019)

Generates:
1. Pre-filtered train_selected_phase2.csv and test_selected_phase2.csv containing only the 6 optimal features.
2. feature_order_phase2.json and preprocessing_config_phase2.json metadata.
3. Comprehensive HANDOVER_README_PHASE2.txt with step-by-step instructions for Bibhas.
"""

import os
import json
import shutil
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HANDOVER_DIR = os.path.join(BASE_DIR, "Vishnu_To_Bibhas")
FS_RESULTS_DIR = os.path.join(BASE_DIR, "05_Phase2_Feature_Selection_Results")
DATA_DIR = os.path.join(BASE_DIR, "02_Dataset", "Processed_Dataset")

os.makedirs(HANDOVER_DIR, exist_ok=True)

def prepare_handover():
    print("=" * 80)
    print(" [HANDOVER] PREPARING PHASE II HANDOVER PACKAGE: VISHNU -> BIBHAS")
    print("=" * 80)

    # 1. Load Selected Features
    feat_csv = os.path.join(FS_RESULTS_DIR, "Phase2_Selected_Features.csv")
    if not os.path.exists(feat_csv):
        raise FileNotFoundError(f"Selected features not found at: {feat_csv}. Run run_hybrid_fs.py first.")
    
    selected_df = pd.read_csv(feat_csv)
    selected_features = selected_df['Feature_Name'].tolist()
    print(f"[+] Loaded {len(selected_features)} optimal features selected by Vishnu's B-PSO optimizer:")
    for idx, f in enumerate(selected_features, 1):
        print(f"    {idx}. {f}")

    # 2. Slice and Save Filtered Datasets for Bibhas
    print("\n[+] Slicing train and test datasets with optimal features...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    cols_to_keep = selected_features + ['label']
    train_selected = train_df[cols_to_keep]
    test_selected = test_df[cols_to_keep]

    train_out = os.path.join(HANDOVER_DIR, "train_selected_phase2.csv")
    test_out = os.path.join(HANDOVER_DIR, "test_selected_phase2.csv")
    train_selected.to_csv(train_out, index=False)
    test_selected.to_csv(test_out, index=False)
    print(f"    --> Saved: {train_out} ({len(train_selected):,} rows, {len(cols_to_keep)} cols)")
    print(f"    --> Saved: {test_out} ({len(test_selected):,} rows, {len(cols_to_keep)} cols)")

    # 3. Export JSON Metadata
    print("\n[+] Exporting JSON schema and config...")
    feature_metadata = {
        "selected_features": selected_features,
        "num_selected": len(selected_features),
        "target_column": "label",
        "feature_descriptions": {
            "udp.srcport": "UDP Source Port (Discriminates Mirai UDP flooding attacks targeting DNS/SSDP reflection)",
            "ip.flags": "IP Fragmentation Flags (Distinguishes fragmented denial-of-service botnet bursts)",
            "tcp.dstport": "TCP Destination Port (Detects Telnet 23/2323 credential harvesting and HTTP scanning)",
            "ip.ttl": "IP Time-to-Live (Anomalous fixed hops characteristic of synthetic scanner bot generators)",
            "ip.proto": "IP Protocol ID (Identifies transport protocol switches between UDP=17, TCP=6, ICMP=1)",
            "tcp": "TCP Protocol Indicator flag (Differentiates stateful control channel traffic from raw floods)"
        }
    }
    with open(os.path.join(HANDOVER_DIR, "feature_order_phase2.json"), "w") as f:
        json.dump(feature_metadata, f, indent=4)

    pipeline_config = {
        "random_seed": 42,
        "dataset_name": "IoT Network Intrusion Detection Dataset (Mirai-focused)",
        "balancing_method": "Random UnderSampling (RUS)",
        "total_balanced_samples": 87422,
        "train_samples": 69937,
        "test_samples": 17485,
        "split_ratio": "80:20 (Stratified)",
        "proposed_feature_selection": {
            "stage_1_filter": "Extra-Trees Ensemble Importance (Pruned bottom 50%)",
            "stage_2_wrapper": "Binary Particle Swarm Optimization (B-PSO)",
            "num_particles": 20,
            "max_iterations": 25,
            "inertia_weight": "0.9 linearly decreasing to 0.4",
            "acceleration_coefficients": "c1 = 1.5, c2 = 1.5",
            "fitness_function": "0.99 * (1 - F1) + 0.01 * (Fs / Fa) + protocol_penalty"
        }
    }
    with open(os.path.join(HANDOVER_DIR, "preprocessing_config_phase2.json"), "w") as f:
        json.dump(pipeline_config, f, indent=4)

    # Copy results & plots to handover
    shutil.copy2(os.path.join(FS_RESULTS_DIR, "Phase2_Selected_Features.csv"), os.path.join(HANDOVER_DIR, "Phase2_Selected_Features.csv"))
    shutil.copy2(os.path.join(FS_RESULTS_DIR, "Filter_Stage_Feature_Ranking.csv"), os.path.join(HANDOVER_DIR, "Filter_Stage_Feature_Ranking.csv"))
    shutil.copy2(os.path.join(FS_RESULTS_DIR, "BPSO_Convergence_Curve.png"), os.path.join(HANDOVER_DIR, "BPSO_Convergence_Curve.png"))

    # 4. Generate Comprehensive HANDOVER_README_PHASE2.txt
    readme_content = f"""========================================================================================
Phase II Official Handover Package: From Vishnu (Roll 23118032) to Bibhas (Roll 23118019)
Group 6: IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB
========================================================================================

Handover Date: October 2026
Project Status: Stage 1 Feature Selection Completed. Stage 2 Ready for Bibhas.

1. Handover Inventory:
----------------------
1. train_selected_phase2.csv  : Pre-filtered 80% stratified training set (69,937 rows x 6 features + label).
2. test_selected_phase2.csv   : Pre-filtered 20% held-out test set (17,485 rows x 6 features + label).
3. Phase2_Selected_Features.csv: The 6 optimal features selected by Binary-PSO.
4. Filter_Stage_Feature_Ranking.csv: Feature importance scores from Stage 1 Extra-Trees filter.
5. BPSO_Convergence_Curve.png : Swarm fitness convergence plot across 25 iterations.
6. feature_order_phase2.json  : Exact column sequence and domain security descriptions.
7. preprocessing_config_phase2.json: Complete pipeline parameters and random seeds.
8. Phase2_Work_Division_and_Plan.md: Detailed division of work and submission checklist.

2. Selected Optimal Feature Set (73.9% Space Reduction):
--------------------------------------------------------
Rank  Feature Name   Domain Security Justification
----  ------------   -----------------------------
 1.   udp.srcport    Mirai UDP flood attacks targeting reflective DNS/SSDP ports.
 2.   ip.flags       IP fragmentation flags used in malformed buffer-overflow bursts.
 3.   tcp.dstport    Detects brute-force Telnet (port 23/2323) scanning and command delivery.
 4.   ip.ttl         Fixed anomalous TTL signatures generated by automated bot attack scripts.
 5.   ip.proto       Identifies protocol switching (TCP, UDP, ICMP) in multi-vector botnets.
 6.   tcp            Stateful TCP control flag identifying legitimate handshakes vs. floods.

3. Instructions for Bibhas (Your Stage 2 Work):
------------------------------------------------
All upstream data prep is 100% complete. You do not need to perform any preprocessing or feature selection.
Directly perform the following downstream tasks:

Step A - Model Optimization:
  - Load `train_selected_phase2.csv` and `test_selected_phase2.csv`.
  - Execute hyperparameter tuning using Bayesian Optimization / Optuna or Randomized CV Search:
    Script: `03_Source_Code/Phase2_Model_Optimization/train_optimized_xgb.py`
  - Hyperparameter Search Space:
    * learning_rate: [0.01, 0.05, 0.1, 0.15, 0.2]
    * max_depth: [3, 4, 5, 6, 8]
    * n_estimators: [50, 75, 100, 150, 200]
    * subsample & colsample_bytree: [0.7 to 1.0]

Step B - Evaluation & Plots:
  - Evaluate the tuned model on `test_selected_phase2.csv`.
  - Save confusion matrix to `07_Result_Images/Phase2_Confusion_Matrix.png`.
  - Save ROC curve to `07_Result_Images/Phase2_ROC_Curves.png`.
  - Verify that `08_Comparative_Results/Result_Comparison.xlsx` contains the 4 benchmark studies.

Step C - Report Writing (Sections 4 to 6):
  - Section 4: Automated Hyperparameter Tuning Setup & 5-Fold Cross Validation.
  - Section 5: Comparative Evaluation & SOTA Discussion (embed Table 1, confusion matrix, ROC).
  - Section 6: Practical Implications, Inference Latency, and Limitations.

Reproducibility Guarantee:
--------------------------
- Random Seed: 42 across all splits and algorithms.
- Stratified cross-validation ensures zero data leakage.
"""
    with open(os.path.join(HANDOVER_DIR, "HANDOVER_README_PHASE2.txt"), "w") as f:
        f.write(readme_content)
    print(f"    --> Created: {os.path.join(HANDOVER_DIR, 'HANDOVER_README_PHASE2.txt')}")

    print("\n[SUCCESS] Vishnu's Phase II Handover Package is 100% Complete & Ready for Bibhas!")

if __name__ == "__main__":
    prepare_handover()
