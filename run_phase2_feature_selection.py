"""
Phase II Research Extension - Stage 1: Two-Stage Hybrid Feature Selection
Author: Dhanishetti Vishnu (Roll No. 23118032)
Collaborator: Bibhas Nath (Roll No. 23118019)
Group 6: IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB

Assigned Base Paper:
  "Efficient and Interpretable IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB"
  (Computer Networks, Elsevier, 2026)

Proposed Extension:
  Stage A (Filter): Extra-Trees & Mutual Information Feature Importance ranking (pruning bottom 50% noisy features).
  Stage B (Metaheuristic Wrapper): Binary-adapted Particle Swarm Optimization (B-PSO)
    Multi-objective Fitness = alpha * (1 - F1_CV) + beta * (Selected_Features / Total_Features)
    Balances maximal detection performance with minimal packet header inspection dimensionality.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

# Resolve base directory dynamically
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(CURRENT_DIR, "02_Dataset", "Processed_Dataset", "train.csv")):
    BASE_DIR = CURRENT_DIR
elif os.path.exists(os.path.join(CURRENT_DIR, "23118032_23118019_Vishnu_Bibhas_PhaseII_ResearchExtension", "02_Dataset", "Processed_Dataset", "train.csv")):
    BASE_DIR = os.path.join(CURRENT_DIR, "23118032_23118019_Vishnu_Bibhas_PhaseII_ResearchExtension")
else:
    BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

TRAIN_PATH = os.path.join(BASE_DIR, "02_Dataset", "Processed_Dataset", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "02_Dataset", "Processed_Dataset", "test.csv")
FS_RESULTS_DIR = os.path.join(BASE_DIR, "05_Phase2_Feature_Selection_Results")
HANDOVER_DIR = os.path.join(BASE_DIR, "Vishnu_To_Bibhas")

os.makedirs(FS_RESULTS_DIR, exist_ok=True)
os.makedirs(HANDOVER_DIR, exist_ok=True)

# Domain Security Descriptions for Candidate Features
FEATURE_JUSTIFICATIONS = {
    "udp.srcport": "Mirai UDP flood attack signature targeting reflective ephemeral DNS/SSDP ports.",
    "ip.flags": "IP fragmentation bitmask manipulation used in anomalous buffer-overflow bursts.",
    "tcp.dstport": "Telnet brute-force exploit attempts targeting ports 23, 2323, and HTTP port 80.",
    "ip.ttl": "Fixed static Time-To-Live hop-count signatures produced by automated bot scripts.",
    "ip.proto": "Identifies rapid transport protocol switching across TCP, UDP, and ICMP floods.",
    "tcp": "Stateful TCP protocol indicator identifying incomplete handshakes vs. benign sessions.",
    "tcp.window_size": "Zero-window exhaustion probes and anomalous TCP buffer allocations.",
    "tcp.len": "Zero-payload TCP SYN flood anomalies and variable-size DDoS bursts.",
    "frame.len": "Abnormal frame size distribution characteristic of high-volume line-rate attacks.",
    "ip.len": "Total IP packet length anomalies distinguishing malformed flood packets.",
    "tcp.flags": "TCP flag combinations (SYN-ACK, FIN, RST, PSH) revealing port scanning patterns.",
    "udp.dstport": "Target destination service port during reflective amplified UDP flooding."
}

def run_phase2_feature_selection():
    print("=" * 80)
    print("  PHASE II: TWO-STAGE HYBRID FEATURE SELECTION PIPELINE")
    print("  Lead Researcher: Dhanishetti Vishnu (Roll No. 23118032)")
    print("  Collaborator:   Bibhas Nath (Roll No. 23118019)")
    print("=" * 80)

    t0 = time.time()

    # 1. Load Data
    print(f"\n[+] Step 1: Loading Processed IoT Traffic Dataset...")
    print(f"    Train Path: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df.drop(columns=['label'])
    y = train_df['label'].astype(int)
    feature_names = list(X.columns)
    total_features = len(feature_names)
    print(f"    Loaded {len(train_df):,} flows with {total_features} candidate features.")

    # =========================================================================
    # STAGE A: FILTER-BASED PRE-SELECTION (Extra-Trees Feature Importance)
    # =========================================================================
    print("\n[+] Step 2: Stage A — Filter Pre-selection (Ensemble Tree Gini Ranking)...")
    print("    Objective: Rank all features and prune bottom 50% noisy/static attributes.")
    
    filter_model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    filter_model.fit(X, y)

    importances = filter_model.feature_importances_
    ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Score': importances
    }).sort_values(by='Importance_Score', ascending=False).reset_index(drop=True)

    filter_csv = os.path.join(FS_RESULTS_DIR, "Filter_Stage_Feature_Ranking.csv")
    ranking_df.to_csv(filter_csv, index=False)
    ranking_df.to_csv(os.path.join(HANDOVER_DIR, "Filter_Stage_Feature_Ranking.csv"), index=False)
    print(f"    --> Feature ranking saved to: {filter_csv}")

    # Retain top 50% features
    k_retain = int(np.ceil(total_features * 0.50))
    candidate_features = ranking_df['Feature'].iloc[:k_retain].tolist()
    print(f"    --> Retained Top {k_retain} features (50% reduction) for Swarm Optimization:")
    for rank, feat in enumerate(candidate_features, 1):
        score = ranking_df.loc[ranking_df['Feature'] == feat, 'Importance_Score'].values[0]
        print(f"        {rank:2d}. {feat:<22} (Importance: {score:.4f})")

    # =========================================================================
    # STAGE B: BINARY PARTICLE SWARM OPTIMIZATION (B-PSO)
    # =========================================================================
    print("\n[+] Step 3: Stage B — Binary-adapted Particle Swarm Optimization (B-PSO)...")
    print("    Objective: Global multi-objective combinatorial search for optimal subset.")

    # Stratified subsample for rapid, stable swarm fitness evaluations
    eval_subsample_size = min(12000, len(X))
    sub_df = train_df.sample(n=eval_subsample_size, random_state=42)
    X_sub = sub_df[candidate_features].values
    y_sub = sub_df['label'].values

    # B-PSO Hyperparameters
    num_particles = 20
    max_iter = 25
    w_max, w_min = 0.9, 0.4
    c1, c2 = 1.5, 1.5
    num_dims = len(candidate_features)

    np.random.seed(42)
    positions = np.random.uniform(0, 1, (num_particles, num_dims))
    velocities = np.random.uniform(-1, 1, (num_particles, num_dims))

    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(-np.clip(v, -10, 10)))

    def evaluate_fitness(binary_mask):
        selected_indices = np.where(binary_mask == 1)[0]
        if len(selected_indices) == 0:
            return 1.0, 0.0, 0
        
        X_sel = X_sub[:, selected_indices]
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        f1_scores = []
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)

        for train_idx, val_idx in cv.split(X_sel, y_sub):
            X_tr, X_val = X_sel[train_idx], X_sel[val_idx]
            y_tr, y_val = y_sub[train_idx], y_sub[val_idx]
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            f1_scores.append(f1_score(y_val, preds, zero_division=0))

        mean_f1 = np.mean(f1_scores)
        fs_count = len(selected_indices)

        # Multi-objective Fitness formulation
        alpha, beta = 0.99, 0.01
        fitness = alpha * (1.0 - mean_f1) + beta * (fs_count / total_features)
        if fs_count < 6:
            fitness += 0.005 * (6 - fs_count)  # Protocol diversity constraint
        return fitness, mean_f1, fs_count

    # Evaluate Initial Population
    pbest_positions = np.copy(positions)
    pbest_fitness = np.full(num_particles, np.inf)
    pbest_f1 = np.zeros(num_particles)

    gbest_position = None
    gbest_fitness = np.inf
    gbest_f1 = 0.0
    gbest_mask = None

    for i in range(num_particles):
        bmask = (sigmoid(positions[i]) >= 0.5).astype(int)
        fit, f1_val, _ = evaluate_fitness(bmask)
        pbest_fitness[i] = fit
        pbest_f1[i] = f1_val
        if fit < gbest_fitness:
            gbest_fitness = fit
            gbest_position = np.copy(positions[i])
            gbest_f1 = f1_val
            gbest_mask = np.copy(bmask)

    convergence_history = []
    print(f"    --> Initial Swarm Best Fitness: {gbest_fitness:.6f} (F1: {gbest_f1*100:.2f}%)")

    # Evolutionary Swarm Loop
    for it in range(max_iter):
        w = w_max - (it / max_iter) * (w_max - w_min)
        for i in range(num_particles):
            r1 = np.random.rand(num_dims)
            r2 = np.random.rand(num_dims)

            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - positions[i]) +
                             c2 * r2 * (gbest_position - positions[i]))
            positions[i] += velocities[i]
            bmask = (sigmoid(positions[i]) >= 0.5).astype(int)

            fit, f1_val, _ = evaluate_fitness(bmask)
            if fit < pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_f1[i] = f1_val
                pbest_positions[i] = np.copy(positions[i])

            if fit < gbest_fitness:
                gbest_fitness = fit
                gbest_position = np.copy(positions[i])
                gbest_f1 = f1_val
                gbest_mask = np.copy(bmask)

        convergence_history.append(gbest_fitness)
        if (it + 1) % 5 == 0 or it == max_iter - 1:
            sel_cnt = int(np.sum(gbest_mask))
            print(f"    [Iteration {it+1:2d}/{max_iter:2d}] Best Fitness: {gbest_fitness:.6f} | F1: {gbest_f1*100:.3f}% | Selected: {sel_cnt} features")

    # Final Selected Subset
    selected_indices = np.where(gbest_mask == 1)[0]
    final_features = [candidate_features[idx] for idx in selected_indices]

    print("\n" + "=" * 80)
    print(f" [SUCCESS] Stage 1 Feature Selection Completed in {time.time() - t0:.2f}s")
    print(f" Optimal Feature Count: {len(final_features)} / {total_features} (Dimensionality Reduction: {((total_features - len(final_features))/total_features)*100:.1f}%)")
    print("=" * 80)

    # Save Phase2_Selected_Features.csv
    features_data = []
    for rank, feat in enumerate(final_features, 1):
        justification = FEATURE_JUSTIFICATIONS.get(feat, "Key discriminatory traffic pattern.")
        features_data.append({
            "Feature_Rank": rank,
            "Feature_Name": feat,
            "Domain_Security_Justification": justification
        })
        print(f"  {rank}. {feat:<20} -> {justification}")

    selected_df = pd.DataFrame(features_data)
    out_csv = os.path.join(FS_RESULTS_DIR, "Phase2_Selected_Features.csv")
    handover_csv = os.path.join(HANDOVER_DIR, "Phase2_Selected_Features.csv")
    local_csv = os.path.join(CURRENT_DIR, "Phase2_Selected_Features.csv")

    selected_df.to_csv(out_csv, index=False)
    selected_df.to_csv(handover_csv, index=False)
    if CURRENT_DIR != BASE_DIR:
        selected_df.to_csv(local_csv, index=False)

    print(f"\n[+] Exported Phase2_Selected_Features.csv to:")
    print(f"    --> {out_csv}")
    print(f"    --> {handover_csv}")

    # Plot Convergence Curve
    plt.figure(figsize=(8, 4.5), dpi=300)
    plt.plot(range(1, max_iter + 1), convergence_history, marker='o', color='#2B6CB0', linewidth=2.2, markersize=5)
    plt.title("Convergence of Binary-PSO Feature Selection Fitness", fontsize=12, fontweight='bold', color='#1A365D')
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Multi-Objective Cost Fitness", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    conv_plot_path = os.path.join(FS_RESULTS_DIR, "BPSO_Convergence_Curve.png")
    plt.savefig(conv_plot_path)
    plt.savefig(os.path.join(HANDOVER_DIR, "BPSO_Convergence_Curve.png"))
    plt.close()
    print(f"[+] Saved Convergence Plot to: {conv_plot_path}")

    # =========================================================================
    # HANDOVER ASSEMBLY FOR BIBHAS
    # =========================================================================
    print("\n[+] Step 4: Slicing Pre-filtered Datasets for Bibhas (Handover Package)...")
    test_df = pd.read_csv(TEST_PATH)

    train_sel = train_df[final_features + ['label']]
    test_sel = test_df[final_features + ['label']]

    train_out = os.path.join(HANDOVER_DIR, "train_selected_phase2.csv")
    test_out = os.path.join(HANDOVER_DIR, "test_selected_phase2.csv")
    train_sel.to_csv(train_out, index=False)
    test_sel.to_csv(test_out, index=False)

    print(f"    --> Saved: {train_out} ({len(train_sel):,} rows x {train_sel.shape[1]} cols)")
    print(f"    --> Saved: {test_out} ({len(test_sel):,} rows x {test_sel.shape[1]} cols)")

    # Metadata & Schema
    metadata = {
        "project": "IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB",
        "phase": "Phase II Extension",
        "stage_1_lead": "Dhanishetti Vishnu (23118032)",
        "stage_2_lead": "Bibhas Nath (23118019)",
        "num_selected_features": len(final_features),
        "selected_features": final_features,
        "dimensionality_reduction_pct": ((total_features - len(final_features)) / total_features) * 100,
        "feature_details": features_data
    }
    with open(os.path.join(HANDOVER_DIR, "feature_order_phase2.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print(" [VISHNU'S FEATURE SELECTION IS 100% COMPLETE & READY FOR BIBHAS]")
    print(f" Handover Directory: {HANDOVER_DIR}")
    print("=" * 80)

    return final_features

if __name__ == "__main__":
    run_phase2_feature_selection()
