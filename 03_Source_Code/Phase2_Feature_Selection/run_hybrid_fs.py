"""
Phase II Extension - Stage 1: Two-Stage Hybrid Feature Selection
Group 6: Vishnu (23118032) & Bibhas (23118019)

Proposed Methodology:
- Stage 1 (Filter): Extra-Trees & Mutual Information Feature Importance (prunes bottom 50%)
- Stage 2 (Wrapper): Binary-adapted Particle Swarm Optimization (B-PSO) multi-objective optimization
  Objective: Minimize [0.99 * (1 - F1) + 0.01 * (Selected_Features / Total_Features)]
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_PATH = os.path.join(BASE_DIR, "02_Dataset", "Processed_Dataset", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "02_Dataset", "Processed_Dataset", "test.csv")
FS_RESULTS_DIR = os.path.join(BASE_DIR, "05_Phase2_Feature_Selection_Results")
HANDOVER_DIR = os.path.join(BASE_DIR, "Vishnu_To_Bibhas")

os.makedirs(FS_RESULTS_DIR, exist_ok=True)
os.makedirs(HANDOVER_DIR, exist_ok=True)

def run_two_stage_hybrid_fs():
    print("=" * 75)
    print(" [STAGE 1] TWO-STAGE HYBRID FEATURE SELECTION PIPELINE")
    print("=" * 75)
    
    start_total_time = time.time()
    
    # 1. Load Data
    print(f"\n[+] Loading training dataset from: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df.drop(columns=['label'])
    y = train_df['label'].astype(int)
    feature_names = list(X.columns)
    total_features = len(feature_names)
    print(f"    --> Total instances: {len(train_df):,}, Initial Candidate Features: {total_features}")

    # =========================================================================
    # PHASE A: FILTER PRE-SELECTION (Extra-Trees Importance)
    # =========================================================================
    print("\n[+] Phase A: Filter-based Pre-selection via Ensemble Tree Importance...")
    filter_model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    filter_model.fit(X, y)
    
    importances = filter_model.feature_importances_
    ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Score': importances
    }).sort_values(by='Importance_Score', ascending=False).reset_index(drop=True)
    
    filter_csv = os.path.join(FS_RESULTS_DIR, "Filter_Stage_Feature_Ranking.csv")
    ranking_df.to_csv(filter_csv, index=False)
    print(f"    --> Filter stage rankings saved to: {filter_csv}")
    
    # Retain top 50% features (e.g., top 12 of 23)
    k_retain = int(np.ceil(total_features * 0.50))
    candidate_features = ranking_df['Feature'].iloc[:k_retain].tolist()
    print(f"    --> Retained Top {k_retain} features (50% reduction) for Metaheuristic Search:")
    for rank, feat in enumerate(candidate_features, 1):
        print(f"        {rank:2d}. {feat:<25} (Score: {ranking_df.loc[ranking_df['Feature'] == feat, 'Importance_Score'].values[0]:.4f})")

    # =========================================================================
    # PHASE B: BINARY-ADAPTED PARTICLE SWARM OPTIMIZATION (B-PSO)
    # =========================================================================
    print("\n[+] Phase B: Binary Particle Swarm Optimization (B-PSO) Feature Selection...")
    
    # Subsample for fast, stable fitness evaluation during swarm iterations
    eval_subsample_size = min(12000, len(X))
    sub_df = train_df.sample(n=eval_subsample_size, random_state=42)
    X_sub = sub_df[candidate_features].values
    y_sub = sub_df['label'].values

    # B-PSO Parameters
    num_particles = 20
    max_iter = 25
    w_max, w_min = 0.9, 0.4
    c1, c2 = 1.5, 1.5
    num_dims = len(candidate_features)
    
    # Initialize Swarm
    np.random.seed(42)
    positions = np.random.uniform(0, 1, (num_particles, num_dims))
    velocities = np.random.uniform(-1, 1, (num_particles, num_dims))
    
    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(-np.clip(v, -10, 10)))
    
    def evaluate_fitness(binary_mask):
        selected_indices = np.where(binary_mask == 1)[0]
        if len(selected_indices) == 0:
            return 1.0  # Maximum error penalty
        
        X_selected = X_sub[:, selected_indices]
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        f1_scores = []
        
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        for train_idx, val_idx in cv.split(X_selected, y_sub):
            X_tr, X_val = X_selected[train_idx], X_selected[val_idx]
            y_tr, y_val = y_sub[train_idx], y_sub[val_idx]
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            f1_scores.append(f1_score(y_val, preds, zero_division=0))
        
        mean_f1 = np.mean(f1_scores)
        fs_count = len(selected_indices)
        
        # Fitness = 0.99 * Error_Rate + 0.01 * (Fs / Fa) + protocol diversity penalty if < 6 features
        alpha, beta = 0.99, 0.01
        fitness = alpha * (1.0 - mean_f1) + beta * (fs_count / total_features)
        if fs_count < 6:
            fitness += 0.005 * (6 - fs_count)  # Ensure sufficient protocol coverage for robust IDS
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
        binary_mask = (sigmoid(positions[i]) >= 0.5).astype(int)
        fit, f1_val, _ = evaluate_fitness(binary_mask)
        pbest_fitness[i] = fit
        pbest_f1[i] = f1_val
        if fit < gbest_fitness:
            gbest_fitness = fit
            gbest_position = np.copy(positions[i])
            gbest_f1 = f1_val
            gbest_mask = np.copy(binary_mask)

    convergence_history = []
    print(f"    --> Initial Swarm Best Fitness: {gbest_fitness:.6f} (F1: {gbest_f1*100:.2f}%)")

    # Swarm Evolution
    for it in range(max_iter):
        w = w_max - (it / max_iter) * (w_max - w_min)
        for i in range(num_particles):
            r1 = np.random.rand(num_dims)
            r2 = np.random.rand(num_dims)
            
            # Velocity update
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest_positions[i] - positions[i]) + 
                             c2 * r2 * (gbest_position - positions[i]))
            
            # Position update via Sigmoid probability
            positions[i] += velocities[i]
            binary_mask = (sigmoid(positions[i]) >= 0.5).astype(int)
            
            # Evaluate
            fit, f1_val, _ = evaluate_fitness(binary_mask)
            if fit < pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_f1[i] = f1_val
                pbest_positions[i] = np.copy(positions[i])
                
            if fit < gbest_fitness:
                gbest_fitness = fit
                gbest_position = np.copy(positions[i])
                gbest_f1 = f1_val
                gbest_mask = np.copy(binary_mask)
        
        convergence_history.append(gbest_fitness)
        if (it + 1) % 5 == 0 or it == max_iter - 1:
            selected_count = np.sum(gbest_mask)
            print(f"    [Iteration {it+1:2d}/{max_iter:2d}] Best Fitness: {gbest_fitness:.6f} | F1: {gbest_f1*100:.3f}% | Features: {selected_count}")

    # =========================================================================
    # FINAL SELECTED SUBSET EXTRACTION
    # =========================================================================
    selected_indices = np.where(gbest_mask == 1)[0]
    final_selected_features = [candidate_features[idx] for idx in selected_indices]
    
    print("\n" + "=" * 75)
    print(f" [SUCCESS] Phase II Hybrid Feature Selection Complete in {time.time() - start_total_time:.2f}s")
    print(f" Optimal Feature Count: {len(final_selected_features)} (Reduction: {((total_features - len(final_selected_features))/total_features)*100:.1f}%)")
    print("=" * 75)
    for idx, feat in enumerate(final_selected_features, 1):
        print(f"  {idx}. {feat}")

    # Save to CSV
    selected_df = pd.DataFrame({
        'Feature_Rank': range(1, len(final_selected_features) + 1),
        'Feature_Name': final_selected_features
    })
    
    out_csv = os.path.join(FS_RESULTS_DIR, "Phase2_Selected_Features.csv")
    handover_csv = os.path.join(HANDOVER_DIR, "Phase2_Selected_Features.csv")
    selected_df.to_csv(out_csv, index=False)
    selected_df.to_csv(handover_csv, index=False)
    print(f"\n[+] Saved selected features to:\n    --> {out_csv}\n    --> {handover_csv}")

    # Plot Convergence Curve
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(range(1, max_iter + 1), convergence_history, marker='o', color='#3182CE', linewidth=2)
    plt.title("Convergence of Binary-PSO Feature Selection Fitness", fontsize=12, fontweight='bold')
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Multi-Objective Fitness", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    conv_plot_path = os.path.join(FS_RESULTS_DIR, "BPSO_Convergence_Curve.png")
    plt.savefig(conv_plot_path)
    plt.close()
    print(f"[+] Convergence curve plot saved to: {conv_plot_path}")

    return final_selected_features

if __name__ == "__main__":
    run_two_stage_hybrid_fs()
