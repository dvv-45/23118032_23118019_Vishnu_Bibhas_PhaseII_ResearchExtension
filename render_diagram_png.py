import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Background canvas color
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Header / Title
    ax.text(9, 9.6, "Proposed Extended Methodology Pipeline for IoT Botnet Detection", 
            ha='center', va='center', fontsize=18, fontweight='bold', color='#1A365D')
    ax.text(9, 9.2, "Phase II Research Extension: Two-Stage Hybrid Feature Selection & Metaheuristic Hyperparameter Optimization", 
            ha='center', va='center', fontsize=12, fontstyle='italic', color='#4A5568')

    # Container 1: Feature Engineering (Top Half)
    rect_box1 = patches.FancyBboxPatch((0.5, 5.0), 17.0, 3.8, boxstyle="round,pad=0.2", 
                                       facecolor='#F8FAFC', edgecolor='#94A3B8', linestyle='--', linewidth=1.5)
    ax.add_patch(rect_box1)
    ax.text(1.0, 8.55, "STAGE I: DATA INGESTION & ADVANCED FEATURE SELECTION", fontsize=11, fontweight='bold', color='#334155')

    # Node 1: Raw Dataset
    rect1 = patches.FancyBboxPatch((0.8, 5.5), 2.5, 2.5, boxstyle="round,pad=0.1", 
                                  facecolor='#EBF8FF', edgecolor='#3182CE', linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.05, 7.3, "IoT Network Traffic", ha='center', fontsize=11, fontweight='bold', color='#2B6CB0')
    ax.text(2.05, 6.7, "• IoT-NID Mirai Subset\n• 18 PCAP captures\n• 2.34M raw packets\n• 35 initial flow attributes", 
            ha='center', va='center', fontsize=9, color='#2D3748')

    # Arrow 1 -> 2
    ax.annotate('', xy=(3.8, 6.75), xytext=(3.3, 6.75),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2, headwidth=8))

    # Node 2: Preprocessing
    rect2 = patches.FancyBboxPatch((3.8, 5.5), 2.8, 2.5, boxstyle="round,pad=0.1", 
                                  facecolor='#F1F5F9', edgecolor='#64748B', linewidth=2)
    ax.add_patch(rect2)
    ax.text(5.2, 7.3, "Data Preprocessing", ha='center', fontsize=11, fontweight='bold', color='#334155')
    ax.text(5.2, 6.6, "• Drop utility IDs (23 left)\n• NaN imputation & encoding\n• Random UnderSampling (RUS)\n• 87,422 balanced flows", 
            ha='center', va='center', fontsize=9, color='#2D3748')

    # Arrow 2 -> Proposed FS Box
    ax.annotate('', xy=(7.1, 6.75), xytext=(6.6, 6.75),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2, headwidth=8))

    # PROPOSED EXTENSION CONTAINER (Highlight Gold)
    rect_fs_ext = patches.FancyBboxPatch((7.1, 5.25), 6.5, 3.0, boxstyle="round,pad=0.15", 
                                         facecolor='#FEFCBF', edgecolor='#D69E2E', linestyle='-', linewidth=2.5)
    ax.add_patch(rect_fs_ext)
    ax.text(10.35, 8.0, "★ PROPOSED EXTENSION: Two-Stage Hybrid Feature Selection ★", 
            ha='center', fontsize=10, fontweight='bold', color='#B7791F')

    # Subnode 3a: Filter
    rect3a = patches.FancyBboxPatch((7.3, 5.5), 2.8, 2.2, boxstyle="round,pad=0.1", 
                                   facecolor='#FFFFFF', edgecolor='#D69E2E', linewidth=1.5)
    ax.add_patch(rect3a)
    ax.text(8.7, 7.1, "Stage A: Filter Ranking", ha='center', fontsize=10, fontweight='bold', color='#744210')
    ax.text(8.7, 6.3, "• Mutual Information (MI)\n• Extra-Trees Importance\n• Prunes bottom 50%\nnoisy & static features", 
            ha='center', va='center', fontsize=8.5, color='#4A5568')

    # Arrow 3a -> 3b
    ax.annotate('', xy=(10.7, 6.6), xytext=(10.1, 6.6),
                arrowprops=dict(facecolor='#DD6B20', edgecolor='#DD6B20', width=2, headwidth=7))
    ax.text(10.4, 6.9, "Top 50%", ha='center', fontsize=8, fontweight='bold', color='#C05621')

    # Subnode 3b: Swarm/Metaheuristic
    rect3b = patches.FancyBboxPatch((10.7, 5.5), 2.7, 2.2, boxstyle="round,pad=0.1", 
                                   facecolor='#FEEBC8', edgecolor='#DD6B20', linewidth=1.5)
    ax.add_patch(rect3b)
    ax.text(12.05, 7.1, "Stage B: Swarm Optimization", ha='center', fontsize=10, fontweight='bold', color='#9C4221')
    ax.text(12.05, 6.3, "• Binary-adapted PSO / RIME\n• Fitness = α·(1-F1) + β·(Fs/Fa)\n• Global multi-feature search\n• Avoids greedy RFE traps", 
            ha='center', va='center', fontsize=8.5, color='#4A5568')

    # Arrow Proposed FS -> Refined Set
    ax.annotate('', xy=(14.2, 6.75), xytext=(13.6, 6.75),
                arrowprops=dict(facecolor='#38A169', edgecolor='#38A169', width=2, headwidth=8))

    # Node 4: Refined Subset
    rect4 = patches.FancyBboxPatch((14.2, 5.5), 2.9, 2.5, boxstyle="round,pad=0.1", 
                                  facecolor='#C6F6D5', edgecolor='#38A169', linewidth=2)
    ax.add_patch(rect4)
    ax.text(15.65, 7.3, "Refined Feature Dataset", ha='center', fontsize=11, fontweight='bold', color='#22543D')
    ax.text(15.65, 6.5, "• Highly discriminative subset\n• 6–8 optimal features\n• >70% dimensionality cut\n• Low memory & CPU footprint", 
            ha='center', va='center', fontsize=9, color='#22543D')

    # Inter-stage Routing Arrow (Downward from Node 4 to Container 2)
    ax.annotate('', xy=(15.65, 4.3), xytext=(15.65, 5.5),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2.5, headwidth=8))
    ax.annotate('', xy=(2.0, 4.3), xytext=(15.65, 4.3),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2.5, headwidth=0))
    ax.annotate('', xy=(2.0, 3.8), xytext=(2.0, 4.3),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2.5, headwidth=8))
    ax.text(8.8, 4.5, "Input: Optimal Feature Subset", ha='center', fontsize=9.5, fontweight='bold', color='#4A5568',
            bbox=dict(boxstyle="square,pad=0.3", facecolor="#EDF2F7", edgecolor="#CBD5E0"))

    # Container 2: Model Optimization & Evaluation (Bottom Half)
    rect_box2 = patches.FancyBboxPatch((0.5, 0.4), 17.0, 3.4, boxstyle="round,pad=0.2", 
                                       facecolor='#F8FAFC', edgecolor='#94A3B8', linestyle='--', linewidth=1.5)
    ax.add_patch(rect_box2)
    ax.text(1.0, 3.5, "STAGE II: AUTOMATED MODEL OPTIMIZATION, BENCHMARKING & INTERPRETABILITY", 
            fontsize=11, fontweight='bold', color='#334155')

    # Node 5: PROPOSED EXTENSION Hyperparameter Optimizer
    rect5 = patches.FancyBboxPatch((0.8, 0.7), 3.5, 2.5, boxstyle="round,pad=0.1", 
                                  facecolor='#E9D8FD', edgecolor='#805AD5', linewidth=2.5)
    ax.add_patch(rect5)
    ax.text(2.55, 2.7, "★ PROPOSED EXTENSION ★\nAutomated Hyperparameter Tuning", ha='center', fontsize=10.5, fontweight='bold', color='#553C9A')
    ax.text(2.55, 1.7, "• Bayesian Optuna (TPE) / CPO\n• 50+ Global Optimization Trials\n• Automatically tunes:\n  max_depth, lr, n_estimators,\n  subsample, colsample_bytree, gamma", 
            ha='center', va='center', fontsize=8.5, color='#44337A')

    # Arrow 5 -> 6
    ax.annotate('', xy=(4.9, 1.95), xytext=(4.3, 1.95),
                arrowprops=dict(facecolor='#805AD5', edgecolor='#805AD5', width=2, headwidth=8))
    ax.text(4.6, 2.2, "Best HPs", ha='center', fontsize=8, fontweight='bold', color='#6B46C1')

    # Node 6: Final Classifier
    rect6 = patches.FancyBboxPatch((4.9, 0.7), 3.4, 2.5, boxstyle="round,pad=0.1", 
                                  facecolor='#BEE3F8', edgecolor='#3182CE', linewidth=2)
    ax.add_patch(rect6)
    ax.text(6.6, 2.7, "Optimized Botnet Detector", ha='center', fontsize=11, fontweight='bold', color='#2A4365')
    ax.text(6.6, 1.7, "• Extreme Gradient Boosting (XGB)\n• Optional Stacking with LightGBM\n• Stratified 5-Fold Cross Validation\n• Zero-leakage train/test validation", 
            ha='center', va='center', fontsize=8.5, color='#2D3748')

    # Arrow 6 -> 7
    ax.annotate('', xy=(8.9, 1.95), xytext=(8.3, 1.95),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2, headwidth=8))

    # Node 7: Multi-Metric Evaluation & Comparison
    rect7 = patches.FancyBboxPatch((8.9, 0.7), 4.1, 2.5, boxstyle="round,pad=0.1", 
                                  facecolor='#FEEBC8', edgecolor='#DD6B20', linewidth=2)
    ax.add_patch(rect7)
    ax.text(10.95, 2.7, "Rigorous Evaluation & SOTA Comparison", ha='center', fontsize=10.5, fontweight='bold', color='#7B341E')
    ax.text(10.95, 1.7, "• Accuracy, Precision, Recall, F1\n• ROC-AUC Curve & Confusion Matrix\n• Per-packet Inference Latency (ms)\n• Tabular Comparison vs 3 SOTA Papers\n  (Result_Comparison.xlsx)", 
            ha='center', va='center', fontsize=8.5, color='#4A5568')

    # Arrow 7 -> 8
    ax.annotate('', xy=(13.6, 1.95), xytext=(13.0, 1.95),
                arrowprops=dict(facecolor='#E53E3E', edgecolor='#E53E3E', width=2, headwidth=8))

    # Node 8: Explainable AI
    rect8 = patches.FancyBboxPatch((13.6, 0.7), 3.5, 2.5, boxstyle="round,pad=0.1", 
                                  facecolor='#FED7D7', edgecolor='#E53E3E', linewidth=2)
    ax.add_patch(rect8)
    ax.text(15.35, 2.7, "Explainable AI (XAI)", ha='center', fontsize=11, fontweight='bold', color='#742A2A')
    ax.text(15.35, 1.7, "• SHAP Global Summary (Beeswarm)\n• Individual Sample Force Plots\n• Security Domain Semantics Check\n• Model Transparency & Trust", 
            ha='center', va='center', fontsize=8.5, color='#4A5568')

    plt.tight_layout()
    output_png = r"c:\PAPER_IMPLENTATION\23118032_23118019_Vishnu_Bibhas_PhaseII_ResearchExtension\Proposed_Methodology.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] High-resolution PNG saved to: {output_png}")

if __name__ == "__main__":
    draw_pipeline()
