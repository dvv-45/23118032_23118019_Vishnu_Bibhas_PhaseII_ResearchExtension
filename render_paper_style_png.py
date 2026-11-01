"""
Renders Published-Paper Style Flow Diagram matching Professor's Reference:
'new (modified) proposed methodology.png'
Group 6: IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_paper_style_diagram():
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # -------------------------------------------------------------
    # TOP SECTION: Feature Engineering
    # -------------------------------------------------------------
    rect_box1 = patches.Rectangle((0.6, 6.2), 14.8, 5.0,
                                  facecolor='none', edgecolor='#B5BD68', linestyle='--', linewidth=1.8)
    ax.add_patch(rect_box1)
    ax.text(15.1, 10.7, "Feature engineering", ha='right', va='center',
            fontsize=20, fontweight='bold', fontfamily='serif', color='#2D3748')

    # Node 1: Dataset Cylinder Icon
    cyl_x, cyl_y, cyl_w, cyl_h = 1.3, 7.8, 1.4, 1.6
    cyl_rect = patches.Rectangle((cyl_x, cyl_y), cyl_w, cyl_h, facecolor='#FFFFFF', edgecolor='#1A202C', linewidth=2.5)
    ax.add_patch(cyl_rect)
    ellipse_top = patches.Ellipse((cyl_x + cyl_w/2, cyl_y + cyl_h), cyl_w, 0.45, facecolor='#FFFFFF', edgecolor='#1A202C', linewidth=2.5)
    ellipse_bot = patches.Arc((cyl_x + cyl_w/2, cyl_y), cyl_w, 0.45, theta1=180, theta2=360, edgecolor='#1A202C', linewidth=2.5)
    ellipse_m1 = patches.Arc((cyl_x + cyl_w/2, cyl_y + 0.5), cyl_w, 0.45, theta1=180, theta2=360, edgecolor='#1A202C', linewidth=2.5)
    ellipse_m2 = patches.Arc((cyl_x + cyl_w/2, cyl_y + 1.0), cyl_w, 0.45, theta1=180, theta2=360, edgecolor='#1A202C', linewidth=2.5)
    ax.add_patch(ellipse_top)
    ax.add_patch(ellipse_bot)
    ax.add_patch(ellipse_m1)
    ax.add_patch(ellipse_m2)
    ax.text(cyl_x + cyl_w/2, 7.3, "IoT-NID Mirai\ndataset", ha='center', va='top',
            fontsize=13, fontweight='bold', fontfamily='serif', color='#1A202C')

    # Arrow 1 -> 2
    ax.annotate('', xy=(4.2, 8.6), xytext=(3.0, 8.6),
                arrowprops=dict(facecolor='#1A202C', edgecolor='#1A202C', width=1.5, headwidth=7))

    # Node 2: Data Preprocessing Gear & Clock
    gear_center = (4.9, 8.7)
    gear = patches.Circle(gear_center, 0.65, facecolor='#FFFFFF', edgecolor='#1A202C', linewidth=2.5)
    ax.add_patch(gear)
    gear_in = patches.Circle(gear_center, 0.38, facecolor='#FFFFFF', edgecolor='#1A202C', linewidth=2.0)
    ax.add_patch(gear_in)
    # Clock hands
    ax.plot([gear_center[0], gear_center[0]], [gear_center[1], gear_center[1] + 0.28], color='#1A202C', linewidth=2.5)
    ax.plot([gear_center[0], gear_center[0] + 0.22], [gear_center[1], gear_center[1]], color='#1A202C', linewidth=2.5)
    # Database icon next to gear
    db_s = patches.Rectangle((5.5, 7.8), 0.75, 0.8, facecolor='#FFFFFF', edgecolor='#1A202C', linewidth=2.2)
    ax.add_patch(db_s)
    ax.plot([5.5, 6.25], [8.08, 8.08], color='#1A202C', linewidth=2)
    ax.plot([5.5, 6.25], [8.35, 8.35], color='#1A202C', linewidth=2)
    ax.text(5.25, 7.3, "Data preprocessing\n(Clean & RUS Balanced)", ha='center', va='top',
            fontsize=13, fontweight='bold', fontfamily='serif', color='#1A202C')

    # Arrow 2 -> Feature Selection Stages Box
    ax.annotate('', xy=(7.2, 8.6), xytext=(6.5, 8.6),
                arrowprops=dict(facecolor='#1A202C', edgecolor='#1A202C', width=1.5, headwidth=7))

    # Feature selection stages inner dashed box
    fss_box = patches.Rectangle((7.2, 6.55), 7.0, 3.85,
                                facecolor='none', edgecolor='#1A202C', linestyle='--', linewidth=1.5)
    ax.add_patch(fss_box)
    ax.text(14.0, 10.0, "Feature selection stages", ha='right', va='center',
            fontsize=16, fontweight='bold', fontfamily='serif', color='#1A202C')

    # Inner Node A: Bar Chart (Feature Importance)
    bc_x, bc_y = 7.7, 7.95
    ax.plot([bc_x, bc_x + 1.8], [bc_y, bc_y], color='#1A202C', linewidth=3)
    ax.fill_between([bc_x + 0.1, bc_x + 0.32], [bc_y, bc_y], [bc_y + 0.45, bc_y + 0.45], color='#1A202C')
    ax.fill_between([bc_x + 0.45, bc_x + 0.67], [bc_y, bc_y], [bc_y + 0.8, bc_y + 0.8], color='#1A202C')
    ax.fill_between([bc_x + 0.8, bc_x + 1.02], [bc_y, bc_y], [bc_y + 1.1, bc_y + 1.1], color='#1A202C')
    ax.fill_between([bc_x + 1.15, bc_x + 1.37], [bc_y, bc_y], [bc_y + 1.45, bc_y + 1.45], color='#1A202C')
    ax.fill_between([bc_x + 1.5, bc_x + 1.72], [bc_y, bc_y], [bc_y + 1.25, bc_y + 1.25], color='#1A202C')
    ax.text(bc_x + 0.9, 7.45, "Extra-Trees & MI\nfeature importance\n(proposed filter score)", ha='center', va='top',
            fontsize=12, fontweight='bold', fontfamily='serif', color='#1A202C')

    # Arrow inside FSS (Filter -> Wrapper)
    ax.annotate('', xy=(10.7, 8.6), xytext=(9.9, 8.6),
                arrowprops=dict(facecolor='#1A202C', edgecolor='#1A202C', width=1.5, headwidth=6))

    # Inner Node B: Light-Green Highlighted Swarm Metaheuristic
    meta_box = patches.Rectangle((10.7, 7.8), 2.8, 1.8,
                                 facecolor='#EDF7ED', edgecolor='#2E7D32', linestyle='--', linewidth=1.8)
    ax.add_patch(meta_box)
    ax.text(12.1, 8.95, "B-PSO", ha='center', va='center', fontsize=22, fontweight='bold', fontfamily='serif', color='#1A202C')
    ax.text(12.1, 8.3, "Binary-adapted", ha='center', va='center', fontsize=12, fontfamily='serif', color='#2D3748')
    ax.text(12.1, 7.45, "Metaheuristic\nfeature selection\n(Stage 2: 6 optimal)", ha='center', va='top',
            fontsize=12, fontweight='bold', fontfamily='serif', color='#1A202C')

    # Routing Line from Top Tier to Bottom Tier
    ax.plot([14.2, 14.8], [8.6, 8.6], color='#1A202C', linewidth=1.5)
    ax.plot([14.8, 14.8], [8.6, 3.6], color='#1A202C', linewidth=1.5)
    ax.annotate('', xy=(13.6, 3.6), xytext=(14.8, 3.6),
                arrowprops=dict(facecolor='#1A202C', edgecolor='#1A202C', width=1.5, headwidth=7))

    # -------------------------------------------------------------
    # BOTTOM SECTION: Model Optimization and Evaluation
    # -------------------------------------------------------------
    rect_box2 = patches.Rectangle((0.6, 0.6), 14.8, 5.0,
                                  facecolor='none', edgecolor='#81C784', linestyle='--', linewidth=1.8)
    ax.add_patch(rect_box2)
    # Label at bottom right below the cards
    ax.text(15.1, 1.0, "Model optimization and evaluation", ha='right', va='center',
            fontsize=20, fontweight='bold', fontfamily='serif', color='#2D3748')

    # Node 4: Refined Feature Dataset (Right side of bottom tier)
    rf_x, rf_y, rf_w, rf_h = 12.0, 2.7, 1.4, 1.6
    rf_rect = patches.Rectangle((rf_x, rf_y), rf_w, rf_h, facecolor='#FFFFFF', edgecolor='#1A202C', linewidth=2.5)
    ax.add_patch(rf_rect)
    rf_top = patches.Ellipse((rf_x + rf_w/2, rf_y + rf_h), rf_w, 0.45, facecolor='#FFFFFF', edgecolor='#1A202C', linewidth=2.5)
    rf_bot = patches.Arc((rf_x + rf_w/2, rf_y), rf_w, 0.45, theta1=180, theta2=360, edgecolor='#1A202C', linewidth=2.5)
    rf_m1 = patches.Arc((rf_x + rf_w/2, rf_y + 0.5), rf_w, 0.45, theta1=180, theta2=360, edgecolor='#1A202C', linewidth=2.5)
    rf_m2 = patches.Arc((rf_x + rf_w/2, rf_y + 1.0), rf_w, 0.45, theta1=180, theta2=360, edgecolor='#1A202C', linewidth=2.5)
    ax.add_patch(rf_top)
    ax.add_patch(rf_bot)
    ax.add_patch(rf_m1)
    ax.add_patch(rf_m2)
    ax.text(rf_x + rf_w/2, 2.3, "Refined feature\ndataset\n(6 Features, >73% cut)", ha='center', va='top',
            fontsize=13, fontweight='bold', fontfamily='serif', color='#1A202C')

    # Arrow Refined -> Tuning (Right to Left)
    ax.annotate('', xy=(10.7, 3.6), xytext=(11.8, 3.6),
                arrowprops=dict(facecolor='#1A202C', edgecolor='#1A202C', width=1.5, headwidth=7))

    # Node 5: Hyperparameter Tuning Sliders + Gear (Highlighted Box)
    tune_box = patches.Rectangle((8.2, 2.6), 2.5, 1.9,
                                 facecolor='#FAF5FF', edgecolor='#805AD5', linestyle='--', linewidth=1.8)
    ax.add_patch(tune_box)
    # Sliders lines
    ax.plot([8.5, 9.3], [3.8, 3.8], color='#1A202C', linewidth=2)
    ax.plot([8.75], [3.8], marker='o', markersize=6, color='#1A202C')
    ax.plot([8.5, 9.3], [3.5, 3.5], color='#1A202C', linewidth=2)
    ax.plot([9.1], [3.5], marker='o', markersize=6, color='#1A202C')
    ax.plot([8.5, 9.3], [3.2, 3.2], color='#1A202C', linewidth=2)
    ax.plot([8.65], [3.2], marker='o', markersize=6, color='#1A202C')
    # Gear
    tune_gear = patches.Circle((9.9, 3.5), 0.5, facecolor='#FFFFFF', edgecolor='#1A202C', linewidth=2.0)
    ax.add_patch(tune_gear)
    tune_gear_in = patches.Circle((9.9, 3.5), 0.22, facecolor='#1A202C', edgecolor='#1A202C')
    ax.add_patch(tune_gear_in)
    ax.text(9.45, 2.3, "Optuna / CPO-based\nhyperparameter tuning\n(5-Fold Stratified CV)", ha='center', va='top',
            fontsize=13, fontweight='bold', fontfamily='serif', color='#1A202C')

    # Arrow Tuning -> Model (Right to Left)
    ax.annotate('', xy=(6.9, 3.6), xytext=(8.0, 3.6),
                arrowprops=dict(facecolor='#1A202C', edgecolor='#1A202C', width=1.5, headwidth=7))

    # Node 6: Best Classifier Model (3D Isometric Cube)
    cube_x, cube_y = 5.2, 3.2
    top_poly = patches.Polygon([[cube_x, cube_y + 0.65], [cube_x + 0.7, cube_y + 0.98], [cube_x + 1.4, cube_y + 0.65], [cube_x + 0.7, cube_y + 0.32]],
                               closed=True, facecolor='#FFFFFF', edgecolor='#1A202C', linewidth=2.2)
    left_poly = patches.Polygon([[cube_x, cube_y + 0.65], [cube_x + 0.7, cube_y + 0.32], [cube_x + 0.7, cube_y - 0.45], [cube_x, cube_y - 0.12]],
                                closed=True, facecolor='#E2E8F0', edgecolor='#1A202C', linewidth=2.2)
    right_poly = patches.Polygon([[cube_x + 0.7, cube_y + 0.32], [cube_x + 1.4, cube_y + 0.65], [cube_x + 1.4, cube_y - 0.12], [cube_x + 0.7, cube_y - 0.45]],
                                 closed=True, facecolor='#CBD5E0', edgecolor='#1A202C', linewidth=2.2)
    ax.add_patch(top_poly)
    ax.add_patch(left_poly)
    ax.add_patch(right_poly)
    ax.text(cube_x + 0.7, 2.3, "Best XGBoost\nclassifier model\n(Tuned Tree Ensemble)", ha='center', va='top',
            fontsize=13, fontweight='bold', fontfamily='serif', color='#1A202C')

    # Arrow Model -> Evaluation (Right to Left)
    ax.annotate('', xy=(3.5, 3.6), xytext=(4.8, 3.6),
                arrowprops=dict(facecolor='#1A202C', edgecolor='#1A202C', width=1.5, headwidth=7))

    # Node 7: Magnifying Glass (Performance Evaluation & SHAP)
    lens_x, lens_y = 2.1, 3.7
    lens = patches.Circle((lens_x, lens_y), 0.6, facecolor='#FFFFFF', edgecolor='#1A202C', linewidth=3.0)
    ax.add_patch(lens)
    lens_inner = patches.Circle((lens_x, lens_y), 0.42, facecolor='none', edgecolor='#CBD5E0', linewidth=1.5)
    ax.add_patch(lens_inner)
    ax.plot([lens_x + 0.42, lens_x + 0.92], [lens_y - 0.42, lens_y - 0.92], color='#1A202C', linewidth=4.8)
    ax.text(lens_x + 0.3, 2.3, "Performance evaluation\n& SHAP interpretability\n(Latency, ROC, F1: 99.91%)", ha='center', va='top',
            fontsize=13, fontweight='bold', fontfamily='serif', color='#1A202C')

    # Save outputs
    out_dir1 = r"c:\PAPER_IMPLENTATION\23118032_23118019_Vishnu_Bibhas_PhaseII_ResearchExtension"
    out_file1 = os.path.join(out_dir1, "Proposed_Methodology_PaperStyle.png")
    out_file2 = r"c:\PAPER_IMPLENTATION\Proposed_Methodology_PaperStyle.png"
    plt.savefig(out_file1, dpi=300, bbox_inches='tight')
    plt.savefig(out_file2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Paper-style flow diagram generated cleanly:")
    print(f"  --> {out_file1}")
    print(f"  --> {out_file2}")

if __name__ == "__main__":
    draw_paper_style_diagram()
