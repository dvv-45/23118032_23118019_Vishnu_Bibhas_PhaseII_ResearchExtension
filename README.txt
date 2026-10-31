========================================================================================
Phase II Research Extension: Two-Stage Hybrid Feature Selection and Automated
Optimization for IoT Botnet Detection
========================================================================================

Team Details:
-------------
- Dhanishetti Vishnu (Roll No. 23118032) — Stage 1: Proposed Feature Selection, Architecture & System Design
- Bibhas Nath (Roll No. 23118019) — Stage 2: Automated Hyperparameter Optimization, Modeling & Comparative Benchmarking

Assigned Research Paper (Phase I Baseline):
-------------------------------------------
Title: Efficient and Interpretable IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB
Journal: Computer Networks, Elsevier, 2026
DOI: https://doi.org/10.1016/j.comnet.2026.112399

Phase II Proposed Methodology Extensions:
------------------------------------------
1. Feature Selection:
   - Original: Greedy Stepwise Recursive Feature Elimination (RFE).
   - Proposed Extension: Two-Stage Hybrid Feature Selection.
     * Stage 1 (Filter): Mutual Information & Extra-Trees Importance to prune bottom 50% noisy features.
     * Stage 2 (Swarm/Wrapper): Binary-adapted Metaheuristic Swarm Optimizer (B-PSO) balancing F1-score and feature compactness.
2. Model Optimization:
   - Original: Heuristic Step-by-Step Grid / Quick SmartSearch candidates.
   - Proposed Extension: Automated Bayesian Optimization (Optuna/TPE) with 5-fold Stratified Cross-Validation.
3. Comparative Benchmarking:
   - Evaluated against 3 external published SOTA papers on IoT intrusion detection.

Folder Structure:
-----------------
23118032_23118019_Vishnu_Bibhas_PhaseII_ResearchExtension/
├── 01_Research_Papers/
│   ├── Original_Assigned_Paper_Antony2026.pdf
│   ├── Comparison_Paper_1.pdf
│   ├── Comparison_Paper_2.pdf
│   └── Comparison_Paper_3.pdf
├── 02_Dataset/
│   ├── Processed_Dataset/
│   │   ├── train.csv (80% Stratified, 69,937 flows)
│   │   ├── test.csv  (20% Stratified, 17,485 flows)
│   │   └── cleaned_features.csv (87,422 flows)
│   └── Dataset_Source_Details.txt
├── 03_Source_Code/
│   ├── Phase2_Feature_Selection/
│   │   └── run_hybrid_fs.py
│   ├── Phase2_Model_Optimization/
│   │   └── train_optimized_xgb.py
│   └── run_phase2_complete_pipeline.py
├── 04_Requirements/
│   ├── requirements.txt
│   └── Software_Environment.txt
├── 05_Phase2_Feature_Selection_Results/
│   ├── Phase2_Selected_Features.csv
│   └── Feature_Importance_Comparison.png
├── 06_Phase2_Final_Results/
│   └── Evaluation_Metrics.csv
├── 07_Result_Images/
│   ├── Phase2_ROC_Curves.png
│   ├── Phase2_Confusion_Matrix.png
│   └── Optimization_Convergence_Curve.png
├── 08_Comparative_Results/
│   └── Result_Comparison.xlsx
├── 09_Trained_Models/
│   └── phase2_optimized_model.joblib
├── 10_Research_Extension_Report/
│   └── PhaseII_ResearchExtension_Report.pdf
├── Proposed_Methodology.drawio
├── Proposed_Methodology.png
├── Vishnu_To_Bibhas/
│   └── Phase2_Work_Division_and_Plan.md
└── README.txt

Execution Instructions:
-----------------------
1. Environment Setup:
   Ensure Python 3.10+ is installed:
   pip install -r 04_Requirements/requirements.txt

2. Run the Complete Phase II Pipeline:
   python 03_Source_Code/run_phase2_complete_pipeline.py

3. Output Generation:
   All metrics, comparison tables (Result_Comparison.xlsx), confusion matrices, and ROC curves will be generated and saved in their respective directories.
