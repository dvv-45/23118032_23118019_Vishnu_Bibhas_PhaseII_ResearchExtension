import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Group 6 — Phase II: Proposed Two-Stage Hybrid Feature Selection\n",
                "### **Assigned Paper:** Efficient and Interpretable IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB (Elsevier Computer Networks, 2026)\n",
                "**Student Name:** Dhanishetti Vishnu  \n",
                "**Roll Number:** 23118032  \n",
                "**Team Partner:** Bibhas Nath (Roll No. 23118019)  \n",
                "**Assignment:** Phase II Research Work Extension & Comparative Evaluation  \n",
                "---\n",
                "## **Overview of Vishnu's Work & Proposed Methodology**\n",
                "This notebook implements Vishnu's Stage 1 responsibilities for the Phase II extension:\n",
                "1. **Motivation & Limitations of Baseline Stepwise RFE:** Standard RFE is greedy, computationally expensive, and prone to local optima.\n",
                "2. **Stage 1 (Filter):** Extra-Trees & Mutual Information Feature Importance ranking to prune bottom 50% noisy features.\n",
                "3. **Stage 2 (Swarm/Wrapper):** Binary Particle Swarm Optimization (B-PSO) with multi-objective fitness balancing classification accuracy and feature subset compactness.\n",
                "4. **Handover Package Assembly:** Slices `train_selected_phase2.csv` and `test_selected_phase2.csv` for Bibhas to directly tune downstream models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import time\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "print('Python environment and libraries initialized!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## **Step 1: Load Preprocessed IoT Traffic Dataset**\n",
                "Loading 87,422 balanced flows (80% train, 20% test) with 23 candidate traffic features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_path = r'02_Dataset/Processed_Dataset/train.csv'\n",
                "test_path = r'02_Dataset/Processed_Dataset/test.csv'\n",
                "\n",
                "train_df = pd.read_csv(train_path)\n",
                "test_df = pd.read_csv(test_path)\n",
                "\n",
                "X_train = train_df.drop(columns=['label'])\n",
                "y_train = train_df['label']\n",
                "print(f'Training shape: {X_train.shape}, Test shape: {test_df.shape}')\n",
                "print('Initial 23 features:', list(X_train.columns))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## **Step 2: Stage 1 Filter Pre-selection (Extra-Trees Importance)**\n",
                "Pruning approximately 50% of least informative attributes to reduce search space complexity."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.ensemble import ExtraTreesClassifier\n",
                "\n",
                "filter_model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)\n",
                "filter_model.fit(X_train, y_train)\n",
                "\n",
                "importances = pd.Series(filter_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)\n",
                "\n",
                "plt.figure(figsize=(10, 6))\n",
                "importances.plot(kind='bar', color='#3182CE')\n",
                "plt.title('Stage 1: Extra-Trees Feature Importance Scores', fontsize=12, fontweight='bold')\n",
                "plt.ylabel('Importance Score')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "top_12_features = importances.head(12).index.tolist()\n",
                "print('Top 50% Candidate Features for Metaheuristic Optimization:')\n",
                "print(top_12_features)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## **Step 3: Stage 2 Binary-adapted Particle Swarm Optimization (B-PSO)**\n",
                "Swarm intelligence algorithm optimizing multi-objective fitness:\n",
                "$$\\text{Fitness} = 0.99 \\times (1 - \\text{F1}) + 0.01 \\times \\left(\\frac{F_s}{F_a}\\right) + \\text{Penalty}$$\n",
                "where $F_s$ is selected features and $F_a$ is total candidate features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from 03_Source_Code.Phase2_Feature_Selection.run_hybrid_fs import run_two_stage_hybrid_fs\n",
                "\n",
                "selected_features = run_two_stage_hybrid_fs()\n",
                "print('Optimal Selected Features by B-PSO:', selected_features)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## **Step 4: Generate Handover Package for Bibhas**\n",
                "Exporting `train_selected_phase2.csv`, `test_selected_phase2.csv`, and JSON schemas for downstream model optimization."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from 03_Source_Code.Phase2_Feature_Selection.prepare_phase2_handover import prepare_handover\n",
                "\n",
                "prepare_handover()\n",
                "print('Handover completed! Bibhas can now run train_optimized_xgb.py.')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = r"c:\PAPER_IMPLENTATION\23118032_23118019_Vishnu_Bibhas_PhaseII_ResearchExtension\Vishnu_Phase2_Pipeline.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=2)
print("Notebook created successfully at:", out_path)
