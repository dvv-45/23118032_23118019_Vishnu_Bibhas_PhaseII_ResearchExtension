# Phase II Assignment: Research Extension & Work Division Plan
**Group 6: IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB**  
**Assigned Paper:** *Efficient and Interpretable IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB* (Computer Networks, Elsevier, 2026)  
**Authors:** Elizabeth Antony & Amit Kumar Roy (IIIT Kottayam)  
**Team Members:**  
- **Dhanishetti Vishnu (Roll No. 23118032)** — Upstream Lead (Feature Selection, Architecture & System Design)  
- **Bibhas Nath (Roll No. 23118019)** — Downstream Lead (Hyperparameter Optimization, Modeling & Comparative Evaluation)

---

## 1. Context & Instructor's Phase II Guidelines

The email notice sent by **Upendra Chaurasiya** included reference attachments (*Metaheuristic-driven two-stage hybrid...*):
> *"These attachments provide an illustrative example. Each individual/pair/group must develop an appropriate extension of its own assigned manuscript."*

This means:
1. The attached paper (Gül & Bakır, KBS 2026) is an **example** illustrating how an original paper was extended.
2. Group 6 extends **our own assigned paper** (*IoT Botnet Detection via XGB*).

---

## 2. Summary of Phase II Required Deliverables

Every group must submit a single ZIP file containing:
1. **`Proposed_Methodology.png`**: High-resolution workflow flowchart highlighting newly modified components.
2. **`Proposed_Methodology.drawio`**: Fully editable source file created via [draw.io](https://app.diagrams.net/).
3. **At Least 3 Published SOTA Comparison Papers (PDFs)**: Full PDFs + Title, Authors, Venue, Year, DOI, and experimental comparison metrics.
4. **`Result_Comparison.xlsx`**: Tabular comparison comparing:
   - Original published paper results
   - Phase I implemented results
   - Phase II proposed extension results
   - At least 3 external published SOTA comparison papers
5. **Complete Executable Implementation**: Source code, dataset instructions, requirements, trained models, and reproduction README.
6. **Updated Research Report (PDF)**: Documenting motivation, mathematical formulation of proposed modifications, comparative evaluation, and limitations.
7. **Submission ZIP**: `23118032_23118019_Vishnu_Bibhas_PhaseII_ResearchExtension.zip`.

---

## 3. 50–50 Independent Work Division

```
       VISHNU (Upstream Focus)                       BIBHAS (Downstream Focus)
 ┌─────────────────────────────────┐           ┌─────────────────────────────────┐
 │ 1. Advanced Feature Selection   │           │ 1. SOTA Comparison Papers Search│
 │    (Hybrid Filter + B-PSO)      │           │    (3 Peer-Reviewed PDFs)       │
 │                                 │           │                                 │
 │ 2. Export Phase 2 Dataset:      │ Handover  │ 2. Advanced Tuning & Modeling   │
 │    `Phase2_Selected_Features.csv│ ────────> │    (5-Fold CV Bayesian XGBoost) │
 │                                 │           │                                 │
 │ 3. Draw.io & PNG Flowchart      │           │ 3. Comprehensive Excel Sheet    │
 │    (`Proposed_Methodology.drawio│           │    (`Result_Comparison.xlsx`)   │
 │                                 │           │                                 │
 │ 4. Report: Sections 1 to 3      │           │ 4. Report: Sections 4 to 6      │
 └─────────────────────────────────┘           └─────────────────────────────────┘
```

### Module Responsibilities:

| Component | Vishnu (23118032) | Bibhas (23118019) | Deliverable / Output File |
| :--- | :--- | :--- | :--- |
| **Feature Selection Extension** | **Lead**: Propose & implement Two-Stage Hybrid (ExtraTrees Filter + Binary PSO) to improve upon greedy RFE. | Reviews code & tests on training split. | `run_hybrid_fs.py`<br>`Phase2_Selected_Features.csv` |
| **Pipeline Diagrams** | **Lead**: Design complete diagram in [draw.io](https://app.diagrams.net/) highlighting extended blocks. Export PNG. | Verifies parameters & layer labels. | `Proposed_Methodology.drawio`<br>`Proposed_Methodology.png` |
| **3 Comparison Research Papers** | Reviews relevance to IoT Botnet detection. | **Lead**: Identify, download 3 peer-reviewed SOTA papers (2023–2026), extract accuracy/F1/features. | 3 Paper PDFs + `Comparison_Papers_Metadata.txt` |
| **Model & Hyperparameter Tuning** | Provides clean handover features. | **Lead**: Implement automated 5-Fold Stratified CV hyperparameter tuner for XGBoost. | `train_optimized_xgb.py`<br>`phase2_optimized_model.joblib` |
| **Results Comparison Sheet** | Reviews metrics & formatting. | **Lead**: Create and populate `Result_Comparison.xlsx` with all 4 result categories. | `Result_Comparison.xlsx` |
| **Report Writing** | **Writes Sections 1–3**: Introduction, Motivation for Extension, Detailed Proposed FS Architecture & Flowchart. | **Writes Sections 4–6**: Model Optimization, Experimental Results Table, Discussion, Ablation & Limitations. | `PhaseII_ResearchExtension_Report.pdf` |
| **Integration & Submission** | **Lead**: Test end-to-end reproducibility, update README, build final submission ZIP. | Reviews final ZIP contents and verifies on his machine. | `RollNo_Name_PhaseII_ResearchExtension.zip` |

---

## 4. Handover Interface (Zero Blocking)

1. **Vishnu's Handover to Bibhas:**
   - Preprocessed dataset: Ready in `02_Dataset/Processed_Dataset/train.csv` and `test.csv`.
   - Feature Selection Output: Generated in `Vishnu_To_Bibhas/Phase2_Selected_Features.csv` (Features: `udp.srcport`, `ip.flags`, `tcp.dstport`, `ip.ttl`, `ip.proto`, `tcp`).

2. **Bibhas's Handover to Vishnu:**
   - 3 SOTA Comparison Paper citations and metrics table.
   - Final experimental metrics (Accuracy: 99.897%, Recall: 99.954%, F1: 99.897%, ROC-AUC: 0.999967, Latency: 0.0011 ms) embedded in `Result_Comparison.xlsx`.
