"""
Phase II Research Extension Report Generator (PDF)
Group 6: Dhanishetti Vishnu (23118032) & Bibhas Nath (23118019)
Department of Information Technology, National Institute of Technology Raipur

Covers:
  - Section 1: Introduction & Research Motivation (Vishnu)
  - Section 2: Detailed Architectural Differences: Baseline vs. Proposed Extension (Vishnu)
  - Section 3: Mathematical Formulation of Two-Stage Hybrid Feature Selection (Vishnu)
  - Section 4: Proposed Methodology Flowcharts (Paper-Style & Detailed Views) (Vishnu)
  - Section 5: Automated Hyperparameter Optimization & Training Setup (Bibhas)
  - Section 6: Comprehensive Comparative Experimental Evaluation (Bibhas & Vishnu)
  - Section 7: Practical Implications, Computational Complexity & Limitations (Both)
"""

import os
import sys
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, "10_Research_Extension_Report", "PhaseII_ResearchExtension_Report.pdf")

def generate_report():
    print(f"[+] Generating Phase II Extension Report: {REPORT_PATH}...")
    doc = SimpleDocTemplate(
        REPORT_PATH,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch
    )

    styles = getSampleStyleSheet()

    # Color Palette
    c_primary = colors.HexColor("#1A365D")   # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0") # Slate Blue
    c_accent = colors.HexColor("#C53030")    # Crimson Accent
    c_bg = colors.HexColor("#F7FAFC")        # Warm White / Light Grey
    c_text = colors.HexColor("#2D3748")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=21,
        textColor=c_primary,
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        alignment=1
    )
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_text,
        alignment=1
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=5
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=c_secondary,
        spaceBefore=7,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=c_text,
        spaceAfter=4
    )
    math_box = ParagraphStyle(
        'MathBox',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=3,
        spaceAfter=4,
        leftIndent=15
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=14,
        bulletIndent=5,
        spaceAfter=3
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.5,
        textColor=c_text
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=c_primary
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    story = []

    # Title & Header
    story.append(Paragraph("Phase II Research Extension Report", title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Two-Stage Hybrid Feature Selection and Metaheuristic Hyperparameter Optimization for IoT Botnet Detection", subtitle_style))
    story.append(Spacer(1, 4))

    meta_text = (
        "<b>Group 6:</b> Dhanishetti Vishnu (Roll No. 23118032) & Bibhas Nath (Roll No. 23118019)<br/>"
        "<b>Assigned Base Paper:</b> <i>Efficient and Interpretable IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB</i> (Computer Networks, Elsevier, 2026)<br/>"
        "<b>Department:</b> Department of Information Technology, National Institute of Technology Raipur (NITRR)"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceBefore=2, spaceAfter=8))

    # =========================================================================
    # SECTION 1: MOTIVATION & RESEARCH OBJECTIVES (Vishnu)
    # =========================================================================
    story.append(Paragraph("1. Research Motivation for Methodology Extension", h1_style))
    p1 = (
        "The exponential expansion of Internet of Things (IoT) deployments in mission-critical domains (smart power grids, "
        "healthcare systems, industrial automation) has dramatically enlarged the cyber-attack surface. Due to constrained computational "
        "and memory budgets, IoT devices frequently fall prey to high-velocity botnet attacks such as Mirai and BASHLITE variants. "
        "In the baseline paper, Antony & Roy (2026) proposed a flow-based intrusion detection framework utilizing Stepwise Recursive "
        "Feature Elimination (RFE) and eXtreme Gradient Boosting (XGBoost). However, an in-depth algorithmic critique reveals "
        "three fundamental architectural limitations that severely restrict its real-world scalability and edge deployment efficacy: "
    )
    story.append(Paragraph(p1, body_style))

    b1 = (
        "<b>(a) Greedy Local Trapping in Stepwise RFE:</b> Deterministic RFE discards features monotonically according to localized "
        "estimator weight gradients. This greedy backward elimination inherently overlooks non-linear inter-feature combinatorial synergies, "
        "easily getting trapped in sub-optimal local basins while requiring costly iterative model re-training across all 23 candidate features."
    )
    b2 = (
        "<b>(b) Sub-optimal Hyperparameter Tuning:</b> The baseline paper employed manual heuristic candidate sets (SmartSearch S1–S8) "
        "or static grid exploration. Such approaches lack stochastic global exploration mechanisms and fail to model complex parameter "
        "co-dependencies (e.g., interaction between tree depth, learning rate, and subsample ratios)."
    )
    b3 = (
        "<b>(c) Edge IoT Latency & Inspection Overhead:</b> While the baseline reduced features from 23 to 8, high-speed line-rate "
        "IoT routers require ultra-compact packet header inspection to achieve sub-millisecond per-packet inference latency and zero queue congestion."
    )
    story.append(Paragraph(b1, bullet_style))
    story.append(Paragraph(b2, bullet_style))
    story.append(Paragraph(b3, bullet_style))

    # =========================================================================
    # SECTION 2: ARCHITECTURAL DIFFERENCES (Vishnu)
    # =========================================================================
    story.append(Paragraph("2. Architectural Differences: Baseline vs. Proposed Extension", h1_style))
    p2 = (
        "To systematically address these bottlenecks, we propose an extended, end-to-end framework featuring two synergistic innovations: "
        "<b>(i) Two-Stage Hybrid Feature Selection (Filter + Binary Swarm Optimizer)</b> led by Vishnu, and "
        "<b>(ii) Automated Bayesian-inspired Hyperparameter Optimization with Stratified 5-Fold Cross-Validation</b> led by Bibhas."
    )
    story.append(Paragraph(p2, body_style))

    comp_table_data = [
        [
            Paragraph("Architectural Dimension", table_header),
            Paragraph("Baseline Methodology (Antony & Roy, 2026)", table_header),
            Paragraph("Proposed Phase II Extension (Group 6)", table_header),
            Paragraph("Technical Justification", table_header)
        ],
        [
            Paragraph("<b>Stage 1: Pre-Filtering</b>", table_cell),
            Paragraph("None (All 23 features fed directly into RFE)", table_cell),
            Paragraph("<b>Extra-Trees &amp; Mutual Information Filter</b>", table_cell_bold),
            Paragraph("Prunes bottom 50% noisy/static attributes in linear time, dramatically shrinking wrapper search complexity.", table_cell)
        ],
        [
            Paragraph("<b>Stage 2: Wrapper Search</b>", table_cell),
            Paragraph("Deterministic Stepwise RFE (Greedy elimination)", table_cell),
            Paragraph("<b>Binary Particle Swarm Optimization (B-PSO)</b>", table_cell_bold),
            Paragraph("Global bio-inspired stochastic exploration with personal-best and swarm-best convergence mechanics.", table_cell)
        ],
        [
            Paragraph("<b>Objective Formulation</b>", table_cell),
            Paragraph("Single-metric F1-score maximization", table_cell),
            Paragraph("<b>Multi-Objective Cost Fitness:</b><br/>0.99·(1 - F1) + 0.01·(F<sub>s</sub> / F<sub>a</sub>)", table_cell_bold),
            Paragraph("Simultaneously penalizes misclassification error and excessive feature counts to produce compact subsets.", table_cell)
        ],
        [
            Paragraph("<b>Selected Feature Count</b>", table_cell),
            Paragraph("8 Features (65.2% reduction)", table_cell),
            Paragraph("<b>6 Features (73.9% dimensionality cut)</b>", table_cell_bold),
            Paragraph("Retains complete protocol attack defense coverage with 25% fewer header attributes to inspect.", table_cell)
        ],
        [
            Paragraph("<b>Hyperparameter Tuning</b>", table_cell),
            Paragraph("Manual candidate list (SmartSearch S1–S8) / Grid", table_cell),
            Paragraph("<b>Automated Optuna (TPE) Bayesian Search</b>", table_cell_bold),
            Paragraph("Global exploration across 6 continuous/discrete parameters guided by historical trial probability densities.", table_cell)
        ],
        [
            Paragraph("<b>Validation Protocol</b>", table_cell),
            Paragraph("Single 80/20 train/test split", table_cell),
            Paragraph("<b>Stratified 5-Fold CV + Held-out Test Set</b>", table_cell_bold),
            Paragraph("Guarantees zero data leakage and unbiased generalization across imbalanced IoT traffic distributions.", table_cell)
        ]
    ]

    t_comp = Table(comp_table_data, colWidths=[1.4 * inch, 1.9 * inch, 2.1 * inch, 2.0 * inch])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 3: MATHEMATICAL FORMULATION (Vishnu)
    # =========================================================================
    story.append(Paragraph("3. Mathematical Formulation of Two-Stage Hybrid Feature Selection", h1_style))
    p3 = (
        "<b>3.1 Stage 1: Filter-Based Importance Pre-filtering</b><br/>"
        "To eliminate noisy, invariant, and non-informative features prior to computationally intensive swarm optimization, an "
        "Extra-Trees ensemble computes Gini impurity reduction across all M decision trees:"
    )
    story.append(Paragraph(p3, body_style))
    story.append(Paragraph("I(f<sub>j</sub>) = (1 / M) &Sigma;<sub>m=1..M</sub> &Delta;i(s<sub>m, j</sub>)", math_box))
    p3_sub = (
        "Features are ranked in descending order, and the candidate set is pruned by 50% ($k = &lceil; 0.50 &times; D &rceil; = 12$ features), "
        "effectively eliminating non-discriminative traffic attributes.<br/><br/>"
        "<b>3.2 Stage 2: Binary-Adapted Particle Swarm Optimization (B-PSO) Dynamics</b><br/>"
        "In B-PSO, a swarm of N = 20 particles navigates the 12-dimensional binary space over T = 25 iterations. Each particle i "
        "maintains a continuous velocity vector V<sub>i</sub> &isin; &real;<sup>D</sup> and a binary position vector X<sub>i</sub> &isin; {0, 1}<sup>D</sup>:"
    )
    story.append(Paragraph(p3_sub, body_style))

    eq_vel = (
        "V<sub>i</sub><sup>t+1</sup> = w(t) &middot; V<sub>i</sub><sup>t</sup> + "
        "c<sub>1</sub> r<sub>1</sub> (P<sub>best, i</sub><sup>t</sup> - X<sub>i</sub><sup>t</sup>) + "
        "c<sub>2</sub> r<sub>2</sub> (G<sub>best</sub><sup>t</sup> - X<sub>i</sub><sup>t</sup>)"
    )
    story.append(Paragraph(eq_vel, math_box))
    p3_inertia = (
        "where w(t) is the time-varying inertia weight linearly decaying from w<sub>max</sub> = 0.9 to w<sub>min</sub> = 0.4 to transition "
        "from early global exploration to fine local exploitation: <i>w(t) = w<sub>max</sub> - t &middot; ((w<sub>max</sub> - w<sub>min</sub>) / T<sub>max</sub>)</i>, "
        "and c<sub>1</sub> = c<sub>2</sub> = 1.5 are cognitive and social acceleration coefficients with random vectors r<sub>1</sub>, r<sub>2</sub> &sim; U(0, 1).<br/>"
        "The continuous velocity is mapped to selection probabilities via the Sigmoid transfer function, followed by stochastic binary thresholding:"
    )
    story.append(Paragraph(p3_inertia, body_style))
    story.append(Paragraph("S(V<sub>i,j</sub><sup>t+1</sup>) = 1 / (1 + e<sup>-V<sub>i,j</sub><sup>t+1</sup></sup>)", math_box))
    story.append(Paragraph("X<sub>i,j</sub><sup>t+1</sup> = 1  if  rand() &le; S(V<sub>i,j</sub><sup>t+1</sup>)  else  0", math_box))

    p3_fit = (
        "Candidate feature masks are evaluated using a multi-objective cost fitness function:"
    )
    story.append(Paragraph(p3_fit, body_style))
    story.append(Paragraph("f(X<sub>i</sub>) = &alpha; &middot; (1 - F1<sub>CV</sub>(X<sub>i</sub>)) + &beta; &middot; (F<sub>selected</sub> / F<sub>total</sub>) + Penalty", math_box))
    p3_fit_exp = (
        "where &alpha; = 0.99 enforces strict detection performance, &beta; = 0.01 rewards subset compactness, and a protocol diversity penalty "
        "ensures that at least 6 critical network layer headers are retained for multi-vector threat robustness.<br/><br/>"
        "<b>3.3 Domain Security Justification of Selected Features:</b><br/>"
        "The final 6 optimal features selected by B-PSO provide comprehensive, multi-protocol defense coverage: "
        "(1) <b>udp.srcport:</b> detects UDP reflection floods targeting ephemeral SSDP/DNS ports; "
        "(2) <b>ip.flags:</b> captures fragmentation bitmask tampering during malformed packet bursts; "
        "(3) <b>tcp.dstport:</b> pinpoints Telnet scanning on ports 23/2323 and HTTP exploits; "
        "(4) <b>ip.ttl:</b> catches static hop-count signatures produced by automated bot scripts; "
        "(5) <b>ip.proto:</b> identifies rapid protocol switching across TCP/UDP/ICMP attacks; "
        "(6) <b>tcp:</b> verifies stateful handshake completion vs. incomplete SYN flood sessions."
    )
    story.append(Paragraph(p3_fit_exp, body_style))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 4: FLOWCHARTS & DIAGRAMS (Vishnu)
    # =========================================================================
    story.append(Paragraph("4. End-to-End Methodology Flowchart & Architecture", h1_style))
    p4_flow = (
        "Figure 1 illustrates the proposed two-tier pipeline matching the published paper style (Figure 3 in Gul & Bakir, 2026), "
        "clearly distinguishing the Top-Tier (Feature Engineering: Filter + B-PSO) and Bottom-Tier (Model Optimization: Optuna Tuning + XGBoost). "
        "The full editable source diagram is provided in <code>Proposed_Methodology.drawio</code>."
    )
    story.append(Paragraph(p4_flow, body_style))

    paper_png = os.path.join(BASE_DIR, "Proposed_Methodology_PaperStyle.png")
    if os.path.exists(paper_png):
        img_w = 7.4 * inch
        img_h = 4.2 * inch
        story.append(Image(paper_png, width=img_w, height=img_h))
        story.append(Spacer(1, 3))
        story.append(Paragraph("<b>Figure 1:</b> Proposed Phase II Methodology Flowchart (Published-Paper Style with highlighted extensions).", meta_style))

    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 5: MODEL TUNING & TRAINING (Bibhas)
    # =========================================================================
    story.append(Paragraph("5. Automated Hyperparameter Optimization and Model Training Setup", h1_style))
    p5 = (
        "Following Stage 1 feature selection, we implemented an automated hyperparameter tuning pipeline utilizing the Tree-structured "
        "Parzen Estimator (TPE) algorithm within the Optuna framework. Unlike the heuristic SmartSearch grid applied in Phase I, Optuna "
        "constructs probabilistic surrogate models over parameter co-dependencies, enabling directed convergence toward the global optimum.<br/>"
        "The XGBoost classifier was tuned across 50 optimization trials targeting: n_estimators [50–300], max_depth [3–10], learning_rate [0.01–0.3], "
        "subsample [0.6–1.0], colsample_bytree [0.6–1.0], and gamma [0–5]. The objective was guided by 5-fold Stratified Cross-Validation on the "
        "69,937 training flows to prevent validation leakage. Final training employed the hist tree-method for hardware-accelerated compiling."
    )
    story.append(Paragraph(p5, body_style))

    # =========================================================================
    # SECTION 6: COMPARATIVE EVALUATION (Bibhas & Vishnu)
    # =========================================================================
    story.append(Paragraph("6. Comparative Experimental Evaluation vs. Published SOTA", h1_style))
    story.append(Paragraph("Table 1: Benchmark Comparison Across Published SOTA Papers and Proposed Phase II Extension", h2_style))

    excel_path = os.path.join(BASE_DIR, "08_Comparative_Results", "Result_Comparison.xlsx")
    try:
        df_excel = pd.read_excel(excel_path)
        results_table_data = [
            [
                Paragraph("Paper / Study Name", table_header),
                Paragraph("Year & Venue", table_header),
                Paragraph("Dataset", table_header),
                Paragraph("Feat.", table_header),
                Paragraph("Acc (%)", table_header),
                Paragraph("Prec (%)", table_header),
                Paragraph("Rec (%)", table_header),
                Paragraph("F1 (%)", table_header),
                Paragraph("ROC-AUC", table_header),
                Paragraph("Latency", table_header)
            ]
        ]

        for _, row in df_excel.iterrows():
            col_study = str(row.get('Paper / Study Name', row.get('Paper / Method', '')))
            col_venue = str(row.get('Publication Venue & Year', row.get('Year', '')))
            name_p = Paragraph(f"<b>{col_study}</b>", table_cell)
            venue_p = Paragraph(f"{col_venue}", table_cell)
            ds_p = Paragraph(f"{row.get('Dataset', '')}", table_cell)
            nf_p = Paragraph(f"{row.get('No. of Features', row.get('No. Feat.', ''))}", table_cell)
            acc_p = Paragraph(f"{row.get('Accuracy (%)', '')}", table_cell)
            pr_p = Paragraph(f"{row.get('Precision (%)', '')}", table_cell)
            rec_p = Paragraph(f"{row.get('Recall (%)', '')}", table_cell)
            f1_p = Paragraph(f"{row.get('F1-Score (%)', row.get('F1 (%)', ''))}", table_cell)
            auc_p = Paragraph(f"{row.get('ROC-AUC', '')}", table_cell)
            lat_p = Paragraph(f"{row.get('Inference Latency (ms)', row.get('Latency (ms)', ''))}", table_cell)
            results_table_data.append([name_p, venue_p, ds_p, nf_p, acc_p, pr_p, rec_p, f1_p, auc_p, lat_p])

        t_res = Table(results_table_data, colWidths=[1.5*inch, 1.2*inch, 1.0*inch, 0.45*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.5*inch])
        t_res.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_primary),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg]),
            ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#EBF8FF")), # Highlight Ours
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ]))
        story.append(t_res)
    except Exception as e:
        story.append(Paragraph(f"<i>[Table loaded with warning: {e}]</i>", body_style))

    story.append(Spacer(1, 6))

    cm_p = os.path.join(BASE_DIR, "07_Result_Images", "Confusion_Matrix_Phase2.png")
    roc_p = os.path.join(BASE_DIR, "07_Result_Images", "ROC_Curve_Phase2.png")

    if os.path.exists(cm_p) and os.path.exists(roc_p):
        t_imgs = Table([
            [
                Image(cm_p, width=3.3 * inch, height=2.4 * inch),
                Image(roc_p, width=3.3 * inch, height=2.4 * inch)
            ],
            [
                Paragraph("<b>Figure 2: (a) Test Confusion Matrix (17,485 Flows)</b>", meta_style),
                Paragraph("<b>Figure 2: (b) Receiver Operating Characteristic (ROC)</b>", meta_style)
            ]
        ], colWidths=[3.7 * inch, 3.7 * inch])
        t_imgs.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_imgs)

    p6_disc = (
        "<b>Performance Discussion:</b> The proposed Two-Stage Hybrid pipeline achieves <b>99.91% Accuracy</b>, <b>99.91% F1-score</b>, "
        "and a near-perfect <b>0.99997 ROC-AUC</b> with an ultra-low inference latency of <b>0.0009 ms per packet</b> (over 1.1 million packets/sec). "
        "Compared to deep learning models in the literature (e.g. CNN-BiLSTM pipelines requiring 5–20 ms GPU processing), our optimized XGBoost "
        "achieves competitive or superior discriminative capability while requiring only 6 packet header attributes, eliminating feature extraction "
        "bottlenecks on resource-constrained edge gateways."
    )
    story.append(Paragraph(p6_disc, body_style))
    story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 7: LIMITATIONS & FUTURE WORK (Both)
    # =========================================================================
    story.append(Paragraph("7. Practical Implications, Limitations & Future Work", h1_style))
    p7 = (
        "While the proposed framework achieves state-of-the-art detection precision and ultra-low edge latency, two practical considerations must be noted: "
        "(1) <b>Offline Optimization Overhead:</b> The Two-Stage Hybrid Feature Selection and Bayesian hyperparameter search require substantial "
        "offline training computation (~10–15 seconds on modern multicore CPUs), making on-device edge re-training impractical; "
        "(2) <b>Adversarial Concept Drift:</b> Dynamic botnets that randomize header fields may degrade static feature representations over time.<br/>"
        "Future research should explore online concept-drift detection and incremental model adaptation to update tree structures dynamically without full retraining."
    )
    story.append(Paragraph(p7, body_style))

    # Build PDF
    doc.build(story)
    print(f"[SUCCESS] PDF Report generated at: {REPORT_PATH} ({os.path.getsize(REPORT_PATH) / 1024:.1f} KB)")

if __name__ == "__main__":
    generate_report()