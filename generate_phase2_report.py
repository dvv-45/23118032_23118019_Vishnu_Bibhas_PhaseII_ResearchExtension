"""
Phase II Research Extension Report Generator (PDF)
Group 6: Dhanishetti Vishnu (23118032) & Bibhas Nath (23118019)
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

BASE_DIR = r"c:\PAPER_IMPLENTATION\23118032_23118019_Vishnu_Bibhas_PhaseII_ResearchExtension"
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
        fontSize=18,
        leading=22,
        textColor=c_primary,
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        alignment=1
    )
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_text,
        alignment=1
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_text,
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=3
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
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
    story.append(Spacer(1, 3))
    story.append(Paragraph("Two-Stage Hybrid Feature Selection and Metaheuristic Hyperparameter Optimization for IoT Botnet Detection", subtitle_style))
    story.append(Spacer(1, 6))

    meta_text = (
        "<b>Group 6:</b> Dhanishetti Vishnu (Roll No. 23118032) & Bibhas Nath (Roll No. 23118019)<br/>"
        "<b>Assigned Base Paper:</b> <i>Efficient and Interpretable IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB</i> (Computer Networks, Elsevier, 2026)<br/>"
        "<b>Department:</b> Information Technology, National Institute of Technology Raipur (NITRR)"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceBefore=4, spaceAfter=10))

    # SECTION 1: MOTIVATION & RESEARCH OBJECTIVES
    story.append(Paragraph("1. Motivation for Extending the Original Methodology", h1_style))
    p1 = (
        "The exponential rise of interconnected Internet of Things (IoT) devices has created an expansive attack surface "
        "susceptible to distributed botnet intrusions (e.g., Mirai and BASHLITE variants). While the baseline paper by "
        "Antony and Roy (2026) introduced an effective flow-based pipeline using Recursive Feature Elimination (RFE) and XGBoost, "
        "rigorous technical analysis reveals critical opportunities for algorithmic extension: "
    )
    story.append(Paragraph(p1, body_style))

    b1 = (
        "<b>(a) Limitations of Greedy Stepwise RFE:</b> Standard RFE eliminates features sequentially based on an estimator's "
        "local weights. This greedy elimination process fails to model multi-feature non-linear interactions, is susceptible to "
        "getting trapped in local feature-space optima, and imposes substantial computational training overhead across each elimination step."
    )
    b2 = (
        "<b>(b) Sub-optimal Hyperparameter Exploration:</b> The baseline paper relies on manual heuristic candidate sets (SmartSearch) "
        "or localized grid search, which lack global search capability, adaptive convergence, and multi-parameter co-dependence tuning."
    )
    b3 = (
        "<b>(c) Edge IoT Deployment Overhead:</b> While the baseline reduced features from 23 to 8, high-speed line-rate IoT edge "
        "devices require minimal packet header inspection and sub-millisecond inference to avoid queuing delays."
    )
    story.append(Paragraph(b1, bullet_style))
    story.append(Paragraph(b2, bullet_style))
    story.append(Paragraph(b3, bullet_style))

    # SECTION 2: PROPOSED EXTENDED METHODOLOGY
    story.append(Paragraph("2. Proposed Extended Methodology & Architecture", h1_style))
    p2 = (
        "To overcome these limitations, we propose an extended, end-to-end framework featuring two synergistic innovations: "
        "<b>(i) Two-Stage Hybrid Feature Selection (Filter + Binary Swarm Optimizer)</b>, and "
        "<b>(ii) Automated Bayesian-inspired Hyperparameter Optimization with Stratified 5-Fold Cross-Validation</b>."
    )
    story.append(Paragraph(p2, body_style))

    story.append(Paragraph("A. Component-Level Comparison: Original vs. Proposed", h2_style))
    comp_table_data = [
        [
            Paragraph("Component / Pipeline Stage", table_header),
            Paragraph("Original Baseline Methodology (Antony & Roy, 2026)", table_header),
            Paragraph("Proposed Extended Methodology (Group 6)", table_header)
        ],
        [
            Paragraph("<b>Feature-Importance Pre-filtering</b>", table_cell),
            Paragraph("None (RFE directly applied to all 23 candidate features)", table_cell),
            Paragraph("<b>Stage 1: Extra-Trees & Mutual Information</b> pre-ranking; eliminates bottom 50% noisy/static attributes.", table_cell_bold)
        ],
        [
            Paragraph("<b>Feature Space Refinement</b>", table_cell),
            Paragraph("Greedy Stepwise Recursive Feature Elimination (RFE) down to 8 features", table_cell),
            Paragraph("<b>Stage 2: Binary Particle Swarm Optimization (B-PSO)</b> multi-objective search; minimizes error while penalizing feature count.", table_cell_bold)
        ],
        [
            Paragraph("<b>Selected Features</b>", table_cell),
            Paragraph("8 features: frame.len, ip.ttl, ip.flags, tcp.dstport, tcp.window_size, udp.srcport, tcp.len, ip.len", table_cell),
            Paragraph("<b>6 features:</b> udp.srcport, ip.flags, tcp.dstport, ip.ttl, ip.proto, tcp (<b>73.9% dimensionality reduction</b>)", table_cell_bold)
        ],
        [
            Paragraph("<b>Hyperparameter Tuning</b>", table_cell),
            Paragraph("Manual candidate list (Quick SmartSearch S1–S8) / Static Grid Search", table_cell),
            Paragraph("<b>Automated Bayesian / Randomized Search</b> over continuous/discrete spaces with <b>Stratified 5-Fold Cross-Validation</b>", table_cell_bold)
        ],
        [
            Paragraph("<b>Validation Protocol</b>", table_cell),
            Paragraph("Single 80/20 train/test split", table_cell),
            Paragraph("<b>Stratified 5-Fold CV</b> on training space + held-out 17,485 test evaluation with zero data leakage", table_cell_bold)
        ]
    ]

    t_comp = Table(comp_table_data, colWidths=[1.8 * inch, 2.7 * inch, 2.9 * inch])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 8))

    # SECTION 3: METHODOLOGY FLOWCHART EMBEDDING
    story.append(Paragraph("3. End-to-End Methodology Flowchart", h1_style))
    p3 = (
        "The complete proposed pipeline, highlighting the newly modified Two-Stage Hybrid Feature Selection "
        "and Automated Hyperparameter Optimization modules, is depicted below (editable source file: <code>Proposed_Methodology.drawio</code>)."
    )
    story.append(Paragraph(p3, body_style))

    png_path = os.path.join(BASE_DIR, "Proposed_Methodology.png")
    if os.path.exists(png_path):
        img_w = 7.4 * inch
        img_h = 3.6 * inch
        story.append(Image(png_path, width=img_w, height=img_h))
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Figure 1:</b> Flow diagram of the proposed Phase II extended methodology highlighting the new components.", meta_style))

    story.append(PageBreak())

    # SECTION 4: COMPARATIVE EXPERIMENTAL EVALUATION
    story.append(Paragraph("4. Experimental Results and Comparative Benchmarking", h1_style))
    p4 = (
        "The proposed methodology was implemented and evaluated on the balanced Mirai IoT dataset (87,422 flows; 69,937 train, 17,485 test). "
        "To satisfy the Phase II submission guidelines, the proposed extension was evaluated against: "
        "(1) Reported results in the original paper, (2) Phase I implemented baseline results, and (3) Three published state-of-the-art studies."
    )
    story.append(Paragraph(p4, body_style))

    story.append(Paragraph("Table 1: Comprehensive Comparative Performance Evaluation Table (Result_Comparison.xlsx)", h2_style))

    # Read Result_Comparison.xlsx
    excel_path = os.path.join(BASE_DIR, "08_Comparative_Results", "Result_Comparison.xlsx")
    df_excel = pd.read_excel(excel_path)

    results_table_data = [
        [
            Paragraph("Paper / Study Name", table_header),
            Paragraph("Year & Venue", table_header),
            Paragraph("Dataset", table_header),
            Paragraph("No. Feat.", table_header),
            Paragraph("Acc (%)", table_header),
            Paragraph("Prec (%)", table_header),
            Paragraph("Rec (%)", table_header),
            Paragraph("F1 (%)", table_header),
            Paragraph("ROC-AUC", table_header),
            Paragraph("Latency (ms)", table_header)
        ]
    ]

    for _, row in df_excel.iterrows():
        name_p = Paragraph(f"<b>{row['Paper / Study']}</b>", table_cell)
        venue_p = Paragraph(f"{row['Publication Venue & Year']}", table_cell)
        ds_p = Paragraph(f"{row['Dataset']}", table_cell)
        nf_p = Paragraph(f"{row['No. of Features']}", table_cell)
        acc_p = Paragraph(f"{row['Accuracy (%)']}", table_cell)
        pr_p = Paragraph(f"{row['Precision (%)']}", table_cell)
        rec_p = Paragraph(f"{row['Recall (%)']}", table_cell)
        f1_p = Paragraph(f"{row['F1-Score (%)']}", table_cell)
        auc_p = Paragraph(f"{row['ROC-AUC']}", table_cell)
        lat_p = Paragraph(f"{row['Inference Latency (ms)']}", table_cell)

        results_table_data.append([name_p, venue_p, ds_p, nf_p, acc_p, pr_p, rec_p, f1_p, auc_p, lat_p])

    t_res = Table(results_table_data, colWidths=[1.5*inch, 1.2*inch, 1.0*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.5*inch])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg]),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#EBF8FF")), # Highlight Ours
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 8))

    # SECTION 5: RESULT IMAGES & DISCUSSION
    story.append(Paragraph("5. Performance Plots and Findings", h1_style))
    p5 = (
        "The experimental evaluation highlights key technical improvements achieved by the Phase II extension: "
        "<br/>• <b>Dimensionality Reduction:</b> The feature space was reduced from 23 features down to 6 features (73.9% reduction), "
        "surpassing the 8 features of the baseline paper and 14 features of Turukmane et al. (2024)."
        "<br/>• <b>Exceptional Discrimination:</b> The proposed model achieved an accuracy of <b>99.897%</b>, recall of <b>99.954%</b>, "
        "and ROC-AUC of <b>0.999967</b>, confirming negligible false alarms."
        "<br/>• <b>Ultra-Low Inference Latency:</b> Per-flow inference latency reached <b>0.0008 ms</b> (13.15 ms for 17,485 flows), "
        "making it extraordinarily suited for line-rate edge deployment on resource-constrained micro-controllers."
    )
    story.append(Paragraph(p5, body_style))
    story.append(Spacer(1, 6))

    cm_p = os.path.join(BASE_DIR, "07_Result_Images", "Phase2_Confusion_Matrix.png")
    roc_p = os.path.join(BASE_DIR, "07_Result_Images", "Phase2_ROC_Curves.png")

    if os.path.exists(cm_p) and os.path.exists(roc_p):
        t_imgs = Table([
            [
                Image(cm_p, width=3.4 * inch, height=2.6 * inch),
                Image(roc_p, width=3.4 * inch, height=2.6 * inch)
            ],
            [
                Paragraph("<b>(a) Test Confusion Matrix (17,485 Flows)</b>", meta_style),
                Paragraph("<b>(b) Receiver Operating Characteristic (ROC) Curve</b>", meta_style)
            ]
        ], colWidths=[3.7 * inch, 3.7 * inch])
        t_imgs.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_imgs)

    story.append(Spacer(1, 8))

    # SECTION 6: LIMITATIONS & FUTURE SCOPE
    story.append(Paragraph("6. Remaining Limitations & Future Scope", h1_style))
    p6 = (
        "While the proposed two-stage hybrid methodology demonstrates superior accuracy and inference speed, certain limitations remain: "
        "<br/>1. <b>Multi-Class Generalization:</b> The current evaluation focuses on binary botnet discrimination (Mirai vs. Benign). "
        "Future research will evaluate multi-class attack classification across complex multi-vector IoT datasets (such as RT-IoT 2022 and CIC-IoT 2023)."
        "<br/>2. <b>Dynamic Concept Drift:</b> Feature distributions in IoT networks evolve as adversaries modify payload packet timings. "
        "Integrating incremental online learning and adaptive swarm updating will be explored to defend against zero-day mutations."
    )
    story.append(Paragraph(p6, body_style))

    # Build PDF
    doc.build(story)
    print(f"[SUCCESS] PDF Report generated at: {REPORT_PATH} ({os.path.getsize(REPORT_PATH) / 1024:.1f} KB)")

if __name__ == "__main__":
    generate_report()
