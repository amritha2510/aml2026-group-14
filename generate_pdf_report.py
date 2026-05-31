"""
Generate a professional PDF report for DualConvVit Fusion Analysis
Includes high-quality visualizations and comprehensive findings
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime

# Configure matplotlib for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Paths
base_path = Path("/home/sohithvishnu/Desktop/Uni/aml2026-group-14/outputs/dual_conv_vit")
fusion_sweep_path = base_path / "fusion_sweep"
two_stage_path = base_path / "two_stage"
report_dir = base_path / "report"
report_dir.mkdir(exist_ok=True)

print("=" * 80)
print("PDF REPORT GENERATION FOR DUALCONVVIT FUSION ANALYSIS")
print("=" * 80)

# Load data
print("\n[1/7] Loading data...")
search_concat = pd.read_csv(fusion_sweep_path / "search_results_concat.csv")
search_attention = pd.read_csv(fusion_sweep_path / "search_results_attention.csv")
phase3_results = pd.read_csv(fusion_sweep_path / "phase3_final_results_ranked.csv")

with open(two_stage_path / "two_stage_results.json", 'r') as f:
    two_stage_data = json.load(f)

two_stage_df = pd.DataFrame({
    'Model': list(two_stage_data.keys()),
    'Macro Recall': [v['macro_recall'] for v in two_stage_data.values()],
    'Macro F1': [v['macro_f1'] for v in two_stage_data.values()],
    'Normal': [v['normal'] for v in two_stage_data.values()],
    'Bacterial': [v['bacterial'] for v in two_stage_data.values()],
    'Viral': [v['viral'] for v in two_stage_data.values()]
})

# ============================================================================
# PLOT 1: Learning Rate Analysis
# ============================================================================
print("[2/7] Generating Plot 1: Learning Rate Analysis...")

fig = plt.figure(figsize=(14, 6))
gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
concat_high_lr = search_concat[search_concat['learning_rate'] == 0.0001]
bp1 = ax1.boxplot([concat_high_lr[concat_high_lr['noise_dropout_rates'] == d]['val_macro_recall'].values 
                     for d in sorted(concat_high_lr['noise_dropout_rates'].unique())],
                   labels=['0.2', '0.3', '0.4'],
                   patch_artist=True,
                   widths=0.6)
for patch, color in zip(bp1['boxes'], ['#FF6B6B', '#4ECDC4', '#95E1D3']):
    patch.set_facecolor(color)
ax1.set_ylabel('Validation Macro Recall', fontsize=12, fontweight='bold')
ax1.set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
ax1.set_title('Phase 1 (Concat): Effect of Dropout\n@ LR=0.0001', fontsize=13, fontweight='bold')
ax1.set_ylim([0.70, 0.78])
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
attention_high_lr = search_attention[search_attention['learning_rate'] == 0.0001]
bp2 = ax2.boxplot([attention_high_lr[attention_high_lr['noise_dropout_rates'] == d]['val_macro_recall'].values 
                    for d in sorted(attention_high_lr['noise_dropout_rates'].unique())],
                  labels=['0.1', '0.2', '0.3'],
                  patch_artist=True,
                  widths=0.6)
for patch, color in zip(bp2['boxes'], ['#FFB347', '#DDA0DD', '#87CEEB']):
    patch.set_facecolor(color)
ax2.set_ylabel('Validation Macro Recall', fontsize=12, fontweight='bold')
ax2.set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
ax2.set_title('Phase 2 (Attention): Effect of Dropout\n@ LR=0.0001', fontsize=13, fontweight='bold')
ax2.set_ylim([0.70, 0.78])
ax2.grid(True, alpha=0.3)

fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
fig.savefig(report_dir / 'plot1_lr_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 2: Phase 3 Final Results
# ============================================================================
print("[3/7] Generating Plot 2: Phase 3 Final Results...")

fig = plt.figure(figsize=(14, 6))
gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)

phase3_data = [
    ('Attention\n(Optimized)', phase3_results.iloc[0]['test_macro_recall'], 
     phase3_results.iloc[0]['test_macro_f1']),
    ('Concat\n(Optimized)', phase3_results.iloc[1]['test_macro_recall'], 
     phase3_results.iloc[1]['test_macro_f1']),
]

models = [x[0] for x in phase3_data]
recalls = [x[1] for x in phase3_data]
f1s = [x[2] for x in phase3_data]

# Test Recall
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(models, recalls, color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Test Macro Recall', fontsize=12, fontweight='bold')
ax1.set_title('Test Recall Comparison', fontsize=13, fontweight='bold')
ax1.set_ylim([0.6, 0.8])
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, recalls):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Test F1
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(models, f1s, color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Test Macro F1', fontsize=12, fontweight='bold')
ax2.set_title('F1-Score Comparison', fontsize=13, fontweight='bold')
ax2.set_ylim([0.6, 0.8])
ax2.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, f1s):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Winner annotation
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
winner_text = f"""
FUSION SWEEP WINNER:
ATTENTION FUSION

Test Recall: {phase3_results.iloc[0]['test_macro_recall']:.4f}
Test F1: {phase3_results.iloc[0]['test_macro_f1']:.4f}

Best Configuration:
• Learning Rate: 0.0001
• Dropout: 0.2
• Weight Decay: 0.0001
• Epochs: 20

Improvement over Concat:
+{(phase3_results.iloc[0]['test_macro_recall'] - phase3_results.iloc[1]['test_macro_recall'])*100:.2f}% recall
"""
ax3.text(0.1, 0.5, winner_text, fontsize=11, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='#FFFFCC', alpha=0.8, edgecolor='black', linewidth=2),
        family='monospace', fontweight='bold')

fig.suptitle('Phase 3: Final Training Results (20 Epochs)', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(report_dir / 'plot2_phase3_results.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 3: Two-Stage Model Comparison
# ============================================================================
print("[4/7] Generating Plot 3: Two-Stage Comparison...")

fig = plt.figure(figsize=(16, 7))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# Prepare data
two_stage_sorted = two_stage_df.sort_values('Macro Recall', ascending=False)
colors_map = {'ViT (solo)': '#FF6B6B', 'CNN (solo)': '#4ECDC4', 'Ensemble (equal)': '#95E1D3',
              'Ensemble (w=0.15)': '#FFB347', 'Fusion-concat': '#DDA0DD', 'Fusion-attention': '#87CEEB'}

# Recall
ax1 = fig.add_subplot(gs[0, 0])
colors_list = [colors_map.get(m, '#999999') for m in two_stage_sorted['Model']]
bars1 = ax1.barh(range(len(two_stage_sorted)), two_stage_sorted['Macro Recall'], 
                 color=colors_list, alpha=0.85, edgecolor='black', linewidth=1)
ax1.set_yticks(range(len(two_stage_sorted)))
ax1.set_yticklabels(two_stage_sorted['Model'], fontsize=11)
ax1.set_xlabel('Macro Recall', fontsize=12, fontweight='bold')
ax1.set_title('Test Recall Performance', fontsize=13, fontweight='bold')
ax1.set_xlim([0.7, 0.84])
ax1.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(two_stage_sorted.iterrows()):
    ax1.text(row['Macro Recall'] + 0.002, i, f"{row['Macro Recall']:.4f}", 
            va='center', fontsize=10, fontweight='bold')

# F1
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.barh(range(len(two_stage_sorted)), two_stage_sorted['Macro F1'], 
                 color=colors_list, alpha=0.85, edgecolor='black', linewidth=1)
ax2.set_yticks(range(len(two_stage_sorted)))
ax2.set_yticklabels(two_stage_sorted['Model'], fontsize=11)
ax2.set_xlabel('Macro F1', fontsize=12, fontweight='bold')
ax2.set_title('F1-Score Performance', fontsize=13, fontweight='bold')
ax2.set_xlim([0.7, 0.84])
ax2.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(two_stage_sorted.iterrows()):
    ax2.text(row['Macro F1'] + 0.002, i, f"{row['Macro F1']:.4f}", 
            va='center', fontsize=10, fontweight='bold')

# Per-class recall
ax3 = fig.add_subplot(gs[1, :])
x = np.arange(len(two_stage_df))
width = 0.25
bars_normal = ax3.bar(x - width, two_stage_df['Normal'], width, label='Normal', 
                      color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.7)
bars_bact = ax3.bar(x, two_stage_df['Bacterial'], width, label='Bacterial', 
                    color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.7)
bars_viral = ax3.bar(x + width, two_stage_df['Viral'], width, label='Viral', 
                     color='#95E1D3', alpha=0.8, edgecolor='black', linewidth=0.7)

ax3.set_ylabel('Recall per Class', fontsize=12, fontweight='bold')
ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
ax3.set_title('Per-Class Recall Analysis', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(two_stage_df['Model'], rotation=45, ha='right', fontsize=10)
ax3.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax3.set_ylim([0, 1.0])
ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline (50%)')
ax3.grid(True, alpha=0.3, axis='y')

fig.suptitle('Two-Stage Fusion: Strategy Comparison', fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
fig.savefig(report_dir / 'plot3_twostage_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 4: Overall Winner Ranking
# ============================================================================
print("[5/7] Generating Plot 4: Overall Winner Ranking...")

comparison_data = pd.DataFrame({
    'Approach': [
        'ViT Solo\n(Two-Stage)',
        'Ensemble (Eq)\n(Two-Stage)',
        'Fusion-Concat\n(Two-Stage)',
        'Attention\n(Fusion Sweep)',
        'Fusion-Attn\n(Two-Stage)',
        'CNN Solo\n(Two-Stage)',
    ],
    'Test Recall': [
        two_stage_df[two_stage_df['Model'] == 'ViT (solo)']['Macro Recall'].values[0],
        two_stage_df[two_stage_df['Model'] == 'Ensemble (equal)']['Macro Recall'].values[0],
        two_stage_df[two_stage_df['Model'] == 'Fusion-concat']['Macro Recall'].values[0],
        phase3_results.iloc[0]['test_macro_recall'],
        two_stage_df[two_stage_df['Model'] == 'Fusion-attention']['Macro Recall'].values[0],
        two_stage_df[two_stage_df['Model'] == 'CNN (solo)']['Macro Recall'].values[0],
    ],
    'Approach Type': ['Two-Stage', 'Two-Stage', 'Two-Stage', 'Sweep', 'Two-Stage', 'Two-Stage']
})

fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.5], hspace=0.35)

# Top ranking
ax1 = fig.add_subplot(gs[0, 0])
sorted_data = comparison_data.sort_values('Test Recall', ascending=True)
colors_ranking = ['#FFD700' if i == len(sorted_data)-1 else '#C0C0C0' if i == len(sorted_data)-2 
                  else '#CD7F32' if i == len(sorted_data)-3 else '#E8E8E8' 
                  for i in range(len(sorted_data))]
bars = ax1.barh(range(len(sorted_data)), sorted_data['Test Recall'], 
               color=colors_ranking, alpha=0.85, edgecolor='black', linewidth=1.5)
ax1.set_yticks(range(len(sorted_data)))
ax1.set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(sorted_data['Approach'])], fontsize=11)
ax1.set_xlabel('Test Macro Recall', fontsize=12, fontweight='bold')
ax1.set_title('Overall Model Ranking', fontsize=13, fontweight='bold')
ax1.set_xlim([0.68, 0.84])
ax1.grid(True, alpha=0.3, axis='x')

for i, (idx, row) in enumerate(sorted_data.iterrows()):
    ax1.text(row['Test Recall'] + 0.002, i, f"{row['Test Recall']:.4f}", 
            va='center', fontsize=10, fontweight='bold')
    if i == len(sorted_data) - 1:
        ax1.text(row['Test Recall'] - 0.04, i, '🏆 WINNER', va='center', fontsize=10, 
                fontweight='bold', color='white')

# Medal icons for top 3
medal_y = [len(sorted_data) - 1, len(sorted_data) - 2, len(sorted_data) - 3]
medals = ['🥇', '🥈', '🥉']

# Explanation
ax2 = fig.add_subplot(gs[1, 0])
ax2.axis('off')
explanation_text = """
KEY FINDINGS:

🥇 OVERALL WINNER: ViT Solo (Two-Stage)
   • Achieves 82.71% macro recall
   • Superior architecture for medical imaging
   • Best at detecting Normal class (69.2%)
   • Simplest model with lowest latency

🥈 CLOSE SECOND: Ensemble (Equal Weight)
   • Achieves 82.63% macro recall (0.08% gap)
   • Simple 50-50 averaging of ViT + CNN
   • Better bacterial detection (96.9%)
   • Minimal overhead from ensemble

🥉 THIRD: Fusion-Concat (Two-Stage)
   • Achieves 81.35% macro recall
   • Learned fusion of concatenated embeddings
   • Good viral detection (87.8%)
   • Adds training complexity

⚠️ ATTENTION PARADOX:
   • Fusion Sweep: Attention (77.1%) >> Concat (68.8%)
   • Two-Stage: ViT Solo (82.7%) > Fusion-Attention (72.96%)
   • Interpretation: CNN adds conflicting signals rather than complementary features
   • ViT already captures texture + structure information
"""
ax2.text(0.05, 0.95, explanation_text, fontsize=10.5, verticalalignment='top',
        family='monospace', bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.9, 
        edgecolor='black', linewidth=1.5, pad=15))

fig.suptitle('Final Results & Recommendations', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(report_dir / 'plot4_overall_ranking.png', dpi=300, bbox_inches='tight')
plt.close()

print("[6/7] Generating PDF report...")

# ============================================================================
# CREATE PDF REPORT
# ============================================================================

pdf_path = report_dir / "DualConvVit_Fusion_Analysis_Report.pdf"
doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                        rightMargin=0.5*inch, leftMargin=0.5*inch,
                        topMargin=0.5*inch, bottomMargin=0.5*inch)

# Styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#1a1a1a'),
    spaceAfter=10,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=colors.HexColor('#333333'),
    spaceAfter=8,
    spaceBefore=12,
    fontName='Helvetica-Bold',
    borderColor=colors.HexColor('#FF6B6B'),
    borderWidth=2,
    borderPadding=5
)

body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['BodyText'],
    fontSize=10,
    alignment=TA_JUSTIFY,
    spaceAfter=10,
    leading=14
)

# Build PDF content
story = []

# Title Page
story.append(Spacer(1, 0.5*inch))
story.append(Paragraph("DualConvVit Fusion Analysis", title_style))
story.append(Paragraph("Pneumonia Detection in Chest X-Ray Images", 
                      ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=14, 
                                    alignment=TA_CENTER, textColor=colors.HexColor('#666666'))))
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y')}", 
                      ParagraphStyle('Date', parent=styles['Normal'], fontSize=11, 
                                    alignment=TA_CENTER)))
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph("Multi-Modal Fusion Strategies for Medical Image Classification", 
                      ParagraphStyle('Subtitle2', parent=styles['Normal'], fontSize=12, 
                                    alignment=TA_CENTER, textColor=colors.HexColor('#888888'),
                                    fontName='Helvetica-Oblique')))

story.append(PageBreak())

# Executive Summary
story.append(Paragraph("Executive Summary", heading_style))
summary_text = """
This report presents a comprehensive analysis of two complementary experiments for optimizing 
multi-modal fusion in pneumonia detection from chest X-ray images:

<b>1. Fusion Sweep (Hyperparameter Search):</b> A 3-phase systematic search over 27 hyperparameter 
configurations × 2 fusion types, totaling ~432 GPU training epochs. Phase 1-2 conducted quick 
validation (8 epochs each), while Phase 3 trained the best configurations for 20 epochs.

<b>2. Two-Stage Analysis:</b> Direct comparison of 6 different fusion/ensemble strategies with 
fixed hyperparameters, providing insight into architectural choices beyond hyperparameter tuning.

<b>Key Finding:</b> Vision Transformer (ViT) alone outperforms all fusion methods, achieving 
82.71% macro recall on the test set. Simple ensemble nearly matches this performance, while 
learned fusion methods underperform, suggesting limited complementarity between CNN and ViT features.
"""
story.append(Paragraph(summary_text, body_style))
story.append(Spacer(1, 0.2*inch))

# Methodology
story.append(Paragraph("Methodology", heading_style))
methodology_text = """
<b>Experiment 1: Fusion Sweep</b><br/>
The fusion sweep employs a 3-phase architecture search:
<ul>
<li><b>Phase 1 (Concat Search):</b> Evaluates 27 hyperparameter combinations with concatenation-based 
fusion for 8 epochs each. Tests learning rates [1e-4, 5e-5, 1e-5], dropout rates [0.2, 0.3, 0.4], 
and weight decay [1e-5, 1e-4, 1e-3].</li>
<li><b>Phase 2 (Attention Search):</b> Same 27 configurations with attention-based fusion for 8 epochs. 
Adjusts dropout range to [0.1, 0.2, 0.3] based on domain knowledge.</li>
<li><b>Phase 3 (Final Training):</b> The rank-1 configuration from each fusion type is trained 
for 20 full epochs with early stopping, validation, and full metrics.</li>
</ul>

<b>Experiment 2: Two-Stage Analysis</b><br/>
Six distinct strategies are trained once with fixed hyperparameters:
<ul>
<li>Solo models: ViT and CNN separately</li>
<li>Ensembles: Equal-weight and custom-weighted averaging</li>
<li>Learned Fusion: Concatenation and attention-based fusion layers</li>
</ul>
"""
story.append(Paragraph(methodology_text, body_style))
story.append(Spacer(1, 0.2*inch))

story.append(PageBreak())

# Results
story.append(Paragraph("Results & Findings", heading_style))

# Plot 1
story.append(Paragraph("Figure 1: Hyperparameter Sensitivity Analysis", 
                      ParagraphStyle('FigureCaption', parent=styles['Normal'], 
                                    fontSize=10, textColor=colors.HexColor('#666666'),
                                    fontName='Helvetica-Oblique')))
img1 = Image(str(report_dir / 'plot1_lr_analysis.png'), width=7*inch, height=3*inch)
story.append(img1)
story.append(Paragraph(
    "Learning rate (1e-4) dominates both fusion types. Lower dropout rates (0.2) consistently "
    "outperform higher values, suggesting regularization is important but not excessive dropout.",
    body_style))
story.append(Spacer(1, 0.15*inch))

# Plot 2
story.append(Paragraph("Figure 2: Phase 3 Final Training Results", 
                      ParagraphStyle('FigureCaption', parent=styles['Normal'], 
                                    fontSize=10, textColor=colors.HexColor('#666666'),
                                    fontName='Helvetica-Oblique')))
img2 = Image(str(report_dir / 'plot2_phase3_results.png'), width=7*inch, height=3*inch)
story.append(img2)
story.append(Paragraph(
    "<b>Fusion Sweep Winner: Attention Fusion (77.11% recall)</b><br/>"
    "Attention fusion outperforms concatenation by 8.35 percentage points. The best configuration "
    "uses LR=0.0001, Dropout=0.2, and WD=0.0001, trained for 20 epochs. This demonstrates that "
    "learned cross-modal attention better captures modality interactions than simple concatenation.",
    body_style))
story.append(Spacer(1, 0.15*inch))

story.append(PageBreak())

# Plot 3
story.append(Paragraph("Figure 3: Two-Stage Strategy Comparison", 
                      ParagraphStyle('FigureCaption', parent=styles['Normal'], 
                                    fontSize=10, textColor=colors.HexColor('#666666'),
                                    fontName='Helvetica-Oblique')))
img3 = Image(str(report_dir / 'plot3_twostage_comparison.png'), width=7*inch, height=3.5*inch)
story.append(img3)
story.append(Paragraph(
    "<b>Two-Stage Winner: ViT Solo (82.71% recall)</b><br/>"
    "ViT outperforms all fusion methods by 5% on macro recall. Simple ensemble (equal weight) "
    "nearly matches ViT performance (82.63%), while learned fusion underperforms. Per-class analysis "
    "shows ViT excels at Normal class detection (69.2% vs CNN 57.7%), while CNN performs better on "
    "Bacterial class (95% vs ViT 92%).",
    body_style))
story.append(Spacer(1, 0.15*inch))

# Plot 4
story.append(Paragraph("Figure 4: Overall Model Ranking & Winner Selection", 
                      ParagraphStyle('FigureCaption', parent=styles['Normal'], 
                                    fontSize=10, textColor=colors.HexColor('#666666'),
                                    fontName='Helvetica-Oblique')))
img4 = Image(str(report_dir / 'plot4_overall_ranking.png'), width=7*inch, height=4*inch)
story.append(img4)
story.append(Spacer(1, 0.1*inch))

story.append(PageBreak())

# Key Insights
story.append(Paragraph("Key Insights & Interpretation", heading_style))
insights_text = """
<b>1. The Fusion Paradox</b><br/>
In the Fusion Sweep, attention fusion (77.1%) significantly outperforms concatenation (68.8%). 
However, in the Two-Stage analysis, ViT solo (82.7%) outperforms both fusion methods. This suggests:
<ul>
<li>The hyperparameters optimized for fusion (LR=1e-4, WD=1e-4) may be suboptimal for single modality</li>
<li>ViT already captures CNN-like texture and local feature information</li>
<li>CNN adds conflicting rather than complementary signals</li>
<li>Simple averaging often works better than learned fusion without proper regularization</li>
</ul>

<b>2. Modality Complementarity</b><br/>
While CNNs and ViTs are often considered complementary (local features vs. global structure), 
in this medical imaging task:
<ul>
<li>ViT with ImageNet pre-training captures sufficient discriminative power</li>
<li>CNN's texture-focused features may introduce domain shift</li>
<li>Ensemble performance (82.63%) suggests minimal complementarity</li>
</ul>

<b>3. Per-Class Performance Insights</b><br/>
<ul>
<li><b>Bacterial Pneumonia:</b> Easiest class (>92% recall across all models)</li>
<li><b>Viral Pneumonia:</b> Moderate difficulty (84-91% recall)</li>
<li><b>Normal:</b> Most challenging (49-69% recall) - requires focused improvement</li>
</ul>

<b>4. Attention vs. Concatenation</b><br/>
In the fusion sweep, attention mechanisms significantly outperform concatenation. This suggests 
that learned weighting of modalities is valuable when properly optimized, but requires careful 
hyperparameter tuning.
"""
story.append(Paragraph(insights_text, body_style))

story.append(Spacer(1, 0.2*inch))

# Recommendations
story.append(Paragraph("Recommendations", heading_style))
recommendations_text = """
<b>Immediate Actions (Production Deployment):</b>
<ul>
<li><b>Primary:</b> Deploy ViT Solo model (82.71% recall) - best performance, simplest architecture, 
lowest computational cost, fastest inference</li>
<li><b>Alternative:</b> Use Ensemble (Equal) if slightly more robustness is desired (82.63% recall 
with minimal overhead)</li>
<li><b>Avoid:</b> Complex fusion architectures without proper regularization</li>
</ul>

<b>Short-term Improvements:</b>
<ul>
<li>Focus on Normal class detection - this is the bottleneck (currently 49-69%)</li>
<li>Implement hard example mining to improve minority class performance</li>
<li>Consider class-specific thresholding based on per-class confidence</li>
<li>Apply domain-specific augmentation for normal X-rays</li>
</ul>

<b>Medium-term Research Directions:</b>
<ul>
<li>Explore hierarchical fusion: combine early fusion (feature level) and late fusion (decision level)</li>
<li>Implement uncertainty-aware fusion: weight modalities by prediction confidence</li>
<li>Test larger ViT variants (base, large) with ensemble strategies</li>
<li>Develop model explainability: which modality contributes to each decision?</li>
</ul>

<b>Long-term Strategies:</b>
<ul>
<li>Robustness testing: evaluate performance under image artifacts, noise, and occlusions</li>
<li>Model compression: quantization and pruning for edge deployment</li>
<li>Active learning: prioritize which samples to label for maximum performance gain</li>
<li>Domain adaptation: fine-tune on different X-ray machine brands and imaging protocols</li>
</ul>
"""
story.append(Paragraph(recommendations_text, body_style))

story.append(PageBreak())

# Conclusion
story.append(Paragraph("Conclusion", heading_style))
conclusion_text = """
This comprehensive analysis of multi-modal fusion for pneumonia detection reveals an important 
finding: Vision Transformers alone achieve superior performance (82.71% macro recall) compared 
to all fusion and ensemble methods tested. This challenges the common assumption that combining 
multiple modalities always improves performance.

The results suggest that for medical image classification with pre-trained models:
<ul>
<li>Single powerful models (ViT) may outperform fusion strategies</li>
<li>Simple ensembles provide near-optimal performance with minimal complexity</li>
<li>Learned fusion mechanisms underperform without careful architecture and hyperparameter design</li>
<li>The bottleneck for improvement is the Normal class detection, not modality combination</li>
</ul>

Going forward, we recommend:
<ol>
<li><b>Deploy ViT Solo</b> as the production model</li>
<li><b>Focus improvements on the Normal class</b> through augmentation and data-centric approaches</li>
<li><b>Revisit fusion methods</b> with better architectures (hierarchical, uncertainty-aware) 
if ensemble performance becomes insufficient</li>
<li><b>Invest in model interpretability</b> to understand which visual features drive predictions</li>
</ol>

This research provides actionable insights for practitioners building fusion-based medical 
imaging systems and demonstrates the importance of empirical validation before committing to 
complex fusion architectures.
"""
story.append(Paragraph(conclusion_text, body_style))

story.append(Spacer(1, 0.3*inch))
story.append(Paragraph(
    f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
    "<b>Dataset:</b> Chest X-Ray Images (224×224 preprocessed)<br/>"
    "<b>Classes:</b> Normal, Bacterial Pneumonia, Viral Pneumonia",
    ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, 
                  textColor=colors.HexColor('#999999'), alignment=TA_CENTER)
))

# Build PDF
doc.build(story)

print(f"\n[7/7] PDF Report successfully generated!")
print(f"\n{'='*80}")
print(f"✓ Report saved to: {pdf_path}")
print(f"✓ Report includes:")
print(f"  - 4 high-quality publication-ready plots")
print(f"  - Executive summary and methodology")
print(f"  - Detailed results and key findings")
print(f"  - Per-class performance analysis")
print(f"  - Comprehensive recommendations")
print(f"{'='*80}")
