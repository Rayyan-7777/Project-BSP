"""
report_generator.py
===================
Generates automated PDF reports for the ECG-HRV Analysis Dashboard.

Author  : BSP Lab OEL
Version : 1.0
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from datetime import datetime

def generate_pdf_report(metrics: dict, interpretations: dict, plots: dict, output_path: str):
    """
    Generate a professional PDF report containing HRV metrics and plots.
    
    Parameters
    ----------
    metrics         : dict  - The HRV metrics dictionary
    interpretations : dict  - The physiological interpretation dictionary
    plots           : dict  - Dictionary mapping plot names to file paths (e.g., {'ecg': 'path.png'})
    output_path     : str   - Where to save the PDF
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
                            
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=15,
        alignment=1 # Center
    )
    
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=10,
        spaceBefore=15
    )
    
    normal_style = styles['Normal']
    
    story = []
    
    # --- HEADER ---
    story.append(Paragraph("ECG-HRV Clinical Analysis Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 20))
    
    # --- TIME DOMAIN METRICS ---
    story.append(Paragraph("Time-Domain HRV Metrics", heading_style))
    
    td_data = [
        ['Metric', 'Value', 'Unit'],
        ['Mean HR', f"{metrics.get('mean_hr', 0):.1f}", 'bpm'],
        ['Mean RR', f"{metrics.get('mean_rr_ms', 0):.1f}", 'ms'],
        ['SDNN', f"{metrics.get('sdnn_ms', 0):.1f}", 'ms'],
        ['RMSSD', f"{metrics.get('rmssd_ms', 0):.1f}", 'ms'],
        ['pNN50', f"{metrics.get('pnn50', 0):.1f}", '%']
    ]
    
    td_table = Table(td_data, colWidths=[200, 100, 100])
    td_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1'))
    ]))
    story.append(td_table)
    story.append(Spacer(1, 15))
    
    # --- FREQUENCY DOMAIN METRICS ---
    story.append(Paragraph("Frequency-Domain HRV Metrics", heading_style))
    
    fd_data = [
        ['Metric', 'Value', 'Unit'],
        ['LF Power', f"{metrics.get('lf_power', 0):.1f}", 'ms²'],
        ['HF Power', f"{metrics.get('hf_power', 0):.1f}", 'ms²'],
        ['LF/HF Ratio', f"{metrics.get('lf_hf_ratio', 0):.2f}", ''],
        ['Total Power', f"{metrics.get('total_power', 0):.1f}", 'ms²']
    ]
    
    fd_table = Table(fd_data, colWidths=[200, 100, 100])
    fd_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1'))
    ]))
    story.append(fd_table)
    story.append(Spacer(1, 20))
    
    # --- INTERPRETATION ---
    story.append(Paragraph("Clinical Interpretation", heading_style))
    for key, text in interpretations.items():
        story.append(Paragraph(f"<b>{key}:</b> {text}", normal_style))
        story.append(Spacer(1, 8))
        
    story.append(Spacer(1, 20))
    
    # --- PLOTS ---
    if plots:
        story.append(Paragraph("Signal & Analysis Visualizations", heading_style))
        for name, img_path in plots.items():
            if os.path.exists(img_path):
                # Calculate aspect ratio to fit the page width
                img = Image(img_path)
                avail_width = 450
                aspect = img.drawHeight / img.drawWidth
                img.drawWidth = avail_width
                img.drawHeight = avail_width * aspect
                
                story.append(Paragraph(f"<b>{name}</b>", normal_style))
                story.append(Spacer(1, 5))
                story.append(img)
                story.append(Spacer(1, 15))
                
    doc.build(story)
