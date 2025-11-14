from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64
from io import BytesIO
from forecasting import extract_model_df

def df_to_pdf_data(df):
    """Convert DataFrame to a list of lists for ReportLab Table."""
    pdf_table = [df.columns.tolist()]
    for _, row in df.iterrows():
        pdf_table.append([str(x) for x in row.values])
    return pdf_table


def generate_pdf(forecast_summary, 
                 chart_base64,
                 ai_insights,
                 forecast_table_data,
                 model_info_text,         
                 output_path="generated_report.pdf"):
    """Create a PDF report with commentary and forecast chart."""
    
    # convert forecast chart from base64 to image
    img_data=base64.b64decode(chart_base64.split(",")[1])
    chart_buffer=BytesIO(img_data)
    
    # PDF setup 
    doc=SimpleDocTemplate(output_path, pagesize=letter)
    styles=getSampleStyleSheet()
    
    flow=[]
    
    # =========================
    # PAGE 1 – Executive Summary
    # =========================
    # Title
    flow.append(Paragraph("<b> Automated Business Insights Report </b>",styles["Title"]))
    flow.append(Spacer(1,0.2*inch))
    
    # AI Forecast Summary
    flow.append(Paragraph("<b> AI Forecast Summary </b>",styles["Heading2"]))
    flow.append(Paragraph(forecast_summary,styles["BodyText"]))
    flow.append(Spacer(1,0.3*inch))
    
    # Forecast chart
    flow.append(Paragraph("<b> Revenue Forecast Chart </b>",styles["Heading2"]))
    flow.append(Image(chart_buffer,width=6*inch,height=4*inch))
    
    # Add page break
    flow.append(PageBreak())
    
    # =========================
    # PAGE 2 – AI Detailed Insights
    # =========================
    flow.append(Paragraph("<b> AI Detailed Insights </b>",styles["Heading1"]))
    flow.append(Spacer(1,0.2*inch))
    
    flow.append(Paragraph(ai_insights.replace("\n","<br/>"),styles["BodyText"]))
    flow.append(Spacer(1,0.3*inch))    
    
    flow.append(Paragraph("<i> (Additional insights, charts or tables can be added here.) </i>",styles["BodyText"]))
    flow.append(PageBreak())
    
    # =========================
    # PAGE 3 – FORECAST TABLE
    # =========================
    flow.append(Paragraph("<b>Forecast Table (Last 30 Days) </b>", styles["Heading1"]))
    flow.append(Spacer(1, 0.2 * inch))

    forecast_pdf_table=df_to_pdf_data(forecast_table_data)

    table = Table(forecast_pdf_table, repeatRows=1)
    
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))

    flow.append(table)
    flow.append(Spacer(1, 0.3 * inch))

    flow.append(PageBreak())


    # =========================
    # PAGE 4 – MODEL INFORMATION
    # =========================
    flow.append(Paragraph("<b>Model Information</b>", styles["Heading1"]))
    flow.append(Spacer(1, 0.2 * inch))
    
    # if model_info is not None:
    #     model_info={
    #         "Model Type": "Prophet Time Series Model",
    #         "Training Period": "Jan 2018 - Dec 2024",
    #         "Forecast Horizon": "6 months",
    #         "Key Features": "Trend, Seasonality, Holidays",
    #         "Performance Metrics": "MAE: $2,500; RMSE: $3,200"
    #     }
    
    flow.append(Paragraph("<b>Model Configuration Details: </b>", styles["BodyText"]))
    flow.append(Spacer(1, 0.2 * inch))
    # model_info_text=extract_model_df(model_info)
    flow.append(Paragraph(model_info_text, styles["BodyText"]))
    flow.append(Spacer(1, 0.3 * inch))
    
    summary_text=""""
        This model summary provides details on predictive performance, 
        training configuration, and important notes for interpretation.
        """
    flow.append(Paragraph(summary_text, styles["BodyText"]))
    
    # Build PDF
    doc.build(flow)
    return output_path


