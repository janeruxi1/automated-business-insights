from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import base64
from io import BytesIO
import matplotlib.pyplot as plt


def generate_pdf(forecast_commentary, chart_base64,output_path="generated_report.pdf"):
    """Create a PDF report with commentary and forecast chart."""
    
    # convert forecast chart from base64 to image
    img_data=base64.b64decode(chart_base64.split(",")[1])
    # buf=BytesIO()
    # plt.savefig(buf,format='png')
    #chart_buffer=buf.seek(0)
    chart_buffer=BytesIO(img_data)
    
    # PDF setup 
    doc=SimpleDocTemplate(output_path, pagesize=letter)
    styles=getSampleStyleSheet()
    
    flow=[]
    
    # Title
    flow.append(Paragraph("<b> Automated Business Insights Report </b>",styles["Title"]))
    flow.append(Spacer(1,0.2*inch))
    
    # AI commentary
    flow.append(Paragraph("<b> AI Forecast Commentary </b>",styles["Heading2"]))
    flow.append(Paragraph(forecast_commentary,styles["BodyText"]))
    flow.append(Spacer(1,0.3*inch))
    
    # Forecast chart
    flow.append(Paragraph("<b> Revenue Forecast Chart </b>",styles["Heading2"]))
    flow.append(Image(chart_buffer,width=6*inch,height=4*inch))
    
    doc.build(flow)
    
    return output_path


