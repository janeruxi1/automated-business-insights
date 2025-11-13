import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./app/.env")

from forecasting import load_data,forecast_revenue,plot_forecast
from ai_insights import generate_forecast_commentary
from report_generator import generate_pdf

st.set_page_config(page_title="Automated Business Insights",layout="wide")
st.title("Automated Business Insights Generator")
st.markdown("Upload your data or use the sample dataset to generate forecasts")

df=load_data()

st.subheader("Recent Sales Data")
st.dataframe(df.tail(10))

model,forecast=forecast_revenue(df)
forecast_plot=plot_forecast(model,forecast)

st.subheader("Forecast Visualization")
st.image(forecast_plot,width="stretch")

st.subheader("Forecast Summary(Next 30 days)")
st.write(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(10))

ai_commentary=generate_forecast_commentary(forecast)
st.subheader("AI-Generated Business Insight")
st.write(ai_commentary)

# Generate PDF when button is clicked
if st.button("Download PDF Report"):
    pdf_path="business_insights_report.pdf"
    
    generate_pdf(
        forecast_commentary=ai_commentary,
        chart_base64=forecast_plot,
        output_path=pdf_path
    )
    
    with open(pdf_path,"rb") as f:
        st.download_button(
            label=" Download Report",
            data=f,
            file_name="business_insights_report.pdf",
            mime="application/pdf"
        )