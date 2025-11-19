import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./app/.env")

from forecasting import load_data,forecast_revenue,plot_forecast,extract_model_df
from ai_insights import generate_forecast_commentary,generate_ai_insights
from report_generator import generate_pdf,df_to_pdf_data

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
forecast_table =forecast.merge(df[['date','revenue']].rename(columns={'date':'ds','revenue':'y'}),on='ds',how='left')
forecast_table=forecast_table[['ds','y','yhat','yhat_lower','yhat_upper']]
st.write(forecast_table.tail(30))

ai_forecast_summary=generate_forecast_commentary(forecast_table)
st.subheader("AI-Generated Forecast Summary")
st.write(ai_forecast_summary)

ai_insights=generate_ai_insights(forecast_table)
st.subheader("AI-Generated AI Insight")
st.write(ai_insights)

model_info_text=extract_model_df(model=model,df=forecast_table)
st.subheader("Model Configuration Details")
st.markdown(model_info_text,unsafe_allow_html=True)

# Generate PDF when button is clicked
if st.button("Download PDF Report"):
    pdf_path="business_insights_report.pdf"
    
    generate_pdf(
        forecast_summary=ai_forecast_summary,
        chart_base64=forecast_plot,
        ai_insights=ai_insights,
        forecast_table_data=forecast_table.tail(30),
        model_info_text=model_info_text,
        output_path=pdf_path
    )
    
    with open(pdf_path,"rb") as f:
        st.download_button(
            label="Download Report",
            data=f,
            file_name="business_insights_report.pdf",
            mime="application/pdf"
        )