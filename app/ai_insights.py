import os
from dotenv import load_dotenv
from openai import OpenAI
import ssl, certifi, httpx


load_dotenv(dotenv_path="./app/.env")

# fix ssl by ensure cerifi CA is used and combined with company corporate root
CORPORATE_CERT_PATH=r"C:\Users\n1309516\corp_root.cer"
context=ssl.create_default_context()
context.load_verify_locations(certifi.where())
context.load_verify_locations(CORPORATE_CERT_PATH)

http_client=httpx.Client(verify=context,timeout=60.0)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=http_client
)

def generate_forecast_commentary(forecast_df):
    """_summary_
    Generate a short AI commentary abut forecast_df
    Args:
        forecast_df (_type_): _description_
    """
    # extract key starts from forecast
    latest=forecast_df.tail(1).iloc[0]
    mean_forecast=forecast_df['yhat'].mean()
    start_forecast=forecast_df['yhat'].iloc[0]
    end_forecast=forecast_df['yhat'].iloc[-1]
    pct_change=((end_forecast-start_forecast)/start_forecast)*100
    
    #build simple text summary
    summary=(
        f"The revenue forecast starts at ${start_forecast:,.0f} "
        f"and ends at ${end_forecast:,.0f},"
        f"a change of {pct_change:.1f}% over the forecast period."
    )
    
    prompt=(
        f"You are business analyst. Based on the following summary, "
        f"please write a 2-sentence, professional insights:\n"
        f"{summary}"
    )
    
    response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are an expert financial data analyst."},
            {"role":"user","content":prompt}
        ],
        temperature=0.6,
    )
    return response.choices[0].message.content.strip()