from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
import certifi
import ssl
import httpx

# Load environment variables
load_dotenv()
print("OpenAI Key:", os.getenv("OPENAI_API_KEY")[:8] + "...")
print("Azure Key:", os.getenv("AZURE_STORAGE_CONNECTION_STRING")[:8] + "...")

# Create an SSL context using certifi's CA bundle
CORPORATE_CERT_PATH=r"C:\Users\n1309516\corp_root.cer"
context=ssl.create_default_context()
context.load_verify_locations(certifi.where())
context.load_verify_locations(CORPORATE_CERT_PATH)

http_client=httpx.Client(verify=context,timeout=60.0)

# Pass it into OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=http_client
)

# Simple test request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hello"}],
)
print(response.choices[0].message.content)
