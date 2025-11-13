###testing method 1 (worked)--------------------------
# import ssl, httpx

# # Create SSL context using system certificates instead of certifi
# context = ssl.create_default_context()

# # Optionally add certifi bundle too (some corporate proxies need this)
# import certifi
# context.load_verify_locations(certifi.where())

# r = httpx.get("https://api.openai.com/v1/models", verify=context)
# print(r.status_code)


####testing method 2 (worked)----------------------
# import httpx, ssl
# context = ssl.create_default_context()
# r = httpx.get("https://api.openai.com/v1/models", verify=context)
# print(r.status_code)


####testing method 3 (worked) ------------------------
# import requests, certifi
# print(certifi.where())
# try:
#     r = requests.get("https://api.openai.com/v1/models", verify=certifi.where())
#     print("Success:", r.status_code)
# except Exception as e:
#     print("Error:", e)



#### testing method 4 (universal pattern, used for production facing that makes HTTPS request)
import ssl, certifi, httpx

context = ssl.create_default_context()
context.load_verify_locations(certifi.where())

with httpx.Client(verify=context, timeout=30) as client:
    response = client.get("https://api.openai.com/v1/models")
    print(response.status_code)
