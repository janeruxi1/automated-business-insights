import httpx
import ssl
import certifi

def get_secure_client(timeout: int = 30) -> httpx.Client:
    """
    Returns a preconfigured httpx.Client with proper SSL verification.
    
    Fixes common SSL certificate errors on Windows and corporate networks
    by merging system and certifi certificate authorities.
    
    Args:
        timeout (int): Request timeout in seconds (default 30).
    
    Returns:
        httpx.Client: A reusable, secure HTTP client.
    """
    # Create SSL context and load certifi CA bundle
    context = ssl.create_default_context()
    context.load_verify_locations(certifi.where())

    # Create reusable HTTPX client
    client = httpx.Client(
        verify=context,
        timeout=timeout,
        http2=True,  # optional, can speed up large responses
    )
    return client
