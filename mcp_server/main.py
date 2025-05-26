import httpx
from mcp.server.fastmcp import FastMCP
from typing import List
import os

mcp = FastMCP("My Agent")

# Tool 1: Google Serper API – Web Search
@mcp.tool()
def web_search(query: str) -> List[str]:
    """
    Perform a web search using Google Serper API and return titles + links.
    """
    api_key = os.getenv("SERPER_API_KEY")
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    json_data = {"q": query}

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        results = response.json()

    return [f"{r['title']}: {r['link']}" for r in results.get("organic", [])]


# Tool 2: ScrapingBee API – Fetch Web Page Content from an URL
@mcp.tool()
def fetch_url_content(url: str) -> str:
    """
    Fetch the content of a web page using ScrapingBee API.
    """
    api_key = os.getenv("SCRAPINGBEE_API_KEY")
    api_url = "https://app.scrapingbee.com/api/v1"

    params = {
        "api_key": api_key,
        "url": url,
        "render_js": "false"
    }

    with httpx.Client() as client:
        response = client.get(api_url, params=params)
        response.raise_for_status()
        return response.text


# Tool 3: SendGrid API – Send Email
@mcp.tool()
def send_email(recipient: str, subject: str, content: str) -> str:
    """
    Send an email using the SendGrid API.
    """
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    sendgrid_url = "https://api.sendgrid.com/v3/mail/send"

    payload = {
        "personalizations": [{"to": [{"email": recipient}]}],
        "from": {"email": "your_email@example.com"},
        "subject": subject,
        "content": [{"type": "text/plain", "value": content}]
    }

    headers = {
        "Authorization": f"Bearer {sendgrid_api_key}",
        "Content-Type": "application/json"
    }

    with httpx.Client() as client:
        response = client.post(sendgrid_url, headers=headers, json=payload)
        response.raise_for_status()
        return "Email sent successfully."
