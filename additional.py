import requests
from bs4 import BeautifulSoup
import urllib3

# --- Disable insecure warnings if you ever skip SSL ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Create a session that always uses certifi for SSL ---
session = requests.Session()
session.verify = False  # No 'verify' needed; python-certifi-win32 handles it

# --- Search SAFLII ---
def search_saflii(query="eviction", max_results=3):
    """
    Search SAFLII for a keyword and return a list of (title, url) pairs.
    """
    search_url = "https://www.saflii.org/search.html"

    # This mimics filling in the search form
    payload = {
        "query": query,
        "submit": "Search"
    }

    r = session.post(search_url, data=payload)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    results = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith("/za/cases/") or href.startswith("/za/journals/"):
            full_url = "https://www.saflii.org" + href
            title = link.get_text(strip=True)
            results.append((title, full_url))
        if len(results) >= max_results:
            break
    return results



# --- Fetch article text ---
def fetch_article_text(url):
    """
    Fetch the article text from SAFLII and extract paragraphs.
    """
    r = session.get(url)
    if r.status_code in (404, 410):
        raise ValueError(f"Article not available at {url} (status {r.status_code})")
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    content = soup.find("div", {"id": "content"}) or soup
    paragraphs = content.find_all("p")
    return "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])


if __name__ == "__main__":
    print(" Searching SAFLII for eviction articles...")
    try:
        articles = search_saflii("eviction", max_results=3)
    except Exception as e:
        print("Error during search:", e)
        exit(1)

    if not articles:
        print("No articles found.")
        exit(0)

    for i, (title, url) in enumerate(articles, 1):
        print(f"{i}. {title}\n   {url}")

    print("\n Fetching first article text...\n")
    try:
        article_text = fetch_article_text(articles[0][1])
        print(article_text[:1500], "...\n[Truncated]")
    except Exception as e:
        print("Error fetching article:", e)
