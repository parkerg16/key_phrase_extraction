import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

""" The goal of this is to potentially gather more data that will allow us to train the extraction model, 
it has yet to be determined whether this process helps or not due to the generalization of the existing models already"""

def get_page_name(url):
    """Extracts the last segment from the URL path."""
    return urlparse(url).path.split("/")[-1]


def crawl_wikipedia(seed_url, max_pages=50, delay=1):
    visited = set()
    queue = [seed_url]
    base_url = "https://en.wikipedia.org"

    while queue and len(visited) < max_pages:
        current_url = queue.pop(0)
        if current_url in visited:
            continue

        print(f"Crawling: {current_url}")
        try:
            response = requests.get(current_url)
            if response.status_code != 200:
                print(f"Failed to retrieve {current_url}")
                continue
        except Exception as e:
            print(f"Error accessing {current_url}: {e}")
            continue

        visited.add(current_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from all <p> tags
        p_tags = soup.find_all("p")
        page_text = "\n".join([p.get_text() for p in p_tags]).strip()

        # Save the text to a file named after the page (e.g., "Machine_learning.txt")
        page_name = get_page_name(current_url)
        file_name = f"{page_name}.txt"
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(page_text)
        print(f"Saved content to {file_name}")

        # Find all internal Wikipedia links
        for link in soup.find_all("a", href=True):
            href = link['href']
            # Only follow internal Wikipedia links and ignore special pages
            if href.startswith("/wiki/") and ":" not in href:
                full_url = urljoin(base_url, href)
                if full_url not in visited and full_url not in queue:
                    queue.append(full_url)

        # Polite crawling: pause between requests
        time.sleep(delay)

    print(f"Crawling finished. Visited {len(visited)} pages.")


# Example usage:
if __name__ == "__main__":
    seed = "https://en.wikipedia.org/wiki/Machine_learning"
    crawl_wikipedia(seed, max_pages=50)
