"""
Simple web scraper for additional data gathering
"""
import httpx
from bs4 import BeautifulSoup

class WebScraper:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)
    
    async def scrape(self, url: str) -> dict:
        """Scrape content from a URL"""
        try:
            response = await self.client.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return {
                "url": url,
                "content": text[:2000],  # Limit content
                "title": soup.title.string if soup.title else "No title",
                "success": True
            }
        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "success": False
            }