"""
Web Researcher Module
-------------------
Handles web research, content extraction, and information gathering from online sources.

Features:
- Web searching with rate limiting
- Content extraction and sanitization
- Result caching
- Robust error handling
"""

import time
import urllib.parse
from typing import Dict, List, Any, Optional
from collections import OrderedDict
from datetime import datetime

from AgentSystem.utils.logger import get_logger
from AgentSystem.modules.knowledge_manager import KnowledgeManager

logger = get_logger("modules.web_researcher")

try:
    import requests
    from bs4 import BeautifulSoup
    import nltk
    from nltk.tokenize import sent_tokenize
    WEB_IMPORTS_AVAILABLE = True
except ImportError:
    logger.warning("Web research dependencies not available. Install with: pip install requests beautifulsoup4 nltk")
    WEB_IMPORTS_AVAILABLE = False

class WebResearcher:
    def __init__(self, knowledge_manager: KnowledgeManager,
                user_agent: str = "Mozilla/5.0 AgentSystem Research Bot"):
        """Initialize web researcher with knowledge manager"""
        self.knowledge_manager = knowledge_manager
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "DNT": "1"
        }
        
        # Initialize session and cache
        self.session = requests.Session() if WEB_IMPORTS_AVAILABLE else None
        self.cache = OrderedDict()
        self.cache_max_size = 100
        self.cache_expiry = 3600  # 1 hour
        
        # Rate limiting
        self.last_request = 0
        self.min_request_interval = 2  # seconds
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests"""
        now = time.time()
        time_since_last = now - self.last_request
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request = now
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web for information
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, url, and snippet
        """
        if not WEB_IMPORTS_AVAILABLE:
            return [{
                "title": "Dependencies Not Available",
                "url": "",
                "snippet": "Web research capabilities are disabled. Install required packages."
            }]
        
        # Check cache
        cache_key = f"search:{query}:{max_results}"
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry["timestamp"] < self.cache_expiry:
                return entry["results"]
        
        try:
            # Apply rate limiting
            self._rate_limit()
            
            # Construct search URL (using DuckDuckGo Lite)
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
            
            # Make request
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse results
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for i, result in enumerate(soup.select('table.result-link')):
                if i >= max_results:
                    break
                    
                link = result.select_one('a')
                if not link:
                    continue
                    
                title = link.get_text(strip=True)
                url = link.get('href')
                
                # Get snippet from next table
                snippet_table = result.find_next('table', {'class': 'result-snippet'})
                snippet = snippet_table.get_text(strip=True) if snippet_table else ""
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })
            
            # Update cache
            self.cache[cache_key] = {
                "results": results,
                "timestamp": time.time()
            }
            
            # Maintain cache size
            if len(self.cache) > self.cache_max_size:
                self.cache.popitem(last=False)
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def fetch_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch and extract content from a web page
        
        Args:
            url: URL to fetch
            
        Returns:
            Dictionary with page title, content, and metadata
        """
        if not WEB_IMPORTS_AVAILABLE:
            return None
            
        try:
            # Apply rate limiting
            self._rate_limit()
            
            # Make request
            response = self.session.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            # Parse content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'meta', 'noscript', 'iframe']):
                tag.decompose()
            
            # Extract title and content
            title = soup.title.string if soup.title else url
            
            # Try to get main content
            content = ""
            main_tags = ['main', 'article', 'div[role="main"]', '.main-content', '#content']
            for selector in main_tags:
                main = soup.select_one(selector)
                if main:
                    content = main.get_text(strip=True)
                    break
            
            # Fallback to body if no main content found
            if not content:
                content = soup.body.get_text(strip=True) if soup.body else ""
            
            # Generate summary
            sentences = sent_tokenize(content)
            summary = " ".join(sentences[:3]) if sentences else ""
            
            return {
                "title": title,
                "content": content,
                "summary": summary,
                "url": url,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def research_topic(self, topic: str, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Research a topic by searching and following links
        
        Args:
            topic: Topic to research
            depth: How many levels of links to follow (1-3)
            
        Returns:
            List of gathered information
        """
        results = []
        visited_urls = set()
        
        # Initial search
        search_results = self.search(topic)
        
        for result in search_results:
            url = result["url"]
            if url in visited_urls:
                continue
                
            # Fetch and process page
            page = self.fetch_page(url)
            if not page:
                continue
                
            # Store in knowledge base
            self.knowledge_manager.add_fact(
                content=page["summary"],
                source=url,
                category=topic,
                confidence=0.8
            )
            
            results.append(page)
            visited_urls.add(url)
            
            # Follow links if depth > 1
            if depth > 1:
                soup = BeautifulSoup(page["content"], 'html.parser')
                for link in soup.find_all('a', href=True):
                    if len(visited_urls) >= 10:  # Limit total pages
                        break
                        
                    href = link.get('href')
                    if not href.startswith('http'):
                        continue
                        
                    if href not in visited_urls:
                        subpage = self.fetch_page(href)
                        if subpage:
                            self.knowledge_manager.add_fact(
                                content=subpage["summary"],
                                source=href,
                                category=topic,
                                confidence=0.6
                            )
                            results.append(subpage)
                            visited_urls.add(href)
        
        return results
