"""
Web Documentation Scraper for MyIAModelChat

Extracts training text from online documentation sites.
Uses trafilatura for main-content extraction and BeautifulSoup
for link discovery with same-domain crawling.
"""

import re
import os
import time
import logging
from urllib.parse import urljoin, urlparse
from typing import List, Set, Optional
from collections import deque

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import trafilatura
except ImportError:
    trafilatura = None

logger = logging.getLogger(__name__)

# Directory for scraped output
WEB_SCRAPER_DIR = os.path.join('datasets_source', 'web')


class WebDocScraper:
    """
    Crawl documentation websites and extract clean text for training data.

    Strategy:
    1. Fetch seed URL
    2. Extract main content with trafilatura
    3. Discover documentation links (same domain, same path prefix)
    4. BFS crawl with depth/page limits and rate limiting
    5. Return list of cleaned text paragraphs
    """

    def __init__(
        self,
        seed_url: str,
        max_pages: int = 50,
        max_depth: int = 3,
        rate_limit: float = 1.0,
        timeout: int = 15,
        path_prefix: Optional[str] = None,
    ):
        self.seed_url = seed_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.path_prefix = path_prefix or self._infer_path_prefix(seed_url)

        parsed = urlparse(seed_url)
        self.domain = parsed.netloc
        self.scheme = parsed.scheme
        self.base_url = f"{self.scheme}://{self.domain}"

        self.visited: Set[str] = set()
        self.texts: List[str] = []
        self._session: Optional[requests.Session] = None

    @staticmethod
    def _infer_path_prefix(url: str) -> str:
        """Infer crawl path prefix from URL structure."""
        path = urlparse(url).path.rstrip('/')
        return path

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': 'MyIAModelChat-Scraper/1.0 (training data collection)'
            })
        return self._session

    def _is_valid_doc_link(self, url: str) -> bool:
        """Check if URL is a valid documentation page to crawl."""
        parsed = urlparse(url)

        if parsed.netloc != self.domain:
            return False

        if self.path_prefix and not parsed.path.startswith(self.path_prefix):
            return False

        skip_extensions = {
            '.pdf', '.zip', '.tar', '.gz', '.png', '.jpg', '.jpeg',
            '.gif', '.svg', '.css', '.js', '.ico', '.woff', '.woff2',
            '.ttf', '.eot', '.mp4', '.mp3', '.xml', '.json', '.rss',
        }
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in skip_extensions):
            return False

        if url.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
            return False

        if parsed.path in self.visited and parsed.fragment:
            return False

        return True

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragment, trailing slash consistency."""
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if normalized != self.base_url and normalized.endswith('/'):
            normalized = normalized.rstrip('/')
        return normalized

    def _extract_links(self, html: str, current_url: str) -> List[str]:
        """Extract documentation links from HTML."""
        if BeautifulSoup is None:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        links = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(current_url, href)
            normalized = self._normalize_url(absolute_url)

            if self._is_valid_doc_link(normalized):
                links.append(normalized)

        return links

    def _extract_content(self, html: str, url: str) -> Optional[str]:
        """Extract main documentation content from HTML using trafilatura."""
        if trafilatura is not None:
            text = trafilatura.extract(
                html,
                include_links=False,
                include_tables=True,
                include_comments=False,
                include_images=False,
                no_fallback=False,
                favor_recall=True,
            )
            if text:
                return text

        return self._basic_extract(html)

    @staticmethod
    def _basic_extract(html: str) -> Optional[str]:
        """Basic content extraction fallback when trafilatura unavailable."""
        if BeautifulSoup is None:
            return None

        soup = BeautifulSoup(html, 'html.parser')

        for tag in soup.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
            tag.decompose()

        content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '#content',
                         '.documentation', '#documentation', '.body', '#body']:
            content = soup.select_one(selector)
            if content:
                break

        if content is None:
            content = soup.find('body')

        if content is None:
            return None

        return content.get_text(separator=' ', strip=True)

    @staticmethod
    def clean_doc_text(text: str) -> str:
        """Clean extracted documentation text."""
        if not text:
            return ""

        # Repair mojibake (UTF-8 bytes decoded as a single-byte encoding)
        if re.search(r'[\x80-\xff]', text):
            for enc in ('latin-1', 'cp1252', 'cp1250', 'cp1251', 'cp1253',
                        'cp1254', 'cp1257', 'iso8859-2', 'iso8859-4',
                        'iso8859-5', 'iso8859-7', 'iso8859-15', 'koi8-r',
                        'koi8-u', 'macroman', 'maccyrillic'):
                try:
                    text = text.encode(enc, errors='strict').decode('utf-8', errors='strict')
                    break
                except (UnicodeDecodeError, UnicodeEncodeError):
                    continue

        text = re.sub(r'```\w*\n', '\n', text)
        text = re.sub(r'```', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^[-~=]{3,}\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def scrape(self) -> List[str]:
        """
        Execute the crawl and return extracted text paragraphs.

        Returns:
            List of cleaned text strings, one per page's content
        """
        if requests is None:
            raise ImportError("requests is required: pip install requests")
        if trafilatura is None and BeautifulSoup is None:
            raise ImportError("trafilatura or beautifulsoup4 required: pip install trafilatura beautifulsoup4")

        session = self._get_session()
        queue: deque = deque()
        queue.append((self.seed_url, 0))

        logger.info(f"Starting web scrape: {self.seed_url}")
        logger.info(f"  Max pages: {self.max_pages}, Max depth: {self.max_depth}")
        logger.info(f"  Path prefix: {self.path_prefix}")

        pages_scraped = 0

        while queue and pages_scraped < self.max_pages:
            current_url, depth = queue.popleft()

            normalized = self._normalize_url(current_url)
            if normalized in self.visited:
                continue
            self.visited.add(normalized)

            if pages_scraped > 0:
                time.sleep(self.rate_limit)

            try:
                response = session.get(current_url, timeout=self.timeout, allow_redirects=True)
                response.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"  Failed to fetch {current_url}: {e}")
                continue

            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type:
                continue

            html = response.text

            text = self._extract_content(html, current_url)
            if text:
                cleaned = self.clean_doc_text(text)
                if cleaned and len(cleaned) > 50:
                    self.texts.append(cleaned)
                    pages_scraped += 1
                    logger.info(f"  [{pages_scraped}/{self.max_pages}] Scraped: {current_url} ({len(cleaned)} chars)")

            if depth < self.max_depth:
                links = self._extract_links(html, current_url)
                for link in links:
                    if link not in self.visited:
                        queue.append((link, depth + 1))

        logger.info(f"Scrape complete: {pages_scraped} pages, {len(self.texts)} text blocks extracted")
        self.close()
        return self.texts

    def close(self):
        """Close the requests session and release connection pool."""
        if self._session is not None:
            self._session.close()
            self._session = None


def load_urls_from_file(filepath: str) -> List[str]:
    """Load URLs from a text file (one URL per line)."""
    if not os.path.exists(filepath):
        return []

    urls = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    return urls


def load_urls_from_json(filepath: str) -> List[dict]:
    """
    Load URL configurations from a JSON file with per-URL settings and shared defaults.

    JSON format:
    {
        "urls": [
            {
                "url": "https://example.com/docs/",
                "max_pages": 50,
                "max_depth": 3,
                "rate_limit": 1.0,
                "timeout": 15,
                "path_prefix": null,
                "enabled": true
            }
        ],
        "defaults": {
            "max_pages": 50,
            "max_depth": 3,
            "rate_limit": 1.0,
            "timeout": 15,
            "path_prefix": null,
            "enabled": true
        }
    }

    Args:
        filepath: Path to the JSON config file

    Returns:
        List of config dicts, one per enabled URL with resolved defaults
    """
    import json as json_mod

    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json_mod.load(f)
    except Exception as e:
        logger.warning(f"Error reading web config: {e}")
        return []

    defaults = config.get('defaults', {})
    urls_config = config.get('urls', [])

    if not urls_config:
        logger.warning("No URLs defined in web config")
        return []

    result = []
    for entry in urls_config:
        url = entry.get('url', '')
        if not url:
            continue

        enabled = entry.get('enabled', defaults.get('enabled', True))
        if not enabled:
            logger.info(f"  Skipping disabled URL: {url}")
            continue

        resolved = {
            'url': url,
            'max_pages': entry.get('max_pages', defaults.get('max_pages', 50)),
            'max_depth': entry.get('max_depth', defaults.get('max_depth', 3)),
            'rate_limit': entry.get('rate_limit', defaults.get('rate_limit', 1.0)),
            'timeout': entry.get('timeout', defaults.get('timeout', 15)),
            'path_prefix': entry.get('path_prefix', defaults.get('path_prefix')),
        }
        result.append(resolved)

    return result


def scrape_web_docs(
    url: Optional[str] = None,
    urls_file: Optional[str] = None,
    max_pages: int = 50,
    max_depth: int = 3,
    rate_limit: float = 1.0,
    output_dir: Optional[str] = None,
) -> List[str]:
    """
    Convenience function to scrape documentation from URL(s).

    Args:
        url: Single seed URL to scrape (optional if urls_file provided)
        urls_file: Path to file with one URL per line (optional if url provided)
        max_pages: Maximum pages per URL (default: 50)
        max_depth: Maximum link-following depth (default: 3)
        rate_limit: Seconds between requests (default: 1.0)
        output_dir: Directory to save scraped text files (optional)

    Returns:
        List of cleaned text strings from all URLs
    """
    all_texts = []

    urls = []
    if url:
        urls.append(url)
    if urls_file:
        urls.extend(load_urls_from_file(urls_file))

    if not urls:
        logger.warning("No URLs provided for web scraping")
        return []

    for seed_url in urls:
        logger.info(f"--- Scraping: {seed_url} ---")
        scraper = WebDocScraper(
            seed_url=seed_url,
            max_pages=max_pages,
            max_depth=max_depth,
            rate_limit=rate_limit,
        )
        try:
            texts = scraper.scrape()
            all_texts.extend(texts)
        except Exception as e:
            logger.warning(f"  Failed to scrape {seed_url}: {e}")
            continue

    # Save scraped text to output directory
    if output_dir and all_texts:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'scraped_text.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in all_texts:
                f.write(text + '\n\n---\n\n')
        logger.info(f"Saved {len(all_texts)} text blocks to {output_file}")

    return all_texts
