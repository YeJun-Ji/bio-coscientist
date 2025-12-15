"""
Web Search Client for PubMed, Semantic Scholar, and arXiv
"""

import os
import logging
from typing import List, Dict, Any, Optional
import asyncio
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlencode

try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout
except ImportError:
    aiohttp = None
    ClientSession = None
    ClientTimeout = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from ..core import Paper

logger = logging.getLogger(__name__)


class WebSearchClient:
    """
    Unified interface for searching scientific literature.
    Supports PubMed, Semantic Scholar, arXiv, and general web search.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.session: Optional[ClientSession] = None
        
        # Initialize timeout only if aiohttp is available
        if ClientTimeout is not None:
            self.timeout = ClientTimeout(total=30)
        else:
            self.timeout = None
        
        # API keys (optional for some services)
        self.semantic_scholar_api_key = self.config.get("semantic_scholar_api_key") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        
        logger.info("WebSearchClient initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        print(f"[DEBUG-WEBSEARCH] 1. Entering WebSearchClient context manager")
        if aiohttp is None:
            print(f"[DEBUG-WEBSEARCH] 2. aiohttp not installed!")
            raise ImportError("aiohttp not installed. Install with: pip install aiohttp")
        print(f"[DEBUG-WEBSEARCH] 3. Creating ClientSession...")
        self.session = ClientSession(timeout=self.timeout)
        print(f"[DEBUG-WEBSEARCH] 4. ClientSession created successfully")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def search_all(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Search across multiple sources.
        
        Args:
            query: Search query
            max_results: Maximum results per source
            sources: List of sources to search (pubmed, semantic_scholar, arxiv)
                    Default: ["pubmed"] only to avoid rate limiting
        
        Returns:
            Combined list of papers from all sources
        """
        if sources is None:
            # Use only PubMed by default - it's free and has no rate limits
            sources = ["pubmed"]
        
        all_papers = []
        
        for source in sources:
            try:
                if source == "pubmed":
                    papers = await self.search_pubmed(query, max_results)
                elif source == "semantic_scholar":
                    papers = await self.search_semantic_scholar(query, max_results)
                elif source == "arxiv":
                    papers = await self.search_arxiv(query, max_results)
                else:
                    logger.warning(f"Unknown source: {source}")
                    continue
                
                all_papers.extend(papers)
            except Exception as e:
                logger.warning(f"Error searching {source}: {e}. Continuing with other sources.")
        
        # Remove duplicates based on DOI or title
        seen = set()
        unique_papers = []
        for paper in all_papers:
            # Safely get identifier - ensure it's a string
            if paper.doi and isinstance(paper.doi, str):
                identifier = paper.doi
            elif paper.title and isinstance(paper.title, str):
                identifier = paper.title.lower()
            else:
                # Skip papers without valid identifiers
                continue
                
            if identifier not in seen:
                seen.add(identifier)
                unique_papers.append(paper)
        
        # Filter recent papers (last 5 years) and sort by citations (descending), then year (descending)
        from datetime import datetime
        current_year = datetime.now().year
        recent_papers = [p for p in unique_papers if p.year >= current_year - 5]
        
        # If 5-year search returned no results, retry with 10-year window
        if len(recent_papers) == 0:
            logger.info(f"No papers found in last 5 years. Retrying with 10-year window...")
            
            # Clear and retry with 10-year filter
            all_papers = []
            for source in sources:
                try:
                    if source == "pubmed":
                        papers = await self.search_pubmed(query, max_results, year_range=10)
                    elif source == "semantic_scholar":
                        papers = await self.search_semantic_scholar(query, max_results, year_range=10)
                    elif source == "arxiv":
                        papers = await self.search_arxiv(query, max_results)
                    else:
                        continue
                    
                    all_papers.extend(papers)
                except Exception as e:
                    logger.warning(f"Error searching {source} (10-year): {e}. Continuing with other sources.")
            
            # Remove duplicates again
            seen = set()
            unique_papers = []
            for paper in all_papers:
                if paper.doi and isinstance(paper.doi, str):
                    identifier = paper.doi
                elif paper.title and isinstance(paper.title, str):
                    identifier = paper.title.lower()
                else:
                    continue
                    
                if identifier not in seen:
                    seen.add(identifier)
                    unique_papers.append(paper)
            
            recent_papers = [p for p in unique_papers if p.year >= current_year - 10]
        
        # If not enough recent papers, include older ones
        if len(recent_papers) < max_results:
            recent_papers = unique_papers
        
        recent_papers.sort(key=lambda p: (p.citations, p.year or 0), reverse=True)
        
        return recent_papers[:max_results * len(sources)]
    
    async def search_pubmed(self, query: str, max_results: int = 10, year_range: int = 5) -> List[Paper]:
        """
        Search PubMed for scientific papers.
        Uses NCBI E-utilities API (free, no API key needed).
        
        Args:
            query: Search query
            max_results: Maximum number of results
            year_range: Number of years to search back (default: 5)
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        papers = []
        
        try:
            # Step 1: Search for PMIDs (last N years)
            from datetime import datetime
            current_year = datetime.now().year
            year_filter = f"({current_year - year_range}:{current_year}[pdat])"
            
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": f"{query} AND {year_filter}",
                "retmax": max_results * 2,  # Get more to filter by citations
                "retmode": "json",
                "sort": "relevance"
            }
            
            async with self.session.get(search_url, params=search_params) as response:
                if response.status != 200:
                    logger.error(f"PubMed search failed: {response.status}")
                    return papers
                
                data = await response.json()
                pmids = data.get("esearchresult", {}).get("idlist", [])
            
            if not pmids:
                return papers
            
            # Step 2: Fetch paper details
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml"
            }
            
            async with self.session.get(fetch_url, params=fetch_params) as response:
                if response.status != 200:
                    logger.error(f"PubMed fetch failed: {response.status}")
                    return papers
                
                xml_data = await response.text()
                root = ET.fromstring(xml_data)
                
                for article in root.findall(".//PubmedArticle"):
                    try:
                        # Extract title
                        title_elem = article.find(".//ArticleTitle")
                        title = title_elem.text if title_elem is not None else "No title"
                        
                        # Extract abstract
                        abstract_texts = article.findall(".//AbstractText")
                        abstract = " ".join([a.text for a in abstract_texts if a.text]) if abstract_texts else ""
                        
                        # Extract authors
                        author_elems = article.findall(".//Author")
                        authors = []
                        for author in author_elems:
                            last_name = author.find("LastName")
                            first_name = author.find("ForeName")
                            if last_name is not None:
                                name = last_name.text
                                if first_name is not None:
                                    name = f"{first_name.text} {name}"
                                authors.append(name)
                        
                        # Extract year
                        year_elem = article.find(".//PubDate/Year")
                        year = int(year_elem.text) if year_elem is not None else None
                        
                        # Extract PMID
                        pmid_elem = article.find(".//PMID")
                        pmid = pmid_elem.text if pmid_elem is not None else None
                        
                        # Extract DOI
                        doi_elem = article.find(".//ArticleId[@IdType='doi']")
                        doi = doi_elem.text if doi_elem is not None else None
                        
                        # Note: PubMed doesn't provide citation counts directly
                        # Use year as proxy - newer papers in results are more relevant
                        
                        paper = Paper(
                            title=title,
                            authors=authors,
                            abstract=abstract,
                            year=year or 0,
                            id=pmid,
                            doi=doi,
                            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                            source="pubmed",
                            citations=0  # PubMed doesn't provide citation counts
                        )
                        papers.append(paper)
                        
                    except Exception as e:
                        logger.error(f"Error parsing PubMed article: {e}")
                        continue
            
            logger.info(f"Found {len(papers)} papers from PubMed (last {year_range} years)")
            
            # Sort by year (proxy for relevance since PubMed doesn't provide citations)
            papers.sort(key=lambda p: p.year, reverse=True)
            
            return papers[:max_results]  # Limit to requested max
            
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return papers
    
    async def search_semantic_scholar(self, query: str, max_results: int = 10, year_range: int = 5) -> List[Paper]:
        """
        Search Semantic Scholar for scientific papers.
        Free API with optional API key for higher rate limits.
        Handles rate limiting gracefully with retry logic.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            year_range: Number of years to search back (default: 5)
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        papers = []
        max_retries = 2
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                from datetime import datetime
                current_year = datetime.now().year
                
                search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {
                    "query": query,
                    "limit": max_results * 2,  # Get more to sort by citations
                    "fields": "title,authors,abstract,year,citationCount,externalIds,url",
                    "yearFilter": f"{current_year - year_range}-{current_year}",  # Configurable year range
                    "minCitationCount": 5  # At least 5 citations
                }
                
                headers = {}
                if self.semantic_scholar_api_key:
                    headers["x-api-key"] = self.semantic_scholar_api_key
                
                async with self.session.get(search_url, params=params, headers=headers) as response:
                    if response.status == 429:
                        # Rate limit hit
                        if attempt < max_retries:
                            wait_time = retry_delay * (attempt + 1)
                            logger.warning(f"Semantic Scholar rate limit (429). Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.warning(f"Semantic Scholar rate limit exceeded after {max_retries} retries. Skipping Semantic Scholar search.")
                            return papers
                    
                    if response.status != 200:
                        logger.warning(f"Semantic Scholar search returned status {response.status}. Skipping.")
                        return papers
                    
                    data = await response.json()
                    
                    # Sort by citations within the results
                    results = data.get("data", [])
                    results.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
                    
                    for item in results[:max_results]:  # Limit to requested max
                        try:
                            # Extract data
                            title = item.get("title", "No title")
                            abstract = item.get("abstract", "")
                            year = item.get("year")
                            citations = item.get("citationCount", 0)
                            
                            # Extract authors
                            authors = [author.get("name", "") for author in item.get("authors", [])]
                            
                            # Extract IDs
                            external_ids = item.get("externalIds", {})
                            doi = external_ids.get("DOI")
                            paper_id = external_ids.get("PubMed") or external_ids.get("ArXiv") or item.get("paperId")
                            
                            paper = Paper(
                                title=title,
                                authors=authors,
                                abstract=abstract,
                                year=year or 0,
                                doi=doi,
                                id=paper_id,
                                url=item.get("url", ""),
                                source="semantic_scholar",
                                citations=citations
                            )
                            papers.append(paper)
                            
                        except Exception as e:
                            logger.error(f"Error parsing Semantic Scholar paper: {e}")
                            continue
                    
                    logger.info(f"Found {len(papers)} papers from Semantic Scholar (last {year_range} years, min 5 citations, sorted by citation count)")
                    return papers
                    
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Semantic Scholar search error: {e}. Retrying...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.warning(f"Semantic Scholar search failed after {max_retries} retries: {e}")
                    return papers
        
        return papers
    
    async def search_arxiv(self, query: str, max_results: int = 10) -> List[Paper]:
        """
        Search arXiv for preprints.
        Free API, no authentication required.
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        papers = []
        
        try:
            # Add date filter for last 5 years
            from datetime import datetime
            current_year = datetime.now().year
            start_date = f"{current_year - 5}0101"
            
            search_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query} AND submittedDate:[{start_date} TO 20991231]",
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",  # Sort by recent first
                "sortOrder": "descending"
            }
            
            async with self.session.get(search_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"arXiv search failed: {response.status}")
                    return papers
                
                xml_data = await response.text()
                
                # Parse using BeautifulSoup if available, otherwise use ElementTree
                if BeautifulSoup:
                    soup = BeautifulSoup(xml_data, "xml")
                    entries = soup.find_all("entry")
                    
                    for entry in entries:
                        try:
                            title = entry.find("title").text.strip() if entry.find("title") else "No title"
                            abstract = entry.find("summary").text.strip() if entry.find("summary") else ""
                            
                            # Extract authors
                            authors = [author.find("name").text for author in entry.find_all("author") if author.find("name")]
                            
                            # Extract arXiv ID
                            id_elem = entry.find("id")
                            arxiv_url = id_elem.text if id_elem else ""
                            arxiv_id = arxiv_url.split("/")[-1] if arxiv_url else None
                            
                            # Extract year from published date
                            published = entry.find("published")
                            year = None
                            if published:
                                year_str = published.text[:4]
                                try:
                                    year = int(year_str)
                                except:
                                    pass
                            
                            paper = Paper(
                                title=title,
                                authors=authors,
                                abstract=abstract,
                                year=year or 0,
                                id=arxiv_id,
                                url=arxiv_url or "",
                                source="arxiv"
                            )
                            papers.append(paper)
                            
                        except Exception as e:
                            logger.error(f"Error parsing arXiv entry: {e}")
                            continue
                else:
                    # Fallback to ElementTree
                    root = ET.fromstring(xml_data)
                    ns = {'atom': 'http://www.w3.org/2005/Atom'}
                    
                    for entry in root.findall('atom:entry', ns):
                        try:
                            title = entry.find('atom:title', ns).text.strip()
                            abstract = entry.find('atom:summary', ns).text.strip()
                            
                            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
                            
                            arxiv_url = entry.find('atom:id', ns).text
                            arxiv_id = arxiv_url.split("/")[-1]
                            
                            published = entry.find('atom:published', ns).text[:4]
                            year = int(published) if published else 0
                            
                            paper = Paper(
                                title=title,
                                authors=authors,
                                abstract=abstract,
                                year=year,
                                id=arxiv_id,
                                url=arxiv_url,
                                source="arxiv"
                            )
                            papers.append(paper)
                            
                        except Exception as e:
                            logger.error(f"Error parsing arXiv entry: {e}")
                            continue
            
            logger.info(f"Found {len(papers)} papers from arXiv")
            
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
        
        return papers
    
    def format_papers_for_context(self, papers: List[Paper], max_papers: int = 5) -> str:
        """
        Format papers into a readable string for LLM context.
        
        Args:
            papers: List of papers
            max_papers: Maximum number of papers to include
        
        Returns:
            Formatted string with paper information
        """
        if not papers:
            return "No papers found."
        
        formatted = []
        for i, paper in enumerate(papers[:max_papers], 1):
            paper_text = f"\n[{i}] {paper.title}\n"
            paper_text += f"    Authors: {', '.join(paper.authors[:3])}"
            if len(paper.authors) > 3:
                paper_text += f" et al."
            paper_text += f"\n    Year: {paper.year or 'Unknown'}"
            if paper.citations:
                paper_text += f" | Citations: {paper.citations}"
            paper_text += f"\n    Source: {paper.source}"
            if paper.abstract:
                # Truncate abstract if too long
                abstract = paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract
                paper_text += f"\n    Abstract: {abstract}"
            if paper.url:
                paper_text += f"\n    URL: {paper.url}"
            formatted.append(paper_text)
        
        return "\n".join(formatted)
