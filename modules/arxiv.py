"""Improved module for interacting with arXiv API using the arxiv package."""

from dataclasses import dataclass, asdict
import os
import json
import arxiv
from datetime import datetime
from typing import List, Dict, Optional, Set

@dataclass
class PaperData:
    id: str
    title: str
    url: str
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    comment: Optional[str] = None
    published: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    summary: Optional[str] = None
    categories: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        """Convert PaperData instance to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PaperData':
        """Create PaperData instance from a dictionary."""
        return cls(**data)

class ArxivClient:
    """A client for interacting with the arXiv API with paper tracking."""
    
    SEEN_PAPERS_FILE = "seen_papers.json"
    SUMMARIES_DIR = "paper_summaries"
    
    def __init__(self):
        """Initialize the arXiv client."""
        self.client = arxiv.Client()
        self.seen_papers = self._load_seen_papers()
        if not os.path.exists(self.SUMMARIES_DIR):
            os.makedirs(self.SUMMARIES_DIR)
    
    def _load_seen_papers(self) -> Dict[str, str]:
        """
        Load the list of previously seen papers from disk.
        
        Returns:
            Dict[str, str]: Map of paper ID to last seen date
        """
        if os.path.exists(self.SEEN_PAPERS_FILE):
            try:
                with open(self.SEEN_PAPERS_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Error reading {self.SEEN_PAPERS_FILE}, starting fresh")
                return {}
        return {}
    
    def _save_seen_papers(self):
        """Save the list of seen papers to disk."""
        try:
            with open(self.SEEN_PAPERS_FILE, 'w') as f:
                json.dump(self.seen_papers, f)
        except IOError as e:
            print(f"Warning: Unable to save seen papers file: {e}")

    def _load_summary_from_file(self, paper_id: str) -> Optional[PaperData]:
        """Load a PaperData object from a JSON summary file if it exists."""
        summary_filepath = os.path.join(self.SUMMARIES_DIR, f"{paper_id}.json")
        if os.path.exists(summary_filepath):
            try:
                with open(summary_filepath, 'r') as f:
                    data = json.load(f)
                    return PaperData.from_dict(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Error reading or parsing summary file {summary_filepath}: {e}")
                return None
        return None

    def save_summary_to_file(self, paper_data: PaperData):
        """Save a PaperData object to a JSON summary file."""
        if not paper_data.summary or not paper_data.summary.strip():
            return

        summary_filepath = os.path.join(self.SUMMARIES_DIR, f"{paper_data.id}.json")
        try:
            with open(summary_filepath, 'w') as f:
                json.dump(paper_data.to_dict(), f, indent=4)
            print(f"Saved summary for paper {paper_data.id} to {summary_filepath}")
        except IOError as e:
            print(f"Warning: Unable to save summary file {summary_filepath}: {e}")
            
    def mark_papers_as_seen(self, papers: List[PaperData]): # Changed type hint from List[Dict]
        """
        Mark papers as seen to avoid duplicates in future searches.
        
        Args:
            papers: List of paper data objects
        """
        current_date = datetime.now().isoformat()
        for paper_data in papers: # Renamed paper to paper_data for clarity
            if paper_data:
                self.seen_papers[paper_data.id] = current_date
        self._save_seen_papers()
    
    def _construct_query(self, search_terms: Optional[List[str]] = None, categories: Optional[List[str]] = None) -> str:
        """
        Construct an arXiv query string with proper URL encoding.
        
        Args:
            search_terms: List of search terms to search for
            categories: List of arXiv categories (e.g., ['cs.AI', 'cs.LG'])
            
        Returns:
            str: Properly formatted query string for arXiv API
        """
        query_parts = []
        
        # Add categories with OR between them
        if categories:
            if len(categories) > 1:
                cats = " OR ".join([f"cat:{cat}" for cat in categories])
                query_parts.append(f"({cats})")
            else:
                query_parts.append(f"cat:{categories[0]}")
        
        # Add search terms with proper encoding
        if search_terms:
            if len(search_terms) > 1:
                # For multiple terms, use OR and encode quotes as %22
                encoded_terms = []
                for term in search_terms:
                    if " " in term:  # Multi-word terms need quotes
                        encoded_terms.append(f'%22{term}%22')
                    else:
                        encoded_terms.append(term)
                terms_str = " OR ".join(encoded_terms)
                query_parts.append(f"({terms_str})")
            else:
                # Single term
                term = search_terms[0]
                if " " in term:  # Multi-word term needs quotes
                    query_parts.append(f'%22{term}%22')
                else:
                    query_parts.append(term)
        
        # Join query parts with AND
        return " AND ".join(query_parts) if query_parts else ""
    
    def search_papers(self, search_terms: Optional[List[str]] = None, categories: Optional[List[str]] = None, 
                     max_results: int = 10, request_size: int = 20, timeout_seconds: float = 1.0) -> List[PaperData]:
        """
        Generic search for papers with configurable terms and categories.
        Makes individual requests and checks for duplicates.
        Continues until we have enough new papers or exhaust the search space.
        
        Args:
            search_terms: List of search terms to search for
            categories: List of arXiv categories (e.g., ['cs.AI', 'cs.LG'])
            max_results: Maximum number of new papers to return
            request_size: Number of papers to fetch in each request to arXiv
            timeout_seconds: Time to wait between requests to be polite to arXiv
            
        Returns:
            List[PaperData]: List of paper data objects
        """
        import time
        
        # Construct the query
        query = self._construct_query(search_terms, categories)
        if not query:
            print("No search terms or categories provided")
            return []
            
        print(f"Searching arXiv with query: {query}")
        
        found_papers = []
        seen_in_this_run = set()
        start_index = 0
        consecutive_seen_requests = 0
        max_consecutive_seen = 3  # Stop if we see 3 consecutive requests with all seen papers
        
        while len(found_papers) < max_results and consecutive_seen_requests < max_consecutive_seen:
            print(f"Making request {start_index // request_size + 1} (papers {start_index}-{start_index + request_size - 1})")
            
            # Create a new search for this batch
            search = arxiv.Search(
                query=query,
                max_results=request_size,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending  # Most recent first
            )
            
            # Set the start index for this request
            search.offset = start_index
            
            try:
                # Fetch this batch of papers
                results = list(self.client.results(search))
                
                if not results:
                    print("No more papers available from arXiv")
                    break
                
                # Check if we found any new papers in this batch
                new_papers_in_batch = 0
                
                for paper in results:
                    paper_id = paper.entry_id.split('/')[-1]
                    
                    # Skip if we've already seen this paper before
                    if paper_id in self.seen_papers or paper_id in seen_in_this_run:
                        print(f"Skipping already seen paper: {paper.title} (already processed or in seen_papers.json)")
                        continue
                    
                    # Try to load from summary file first
                    existing_summary_paper = self._load_summary_from_file(paper_id)
                    if existing_summary_paper and existing_summary_paper.summary: # Check if summary exists
                        print(f"Loaded paper {paper_id} from summary file: {existing_summary_paper.title}")
                        paper_data = existing_summary_paper
                    else:
                        # Convert the paper to our format and add it to the results
                        print(f"Fetching paper {paper_id} from arXiv: {paper.title}")
                        paper_data = self._convert_result(paper)

                    found_papers.append(paper_data)
                    seen_in_this_run.add(paper_id) # Mark as seen in this run to avoid re-processing in this session
                    new_papers_in_batch += 1
                    
                    print(f"Found new paper: {paper.title}")
                    
                    # Check if we have enough papers
                    if len(found_papers) >= max_results:
                        break
                
                # Track consecutive requests with no new papers
                if new_papers_in_batch == 0:
                    consecutive_seen_requests += 1
                    print(f"No new papers in this batch ({consecutive_seen_requests}/{max_consecutive_seen})")
                else:
                    consecutive_seen_requests = 0
                
                # Move to the next batch
                start_index += request_size
                
                # Wait between requests to be polite to arXiv
                if len(found_papers) < max_results and consecutive_seen_requests < max_consecutive_seen:
                    print(f"Waiting {timeout_seconds} seconds before next request...")
                    time.sleep(timeout_seconds)
                    
            except Exception as e:
                print(f"Error in request: {e}")
                break
        
        print(f"Search completed. Found {len(found_papers)} new papers.")
        
        # Mark all new papers as seen
        self.mark_papers_as_seen(found_papers)
        
        return found_papers
    
    def search_interpretability_papers(self, max_results: int = 10, request_size: int = 20, timeout_seconds: float = 1.0) -> List[PaperData]:
        """
        Search for interpretability papers, making individual requests and checking for duplicates.
        Continues until we have enough new papers or exhaust the search space.
        
        Args:
            max_results: Maximum number of new papers to return
            request_size: Number of papers to fetch in each request to arXiv
            timeout_seconds: Time to wait between requests to be polite to arXiv
            
        Returns:
            List[PaperData]: List of paper data objects
        """
        # Use the generic search function with interpretability-specific terms
        return self.search_papers(
            search_terms=["mechanistic interpretability"],
            categories=["cs.AI", "cs.LG", "cs.CL"],
            max_results=max_results,
            request_size=request_size,
            timeout_seconds=timeout_seconds
        )
    
    def search(self, search_terms=None, categories=None, max_results=10):
        """
        Search arXiv for papers matching the given criteria.
        
        Args:
            search_terms: Search terms or phrases
            categories: arXiv categories to search in
            max_results: Maximum number of results to return
            
        Returns:
            list: A list of dictionaries containing paper metadata
        """
        # Build a query string in the format from working_interp_search.py
        query_parts = []
        
        # Add categories with OR between them
        if categories:
            if isinstance(categories, list) and len(categories) > 0:
                cats = " OR ".join([f"cat:{cat}" for cat in categories])
                if len(categories) > 1:
                    query_parts.append(f"({cats})")
                else:
                    query_parts.append(cats)
        
        # Add search terms with quotes for exact match if multiple words
        if search_terms:
            if isinstance(search_terms, list):
                terms = []
                for term in search_terms:
                    if " " in term:  # If term contains spaces, use quotes
                        terms.append(f'"{term}"')
                    else:
                        terms.append(term)
                terms_str = " OR ".join(terms)
                query_parts.append(f"({terms_str})")
            else:
                if " " in search_terms:  # If term contains spaces, use quotes
                    query_parts.append(f'"{search_terms}"')
                else:
                    query_parts.append(search_terms)
        
        # Join query parts with AND
        query = " AND ".join(query_parts) if query_parts else ""
        
        print(f"Searching arXiv with query: {query}")
        
        # Create the search object
        search = arxiv.Search(
            query=query,
            max_results=100,  # Fetch more than we need to account for duplicates
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending  # Most recent first
        )
        
        # Fetch results
        results_generator = self.client.results(search)
        
        # Track papers we've found in this search session
        found_papers = []
        seen_in_this_run = set()
        
        # Get up to max_results papers we haven't seen before
        for paper in results_generator:
            paper_id = paper.entry_id.split('/')[-1]
            
            # Skip if we've already seen this paper before
            if paper_id in self.seen_papers or paper_id in seen_in_this_run:
                print(f"Skipping already seen paper: {paper.title} (already processed or in seen_papers.json)")
                continue
            
            # Try to load from summary file first
            existing_summary_paper = self._load_summary_from_file(paper_id)
            if existing_summary_paper and existing_summary_paper.summary: # Check if summary exists
                print(f"Loaded paper {paper_id} from summary file: {existing_summary_paper.title}")
                paper_data = existing_summary_paper
            else:
                # Convert the paper to our format and add it to the results
                print(f"Fetching paper {paper_id} from arXiv: {paper.title}")
                paper_data = self._convert_result(paper) # paper_dict renamed to paper_data

            found_papers.append(paper_data) # paper_dict renamed to paper_data
            seen_in_this_run.add(paper_id) # Mark as seen in this run
            
            # Check if we have enough papers
            if len(found_papers) >= max_results:
                break
        
        # Mark all new papers as seen
        self.mark_papers_as_seen(found_papers)
        
        return found_papers
    
    def get_paper_by_id(self, paper_id):
        """
        Retrieve a specific paper by its arXiv ID.
        
        Args:
            paper_id: The arXiv ID of the paper
            
        Returns:
            PaperData: A PaperData object or None
        """
        # Try to load from summary file first
        existing_summary_paper = self._load_summary_from_file(paper_id)
        if existing_summary_paper and existing_summary_paper.summary: # Check if summary exists
            print(f"Loaded paper {paper_id} from summary file.")
             # Ensure it's marked in seen_papers if loaded from summary
            if paper_id not in self.seen_papers:
                self.seen_papers[paper_id] = datetime.now().isoformat()
                self._save_seen_papers()
            return existing_summary_paper

        print(f"Fetching paper {paper_id} from arXiv as no local summary found.")
        search = arxiv.Search(id_list=[paper_id])
        try:
            result = next(self.client.results(search))
            paper_data = self._convert_result(result)
            # Mark as seen if fetched from arXiv
            if paper_id not in self.seen_papers:
                 self.seen_papers[paper_id] = datetime.now().isoformat()
                 self._save_seen_papers()
            return paper_data
        except StopIteration:
            print(f"Paper with ID {paper_id} not found on arXiv.")
            return None
    
    def get_pdf_url(self, paper_id):
        """
        Get the PDF URL for a paper.
        
        Args:
            paper_id: The arXiv ID of the paper
            
        Returns:
            str: The URL to the PDF
        """
        paper = self.get_paper_by_id(paper_id)
        if paper and paper.pdf_url:
            return paper.pdf_url
        
        raise ValueError(f"Paper with ID {paper_id} not found or has no PDF URL.")
    
    def _convert_result(self, result):
        """
        Convert an arxiv.Result object to a standardized dictionary.
        
        Args:
            result: An arxiv.Result object
            
        Returns:
            PaperData: A PaperData object
        """
        # Extract the arXiv ID from the entry ID URL
        arxiv_id = result.entry_id.split('/')[-1]
        
        # Get PDF URL and ensure it uses HTTPS
        pdf_url = result.pdf_url if hasattr(result, 'pdf_url') else f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if pdf_url.startswith('http:'):
            pdf_url = 'https' + pdf_url[4:]

        paper = PaperData(
            id=arxiv_id,
            categories=result.categories,
            title=result.title,
            url=result.entry_id,
            published=result.published.isoformat() if hasattr(result, 'published') else None,
            authors=[author.name for author in result.authors],
            abstract=result.summary,
            keywords=result.categories,
            pdf_url=pdf_url
        )
        
        # Add DOI if available
        if hasattr(result, 'doi'):
            paper.doi = result.doi
        
        # Add comment if available
        if hasattr(result, 'comment'):
            paper.comment = result.comment
        
        # The summary field will be None initially unless loaded from a file.
        # It will be populated later by a different part of the pipeline.
            
        return paper