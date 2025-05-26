import re
from typing import List, Optional

from tqdm import tqdm
from modules.api_clients import AnthropicClient
from modules.arxiv import PaperData


def extract_xml_content(text: str) -> dict[str, Optional[str]]:
    """
    Extract content between specific XML tags from LLM output.
    
    Args:
        text (str): The input text containing XML tags
        
    Returns:
        Dict[str, Optional[str]]: Dictionary with tag names as keys and content as values
    """
    tags = ['summary', 'methods', 'contributions', 'limitations']
    results = {}
    
    for tag in tags:
        # Create regex pattern to match content between opening and closing tags
        pattern = f'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            # Strip whitespace and store the content
            results[tag] = match.group(1).strip()
        else:
            results[tag] = None
    
    return results

def format_summary_html(results: dict[str, Optional[str]]) -> str:
    """
    Format the extracted XML content into HTML.
    
    Args:
        results (Dict[str, Optional[str]]): Dictionary with tag names as keys and content as values
        
    Returns:
        str: HTML formatted string
    """
    html_content = []
    
    for tag, content in results.items():
        if content:
            html_content.append(f"<h3>{tag.capitalize()}</h3>")
            html_content.append(f"<p>{content}</p>")
    
    return "\n".join(html_content)


class PaperSummarizer:
    
    MAX_REQ_BYTES = 32 * 1000000  # 32MB
    MAX_REQ_PAGES = 100

    def __init__(self, client: AnthropicClient):
        self._client = client

    def _generate_summary_prompt(self) -> str:
        """
        Generate a prompt for the AI model to summarize a paper.
        
        Returns:
            str: The prompt for the AI model.
        """
        return """
        I'm sharing a research paper with you as a PDF attachment. Please provide a comprehensive summary of this paper.
        
        Please analyze the full PDF and provide:
        
        1. A concise summary (250-300 words) of the paper's main contributions and findings
        2. The key methodologies used
        3. The key contributions
        4. Any notable limitations mentioned
        
        Focus especially on the paper's relevance to interpretability research, mechanistic interpretability, and explainable AI. 

        Output your response in the following XML tags:
        <summary></summary>
        <methods></methods>
        <contributions></contributions>
        <limitations></limitations>

        Plan your response outside of the XML tags before writing the final output.
        """

    def summarize_paper(self, pdf_url: str) -> str:
        """
        Summarize a single paper given its PDF URL.
        
        Args:
            pdf_url (str): URL to the paper's PDF.
        
        Returns:
            str: HTML formatted summary of the paper.
        """
        # Generate prompt for the AI model
        prompt = self._generate_summary_prompt()
        
        # Check if the prompt is too large
        if len(prompt.encode('utf-8')) > self.MAX_REQ_BYTES:
            raise ValueError("Prompt exceeds maximum request size.")
        
        # Send request to the AI model with PDF URL
        response = self._client.send_request(
            prompt=prompt, 
            pdf_url=pdf_url,
            max_tokens_to_sample=5000
        )

        processed_response = extract_xml_content(response)
        return format_summary_html(processed_response)
    
    def summarize_papers(self, papers: List[PaperData]) -> List[PaperData]:
        """
        Summarize multiple papers.
        
        Args:
            papers (List[PaperData]): List of paper data objects.
        
        Returns:
            List[PaperData]: List of papers with summaries added.
        """
        summarized_papers = []
        for paper in tqdm(papers, desc="Summarizing papers", unit="paper"):
            try:
                # Check if summary already exists
                if paper.summary and paper.summary.strip():
                    print(f"Skipping summarization for paper {paper.id} as summary already exists.")
                    summarized_papers.append(paper)
                    continue

                if paper.pdf_url:
                    summary = self.summarize_paper(paper.pdf_url)
                    paper.summary = summary
                    summarized_papers.append(paper)
                else:
                    print(f"Skipping paper {paper.id} - no PDF URL, cannot summarize.")
            except Exception as e:
                print(f"Error summarizing paper {paper.id}: {e}")
                
        return summarized_papers