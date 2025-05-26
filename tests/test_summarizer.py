import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys

# Add project root to sys.path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.summarizer import PaperSummarizer, format_summary_html, extract_xml_content
from modules.arxiv import PaperData
from modules.api_clients import AnthropicClient # Needed for type hinting if not mocking constructor directly

class TestPaperSummarizer(unittest.TestCase):

    def setUp(self):
        """Set up for each test."""
        self.mock_anthropic_client = MagicMock(spec=AnthropicClient)
        self.summarizer = PaperSummarizer(client=self.mock_anthropic_client)

    @patch('builtins.print')
    def test_skip_paper_with_existing_summary(self, mock_print):
        """Test that a paper with an existing summary is skipped."""
        paper_with_summary = PaperData(
            id="paper1",
            title="Paper With Summary",
            summary="This is an existing summary.",
            pdf_url="http://example.com/paper1.pdf" # pdf_url needed to not be skipped for other reasons
        )
        
        results = self.summarizer.summarize_papers([paper_with_summary])
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].summary, "This is an existing summary.")
        self.mock_anthropic_client.send_request.assert_not_called()
        mock_print.assert_any_call("Skipping summarization for paper paper1 as summary already exists.")

    @patch('builtins.print')
    def test_summarize_paper_without_existing_summary(self, mock_print):
        """Test that a paper without an existing summary is summarized."""
        paper_no_summary = PaperData(
            id="paper2",
            title="Paper Without Summary",
            pdf_url="http://example.com/paper2.pdf",
            summary=None
        )
        
        mock_response_xml = "<summary>New summary from API</summary><methods>Test methods</methods><contributions>Test contributions</contributions><limitations>Test limitations</limitations>"
        self.mock_anthropic_client.send_request.return_value = mock_response_xml
        
        results = self.summarizer.summarize_papers([paper_no_summary])
        
        self.assertEqual(len(results), 1)
        self.assertIn("New summary from API", results[0].summary) # format_summary_html will wrap this
        self.mock_anthropic_client.send_request.assert_called_once()
        # Check arguments of send_request if necessary (e.g. prompt, pdf_url)
        args, kwargs = self.mock_anthropic_client.send_request.call_args
        self.assertEqual(kwargs.get('pdf_url'), "http://example.com/paper2.pdf")


    @patch('builtins.print')
    def test_summarize_mixed_list_of_papers(self, mock_print):
        """Test with a mixed list of papers: one with summary, one without, one without PDF."""
        paper_A_with_summary = PaperData(
            id="paperA",
            title="Paper A (has summary)",
            summary="Existing summary for Paper A",
            pdf_url="http://example.com/paperA.pdf"
        )
        paper_B_needs_summary = PaperData(
            id="paperB",
            title="Paper B (needs summary)",
            pdf_url="http://example.com/paperB.pdf",
            summary=None
        )
        paper_C_no_pdf = PaperData(
            id="paperC",
            title="Paper C (no PDF)",
            pdf_url=None,
            summary=None
        )
        
        papers_in = [paper_A_with_summary, paper_B_needs_summary, paper_C_no_pdf]
        
        mock_response_xml_for_B = "<summary>Summary for Paper B</summary><methods>B methods</methods><contributions>B contribs</contributions><limitations>B limits</limitations>"
        self.mock_anthropic_client.send_request.return_value = mock_response_xml_for_B
        
        results = self.summarizer.summarize_papers(papers_in)
        
        # Expected: paperA (skipped, existing summary), paperB (summarized), paperC (skipped, no PDF)
        self.assertEqual(len(results), 2) # paperC is not added to results
        
        found_paper_A = next((p for p in results if p.id == "paperA"), None)
        found_paper_B = next((p for p in results if p.id == "paperB"), None)
        
        self.assertIsNotNone(found_paper_A)
        self.assertEqual(found_paper_A.summary, "Existing summary for Paper A")
        
        self.assertIsNotNone(found_paper_B)
        self.assertIn("Summary for Paper B", found_paper_B.summary)
        
        # API should only be called for paper_B
        self.mock_anthropic_client.send_request.assert_called_once_with(
            prompt=unittest.mock.ANY, # or specific prompt if needed
            pdf_url="http://example.com/paperB.pdf",
            max_tokens_to_sample=5000
        )
        
        # Check print outputs
        mock_print.assert_any_call("Skipping summarization for paper paperA as summary already exists.")
        mock_print.assert_any_call("Skipping paper paperC - no PDF URL, cannot summarize.")


    def test_extract_xml_content(self):
        """Test the helper function extract_xml_content."""
        xml_text = """
        <summary>This is the summary.</summary>
        <methods>Method A, Method B.</methods>
        <contributions>Contribution 1. Contribution 2.</contributions>
        <limitations>Limitation X.</limitations>
        Some other text here.
        """
        expected = {
            'summary': 'This is the summary.',
            'methods': 'Method A, Method B.',
            'contributions': 'Contribution 1. Contribution 2.',
            'limitations': 'Limitation X.'
        }
        self.assertEqual(extract_xml_content(xml_text), expected)

    def test_extract_xml_content_missing_tags(self):
        """Test extract_xml_content with some tags missing."""
        xml_text = "<summary>Only summary here.</summary>"
        expected = {
            'summary': 'Only summary here.',
            'methods': None,
            'contributions': None,
            'limitations': None
        }
        self.assertEqual(extract_xml_content(xml_text), expected)

    def test_format_summary_html(self):
        """Test the helper function format_summary_html."""
        extracted_content = {
            'summary': 'This is the summary.',
            'methods': 'Method A, Method B.',
            'contributions': None, # Test with a missing item
            'limitations': 'Limitation X.'
        }
        html_output = format_summary_html(extracted_content)
        self.assertIn("<h3>Summary</h3>", html_output)
        self.assertIn("<p>This is the summary.</p>", html_output)
        self.assertIn("<h3>Methods</h3>", html_output)
        self.assertIn("<p>Method A, Method B.</p>", html_output)
        self.assertNotIn("<h3>Contributions</h3>", html_output) # Should not be present if content is None
        self.assertIn("<h3>Limitations</h3>", html_output)
        self.assertIn("<p>Limitation X.</p>", html_output)

if __name__ == '__main__':
    unittest.main()
