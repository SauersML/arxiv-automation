import unittest
import os
import json
import shutil
import sys
from unittest.mock import patch, MagicMock, call
from datetime import datetime

# Add project root to sys.path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from run_once import main as run_once_main
from modules.arxiv import PaperData # For creating expected PaperData objects
import arxiv # For creating mock arxiv.Result objects

# Define constants for test-specific paths
TEST_BASE_DIR = os.path.join(project_root, "tests", "test_run_once_artifacts")
TEST_SUMMARIES_DIR = os.path.join(TEST_BASE_DIR, "paper_summaries")
TEST_SEEN_PAPERS_FILE = os.path.join(TEST_BASE_DIR, "seen_papers.json")

class TestRunOnce(unittest.TestCase):

    def setUp(self):
        """Set up for each test."""
        # Mock environment variables
        self.mock_env = patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "fake_anthropic_key",
            "SENDGRID_API_KEY": "fake_sendgrid_key",
            "SENDER_EMAIL": "sender@example.com",
            "RECIPIENT_EMAIL": "recipient@example.com",
            # Add other env vars if run_once.py or its imports require them
        })
        self.mock_env.start()

        # Clean up and create test directories
        if os.path.exists(TEST_BASE_DIR):
            shutil.rmtree(TEST_BASE_DIR)
        os.makedirs(TEST_SUMMARIES_DIR, exist_ok=True)
        
        # Ensure seen_papers.json does not exist from a previous run initially
        if os.path.exists(TEST_SEEN_PAPERS_FILE):
            os.remove(TEST_SEEN_PAPERS_FILE)

    def tearDown(self):
        """Clean up after each test."""
        self.mock_env.stop()
        if os.path.exists(TEST_BASE_DIR):
            shutil.rmtree(TEST_BASE_DIR)

    # Patch order: from bottom up for decorators
    @patch('modules.arxiv.arxiv.Client') # Mock the low-level arxiv library client
    @patch('modules.api_clients.AnthropicClient') # Mock the AnthropicClient class
    @patch('modules.email_sender.EmailSender') # Mock the EmailSender class
    @patch('modules.arxiv.ArxivClient.SUMMARIES_DIR', TEST_SUMMARIES_DIR) # Patch class attribute
    @patch('modules.arxiv.ArxivClient.SEEN_PAPERS_FILE', TEST_SEEN_PAPERS_FILE) # Patch class attribute
    def test_first_run_fetch_summarize_save(self,
                                             mock_arxiv_lib_client_class,
                                             mock_anthropic_class,
                                             mock_email_sender_class):
        # Get mock instances from the patched classes
        mock_arxiv_lib_instance = mock_arxiv_lib_client_class.return_value
        mock_anthropic_instance = mock_anthropic_class.return_value
        mock_email_sender_instance = mock_email_sender_class.return_value

        # Configure mock for arxiv.Client().results()
        paper1_id = "2301.00001"
        paper1_api_result = arxiv.Result(
            entry_id=f"http://arxiv.org/abs/{paper1_id}",
            title="Test Paper Title 1",
            authors=[arxiv.Result.Author("Author One")],
            summary="This is the abstract of test paper 1.",
            published=datetime.now(),
            pdf_url=f"http://arxiv.org/pdf/{paper1_id}.pdf",
            categories=['cs.AI']
        )
        mock_arxiv_lib_instance.results.return_value = iter([paper1_api_result])

        # Configure mock for AnthropicClient().send_request()
        expected_summary_html = "<h3>Summary</h3><p>Test Summary for Paper 1</p>" # Simplified HTML
        mock_anthropic_instance.send_request.return_value = "<summary>Test Summary for Paper 1</summary><methods>m</methods><contributions>c</contributions><limitations>l</limitations>"

        # Call the main function from run_once.py
        run_once_main()

        # Assertions
        mock_arxiv_lib_instance.results.assert_called_once()
        
        mock_anthropic_instance.send_request.assert_called_once()
        # Check if it was called with the correct PDF URL
        self.assertEqual(mock_anthropic_instance.send_request.call_args[1]['pdf_url'], paper1_api_result.pdf_url)

        # Assert summary file was created and contains correct data
        summary_filepath = os.path.join(TEST_SUMMARIES_DIR, f"{paper1_id}.json")
        self.assertTrue(os.path.exists(summary_filepath))
        
        with open(summary_filepath, 'r') as f:
            saved_summary_data = json.load(f)
        
        self.assertEqual(saved_summary_data['id'], paper1_id)
        self.assertEqual(saved_summary_data['title'], paper1_api_result.title)
        self.assertEqual(saved_summary_data['summary'], expected_summary_html) # ArxivClient saves the HTML formatted summary

        # Assert seen papers file was created and contains paper1_id
        self.assertTrue(os.path.exists(TEST_SEEN_PAPERS_FILE))
        with open(TEST_SEEN_PAPERS_FILE, 'r') as f:
            seen_papers_data = json.load(f)
        self.assertIn(paper1_id, seen_papers_data)

        # Assert email sender was called
        mock_email_sender_instance.send_email.assert_called_once()


    @patch('modules.arxiv.arxiv.Client')
    @patch('modules.api_clients.AnthropicClient')
    @patch('modules.email_sender.EmailSender')
    @patch('modules.arxiv.ArxivClient.SUMMARIES_DIR', TEST_SUMMARIES_DIR)
    @patch('modules.arxiv.ArxivClient.SEEN_PAPERS_FILE', TEST_SEEN_PAPERS_FILE)
    def test_second_run_load_no_summarize(self,
                                            mock_arxiv_lib_client_class,
                                            mock_anthropic_class,
                                            mock_email_sender_class):
        mock_arxiv_lib_instance = mock_arxiv_lib_client_class.return_value
        mock_anthropic_instance = mock_anthropic_class.return_value
        mock_email_sender_instance = mock_email_sender_class.return_value

        # --- Setup for the second run: Simulate paper1 already processed ---
        paper1_id = "2301.00001" # Same ID as in the first test
        pre_existing_summary_html = "<h3>Summary</h3><p>Pre-existing Summary for Paper 1</p>"
        
        # Create a PaperData dictionary as it would be saved
        paper1_data_for_json = PaperData(
            id=paper1_id,
            title="Test Paper Title 1 (from file)",
            url=f"http://arxiv.org/abs/{paper1_id}",
            pdf_url=f"http://arxiv.org/pdf/{paper1_id}.pdf",
            authors=["Author One"],
            abstract="Abstract from file.", # Abstract is part of PaperData
            summary=pre_existing_summary_html, # The crucial part
            published=datetime.now().isoformat(),
            categories=['cs.AI']
        ).to_dict()

        summary_filepath = os.path.join(TEST_SUMMARIES_DIR, f"{paper1_id}.json")
        with open(summary_filepath, 'w') as f:
            json.dump(paper1_data_for_json, f, indent=4)
        
        # Also, ensure it's in seen_papers.json
        with open(TEST_SEEN_PAPERS_FILE, 'w') as f:
            json.dump({paper1_id: datetime.now().isoformat()}, f)


        # Configure mock for arxiv.Client().results() to return the same paper
        # This simulates the paper still being found by the arXiv search query
        paper1_api_result = arxiv.Result(
            entry_id=f"http://arxiv.org/abs/{paper1_id}",
            title="Test Paper Title 1 (from API, should be ignored if loaded from file)",
            authors=[arxiv.Result.Author("Author One")],
            summary="This is the abstract of test paper 1 (from API).",
            published=datetime.now(),
            pdf_url=f"http://arxiv.org/pdf/{paper1_id}.pdf",
            categories=['cs.AI']
        )
        mock_arxiv_lib_instance.results.return_value = iter([paper1_api_result])
        
        # Call the main function from run_once.py
        run_once_main()

        # Assertions
        mock_arxiv_lib_instance.results.assert_called_once() # ArxivClient search is still performed

        # Crucially, AnthropicClient should NOT be called as summary exists
        mock_anthropic_instance.send_request.assert_not_called()

        # Assert email sender was still called
        mock_email_sender_instance.send_email.assert_called_once()
        
        # Verify the email sender received the paper with the pre-existing summary
        # The argument to send_email is `paper_summaries`, which is a list of PaperData
        call_args = mock_email_sender_instance.send_email.call_args
        self.assertIsNotNone(call_args)
        sent_paper_summaries = call_args[1]['paper_summaries'] # Accessing by kwarg name
        self.assertEqual(len(sent_paper_summaries), 1)
        self.assertEqual(sent_paper_summaries[0].id, paper1_id)
        self.assertEqual(sent_paper_summaries[0].summary, pre_existing_summary_html)
        self.assertEqual(sent_paper_summaries[0].title, "Test Paper Title 1 (from file)") # Title from loaded file

if __name__ == '__main__':
    unittest.main()
