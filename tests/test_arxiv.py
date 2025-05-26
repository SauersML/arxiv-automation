import unittest
import os
import json
import shutil
import sys
from unittest.mock import patch, MagicMock, call

# Add project root to sys.path to allow importing modules
# This is often necessary when running tests directly from the tests directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import arxiv # For live tests and its specific exceptions
import requests # For requests.exceptions.ConnectionError

from modules.arxiv import ArxivClient, PaperData

class TestArxivClientSummaries(unittest.TestCase):
    TEST_SUMMARIES_DIR = os.path.join(project_root, "tests", "test_paper_summaries")
    TEST_LIVE_PAPER_ID = "1605.08386" # A known, stable paper (e.g., "Batch Normalization")
    ORIGINAL_SUMMARIES_DIR = None
    ORIGINAL_SEEN_PAPERS_FILE = None
    TEST_SEEN_PAPERS_FILE = os.path.join(project_root, "tests", "test_seen_papers.json")


    @classmethod
    def setUpClass(cls):
        """
        Store original class variables from ArxivClient that will be patched.
        """
        cls.ORIGINAL_SUMMARIES_DIR = ArxivClient.SUMMARIES_DIR
        cls.ORIGINAL_SEEN_PAPERS_FILE = ArxivClient.SEEN_PAPERS_FILE

    @classmethod
    def tearDownClass(cls):
        """
        Restore original class variables to ArxivClient.
        """
        ArxivClient.SUMMARIES_DIR = cls.ORIGINAL_SUMMARIES_DIR
        ArxivClient.SEEN_PAPERS_FILE = cls.ORIGINAL_SEEN_PAPERS_FILE

    def setUp(self):
        """
        Set up for each test:
        - Patch ArxivClient.SUMMARIES_DIR to use the test-specific directory.
        - Patch ArxivClient.SEEN_PAPERS_FILE to use a test-specific file.
        - Create the test summaries directory.
        - Instantiate ArxivClient.
        """
        ArxivClient.SUMMARIES_DIR = self.TEST_SUMMARIES_DIR
        ArxivClient.SEEN_PAPERS_FILE = self.TEST_SEEN_PAPERS_FILE

        if os.path.exists(self.TEST_SUMMARIES_DIR):
            shutil.rmtree(self.TEST_SUMMARIES_DIR)
        os.makedirs(self.TEST_SUMMARIES_DIR, exist_ok=True)

        # Remove test seen_papers.json if it exists from a previous run
        if os.path.exists(self.TEST_SEEN_PAPERS_FILE):
            os.remove(self.TEST_SEEN_PAPERS_FILE)

        self.arxiv_client = ArxivClient()
        # Ensure the client's internal state for summaries_dir is also updated
        self.arxiv_client.SUMMARIES_DIR = self.TEST_SUMMARIES_DIR
        self.arxiv_client.SEEN_PAPERS_FILE = self.TEST_SEEN_PAPERS_FILE


    def tearDown(self):
        """
        Clean up after each test:
        - Remove the test summaries directory and its contents.
        - Remove the test seen_papers.json file.
        """
        if os.path.exists(self.TEST_SUMMARIES_DIR):
            shutil.rmtree(self.TEST_SUMMARIES_DIR)
        if os.path.exists(self.TEST_SEEN_PAPERS_FILE):
            os.remove(self.TEST_SEEN_PAPERS_FILE)

    def test_paper_data_serialization(self):
        """Test PaperData.to_dict() and PaperData.from_dict() methods."""
        original_data = PaperData(
            id="test001",
            title="Test Paper Title",
            url="http://example.com/test001",
            pdf_url="http://example.com/pdf/test001",
            authors=["Author One", "Author Two"],
            abstract="This is a test abstract.",
            summary="This is a test summary.",
            keywords=["test", "serialization"],
            published="2023-01-01T00:00:00Z",
            categories=["cs.AI", "cs.LG"]
        )
        
        data_dict = original_data.to_dict()
        self.assertIsInstance(data_dict, dict)
        
        reconstructed_data = PaperData.from_dict(data_dict)
        self.assertIsInstance(reconstructed_data, PaperData)
        self.assertEqual(original_data, reconstructed_data)
        # Check a few fields explicitly
        self.assertEqual(original_data.id, reconstructed_data.id)
        self.assertEqual(original_data.summary, reconstructed_data.summary)
        self.assertEqual(original_data.authors, reconstructed_data.authors)

    def test_save_summary_to_file(self):
        """Test saving a PaperData object with a summary to a file."""
        paper_data = PaperData(
            id="save002",
            title="Paper to Save",
            url="http://example.com/save002",
            summary="This summary should be saved to a file.",
            authors=["Saver One"],
            published="2023-01-02T00:00:00Z"
        )
        
        self.arxiv_client.save_summary_to_file(paper_data)
        
        expected_filepath = os.path.join(self.TEST_SUMMARIES_DIR, f"{paper_data.id}.json")
        self.assertTrue(os.path.exists(expected_filepath), f"File not found: {expected_filepath}")
        
        with open(expected_filepath, 'r') as f:
            saved_dict = json.load(f)
        
        # Convert original paper_data to dict for comparison, as it might have more Nones
        # than what's saved if from_dict has defaults.
        reloaded_paper_data = PaperData.from_dict(saved_dict)
        self.assertEqual(paper_data.id, reloaded_paper_data.id)
        self.assertEqual(paper_data.title, reloaded_paper_data.title)
        self.assertEqual(paper_data.summary, reloaded_paper_data.summary)
        self.assertEqual(paper_data.to_dict(), saved_dict) # Compare dicts directly

    def test_save_summary_to_file_no_summary(self):
        """Test that save_summary_to_file does not save if summary is None or empty."""
        paper_data_no_summary = PaperData(
            id="no_summary_003",
            title="Paper Without Summary",
            url="http://example.com/no_summary_003",
            summary=None # Explicitly None
        )
        paper_data_empty_summary = PaperData(
            id="empty_summary_004",
            title="Paper With Empty Summary",
            url="http://example.com/empty_summary_004",
            summary="   " # Whitespace only
        )

        self.arxiv_client.save_summary_to_file(paper_data_no_summary)
        expected_filepath_no_summary = os.path.join(self.TEST_SUMMARIES_DIR, f"{paper_data_no_summary.id}.json")
        self.assertFalse(os.path.exists(expected_filepath_no_summary), "File should not be created if summary is None.")

        paper_data_truly_empty_summary = PaperData(
            id="truly_empty_summary_004b",
            title="Paper With Truly Empty Summary",
            url="http://example.com/truly_empty_summary_004b",
            summary="" 
        )
        self.arxiv_client.save_summary_to_file(paper_data_truly_empty_summary)
        expected_filepath_truly_empty_summary = os.path.join(self.TEST_SUMMARIES_DIR, f"{paper_data_truly_empty_summary.id}.json")
        self.assertFalse(os.path.exists(expected_filepath_truly_empty_summary), "File should not be created if summary is an empty string.")

        # This one should NOT be saved as "   ".strip() is False
        self.arxiv_client.save_summary_to_file(paper_data_empty_summary) 
        expected_filepath_empty_summary = os.path.join(self.TEST_SUMMARIES_DIR, f"{paper_data_empty_summary.id}.json")
        self.assertFalse(os.path.exists(expected_filepath_empty_summary), "File should NOT be created if summary is only whitespace.")


    def test_load_summary_from_file(self):
        """Test loading a PaperData object from a JSON summary file."""
        sample_paper_dict = {
            "id": "load003",
            "title": "Loaded Paper",
            "url": "http://example.com/load003",
            "summary": "This summary was loaded from a file.",
            "authors": ["Loader One"],
            "published": "2023-01-03T00:00:00Z",
            "pdf_url": None, "doi": None, "comment": None, 
            "keywords": None, "categories": None
        }
        
        # Manually create the JSON file
        filepath = os.path.join(self.TEST_SUMMARIES_DIR, f"{sample_paper_dict['id']}.json")
        with open(filepath, 'w') as f:
            json.dump(sample_paper_dict, f)
            
        loaded_paper = self.arxiv_client._load_summary_from_file(sample_paper_dict['id'])
        
        self.assertIsNotNone(loaded_paper)
        self.assertIsInstance(loaded_paper, PaperData)
        self.assertEqual(sample_paper_dict['id'], loaded_paper.id)
        self.assertEqual(sample_paper_dict['title'], loaded_paper.title)
        self.assertEqual(sample_paper_dict['summary'], loaded_paper.summary)
        self.assertEqual(sample_paper_dict['authors'], loaded_paper.authors)
        # Ensure all fields are loaded correctly by comparing dicts
        self.assertEqual(sample_paper_dict, loaded_paper.to_dict())

    def test_load_summary_from_file_not_exists(self):
        """Test loading when the summary file does not exist."""
        loaded_paper = self.arxiv_client._load_summary_from_file("non_existent_id_004")
        self.assertIsNone(loaded_paper)

    @patch('builtins.print') # To suppress "Warning: Error reading or parsing..."
    def test_load_summary_from_file_malformed_json(self, mock_print):
        """Test loading when the JSON file is malformed."""
        malformed_id = "malformed005"
        filepath = os.path.join(self.TEST_SUMMARIES_DIR, f"{malformed_id}.json")
        with open(filepath, 'w') as f:
            f.write("{'id': 'malformed005', 'title': 'Malformed JSON', ...") # Invalid JSON
            
        loaded_paper = self.arxiv_client._load_summary_from_file(malformed_id)
        self.assertIsNone(loaded_paper)
        mock_print.assert_called_once() # Check if the warning was printed

    @patch('modules.arxiv.arxiv.Client') # Mock the actual arxiv.Client
    def test_get_paper_by_id_integration(self, MockArxivLowLevelClient):
        """
        Test integration of summary loading with get_paper_by_id.
        Mocks the low-level arxiv API client.
        """
        # Setup mock for the low-level arxiv.Client().results()
        mock_arxiv_search_instance = MagicMock()
        
        # This mock_paper_result needs to have attributes that _convert_result expects
        mock_api_paper_result = MagicMock()
        mock_api_paper_result.entry_id = "http://arxiv.org/abs/api_paper_id_001"
        mock_api_paper_result.title = "API Paper Title"
        mock_api_paper_result.authors = [MagicMock(name="Author A")]
        mock_api_paper_result.summary = "API abstract."
        mock_api_paper_result.published = MagicMock() # Needs to be a datetime object
        mock_api_paper_result.published.isoformat.return_value = "2023-01-05T00:00:00Z"
        mock_api_paper_result.categories = ["cs.TEST"]
        mock_api_paper_result.pdf_url = "http://arxiv.org/pdf/api_paper_id_001"
        # Ensure doi and comment are not present or None to avoid AttributeError
        del mock_api_paper_result.doi 
        del mock_api_paper_result.comment


        # Configure the client instance's results method
        # This is a bit tricky because ArxivClient instantiates its own arxiv.Client
        # So, MockArxivLowLevelClient is the class, its return_value is the instance
        mock_low_level_client_instance = MockArxivLowLevelClient.return_value
        mock_low_level_client_instance.results.return_value = iter([mock_api_paper_result])


        # --- Scenario 1: Paper not in local cache, fetched from API ---
        paper_id_api = "api_paper_id_001"
        print(f"Testing get_paper_by_id for {paper_id_api} (expect API call)")
        
        # Ensure no local file exists for this ID
        api_paper_filepath = os.path.join(self.TEST_SUMMARIES_DIR, f"{paper_id_api}.json")
        if os.path.exists(api_paper_filepath): os.remove(api_paper_filepath)

        retrieved_paper_api = self.arxiv_client.get_paper_by_id(paper_id_api)
        
        # Check that arxiv.Search was called with the ID
        # The actual call is self.client.results(search_object)
        # where search_object = arxiv.Search(id_list=[paper_id])
        # So we check if self.client.results was called.
        mock_low_level_client_instance.results.assert_called_once()
        
        self.assertIsNotNone(retrieved_paper_api)
        self.assertEqual(paper_id_api, retrieved_paper_api.id)
        self.assertEqual("API Paper Title", retrieved_paper_api.title)
        # This paper should now be in seen_papers.json
        self.assertTrue(os.path.exists(self.TEST_SEEN_PAPERS_FILE))


        # --- Scenario 2: Paper exists in local cache (summary file) ---
        paper_id_local = "local_paper_id_002"
        local_paper_data = PaperData(
            id=paper_id_local,
            title="Local Paper Title",
            url=f"http://arxiv.org/abs/{paper_id_local}",
            summary="This is a locally cached summary.",
            authors=["Local Author"],
            published="2023-01-04T00:00:00Z",
            categories=["cs.LOCAL"],
            pdf_url=f"http://arxiv.org/pdf/{paper_id_local}"
        )
        # Save this paper to the test summaries directory
        self.arxiv_client.save_summary_to_file(local_paper_data)
        print(f"Testing get_paper_by_id for {paper_id_local} (expect local load, no API call)")

        # Reset the mock for the API call check
        mock_low_level_client_instance.results.reset_mock()
        
        retrieved_paper_local = self.arxiv_client.get_paper_by_id(paper_id_local)
        
        mock_low_level_client_instance.results.assert_not_called() # API should NOT be called
        
        self.assertIsNotNone(retrieved_paper_local)
        self.assertEqual(local_paper_data.id, retrieved_paper_local.id)
        self.assertEqual(local_paper_data.title, retrieved_paper_local.title)
        self.assertEqual(local_paper_data.summary, retrieved_paper_local.summary)
        self.assertEqual(local_paper_data, retrieved_paper_local) # Full object comparison
        # This paper should also be in seen_papers.json if loaded from summary
        # The logic in get_paper_by_id adds it to seen_papers if loaded from summary
        # and not already there. Let's verify this:
        seen_papers_content = {}
        if os.path.exists(self.TEST_SEEN_PAPERS_FILE):
            with open(self.TEST_SEEN_PAPERS_FILE, 'r') as f_seen:
                seen_papers_content = json.load(f_seen)
        self.assertIn(paper_id_local, seen_papers_content)


    @patch('modules.arxiv.arxiv.Client')
    def test_search_papers_integration_loads_from_summary(self, MockArxivLowLevelClient):
        """
        Test that search_papers attempts to load from summary files first.
        """
        mock_low_level_client_instance = MockArxivLowLevelClient.return_value

        # Paper 1: Will be available locally
        local_paper_id = "search_local_001"
        local_paper_data = PaperData(
            id=local_paper_id, title="Local Search Result", url=f"http://arxiv.org/abs/{local_paper_id}",
            summary="Summary for local search result.", authors=["Search Local Author"],
            published="2023-10-01T00:00:00Z", categories=["cs.CACHE"]
        )
        self.arxiv_client.save_summary_to_file(local_paper_data)

        # Paper 2: Will be fetched from API (mocked)
        api_paper_id = "search_api_002"
        mock_api_result = MagicMock()
        mock_api_result.entry_id = f"http://arxiv.org/abs/{api_paper_id}"
        mock_api_result.title = "API Search Result"
        mock_api_result.authors = [MagicMock(name="Search API Author")]
        mock_api_result.summary = "API abstract for search."
        mock_api_result.published = MagicMock()
        mock_api_result.published.isoformat.return_value = "2023-10-02T00:00:00Z"
        mock_api_result.categories = ["cs.API"]
        mock_api_result.pdf_url = f"http://arxiv.org/pdf/{api_paper_id}"
        del mock_api_result.doi
        del mock_api_result.comment

        # Mock arxiv.Search results to return both paper IDs conceptually
        # The actual search result from client.results will be just the API one,
        # as the local one should be picked up before the API call for it.
        # However, arxiv.Search is complex. The logic in ArxivClient.search_papers is:
        # 1. search = arxiv.Search(...)
        # 2. results = list(self.client.results(search))
        # 3. For paper in results:
        # 4.   paper_id = ...
        # 5.   if paper_id in seen_papers or paper_id in seen_in_this_run: skip
        # 6.   existing_summary_paper = self._load_summary_from_file(paper_id)
        # 7.   if existing_summary_paper and existing_summary_paper.summary: use it
        # 8.   else: paper_data = self._convert_result(paper)
        #
        # So, the mock needs to simulate `self.client.results(search)` returning a list
        # that would include *both* papers if neither were local.
        # Then we check if the API was called only for the one not available locally.

        # Let's simulate the API returning both. The code should then only "fetch" the second one.
        # The arxiv.Search object itself is not directly called for results in ArxivClient,
        # rather self.client.results(search_object) is.
        
        # We need a mock that can be used in arxiv.Search(id_list=...) and for query search
        # For this test, we are doing a query search.
        
        # Mock the response from self.client.results(search_object)
        # The search_object will have query="cat:cs.TEST" and max_results=10 (default in ArxivClient.search_papers via search_interpretability_papers)
        # Let's make the API return only the api_paper_id. The local one should be found without an API call.
        # The challenge is, search_papers fetches a batch, then iterates.
        # If local_paper_id was part of that batch, it would be loaded from file.
        
        # Simpler: Let the API return a list containing metadata for BOTH papers.
        # The code should then load local_paper_id from file and api_paper_id via _convert_result.
        
        # This mock represents the paper that would be constructed by `arxiv.Result`
        # for the local paper, if it were fetched from the API.
        mock_local_paper_as_api_result = MagicMock()
        mock_local_paper_as_api_result.entry_id = f"http://arxiv.org/abs/{local_paper_id}"
        mock_local_paper_as_api_result.title = local_paper_data.title # Match to avoid confusion
        mock_local_paper_as_api_result.authors = [MagicMock(name=a) for a in local_paper_data.authors]
        mock_local_paper_as_api_result.summary = "Abstract for local paper if fetched from API"
        mock_local_paper_as_api_result.published = MagicMock()
        mock_local_paper_as_api_result.published.isoformat.return_value = local_paper_data.published
        mock_local_paper_as_api_result.categories = local_paper_data.categories
        mock_local_paper_as_api_result.pdf_url = local_paper_data.pdf_url
        del mock_local_paper_as_api_result.doi
        del mock_local_paper_as_api_result.comment


        mock_low_level_client_instance.results.return_value = iter([mock_local_paper_as_api_result, mock_api_result])
        
        # Patch _convert_result to spy on it
        with patch.object(self.arxiv_client, '_convert_result', wraps=self.arxiv_client._convert_result) as mock_convert_result:
            search_results = self.arxiv_client.search_papers(categories=["cs.TEST"], max_results=2)

            self.assertEqual(len(search_results), 2)

            found_local = any(p.id == local_paper_id for p in search_results)
            found_api = any(p.id == api_paper_id for p in search_results)
            self.assertTrue(found_local, "Local paper not found in search results")
            self.assertTrue(found_api, "API paper not found in search results")

            # Check which papers were processed by _convert_result
            # It should only be called for the API paper, as the local one is loaded from file.
            
            calls_to_convert = []
            for call_args in mock_convert_result.call_args_list:
                # The first argument to _convert_result is the arxiv.Result object
                # We check its entry_id to see which paper it was.
                arxiv_result_arg = call_args[0][0]
                calls_to_convert.append(arxiv_result_arg.entry_id.split('/')[-1])

            self.assertIn(api_paper_id, calls_to_convert, "_convert_result not called for API paper")
            self.assertNotIn(local_paper_id, calls_to_convert, "_convert_result was called for local paper, but it should have been loaded from file.")

            # Verify the content of the loaded local paper
            for p in search_results:
                if p.id == local_paper_id:
                    self.assertEqual(p.summary, local_paper_data.summary) # Ensure it's the summary from the file

    def test_live_arxiv_fetch_save_load(self):
        """
        Tests a live fetch from arXiv, saves a dummy summary,
        and then verifies that a new client instance loads this summary from file.
        This test makes actual network calls to arXiv.
        """
        # ArxivClient.SUMMARIES_DIR and ArxivClient.SEEN_PAPERS_FILE are patched by setUp

        # Instantiate the first client - this will use the patched SUMMARIES_DIR
        live_arxiv_client1 = ArxivClient() 

        summary_filepath = os.path.join(live_arxiv_client1.SUMMARIES_DIR, f"{self.TEST_LIVE_PAPER_ID}.json")

        # Cleanup before test: Remove summary file if it exists from a previous failed run
        if os.path.exists(summary_filepath):
            os.remove(summary_filepath)
        # Ensure seen_papers.json is also clean for this specific paper ID or test run if needed
        # For this test, a fresh seen_papers.json (handled by setUp) is usually sufficient

        paper1 = None
        try:
            print(f"\nAttempting live fetch for paper ID: {self.TEST_LIVE_PAPER_ID}")
            paper1 = live_arxiv_client1.get_paper_by_id(self.TEST_LIVE_PAPER_ID)
        except (arxiv.arxiv.HTTPError, requests.exceptions.ConnectionError, arxiv.arxiv.UnexpectedEmptyPageError) as e:
            self.skipTest(f"Skipping live arXiv test for paper {self.TEST_LIVE_PAPER_ID}: Network error or API issue ({type(e).__name__}: {e})")
        except Exception as e: # Catch any other unexpected errors during live fetch
            self.skipTest(f"Skipping live arXiv test for paper {self.TEST_LIVE_PAPER_ID}: Unexpected error ({type(e).__name__}: {e})")


        self.assertIsNotNone(paper1, f"Paper {self.TEST_LIVE_PAPER_ID} could not be fetched from arXiv.")
        self.assertEqual(paper1.id, self.TEST_LIVE_PAPER_ID)
        self.assertIsNotNone(paper1.title, "Fetched paper is missing a title.") # Sanity check

        # Add dummy summary and save
        original_title = paper1.title # Store original title for later comparison
        paper1.summary = "LIVE_TEST_DUMMY_SUMMARY"
        live_arxiv_client1.save_summary_to_file(paper1)
        self.assertTrue(os.path.exists(summary_filepath), f"Summary file was not saved at {summary_filepath}")

        # Second fetch (should load from file)
        # Instantiate a new ArxivClient instance
        # It will also use the TEST_SUMMARIES_DIR due to how setUp patches the class attribute
        live_arxiv_client2 = ArxivClient()
        
        # Before fetching with client2, ensure its internal seen_papers is fresh or doesn't interfere
        # If get_paper_by_id has side effects on seen_papers that affect loading, this might be needed.
        # However, get_paper_by_id's primary loading path from summary file doesn't depend on seen_papers.
        # It *does* add to seen_papers if loaded from summary and not already there.

        paper2 = live_arxiv_client2.get_paper_by_id(self.TEST_LIVE_PAPER_ID)

        self.assertIsNotNone(paper2, "Paper could not be retrieved by the second client.")
        self.assertEqual(paper2.summary, "LIVE_TEST_DUMMY_SUMMARY", "Loaded paper summary does not match the saved dummy summary.")
        self.assertEqual(paper2.title, original_title, "Loaded paper title does not match the original.")
        self.assertEqual(paper2.id, self.TEST_LIVE_PAPER_ID)

        # Optional: Cleanup of the specific file can be done here,
        # but tearDown handles the entire TEST_SUMMARIES_DIR.
        # if os.path.exists(summary_filepath):
        #     os.remove(summary_filepath)


if __name__ == '__main__':
    unittest.main()
