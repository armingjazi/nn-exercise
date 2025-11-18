"""
Test suite for Exercise 4 LLM functionality
"""

import unittest
import pytest
import random
from unittest.mock import patch, MagicMock
from io import StringIO
from exercise4 import (
    select_random_people,
    display_identifications,
    categorize_identification,
    display_summary_statistics
)
from llm_backend import LLMBackend


class TestSelectRandomPeople(unittest.TestCase):
    """Test suite for select_random_people function"""

    def test_select_fewer_than_available(self):
        """Test selecting fewer people than available"""
        names = [
            "Alice Smith",
            "Bob Jones",
            "Charlie Brown",
            "Diana Prince",
            "Eve Wilson",
        ]
        result = select_random_people(names, count=3)

        assert len(result) == 3
        assert all(isinstance(person, tuple) and len(person) == 2 for person in result)

        # Verify all returned names are from the original list
        result_full_names = [f"{first} {last}" for first, last in result]
        assert all(name in names for name in result_full_names)

    def test_select_more_than_available(self):
        """Test selecting more people than available (should return all)"""
        names = ["Alice Smith", "Bob Jones"]
        result = select_random_people(names, count=10)

        assert len(result) == 2
        assert ('Alice', 'Smith') in result or ('Bob', 'Jones') in result

    def test_correct_name_parsing(self):
        """Test that names are correctly split into (first, last) tuples"""
        names = ["John Doe", "Jane Smith"]
        result = select_random_people(names, count=2)

        # Verify exact parsing
        assert len(result) == 2
        name_dict = {f"{first} {last}": (first, last) for first, last in result}

        if 'John Doe' in name_dict:
            first, last = name_dict['John Doe']
            assert first == 'John'
            assert last == 'Doe'

        if 'Jane Smith' in name_dict:
            first, last = name_dict['Jane Smith']
            assert first == 'Jane'
            assert last == 'Smith'

    def test_names_with_multiple_spaces_in_last_name(self):
        """Test parsing names with spaces in last name (e.g., 'Van Der Berg')"""
        names = ['John Van Der Berg']
        result = select_random_people(names, count=1)

        assert len(result) == 1
        first, last = result[0]
        assert first == 'John'
        assert last == 'Van Der Berg'  # Should keep the full last name

    def test_randomness_with_seed(self):
        """Test that random selection is actually random but reproducible with seed"""
        names = ['Alice Smith', 'Bob Jones', 'Charlie Brown', 'Diana Prince', 'Eve Wilson']

        # Set seed and get results
        random.seed(42)
        result1 = select_random_people(names, count=3)

        # Reset seed and get results again
        random.seed(42)
        result2 = select_random_people(names, count=3)

        # Should be identical with same seed
        assert result1 == result2

        # Different seed should give different results (very likely)
        random.seed(99)
        result3 = select_random_people(names, count=3)
        # Can't guarantee difference but statistically unlikely to be same
        # Just verify it's still valid
        assert len(result3) == 3

    def test_empty_list(self):
        """Test with empty list"""
        result = select_random_people([], count=5)
        assert len(result) == 0

    def test_single_word_names_excluded(self):
        """Test that names without spaces (no last name) are excluded"""
        names = ['Madonna', 'John Doe', 'Cher']
        result = select_random_people(names, count=3)

        # Only 'John Doe' should be included
        assert len(result) <= 1
        if len(result) == 1:
            assert result[0] == ('John', 'Doe')


class TestLLMBackendInit:
    """Test suite for LLMBackend initialization"""

    @patch('config.Config.get_openrouter_api_key')
    @patch('config.Config.get_openrouter_model')
    @patch('config.Config.get_openrouter_base_url')
    def test_init_uses_config_methods(self, mock_base_url, mock_model, mock_api_key):
        """Test that initialization properly calls Config methods"""
        mock_api_key.return_value = "test_key_123"
        mock_model.return_value = "openai/gpt-4o-mini"
        mock_base_url.return_value = "https://openrouter.ai/api/v1"

        llm = LLMBackend()

        # Verify Config methods were called
        mock_api_key.assert_called_once()
        mock_model.assert_called_once()
        mock_base_url.assert_called_once()

        # Verify values are set correctly
        assert llm.api_key == "test_key_123"
        assert llm.model == "openai/gpt-4o-mini"
        assert llm.base_url == "https://openrouter.ai/api/v1/chat/completions"

    @patch('config.Config.get_openrouter_api_key')
    @patch('config.Config.get_openrouter_model')
    @patch('config.Config.get_openrouter_base_url')
    def test_init_with_explicit_parameters(self, mock_base_url, mock_model, mock_api_key):
        """Test that explicit parameters override config"""
        llm = LLMBackend(
            api_key="explicit_key",
            model="openai/gpt-4o",
            base_url="https://custom.url/api/v1"
        )

        # Config methods should NOT be called when explicit params provided
        mock_api_key.assert_not_called()
        mock_model.assert_not_called()
        mock_base_url.assert_not_called()

        # Verify explicit values are used
        assert llm.api_key == "explicit_key"
        assert llm.model == "openai/gpt-4o"
        assert llm.base_url == "https://custom.url/api/v1"

    @patch('config.Config.get_openrouter_api_key')
    def test_init_no_api_key_raises_error(self, mock_api_key):
        """Test that ValueError is raised when no API key is available"""
        mock_api_key.side_effect = ValueError("OpenRouter API key not found")

        with pytest.raises(ValueError, match="OpenRouter API key"):
            LLMBackend()

    @patch('config.Config.get_openrouter_api_key')
    @patch('config.Config.get_openrouter_model')
    @patch('config.Config.get_openrouter_base_url')
    def test_base_url_endpoint_construction(self, mock_base_url, mock_model, mock_api_key):
        """Test that /chat/completions is properly appended to base URL"""
        mock_api_key.return_value = "test_key"
        mock_model.return_value = "openai/gpt-4o-mini"
        mock_base_url.return_value = "https://openrouter.ai/api/v1"

        llm = LLMBackend()

        # Should append /chat/completions
        assert llm.base_url == "https://openrouter.ai/api/v1/chat/completions"


class TestLLMBackendIdentify:
    """Test suite for LLM identification methods"""

    @patch("llm_backend.requests.post")
    def test_identify_person_success(self, mock_post):
        """Test successful person identification with proper API call"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Albert Einstein was a theoretical physicist who developed the theory of relativity."}}
            ]
        }
        mock_post.return_value = mock_response

        # Test
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}):
            llm = LLMBackend()
            result = llm.identify_person("Albert", "Einstein")

            # Verify the exact response content
            assert result == "Albert Einstein was a theoretical physicist who developed the theory of relativity."

            # Verify API was called with correct parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args

            # Check headers
            assert "Authorization" in call_args[1]["headers"]
            assert call_args[1]["headers"]["Authorization"] == "Bearer test_key"

    @patch("llm_backend.requests.post")
    def test_identify_person_api_error(self, mock_post):
        """Test handling of API errors"""
        # Setup mock to raise error
        mock_post.side_effect = Exception("API Error")

        # Test
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}):
            llm = LLMBackend()
            result = llm.identify_person("Test", "Person")

            assert "Error" in result
            assert "API Error" in result

    @patch("llm_backend.requests.post")
    def test_identify_person_http_error_status(self, mock_post):
        """Test handling of non-200 HTTP status codes"""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}):
            llm = LLMBackend()
            result = llm.identify_person("Test", "Person")

            assert "Error" in result
            assert "401" in result

    @patch("llm_backend.requests.post")
    @patch("search_tools.WikipediaSearch")
    def test_identify_person_with_search_wikipedia_found(self, mock_wiki_class, mock_post):
        """Test identification when Wikipedia article is found"""
        # Mock Wikipedia search result
        mock_wiki = MagicMock()
        mock_wiki.search_person.return_value = {
            "found": True,
            "name": "Isaac Newton",
            "summary": "Sir Isaac Newton was an English mathematician, physicist, and astronomer.",
            "url": "https://en.wikipedia.org/wiki/Isaac_Newton"
        }
        mock_wiki_class.return_value = mock_wiki

        # Mock LLM response for summarization
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Isaac Newton was a renowned physicist and mathematician known for his laws of motion."}}
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}):
            llm = LLMBackend()
            result = llm.identify_person_with_search("Isaac", "Newton")

            # Should include LLM summary and Wikipedia source
            assert "Isaac Newton" in result or "physicist" in result
            assert "Wikipedia" in result
            assert "https://en.wikipedia.org/wiki/Isaac_Newton" in result

    @patch("llm_backend.requests.post")
    @patch("search_tools.WikipediaSearch")
    def test_identify_person_with_search_wikipedia_not_found(self, mock_wiki_class, mock_post):
        """Test identification when Wikipedia article is NOT found"""
        # Mock Wikipedia search result (not found)
        mock_wiki = MagicMock()
        mock_wiki.search_person.return_value = {
            "found": False,
            "name": "Random Person"
        }
        mock_wiki_class.return_value = mock_wiki

        # Mock LLM fallback response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Unknown person"}}
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}):
            llm = LLMBackend()
            result = llm.identify_person_with_search("Random", "Person")

            # Should include fallback and note about no Wikipedia
            assert "Unknown person" in result
            assert "No Wikipedia article found" in result

    @patch("llm_backend.requests.post")
    def test_batch_identify_people(self, mock_post):
        """Test batch identification of multiple people"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Test identification"}}
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}):
            llm = LLMBackend()
            people = [("Alice", "Smith"), ("Bob", "Jones"), ("Charlie", "Brown")]
            results = llm.batch_identify_people(people)

            # Should return dict with all people
            assert len(results) == 3
            assert "Alice Smith" in results
            assert "Bob Jones" in results
            assert "Charlie Brown" in results

            # Should have called API 3 times (once per person)
            assert mock_post.call_count == 3

    @patch("llm_backend.requests.post")
    def test_invoke_custom_prompt(self, mock_post):
        """Test the flexible invoke method with custom prompts"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Custom response from LLM"}}
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}):
            llm = LLMBackend()
            result = llm.invoke("What is the meaning of life?", max_tokens=100)

            assert result == "Custom response from LLM"

            # Verify API call includes custom max_tokens
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["max_tokens"] == 100

    def test_get_model_info(self):
        """Test model info retrieval"""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}):
            llm = LLMBackend(model="openai/gpt-4o")
            info = llm.get_model_info()

            assert "openai/gpt-4o" in info


class TestCategorizeIdentification:
    """Test suite for categorize_identification function"""

    def test_verified_notable_person(self):
        """Test categorization of verified Wikipedia person"""
        identification = "Isaac Newton was a physicist. (Source: Wikipedia - https://en.wikipedia.org/wiki/Isaac_Newton)"
        result = categorize_identification("Isaac Newton", identification)

        assert result["category"] == "✅ Verified Notable Person"
        assert result["confidence"] == "High"
        assert result["name"] == "Isaac Newton"

    def test_unknown_person_no_wikipedia(self):
        """Test categorization of unknown person"""
        identification = "Unknown person\n\n(Note: No Wikipedia article found)"
        result = categorize_identification("Random Name", identification)

        assert result["category"] == "❌ Unknown/Non-Notable"
        assert result["confidence"] == "Low"

    def test_fictional_character(self):
        """Test categorization of fictional character"""
        identification = "This is a fictional character from a novel."
        result = categorize_identification("Harry Potter", identification)

        assert result["category"] == "🎭 Fictional Character"
        assert result["confidence"] == "High"

    def test_error_result(self):
        """Test categorization of error response"""
        identification = "Error identifying person: API timeout"
        result = categorize_identification("Test Person", identification)

        assert result["category"] == "⚠️ Error"
        assert result["confidence"] == "N/A"

    def test_possibly_notable_unverified(self):
        """Test categorization of possibly notable but unverified person"""
        identification = "This person may be notable in their field.\n\n(Note: No Wikipedia article found)"
        result = categorize_identification("Someone Famous", identification)

        assert result["category"] == "⚠️ Possibly Notable (Unverified)"
        assert result["confidence"] == "Medium"


class TestDisplayFunctions:
    """Test suite for display functions"""

    @patch('sys.stdout', new_callable=StringIO)
    def test_display_identifications_output_format(self, mock_stdout):
        """Test that display_identifications produces expected output"""
        identifications = {
            "Isaac Newton": "English physicist known for laws of motion. (Source: Wikipedia - https://en.wikipedia.org/wiki/Isaac_Newton)",
            "Random Person": "Unknown person\n\n(Note: No Wikipedia article found)"
        }

        display_identifications(identifications)
        output = mock_stdout.getvalue()

        # Check for key sections
        assert "IDENTIFICATION RESULTS" in output
        assert "✅ Verified Notable Person" in output
        assert "❌ Unknown/Non-Notable" in output
        assert "Isaac Newton" in output
        assert "Random Person" in output
        assert "Quick Stats" in output

    @patch('sys.stdout', new_callable=StringIO)
    def test_display_summary_statistics(self, mock_stdout):
        """Test summary statistics display"""
        categorized_results = [
            {"category": "✅ Verified Notable Person", "confidence": "High", "name": "Isaac Newton", "details": "Physicist"},
            {"category": "❌ Unknown/Non-Notable", "confidence": "Low", "name": "Random Person", "details": "Unknown"}
        ]

        display_summary_statistics(categorized_results)
        output = mock_stdout.getvalue()

        assert "SUMMARY STATISTICS" in output
        assert "Total People Processed: 2" in output
        assert "Verification Success Rate" in output

    @patch('sys.stdout', new_callable=StringIO)
    def test_display_identifications_empty_results(self, mock_stdout):
        """Test display with no identifications"""
        identifications = {}

        display_identifications(identifications)
        output = mock_stdout.getvalue()

        # Should still display structure but with 0 counts
        assert "IDENTIFICATION RESULTS" in output
        assert "0 people processed" in output
