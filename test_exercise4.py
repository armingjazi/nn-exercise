"""
Test suite for Exercise 4 LLM functionality
"""

import unittest
import pytest
from unittest.mock import patch, MagicMock
from exercise4 import select_random_people, display_identifications
from llm_backend import LLMBackend


class TestSelectRandomPeople(unittest.TestCase):
    """Test suite for select_random_people function"""

    def test_select_fewer_than_available(self):
        """Test selecting fewer people than available"""
        names = ['Alice Smith', 'Bob Jones', 'Charlie Brown', 'Diana Prince', 'Eve Wilson']
        result = select_random_people(names, count=3)

        assert len(result) == 3
        assert all(isinstance(person, tuple) and len(person) == 2 for person in result)

    def test_select_more_than_available(self):
        """Test selecting more people than available (should return all)"""
        names = ['Alice Smith', 'Bob Jones']
        result = select_random_people(names, count=10)

        assert len(result) == 2

    def test_name_format_parsing(self):
        """Test that names are correctly split into (first, last) tuples"""
        names = ['John Doe', 'Jane Smith']
        result = select_random_people(names, count=2)

        # Check that we got the right format
        for first, last in result:
            assert isinstance(first, str)
            assert isinstance(last, str)
            assert len(first) > 0
            assert len(last) > 0

    def test_empty_list(self):
        """Test with empty list"""
        result = select_random_people([], count=5)

        assert len(result) == 0


class TestLLMBackendInit(unittest.TestCase):
    """Test suite for LLMBackend initialization"""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key"""
        with patch.dict('os.environ', {}, clear=True):
            llm = LLMBackend(api_key="test_key_123")
            assert llm.api_key == "test_key_123"

    def test_init_from_env_var(self):
        """Test initialization from environment variable"""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'env_key_456'}):
            llm = LLMBackend()
            assert llm.api_key == 'env_key_456'

    def test_init_no_api_key_raises_error(self):
        """Test that ValueError is raised when no API key is provided"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key"):
                LLMBackend()

    def test_model_from_env_var(self):
        """Test that model can be loaded from environment variable"""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key', 'OPENROUTER_MODEL': 'openai/gpt-4o'}):
            llm = LLMBackend()
            assert llm.model == "openai/gpt-4o"


class TestLLMBackendIdentify:
    """Test suite for LLM identification methods"""

    @patch('llm_backend.requests.post')
    def test_identify_person_success(self, mock_post):
        """Test successful person identification"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Albert Einstein was a theoretical physicist."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        # Test
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            llm = LLMBackend()
            result = llm.identify_person("Albert", "Einstein")

            assert "Albert Einstein" in result or "physicist" in result

    @patch('llm_backend.requests.post')
    def test_identify_person_api_error(self, mock_post):
        """Test handling of API errors"""
        # Setup mock to raise error
        mock_post.side_effect = Exception("API Error")

        # Test
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            llm = LLMBackend()
            result = llm.identify_person("Test", "Person")

            assert "Error" in result
