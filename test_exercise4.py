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
    
    def test_select_exact_count(self):
        """Test selecting exact number available"""
        names = ['Alice Smith', 'Bob Jones', 'Charlie Brown']
        result = select_random_people(names, count=3)
        
        assert len(result) == 3
    
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
    
    def test_single_person(self):
        """Test selecting from single person list"""
        names = ['Single Person']
        result = select_random_people(names, count=1)
        
        assert len(result) == 1
        assert result[0] == ('Single', 'Person')


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
    
    def test_default_model(self):
        """Test that default model is set"""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            llm = LLMBackend()
            assert llm.model == "openai/gpt-4o"
    
    def test_custom_model(self):
        """Test initialization with custom model"""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            llm = LLMBackend(model="gpt-3.5-turbo")
            assert llm.model == "gpt-3.5-turbo"
    
    def test_get_model_info(self):
        """Test get_model_info returns correct string"""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            llm = LLMBackend(model="test-model")
            info = llm.get_model_info()
            assert "test-model" in info


class TestLLMBackendIdentify:
    """Test suite for LLM identification methods"""
    
    @patch('llm_backend.OpenAI')
    def test_identify_person_success(self, mock_openai_class):
        """Test successful person identification"""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Albert Einstein was a theoretical physicist.")]
        mock_client.messages.create.return_value = mock_message
        
        # Test
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            llm = LLMBackend()
            result = llm.identify_person("Albert", "Einstein")
            
            assert "Albert Einstein" in result or "physicist" in result
    
    @patch('llm_backend.OpenAI')
    def test_identify_person_api_error(self, mock_openai_class):
        """Test handling of API errors"""
        # Setup mock to raise error
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")
        
        # Test
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            llm = LLMBackend()
            result = llm.identify_person("Test", "Person")
            
            assert "Error" in result
    
    @patch('llm_backend.OpenAI')
    def test_batch_identify_people(self, mock_openai_class):
        """Test batch identification of multiple people"""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="A famous person.")]
        mock_client.messages.create.return_value = mock_message
        
        # Test
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            llm = LLMBackend()
            people = [("John", "Doe"), ("Jane", "Smith")]
            results = llm.batch_identify_people(people)
            
            assert len(results) == 2
            assert "John Doe" in results
            assert "Jane Smith" in results


class TestDisplayIdentifications:
    """Test suite for display_identifications function"""
    
    def test_display_output(self, capsys):
        """Test that display function outputs correctly formatted results"""
        identifications = {
            'Albert Einstein': 'A theoretical physicist.',
            'Marie Curie': 'A chemist and physicist.'
        }
        
        display_identifications(identifications)
        captured = capsys.readouterr()
        
        assert 'Albert Einstein' in captured.out
        assert 'Marie Curie' in captured.out
        assert 'theoretical physicist' in captured.out
        assert 'chemist' in captured.out
    
    def test_display_empty_dict(self, capsys):
        """Test display with empty results"""
        display_identifications({})
        captured = capsys.readouterr()
        
        assert '=' in captured.out  # Should have formatting
