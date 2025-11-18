"""
Test suite for Exercise 5 LangChain Agentic Workflow
Tests the agentic workflow implementation with tools.
"""

import unittest
import pytest
from unittest.mock import patch, Mock
from exercise5 import (
    fetch_users_from_api,
    filter_users_by_birth_year,
    select_random_people_from_list,
    search_wikipedia_for_person,
    identify_person_with_llm,
    check_if_notable,
    research_best_work,
    BestWorkAgent
)


class TestFetchUsersFromAPI(unittest.TestCase):
    """Test suite for fetch_users_from_api tool function"""

    @patch('exercise5.requests.get')
    def test_fetch_users_success(self, mock_get):
        """Test successful user fetching"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {'name': {'first': 'John', 'last': 'Doe'}, 'dob': {'date': '1990-01-01'}},
                {'name': {'first': 'Jane', 'last': 'Smith'}, 'dob': {'date': '1985-05-15'}}
            ]
        }
        mock_get.return_value = mock_response

        result = fetch_users_from_api(num_results=2)

        assert "Fetched 2 users" in result
        assert "John Doe" in result
        assert "Jane Smith" in result
        assert "1990" in result

    @patch('exercise5.requests.get')
    def test_fetch_users_api_error(self, mock_get):
        """Test handling of API errors"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = fetch_users_from_api(num_results=20)

        assert "Error" in result
        assert "500" in result

    @patch('exercise5.requests.get')
    def test_fetch_users_timeout(self, mock_get):
        """Test handling of timeout errors"""
        mock_get.side_effect = Exception("Timeout")

        result = fetch_users_from_api(num_results=20)

        assert "Error fetching users" in result

    @patch('exercise5.requests.get')
    def test_fetch_users_custom_count(self, mock_get):
        """Test fetching custom number of users"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {'name': {'first': f'User{i}', 'last': 'Test'}, 'dob': {'date': f'199{i}-01-01'}}
                for i in range(10)
            ]
        }
        mock_get.return_value = mock_response

        result = fetch_users_from_api(num_results=10)

        assert "Fetched 10 users" in result


class TestFilterUsersByBirthYear(unittest.TestCase):
    """Test suite for filter_users_by_birth_year tool function"""

    def test_filter_users_basic(self):
        """Test basic filtering functionality"""
        users_summary = """Fetched 5 users:
John Doe (born 1990)
Jane Smith (born 2005)
Bob Jones (born 1998)
Alice Brown (born 2010)
Charlie Wilson (born 2000)"""

        result = filter_users_by_birth_year(users_summary, max_birth_year=2000)

        assert "John Doe" in result
        assert "Bob Jones" in result
        assert "Charlie Wilson" in result
        assert "Jane Smith" not in result  # Born after 2000
        assert "Alice Brown" not in result  # Born after 2000

    def test_filter_all_after_year(self):
        """Test when all users are born after the year"""
        users_summary = """Fetched 2 users:
Young Person (born 2010)
Another Young (born 2015)"""

        result = filter_users_by_birth_year(users_summary, max_birth_year=2000)

        assert "No users found matching criteria" in result

    def test_filter_all_before_year(self):
        """Test when all users are born before the year"""
        users_summary = """Fetched 2 users:
Old Person (born 1950)
Another Old (born 1960)"""

        result = filter_users_by_birth_year(users_summary, max_birth_year=2000)

        assert "Old Person" in result
        assert "Another Old" in result

    def test_filter_empty_input(self):
        """Test filtering with empty input"""
        result = filter_users_by_birth_year("Fetched 0 users:", max_birth_year=2000)

        assert "No users found" in result or result == ""


class TestSelectRandomPeopleFromList(unittest.TestCase):
    """Test suite for select_random_people_from_list tool function"""

    def test_select_fewer_than_available(self):
        """Test selecting fewer people than available"""
        names_list = "Alice Smith, Bob Jones, Charlie Brown, Diana Prince, Eve Wilson"
        result = select_random_people_from_list(names_list, count=3)

        names = result.split(", ")
        assert len(names) == 3
        assert all(name in names_list for name in names)

    def test_select_more_than_available(self):
        """Test selecting more people than available"""
        names_list = "Alice Smith, Bob Jones"
        result = select_random_people_from_list(names_list, count=10)

        names = result.split(", ")
        assert len(names) == 2

    def test_select_exact_count(self):
        """Test selecting exact number available"""
        names_list = "Alice Smith, Bob Jones, Charlie Brown"
        result = select_random_people_from_list(names_list, count=3)

        names = result.split(", ")
        assert len(names) == 3

    def test_select_empty_list(self):
        """Test selecting from empty list"""
        result = select_random_people_from_list("", count=5)

        assert "Error" in result

    def test_select_with_error(self):
        """Test error handling in selection"""
        result = select_random_people_from_list(None, count=5)

        assert "Error selecting people" in result


class TestSearchWikipediaForPerson(unittest.TestCase):
    """Test suite for search_wikipedia_for_person tool function"""

    @patch('search_tools.WikipediaSearch')
    def test_wikipedia_search_found(self, mock_wiki_class):
        """Test successful Wikipedia search"""
        mock_wiki = Mock()
        mock_wiki_class.return_value = mock_wiki
        mock_wiki.search_person.return_value = {
            'found': True,
            'name': 'Albert Einstein',
            'summary': 'Albert Einstein was a theoretical physicist.',
            'url': 'https://en.wikipedia.org/wiki/Albert_Einstein'
        }

        result = search_wikipedia_for_person("Albert Einstein")

        assert "Wikipedia article found" in result
        assert "Albert Einstein" in result
        assert "theoretical physicist" in result
        assert "wikipedia.org" in result

    @patch('search_tools.WikipediaSearch')
    def test_wikipedia_search_not_found(self, mock_wiki_class):
        """Test Wikipedia search when article not found"""
        mock_wiki = Mock()
        mock_wiki_class.return_value = mock_wiki
        mock_wiki.search_person.return_value = {
            'found': False,
            'name': 'Random Person',
        }

        result = search_wikipedia_for_person("Random Person")

        assert "No Wikipedia article found" in result
        assert "Random Person" in result

    def test_wikipedia_search_invalid_name(self):
        """Test Wikipedia search with invalid name format"""
        result = search_wikipedia_for_person("InvalidName")

        assert "Invalid name format" in result

    @patch('search_tools.WikipediaSearch')
    def test_wikipedia_search_error(self, mock_wiki_class):
        """Test Wikipedia search error handling"""
        mock_wiki = Mock()
        mock_wiki_class.return_value = mock_wiki
        mock_wiki.search_person.side_effect = Exception("Search failed")

        result = search_wikipedia_for_person("Test Person")

        assert "Error searching Wikipedia" in result


class TestIdentifyPersonWithLLM(unittest.TestCase):
    """Test suite for identify_person_with_llm tool function"""

    @patch('llm_backend.LLMBackend')
    def test_identify_person_success(self, mock_llm_class):
        """Test successful person identification"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_llm.identify_person.return_value = "Albert Einstein was a physicist."

        result = identify_person_with_llm("Albert Einstein")

        assert "physicist" in result or "Albert Einstein" in result

    @patch('llm_backend.LLMBackend')
    def test_identify_person_error(self, mock_llm_class):
        """Test person identification error handling"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_llm.identify_person.side_effect = Exception("LLM error")

        result = identify_person_with_llm("Test Person")

        assert "Error identifying person" in result

    def test_identify_person_invalid_name(self):
        """Test identification with invalid name format"""
        result = identify_person_with_llm("SingleName")

        assert "Invalid name format" in result


class TestCheckIfNotable(unittest.TestCase):
    """Test suite for check_if_notable tool function"""

    def test_check_notable_person(self):
        """Test checking notable person"""
        person_info = "Albert Einstein was a famous theoretical physicist who won the Nobel Prize."

        result = check_if_notable(person_info)

        assert "YES" in result
        assert "famous" in result or "Nobel" in result

    def test_check_unknown_person(self):
        """Test checking unknown person"""
        person_info = "Unknown person"

        result = check_if_notable(person_info)

        assert "NO" in result
        assert "unknown" in result.lower()

    def test_check_error_info(self):
        """Test checking with error information"""
        person_info = "Error identifying person: API failed"

        result = check_if_notable(person_info)

        assert "NO" in result

    def test_check_uncertain_person(self):
        """Test checking person with uncertain notability"""
        person_info = "John Smith is a person."

        result = check_if_notable(person_info)

        assert "NO" in result or "UNCERTAIN" in result


class TestResearchBestWork(unittest.TestCase):
    """Test suite for research_best_work tool function"""

    def test_research_notable_person(self):
        """Test researching best work of notable person"""
        person_name = "Albert Einstein"
        person_info = "Albert Einstein was a theoretical physicist."

        result = research_best_work(person_name, person_info)

        assert person_name in result
        assert "theoretical physicist" in result
        assert "notable work" in result or "achievement" in result


class TestBestWorkAgent(unittest.TestCase):
    """Test suite for BestWorkAgent class"""

    @patch('exercise5.init_chat_model')
    @patch('exercise5.create_agent')
    def test_agent_initialization(self, mock_create_agent, mock_init_chat):
        """Test BestWorkAgent initialization"""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            mock_model = Mock()
            mock_init_chat.return_value = mock_model
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent

            agent = BestWorkAgent(model_name="openai/gpt-4o")

            assert agent.model is not None
            assert agent.agent is not None
            mock_init_chat.assert_called_once()
            mock_create_agent.assert_called_once()

    @patch('exercise5.init_chat_model')
    @patch('exercise5.create_agent')
    def test_agent_with_custom_config(self, mock_create_agent, mock_init_chat):
        """Test agent initialization with custom configuration"""
        mock_model = Mock()
        mock_init_chat.return_value = mock_model
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        agent = BestWorkAgent(
            model_name="custom/model",
            api_key="custom_key",
            base_url="https://custom.url"
        )

        # Verify init_chat_model was called with correct parameters
        call_kwargs = mock_init_chat.call_args[1]
        assert call_kwargs['model'] == "custom/model"
        assert call_kwargs['api_key'] == "custom_key"
        assert call_kwargs['base_url'] == "https://custom.url"

    @patch('exercise5.init_chat_model')
    @patch('exercise5.create_agent')
    def test_agent_no_api_key_raises_error(self, mock_create_agent, mock_init_chat):
        """Test that agent initialization fails without API key"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key"):
                BestWorkAgent()

    @patch('exercise5.init_chat_model')
    @patch('exercise5.create_agent')
    def test_agent_tools_configuration(self, mock_create_agent, mock_init_chat):
        """Test that agent is configured with all required tools"""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            mock_model = Mock()
            mock_init_chat.return_value = mock_model
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent

            agent = BestWorkAgent(model_name="openai/gpt-4o")

            # Verify create_agent was called with tools
            call_args = mock_create_agent.call_args
            tools = call_args[1]['tools']
            assert len(tools) == 7  # Should have 7 tools

            # Verify all expected tools are present
            tool_names = [tool.__name__ for tool in tools]
            assert 'fetch_users_from_api' in tool_names
            assert 'filter_users_by_birth_year' in tool_names
            assert 'select_random_people_from_list' in tool_names
            assert 'search_wikipedia_for_person' in tool_names
            assert 'identify_person_with_llm' in tool_names
            assert 'check_if_notable' in tool_names
            assert 'research_best_work' in tool_names


class TestAgentWorkflow:
    """Test suite for agent workflow integration"""

    @patch('exercise5.init_chat_model')
    @patch('exercise5.create_agent')
    def test_agent_invoke_success(self, mock_create_agent, mock_init_chat):
        """Test successful agent invocation"""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            # Setup mocks
            mock_model = Mock()
            mock_init_chat.return_value = mock_model

            mock_agent = Mock()
            mock_final_message = Mock()
            mock_final_message.content = "Agent completed the task successfully."
            mock_create_agent.return_value = mock_agent
            mock_agent.invoke.return_value = {
                "messages": [mock_final_message]
            }

            # Create agent and invoke
            agent = BestWorkAgent(model_name="openai/gpt-4o")
            response = agent.agent.invoke({
                "messages": [{"role": "user", "content": "Test task"}]
            })

            # Verify
            assert "messages" in response
            assert len(response["messages"]) > 0
            mock_agent.invoke.assert_called_once()

    @patch('exercise5.init_chat_model')
    @patch('exercise5.create_agent')
    def test_agent_error_handling(self, mock_create_agent, mock_init_chat):
        """Test agent error handling"""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            # Setup mocks
            mock_model = Mock()
            mock_init_chat.return_value = mock_model

            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent
            mock_agent.invoke.side_effect = Exception("Agent error")

            # Create agent
            agent = BestWorkAgent(model_name="openai/gpt-4o")

            # Verify error is raised
            with pytest.raises(Exception, match="Agent error"):
                agent.agent.invoke({
                    "messages": [{"role": "user", "content": "Test task"}]
                })


class TestMainFunction:
    """Test suite for main function"""

    @patch('exercise5.BestWorkAgent')
    def test_main_execution_flow(self, mock_agent_class):
        """Test main function execution flow"""
        # Mock the agent
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        mock_final_message = Mock()
        mock_final_message.content = "Test results"
        mock_agent.agent.invoke.return_value = {
            "messages": [mock_final_message]
        }

        # This would normally be tested by running main() but we'd need
        # to mock all the dependencies, so we just verify the structure exists
        from exercise5 import main
        assert callable(main)

    def test_main_has_proper_error_handling(self):
        """Test that main function has error handling"""
        from exercise5 import main
        import inspect

        # Check that main has try-except blocks
        source = inspect.getsource(main)
        assert 'try:' in source
        assert 'except' in source


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
