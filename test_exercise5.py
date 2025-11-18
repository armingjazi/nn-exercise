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
    User,
)


class TestFetchUsersFromAPI(unittest.TestCase):
    """Test suite for fetch_users_from_api tool function"""

    @patch("exercise5.requests.get")
    def test_fetch_users_success(self, mock_get):
        """Test successful user fetching"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "name": {"first": "John", "last": "Doe"},
                    "dob": {"date": "1990-01-01"},
                },
                {
                    "name": {"first": "Jane", "last": "Smith"},
                    "dob": {"date": "1985-05-15"},
                },
            ]
        }
        mock_get.return_value = mock_response

        result = fetch_users_from_api.invoke({"num_results": 2})

        assert "Fetched 2 users" in result
        assert "John Doe" in result
        assert "Jane Smith" in result
        assert "1990" in result

    @patch("exercise5.requests.get")
    def test_fetch_users_api_error(self, mock_get):
        """Test handling of API errors"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = fetch_users_from_api.invoke({"num_results": 20})

        assert "Error" in result
        assert "500" in result

    @patch("exercise5.requests.get")
    def test_fetch_users_timeout(self, mock_get):
        """Test handling of timeout errors"""
        mock_get.side_effect = Exception("Timeout")

        result = fetch_users_from_api.invoke({"num_results": 20})

        assert "Error fetching users" in result


class TestFilterUsersByBirthYear(unittest.TestCase):
    """Test suite for filter_users_by_birth_year tool function"""

    def test_filter_users_basic(self):
        """Test basic filtering functionality"""
        users = [
            User(name="John Doe", born="1990"),
            User(name="Jane Smith", born="2005"),
            User(name="Bob Jones", born="1998"),
            User(name="Alice Brown", born="2010"),
            User(name="Charlie Wilson", born="2000"),
        ]

        result = filter_users_by_birth_year.invoke(
            {"users": users, "max_birth_year": 2000}
        )

        assert "John Doe" in result
        assert "Bob Jones" in result
        assert "Charlie Wilson" in result
        assert "Jane Smith" not in result  # Born after 2000
        assert "Alice Brown" not in result  # Born after 2000

    def test_filter_all_after_year(self):
        """Test when all users are born after the year"""
        users = [
            User(name="Young Person", born="2010"),
            User(name="Another Young", born="2015"),
        ]

        result = filter_users_by_birth_year.invoke(
            {"users": users, "max_birth_year": 2000}
        )

        assert "No users found matching criteria" in result

    def test_filter_empty_input(self):
        """Test filtering with empty input"""
        users = []

        result = filter_users_by_birth_year.invoke(
            {"users": users, "max_birth_year": 2000}
        )

        assert "No users found matching criteria" in result

    def test_filter_with_invalid_year_data(self):
        """Test filtering with malformed birth year data"""
        users = [
            User(name="Valid Person", born="1990"),
            User(name="Invalid Year", born="unknown"),
            User(name="Empty Year", born=""),
            User(name="Another Valid", born="1995"),
        ]

        result = filter_users_by_birth_year.invoke(
            {"users": users, "max_birth_year": 2000}
        )

        # Should only include users with valid numeric years
        assert "Valid Person" in result
        assert "Another Valid" in result
        assert "Invalid Year" not in result
        assert "Empty Year" not in result


class TestSelectRandomPeopleFromList(unittest.TestCase):
    """Test suite for select_random_people_from_list tool function"""

    def test_select_fewer_than_available(self):
        """Test selecting fewer people than available"""
        names_list = "Alice Smith, Bob Jones, Charlie Brown, Diana Prince, Eve Wilson"
        result = select_random_people_from_list.invoke(
            {"names_list": names_list, "count": 3}
        )

        names = result.split(", ")
        assert len(names) == 3
        assert all(name in names_list for name in names)

    def test_select_more_than_available(self):
        """Test selecting more people than available"""
        names_list = "Alice Smith, Bob Jones"
        result = select_random_people_from_list.invoke(
            {"names_list": names_list, "count": 10}
        )

        names = result.split(", ")
        assert len(names) == 2

    def test_select_empty_list(self):
        """Test selecting from empty list"""
        result = select_random_people_from_list.invoke({"names_list": "", "count": 5})

        assert "Error" in result

    def test_select_with_whitespace_handling(self):
        """Test that function properly handles extra whitespace in names"""
        names_list = " Alice Smith ,  Bob Jones  , Charlie Brown "
        result = select_random_people_from_list.invoke(
            {"names_list": names_list, "count": 2}
        )

        names = result.split(", ")
        assert len(names) == 2
        # Names should be trimmed of whitespace
        assert all(name.strip() == name for name in names)
        assert all(
            name in ["Alice Smith", "Bob Jones", "Charlie Brown"] for name in names
        )


class TestSearchWikipediaForPerson(unittest.TestCase):
    """Test suite for search_wikipedia_for_person tool function"""

    @patch("search_tools.WikipediaSearch")
    def test_wikipedia_search_found(self, mock_wiki_class):
        """Test successful Wikipedia search"""
        mock_wiki = Mock()
        mock_wiki_class.return_value = mock_wiki
        mock_wiki.search_person.return_value = {
            "found": True,
            "name": "Albert Einstein",
            "summary": "Albert Einstein was a theoretical physicist.",
            "url": "https://en.wikipedia.org/wiki/Albert_Einstein",
        }

        result = search_wikipedia_for_person.invoke({"person_name": "Albert Einstein"})

        assert "Wikipedia article found" in result
        assert "Albert Einstein" in result
        assert "theoretical physicist" in result
        assert "wikipedia.org" in result

    @patch("search_tools.WikipediaSearch")
    def test_wikipedia_search_not_found(self, mock_wiki_class):
        """Test Wikipedia search when article not found"""
        mock_wiki = Mock()
        mock_wiki_class.return_value = mock_wiki
        mock_wiki.search_person.return_value = {
            "found": False,
            "name": "Random Person",
        }

        result = search_wikipedia_for_person.invoke({"person_name": "Random Person"})

        assert "No Wikipedia article found" in result
        assert "Random Person" in result

    @patch("search_tools.WikipediaSearch")
    def test_wikipedia_search_found_without_url(self, mock_wiki_class):
        """Test Wikipedia search with result but missing URL"""
        mock_wiki = Mock()
        mock_wiki_class.return_value = mock_wiki
        mock_wiki.search_person.return_value = {
            "found": True,
            "name": "Jane Doe",
            "summary": "Jane Doe was a researcher.",
        }

        result = search_wikipedia_for_person.invoke({"person_name": "Jane Doe"})

        assert "Wikipedia article found" in result
        assert "Jane Doe" in result
        assert "researcher" in result
        # Should handle missing URL gracefully
        assert "Source:" not in result or result.count("Source:") == 0

    def test_wikipedia_search_invalid_name(self):
        """Test Wikipedia search with invalid name format"""
        result = search_wikipedia_for_person.invoke({"person_name": "InvalidName"})

        assert "Invalid name format" in result

    @patch("search_tools.WikipediaSearch")
    def test_wikipedia_search_error(self, mock_wiki_class):
        """Test Wikipedia search error handling"""
        mock_wiki = Mock()
        mock_wiki_class.return_value = mock_wiki
        mock_wiki.search_person.side_effect = Exception("Search failed")

        result = search_wikipedia_for_person.invoke({"person_name": "Test Person"})

        assert "Error searching Wikipedia" in result


class TestIdentifyPersonWithLLM(unittest.TestCase):
    """Test suite for identify_person_with_llm tool function"""

    @patch("llm_backend.LLMBackend")
    def test_identify_person_success(self, mock_llm_class):
        """Test successful person identification"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        expected_response = "Albert Einstein was a theoretical physicist."
        mock_llm.identify_person.return_value = expected_response

        result = identify_person_with_llm.invoke({"person_name": "Albert Einstein"})

        # Verify the backend was called with correct arguments
        mock_llm.identify_person.assert_called_once_with("Albert", "Einstein")
        # Verify the response is returned correctly
        assert result == expected_response

    @patch("llm_backend.LLMBackend")
    def test_identify_person_error(self, mock_llm_class):
        """Test person identification error handling"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_llm.identify_person.side_effect = Exception("LLM error")

        result = identify_person_with_llm.invoke({"person_name": "Test Person"})

        assert "Error identifying person" in result

    def test_identify_person_invalid_name(self):
        """Test identification with invalid name format"""
        result = identify_person_with_llm.invoke({"person_name": "SingleName"})

        assert "Invalid name format" in result


class TestCheckIfNotable(unittest.TestCase):
    """Test suite for check_if_notable tool function"""

    def test_check_notable_person(self):
        """Test checking notable person"""
        person_info = "Albert Einstein was a famous theoretical physicist who won the Nobel Prize."

        result = check_if_notable.invoke({"person_info": person_info})

        assert "YES" in result
        assert "famous" in result or "Nobel" in result

    def test_check_unknown_person(self):
        """Test checking unknown person"""
        person_info = "Unknown person"

        result = check_if_notable.invoke({"person_info": person_info})

        assert "NO" in result
        assert "unknown" in result.lower()

    def test_check_error_info(self):
        """Test checking with error information"""
        person_info = "Error identifying person: API failed"

        result = check_if_notable.invoke({"person_info": person_info})

        assert "NO" in result

    def test_check_non_notable_person(self):
        """Test checking person without notable keywords"""
        person_info = "John Smith is a software developer at a small company."

        result = check_if_notable.invoke({"person_info": person_info})

        # Should return UNCERTAIN since no notable keywords are present
        assert "UNCERTAIN" in result
        assert "John Smith" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
