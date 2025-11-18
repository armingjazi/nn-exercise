"""
Exercise 3: Test suite for Exercise 2 functionality
This module contains comprehensive tests for the user data fetching and filtering logic.
"""

import pytest
import requests
from unittest.mock import Mock, patch
from exercise2 import fetch_random_users, format_and_filter_users


# Sample mock data for testing
MOCK_API_RESPONSE = {
    'results': [
        {
            'name': {'first': 'James', 'last': 'Bond'},
            'dob': {'date': '1980-04-13T10:28:45.078Z'}
        },
        {
            'name': {'first': 'Ash', 'last': 'Ketchum'},
            'dob': {'date': '1995-07-22T14:15:30.500Z'}
        },
        {
            'name': {'first': 'John', 'last': 'Doe'},
            'dob': {'date': '2001-12-25T08:00:00.000Z'}
        },
        {
            'name': {'first': 'Jane', 'last': 'Smith'},
            'dob': {'date': '2000-01-01T12:00:00.000Z'}
        },
        {
            'name': {'first': 'Robert', 'last': 'Johnson'},
            'dob': {'date': '1990-06-15T09:30:45.123Z'}
        },
    ],
    'info': {'seed': 'test', 'results': 5, 'page': 1, 'version': '1.4'}
}


class TestFormatAndFilterUsers:
    """Test suite for format_and_filter_users function"""

    def test_includes_users_born_in_2000_or_earlier(self):
        """Test that users born in 2000 or earlier are included"""
        result = format_and_filter_users(MOCK_API_RESPONSE)

        # These SHOULD be in results (born 2000 or earlier)
        assert 'James Bond' in result
        assert 'Ash Ketchum' in result
        assert 'Jane Smith' in result  # Born exactly in 2000
        assert 'Robert Johnson' in result

    def test_filters_out_users_born_after_2000(self):
        """Test that users born after 2000 are filtered out"""
        result = format_and_filter_users(MOCK_API_RESPONSE)

        # John Doe (born 2001) should NOT be in results
        assert 'John Doe' not in result

    def test_full_name_format(self):
        """Test that full names are formatted correctly (first + last)"""
        result = format_and_filter_users(MOCK_API_RESPONSE)
        
        # Check format is "FirstName LastName"
        for name in result:
            parts = name.split(' ')
            assert len(parts) == 2, f"Name '{name}' should have exactly 2 parts"
    
    def test_empty_results_list(self):
        """Test handling of API response with no results"""
        empty_response = {'results': []}
        result = format_and_filter_users(empty_response)

        assert result == []
    
    def test_all_users_born_after_2000(self):
        """Test when all users are born after 2000"""
        response = {
            'results': [
                {
                    'name': {'first': 'Alice', 'last': 'Wonder'},
                    'dob': {'date': '2005-03-10T10:00:00.000Z'}
                },
                {
                    'name': {'first': 'Bob', 'last': 'Builder'},
                    'dob': {'date': '2010-06-20T12:00:00.000Z'}
                },
            ]
        }
        result = format_and_filter_users(response)

        assert len(result) == 0

    def test_preserves_name_order(self):
        """Test that the order of names is preserved from API response"""
        result = format_and_filter_users(MOCK_API_RESPONSE)

        # 4 users born in 2000 or earlier, in original order
        assert len(result) == 4
        assert result[0] == 'James Bond'
        assert result[1] == 'Ash Ketchum'
        assert result[2] == 'Jane Smith'
        assert result[3] == 'Robert Johnson'
    
    def test_handles_unicode_names(self):
        """Test that unicode characters in names are handled correctly"""
        response = {
            'results': [
                {
                    'name': {'first': 'José', 'last': 'García'},
                    'dob': {'date': '1995-05-10T10:00:00.000Z'}
                },
                {
                    'name': {'first': '李', 'last': '明'},
                    'dob': {'date': '1998-08-15T14:00:00.000Z'}
                },
            ]
        }
        result = format_and_filter_users(response)

        # Both born before 2000, so should be included
        assert 'José García' in result
        assert '李 明' in result
        assert len(result) == 2
    
    def test_invalid_date_format_raises_error(self):
        """Test that invalid date format raises ValueError"""
        invalid_response = {
            'results': [
                {
                    'name': {'first': 'Test', 'last': 'User'},
                    'dob': {'date': 'invalid-date'}
                }
            ]
        }
        
        with pytest.raises(ValueError):
            format_and_filter_users(invalid_response)
    
    def test_missing_dob_raises_error(self):
        """Test that missing 'dob' field raises KeyError"""
        invalid_response = {
            'results': [
                {
                    'name': {'first': 'Test', 'last': 'User'},
                }
            ]
        }
        
        with pytest.raises(KeyError):
            format_and_filter_users(invalid_response)
    
    def test_missing_name_raises_error(self):
        """Test that missing 'name' field raises KeyError"""
        invalid_response = {
            'results': [
                {
                    'dob': {'date': '1995-01-01T10:00:00.000Z'},  # Born before 2000, would be included
                }
            ]
        }

        with pytest.raises(KeyError):
            format_and_filter_users(invalid_response)


class TestFetchRandomUsers:
    """Test suite for fetch_random_users function"""
    
    @patch('exercise2.requests.get')
    def test_fetch_random_users_success(self, mock_get):
        """Test successful API call"""
        mock_response = Mock()
        mock_response.json.return_value = MOCK_API_RESPONSE
        mock_get.return_value = mock_response

        result = fetch_random_users(results=5)

        assert result == MOCK_API_RESPONSE
        mock_get.assert_called_once_with('https://randomuser.me/api/?results=5')

    @patch('exercise2.requests.get')
    def test_fetch_random_users_http_error(self, mock_get):
        """Test that HTTP errors are properly raised"""
        mock_get.side_effect = requests.RequestException('HTTP Error 500')

        with pytest.raises(requests.RequestException):
            fetch_random_users()


class TestIntegration:
    """Integration tests combining fetch and filter functions"""
    
    @patch('exercise2.requests.get')
    def test_end_to_end_fetch_and_filter(self, mock_get):
        """Test complete workflow: fetch API data and filter"""
        mock_response = Mock()
        mock_response.json.return_value = MOCK_API_RESPONSE
        mock_get.return_value = mock_response

        # Fetch data
        api_data = fetch_random_users(results=5)

        # Filter data
        filtered_users = format_and_filter_users(api_data)

        # Verify results - John Doe (born 2001) should be filtered out
        # James, Ash, Jane, Robert (born 2000 or earlier) should be included
        assert len(filtered_users) == 4
        assert 'John Doe' not in filtered_users
        assert 'James Bond' in filtered_users
        assert 'Ash Ketchum' in filtered_users
        assert 'Jane Smith' in filtered_users
        assert 'Robert Johnson' in filtered_users


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
