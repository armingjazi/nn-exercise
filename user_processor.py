"""
User processor module - core functionality for fetching and processing user data.
This module contains reusable functions for Exercise 2 and beyond.
"""

import requests
from typing import List


def fetch_random_users(results: int = 20) -> dict:
    """
    Fetches random user data from the Random User API.
    
    Args:
        results: Number of user records to fetch (default: 20)
    
    Returns:
        dict: The JSON response from the API
    
    Raises:
        requests.RequestException: If the API request fails
    """
    api_url = f"https://randomuser.me/api/?results={results}"
    
    response = requests.get(api_url)
    response.raise_for_status()
    
    return response.json()


def format_and_filter_users(api_response: dict) -> List[str]:
    """
    Formats and filters the API response.
    
    Extracts full names from user records and filters out anyone born after 2000.
    
    Args:
        api_response: The JSON response from the Random User API
    
    Returns:
        List[str]: Array of full names (first name + last name) for users born in 2000 or earlier
    
    Raises:
        KeyError: If the API response structure is invalid
        ValueError: If the date format is invalid
    """
    filtered_users = []
    
    for user in api_response['results']:
        # Extract birth year from DOB
        dob_string = user['dob']['date']
        birth_year = int(dob_string.split('-')[0])
        
        # Filter: only include users born in 2000 or earlier
        if birth_year <= 2000:
            first_name = user['name']['first']
            last_name = user['name']['last']
            full_name = f"{first_name} {last_name}"
            filtered_users.append(full_name)
    
    return filtered_users


def display_results(users: List[str]) -> None:
    """
    Display the formatted results in a clean array format.
    
    Args:
        users: List of user full names
    """
    print("[")
    for user in users:
        print(f"    '{user}',")
    print("]")
