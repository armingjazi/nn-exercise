"""
Exercise 2: Format and filter user data
This program fetches random users, filters those born after 2000,
and formats the results as an array of full names.
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
    """
    filtered_users = []

    for user in api_response['results']:
        # Extract birth year from DOB
        dob_string = user['dob']['date']
        birth_year = int(dob_string.split('-')[0])

        # Filter out users born after 2000: only include users born in 2000 or earlier
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


def main():
    """Main entry point for Exercise 2"""
    try:
        print("Fetching 20 random users from the Random User API...\n")
        data = fetch_random_users(results=20)
        
        print(f"Fetched {len(data['results'])} user records")
        print("Filtering out users born after 2000...\n")

        filtered_users = format_and_filter_users(data)

        print(f"Found {len(filtered_users)} users born in 2000 or earlier:\n")
        display_results(filtered_users)
        
    except requests.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return 1
    except (KeyError, ValueError, IndexError) as e:
        print(f"Error processing API response: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
