"""
Exercise 1: Fetch data from the Random User API
This simple program fetches 20 random user records from the API endpoint.
"""

import requests


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
    response.raise_for_status()  # Raise exception for bad status codes
    
    return response.json()


def main():
    """Main entry point for Exercise 1"""
    try:
        print("Fetching 20 random users from the Random User API...")
        data = fetch_random_users(results=20)
        

        print(f"\nSuccessfully fetched {len(data['results'])} user records")
        print(f"API returned: {data['info']}")
        
        if data['results']:
            first_user = data['results'][0]
            print("\nExample first user:")
            print(f"  Name: {first_user['name']['first']} {first_user['name']['last']}")
            print(f"  Email: {first_user['email']}")
            print(f"  DOB: {first_user['dob']['date']}")
            
    except requests.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
