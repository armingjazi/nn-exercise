"""
Exercise 4: Identify people using LLM with Wikipedia Search
This program fetches random users, filters those born before 2001,
and uses Wikipedia search + LLM to identify who they are.
If a Wikipedia article exists, factual information is provided.
Otherwise, the LLM uses its knowledge (limiting to 5 people to respect rate limits).
"""

import random
from typing import List, Tuple
from exercise2 import fetch_random_users, format_and_filter_users
from llm_backend import LLMBackend


def select_random_people(filtered_names: List[str], count: int = 5) -> List[Tuple[str, str]]:
    """
    Select random people from the filtered list and return as (first_name, last_name) tuples.
    
    Args:
        filtered_names: List of full names formatted as "FirstName LastName"
        count: Number of people to select (default: 5)
    
    Returns:
        List of tuples (first_name, last_name)
    """
    # Ensure we don't select more than available
    sample_size = min(count, len(filtered_names))
    selected = random.sample(filtered_names, sample_size)
    
    # Convert "FirstName LastName" to (first_name, last_name) tuples
    people = []
    for full_name in selected:
        parts = full_name.split(' ', 1)
        if len(parts) == 2:
            people.append((parts[0], parts[1]))
    
    return people


def display_identifications(identifications: dict[str, str]) -> None:
    """
    Display the LLM identifications in a formatted manner.
    
    Args:
        identifications: Dictionary mapping full_name -> identification
    """
    print("\n" + "="*70)
    print("LLM IDENTIFICATIONS")
    print("="*70)
    
    for full_name, identification in identifications.items():
        print(f"\n📌 {full_name}:")
        print(f"   {identification}")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main entry point for Exercise 4"""
    try:
        # Step 1: Fetch random users
        print("🔄 Fetching 20 random users from the Random User API...")
        api_response = fetch_random_users(results=20)
        print(f"✅ Fetched {len(api_response['results'])} user records\n")
        
        # Step 2: Filter users born in 2000 or earlier
        print("🔍 Filtering users born in 2000 or earlier...")
        filtered_users = format_and_filter_users(api_response)
        print(f"✅ Found {len(filtered_users)} eligible users\n")
        
        # Step 3: Select 5 random people
        print("🎲 Selecting 5 random people from the filtered list...")
        selected_people = select_random_people(filtered_users, count=5)
        print(f"✅ Selected {len(selected_people)} people:\n")
        for first_name, last_name in selected_people:
            print(f"   - {first_name} {last_name}")
        
        # Step 4: Initialize LLM backend and identify people
        print("\n🤖 Initializing LLM backend (OpenRouter)...")
        llm = LLMBackend()
        print(f"✅ {llm.get_model_info()}\n")
        
        print("🔍 Searching Wikipedia and identifying selected people...")
        print("(Searching Wikipedia first, then using LLM for interpretation)\n")
        identifications = llm.batch_identify_people_with_search(selected_people)
        
        # Step 5: Display results
        display_identifications(identifications)
        
        print("✨ Exercise 4 completed successfully!")
        return 0
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\n💡 To use OpenRouter:")
        print("   1. Sign up at https://openrouter.io (free tier available)")
        print("   2. Get your API key from the dashboard")
        print("   3. Set the OPENROUTER_API_KEY environment variable:")
        print("      export OPENROUTER_API_KEY='your_api_key_here'")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
