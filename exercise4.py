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


def categorize_identification(full_name: str, identification: str) -> dict:
    """
    Categorize identification results for better presentation.

    Args:
        full_name: Person's full name
        identification: Identification text from LLM/Wikipedia

    Returns:
        dict: Category, confidence, and details
    """
    # Check for Wikipedia verification
    if "(Source: Wikipedia -" in identification:
        category = "✅ Verified Notable Person"
        confidence = "High"
    # Check for errors
    elif "Error" in identification or "error" in identification:
        category = "⚠️ Error"
        confidence = "N/A"
    # Check for no Wikipedia article
    elif "No Wikipedia article found" in identification:
        if "Unknown person" in identification:
            category = "❌ Unknown/Non-Notable"
            confidence = "Low"
        else:
            category = "⚠️ Possibly Notable (Unverified)"
            confidence = "Medium"
    # Check for fictional characters
    elif any(keyword in identification.lower()
             for keyword in ["fictional", "character", "anime", "movie", "tv show", "novel"]):
        category = "🎭 Fictional Character"
        confidence = "High"
    # Default: Unknown
    else:
        category = "❌ Unknown/Non-Notable"
        confidence = "Low"

    return {
        "name": full_name,
        "category": category,
        "confidence": confidence,
        "details": identification
    }


def display_identifications(identifications: dict[str, str]) -> None:
    """
    Display the LLM identifications in a formatted, categorized manner.

    Args:
        identifications: Dictionary mapping full_name -> identification
    """
    # Categorize all results
    categorized = [
        categorize_identification(name, ident)
        for name, ident in identifications.items()
    ]

    # Group by category
    categories = {
        "✅ Verified Notable Person": [],
        "⚠️ Possibly Notable (Unverified)": [],
        "🎭 Fictional Character": [],
        "❌ Unknown/Non-Notable": [],
        "⚠️ Error": []
    }

    for result in categorized:
        categories[result["category"]].append(result)

    # Display header
    print("\n" + "="*80)
    print("IDENTIFICATION RESULTS - CATEGORIZED")
    print("="*80)

    # Quick stats
    total = len(categorized)
    verified = len(categories["✅ Verified Notable Person"])
    unknown = len(categories["❌ Unknown/Non-Notable"])

    print(f"\n📊 Quick Stats: {total} people processed | "
          f"{verified} verified ({verified/total*100:.0f}%) | "
          f"{unknown} unknown ({unknown/total*100:.0f}%)")

    # Display each category
    for category, items in categories.items():
        if items:
            print(f"\n{category} ({len(items)} {'person' if len(items) == 1 else 'people'})")
            print("-" * 80)

            for item in items:
                print(f"\n  {item['name']} (Confidence: {item['confidence']})")

                # Truncate long details for better readability
                details = item['details']
                if len(details) > 300:
                    # Find a good break point (sentence or newline)
                    truncate_at = details.rfind('.', 0, 300)
                    if truncate_at == -1:
                        truncate_at = details.rfind('\n', 0, 300)
                    if truncate_at == -1:
                        truncate_at = 300

                    # Split into lines for indentation
                    for line in details[:truncate_at + 1].split('\n'):
                        if line.strip():
                            print(f"    {line.strip()}")
                    print("    ... (truncated)")
                else:
                    # Split into lines for proper indentation
                    for line in details.split('\n'):
                        if line.strip():
                            print(f"    {line.strip()}")
        else:
            print(f"\n{category} (0 people)")
            print("-" * 80)
            print("  (none found)")

    # Display summary statistics
    display_summary_statistics(categorized)

    print("\n" + "="*80 + "\n")


def display_summary_statistics(categorized_results: list[dict]) -> None:
    """
    Display summary statistics of the identification results.

    Args:
        categorized_results: List of categorized identification results
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    total = len(categorized_results)
    if total == 0:
        print("No results to display")
        return

    # Count by category
    category_counts = {}
    for result in categorized_results:
        cat = result["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Display counts
    print(f"\nTotal People Processed: {total}")
    for category, count in category_counts.items():
        percentage = (count / total) * 100
        print(f"{category}: {count} ({percentage:.1f}%)")

    # Success rate
    verified = category_counts.get("✅ Verified Notable Person", 0)
    print(f"\nVerification Success Rate: {verified}/{total} ({verified/total*100:.1f}%)")

    # Helpful note
    if verified == 0:
        print("\n💡 Note: Most random users from randomuser.me are fictional and won't have")
        print("   Wikipedia articles. This is expected behavior. To see verified results,")
        print("   the API would need to return names matching real notable people.")


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
