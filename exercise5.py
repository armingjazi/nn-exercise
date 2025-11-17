"""
Exercise 5: Agentic workflow using LangChain to get details on best work
This program uses LangChain's agent framework to create an intelligent agent
that researches and provides details about notable people's best work.
"""

import random
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
import requests
from config import Config


def fetch_users_from_api(num_results: int = 20) -> str:
    """
    Fetch random users from the Random User API.

    Args:
        num_results: Number of users to fetch (default: 20)

    Returns:
        str: Summary of fetched users with their names and birth years
    """
    try:
        response = requests.get(
            f"https://randomuser.me/api/?results={num_results}",
            timeout=10
        )
        if response.status_code != 200:
            return f"Error: API returned status code {response.status_code}"

        data = response.json()
        users_summary = []
        for user in data.get('results', []):
            first = user['name']['first'].title()
            last = user['name']['last'].title()
            year = user.get('dob', {}).get('date', '')[:4]
            users_summary.append(f"{first} {last} (born {year})")

        return f"Fetched {len(users_summary)} users:\n" + "\n".join(users_summary)
    except Exception as e:
        return f"Error fetching users: {str(e)}"


def filter_users_by_birth_year(users_summary: str, max_birth_year: int = 2000) -> str:
    """
    Filter users to only include those born in or before the specified year.

    Args:
        users_summary: Summary string from fetch_users_from_api
        max_birth_year: Maximum birth year to include (default: 2000)

    Returns:
        str: Comma-separated list of full names born before or in the year
    """
    try:
        filtered_names = []
        lines = users_summary.split('\n')[1:]  # Skip first summary line

        for line in lines:
            if '(born' in line:
                name_part = line.split('(born')[0].strip()
                year_part = line.split('(born')[1].strip(' )')

                if year_part.isdigit() and int(year_part) <= max_birth_year:
                    filtered_names.append(name_part)

        return ", ".join(filtered_names) if filtered_names else "No users found matching criteria"
    except Exception as e:
        return f"Error filtering users: {str(e)}"


def select_random_people_from_list(names_list: str, count: int = 5) -> str:
    """
    Select random people from a comma-separated list of names.

    Args:
        names_list: Comma-separated list of full names
        count: Number of people to select (default: 5)

    Returns:
        str: Comma-separated list of selected names
    """
    try:
        names = [name.strip() for name in names_list.split(',') if name.strip()]
        if not names:
            return "Error: No names provided"

        sample_size = min(count, len(names))
        selected = random.sample(names, sample_size)
        return ", ".join(selected)
    except Exception as e:
        return f"Error selecting people: {str(e)}"


def search_wikipedia_for_person(person_name: str) -> str:
    """
    Search Wikipedia for information about a person.

    Args:
        person_name: Full name of the person (FirstName LastName)

    Returns:
        str: Wikipedia summary if found, or message indicating not found
    """
    try:
        from search_tools import WikipediaSearch
        parts = person_name.split(' ', 1)
        if len(parts) != 2:
            return "Invalid name format. Expected 'FirstName LastName'"

        first_name, last_name = parts
        wiki = WikipediaSearch()
        result = wiki.search_person(first_name, last_name)

        if result["found"] and result.get("summary"):
            output = f"Wikipedia article found for {result['name']}:\n\n"
            output += result['summary']
            if result.get("url"):
                output += f"\n\nSource: {result['url']}"
            return output
        else:
            return f"No Wikipedia article found for {person_name}"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


def identify_person_with_llm(person_name: str) -> str:
    """
    Identify who a person is using the LLM backend.

    Args:
        person_name: Full name of the person (FirstName LastName)

    Returns:
        str: Information about who the person is
    """
    try:
        from llm_backend import LLMBackend
        parts = person_name.split(' ', 1)
        if len(parts) != 2:
            return "Invalid name format. Expected 'FirstName LastName'"

        first_name, last_name = parts
        llm = LLMBackend()
        return llm.identify_person(first_name, last_name)
    except Exception as e:
        return f"Error identifying person: {str(e)}"


def check_if_notable(person_info: str) -> str:
    """
    Analyze if a person is notable based on their initial identification.

    Args:
        person_info: Initial identification information about the person

    Returns:
        str: YES or NO with reasoning
    """
    # This is a tool that the agent can use
    # The actual LLM will be called by the agent to make this decision
    if "Unknown" in person_info or "Error" in person_info:
        return "NO - Person could not be identified or is unknown"

    # Keywords indicating notability
    notable_keywords = [
        "famous", "renowned", "acclaimed", "Nobel", "Oscar", "Grammy",
        "president", "prime minister", "CEO", "founder", "inventor",
        "scientist", "author", "actor", "actress", "musician", "artist",
        "director", "producer", "athlete", "champion"
    ]

    person_info_lower = person_info.lower()
    if any(keyword in person_info_lower for keyword in notable_keywords):
        return f"YES - Contains notable indicators: {person_info}"

    return f"UNCERTAIN - Needs further research: {person_info}"


def research_best_work(person_name: str, person_info: str) -> str:
    """
    Research and return details about a person's best work or notable achievement.

    Args:
        person_name: Full name of the person
        person_info: Initial identification information

    Returns:
        str: Detailed description of their best work
    """
    # This tool will be called by the agent when it needs to research best work
    prompt = f"""Based on this information about {person_name}:
{person_info}

Provide a detailed description (3-5 sentences) of their most notable work or achievement.
Focus on:
1. What they are most famous for
2. The impact or significance of this work
3. Any awards or recognition received

If they have multiple notable works, focus on the most significant one."""

    # This will be handled by the agent's LLM
    return prompt


class BestWorkAgent:
    """
    LangChain agent that intelligently researches people's best work
    using an agentic workflow with tools.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None
    ):
        """
        Initialize the agent with OpenRouter configuration.

        Args:
            model_name: The model to use (default: openai/gpt-4o)
            api_key: OpenRouter API key. If None, will load from config
            base_url: OpenRouter base URL. If None, will load from config
        """
        if api_key is None:
            api_key = Config.get_openrouter_api_key()

        if base_url is None:
            base_url = Config.get_openrouter_base_url()

        # Initialize the chat model using LangChain's init_chat_model
        self.model = init_chat_model(
            model=model_name,
            model_provider="openai",
            base_url=base_url,
            api_key=api_key,
        )

        # Create the agent with all available tools
        self.agent = create_agent(
            model=self.model,
            tools=[
                fetch_users_from_api,
                filter_users_by_birth_year,
                select_random_people_from_list,
                search_wikipedia_for_person,
                identify_person_with_llm,
                check_if_notable,
                research_best_work
            ],
            system_prompt="""You are a research assistant specialized in identifying notable people
and their achievements. You have access to tools to:

1. Fetch random users from an API
2. Filter users by birth year
3. Select random people from a list
4. Search Wikipedia for person information (use this FIRST before LLM identification)
5. Identify who people are using an LLM (fallback if Wikipedia has nothing)
6. Determine if a person is notable
7. Research their best work or most significant achievement

Be autonomous, use the tools intelligently, and provide detailed, informative results.
Always try Wikipedia search first before using LLM identification."""
        )


def main():
    """Main entry point for Exercise 5"""
    try:
        # Initialize the LangChain agent
        print("🤖 Initializing LangChain agent with init_chat_model...")
        agent = BestWorkAgent(model_name="openai/gpt-4o")
        print("✅ Agent initialized with 7 research tools:")
        print("   1. fetch_users_from_api")
        print("   2. filter_users_by_birth_year")
        print("   3. select_random_people_from_list")
        print("   4. search_wikipedia_for_person (NEW!)")
        print("   5. identify_person_with_llm")
        print("   6. check_if_notable")
        print("   7. research_best_work")

        # Let the agent autonomously execute the workflow
        print("\n🚀 Starting agentic workflow...")
        print("The agent will autonomously:")
        print("   - Fetch users from the API")
        print("   - Filter by birth year (≤2000)")
        print("   - Select 5 random people")
        print("   - Identify each person")
        print("   - Assess notability")
        print("   - Research their best work")
        print("\n" + "="*80)

        # Task for the agent
        task_message = """Please complete the following research task:

1. Fetch 20 random users from the API using fetch_users_from_api
2. Filter those users to only include people born in 2000 or earlier using filter_users_by_birth_year
3. Select 5 random people from the filtered list using select_random_people_from_list
4. For each of the 5 selected people:
   a. FIRST search Wikipedia using search_wikipedia_for_person
   b. If Wikipedia has no info, use identify_person_with_llm as fallback
   c. Check if they are notable using check_if_notable
   d. If notable, research their best work using research_best_work

Provide a comprehensive summary of your findings for each person."""

        # Invoke the agent
        response = agent.agent.invoke({
            "messages": [{"role": "user", "content": task_message}]
        })

        # Display the agent's response
        print("\n🎯 AGENT WORKFLOW COMPLETE")
        print("="*80)
        print("\nAgent's Final Report:")
        print("="*80)

        # Extract the final message content
        final_message = response["messages"][-1]
        if hasattr(final_message, 'content'):
            agent_output = final_message.content
        else:
            agent_output = str(final_message)

        print(agent_output)

        print("\n✨ Exercise 5 completed successfully!")
        print("\n💡 This agentic workflow:")
        print("   1. Uses LangChain's init_chat_model with OpenRouter")
        print("   2. Uses create_agent framework with 7 custom tools")
        print("   3. Includes Wikipedia search for factual person identification")
        print("   4. Agent autonomously decides which tools to use and when")
        print("   5. Complete workflow delegation - agent drives the entire process")
        print("   6. Multi-step reasoning from data fetching to best work research")

        return 0

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("\n💡 To use OpenRouter:")
        print("   1. Sign up at https://openrouter.io (free tier available)")
        print("   2. Get your API key from the dashboard")
        print("   3. Set the OPENROUTER_API_KEY environment variable:")
        print("      export OPENROUTER_API_KEY='your_api_key_here'")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
