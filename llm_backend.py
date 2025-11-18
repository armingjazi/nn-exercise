"""
LLM Backend module for interfacing with OpenRouter API.
Provides a clean abstraction for LLM calls to identify notable people.
"""

from typing import Optional
import requests
import json
from config import Config


class LLMBackend:
    """
    Backend for interacting with OpenRouter LLM API.
    
    OpenRouter provides free tier access to various open-source models.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the LLM backend.

        Args:
            api_key: OpenRouter API key. If None, will load from .env or environment
            model: The model to use. If None, will load from .env or use default (openai/gpt-4o-mini)
            base_url: OpenRouter base URL. If None, will load from config

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = Config.get_openrouter_api_key()

        if model:
            self.model = model
        else:
            self.model = Config.get_openrouter_model()

        if base_url:
            self.base_url = base_url
        else:
            # Get base URL from config and append the chat completions endpoint
            openrouter_base = Config.get_openrouter_base_url()
            self.base_url = f"{openrouter_base}/chat/completions"
    
    def identify_person(self, first_name: str, last_name: str) -> str:
        """
        Identify who a person is using the LLM.
        
        Args:
            first_name: Person's first name
            last_name: Person's last name
        
        Returns:
            str: Information about who the person is
        
        Raises:
            Exception: If the API call fails
        """
        prompt = f"""Briefly identify who {first_name} {last_name} is. 
        
If this is a famous person, provide their profession/notability in 1-2 sentences.
If this is a fictional character, identify them as such.
If you don't recognize this name, respond with "Unknown person".

Keep the response concise (max 3 sentences)."""
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            
            payload = {
                "model": self.model,    
                "max_tokens": 150,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code != 200:
                return f"Error identifying person: Error code: {response.status_code}"
            
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            return f"Error identifying person: {str(e)}"
    
    def identify_person_with_search(self, first_name: str, last_name: str) -> str:
        """
        Identify a person using Wikipedia search + LLM interpretation.

        This approach:
        1. Searches Wikipedia first for factual information
        2. If found, uses LLM to summarize nicely
        3. If not found, falls back to LLM knowledge

        Args:
            first_name: Person's first name
            last_name: Person's last name

        Returns:
            str: Information about who the person is
        """
        from search_tools import WikipediaSearch

        wiki = WikipediaSearch()
        result = wiki.search_person(first_name, last_name)

        if result["found"] and result.get("summary"):
            # Found on Wikipedia - use LLM to create concise summary
            prompt = f"""Based on this Wikipedia information about {result['name']}:

{result['summary']}

Provide a concise 1-2 sentence summary of who they are and what they're known for.
Include their profession/field and main achievement."""

            identification = self.invoke(prompt, max_tokens=150)

            # Add Wikipedia reference
            if result.get("url"):
                identification += f"\n\n(Source: Wikipedia - {result['url']})"

            return identification
        else:
            # Not on Wikipedia or search failed - use pure LLM knowledge
            fallback_result = self.identify_person(first_name, last_name)

            if result.get("error"):
                return f"{fallback_result}\n\n(Note: Wikipedia search encountered an error)"
            else:
                return f"{fallback_result}\n\n(Note: No Wikipedia article found)"

    def batch_identify_people(self, people: list[tuple[str, str]]) -> dict[str, str]:
        """
        Identify multiple people in batch.

        Args:
            people: List of tuples (first_name, last_name)

        Returns:
            dict: Mapping of full_name -> identification
        """
        results = {}

        for first_name, last_name in people:
            full_name = f"{first_name} {last_name}"
            identification = self.identify_person(first_name, last_name)
            results[full_name] = identification

        return results

    def batch_identify_people_with_search(self, people: list[tuple[str, str]]) -> dict[str, str]:
        """
        Identify multiple people in batch using Wikipedia search.

        Args:
            people: List of tuples (first_name, last_name)

        Returns:
            dict: Mapping of full_name -> identification
        """
        results = {}

        for first_name, last_name in people:
            full_name = f"{first_name} {last_name}"
            identification = self.identify_person_with_search(first_name, last_name)
            results[full_name] = identification

        return results
    
    def get_model_info(self) -> str:
        """Get information about the current model being used."""
        return f"Using model: {self.model}"
    
    def invoke(self, prompt: str, max_tokens: int = 300) -> str:
        """
        Send a prompt to the LLM and get a response.
        
        This is a flexible method for custom prompts and agentic workflows.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in the response (default: 300)
        
        Returns:
            str: The LLM's response
        
        Raises:
            Exception: If the API call fails
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/",
                "X-Title": "LLM Backend Agent",
            }
            
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                return f"Error: API returned status {response.status_code}"
            
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error: {str(e)}"
