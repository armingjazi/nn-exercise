"""
Search tools for identifying people via Wikipedia.
Provides utilities to search Wikipedia and retrieve person information.
"""

import requests
from typing import Dict


class WikipediaSearch:
    """Wikipedia search tool for identifying notable people."""

    BASE_URL = "https://en.wikipedia.org/w/api.php"
    REST_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"

    def __init__(self):
        """Initialize Wikipedia search with appropriate headers."""
        self.headers = {
            "User-Agent": "NN-Exercise/1.0 (educational project)"
        }

    def search_person(self, first_name: str, last_name: str) -> Dict[str, any]:
        """
        Search Wikipedia for a person and return their information.

        Args:
            first_name: Person's first name
            last_name: Person's last name

        Returns:
            dict: Dictionary containing:
                - found (bool): Whether a Wikipedia article was found
                - name (str): Full name searched
                - title (str): Wikipedia article title (if found)
                - summary (str): Article summary/extract (if found)
                - url (str): Wikipedia article URL (if found)
                - error (str): Error message (if search failed)
        """
        full_name = f"{first_name} {last_name}"

        # Search Wikipedia using OpenSearch API
        search_params = {
            "action": "opensearch",
            "search": full_name,
            "limit": 1,
            "namespace": 0,  # Main namespace only
            "format": "json"
        }

        try:
            response = requests.get(
                self.BASE_URL,
                params=search_params,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            search_results = response.json()

            # Check if any results were found
            if not search_results[1]:  # No results in titles array
                return {
                    "found": False,
                    "name": full_name,
                    "summary": None,
                    "url": None
                }

            # Extract first result
            title = search_results[1][0]
            url = search_results[3][0] if len(search_results) > 3 and search_results[3] else None

            # Get detailed summary using REST API
            summary_url = f"{self.REST_URL}/{title.replace(' ', '_')}"
            summary_response = requests.get(
                summary_url,
                headers=self.headers,
                timeout=10
            )

            if summary_response.status_code == 200:
                summary_data = summary_response.json()
                summary = summary_data.get("extract", "")
            else:
                # Fallback to description from OpenSearch
                summary = search_results[2][0] if len(search_results) > 2 and search_results[2] else ""

            return {
                "found": True,
                "name": full_name,
                "title": title,
                "summary": summary,
                "url": url
            }

        except requests.RequestException as e:
            # Network or API error
            return {
                "found": False,
                "name": full_name,
                "summary": None,
                "url": None,
                "error": f"Wikipedia search failed: {str(e)}"
            }
        except (IndexError, KeyError) as e:
            # Unexpected response format
            return {
                "found": False,
                "name": full_name,
                "summary": None,
                "url": None,
                "error": f"Unexpected Wikipedia response format: {str(e)}"
            }
