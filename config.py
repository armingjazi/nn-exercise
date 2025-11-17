"""
Configuration module for Exercise 4.
Handles environment variable loading from .env file and provides configuration access.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """
    Configuration handler for the application.
    Loads environment variables from .env file and provides typed access.
    """
    
    _loaded = False
    
    @classmethod
    def load(cls) -> None:
        """
        Load environment variables from .env file.
        This is called automatically on first access but can be called explicitly.
        """
        if cls._loaded:
            return
        
        # Locate .env file in the project root
        env_path = Path(__file__).parent / ".env"
        
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # If .env doesn't exist, still load from system environment
            load_dotenv()
        
        cls._loaded = True
    
    @classmethod
    def get_openrouter_api_key(cls) -> str:
        """
        Get the OpenRouter API key from environment.

        Returns:
            str: The API key

        Raises:
            ValueError: If the API key is not set
        """
        cls.load()

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Please set OPENROUTER_API_KEY in your .env file or environment. "
                "Get a free API key at https://openrouter.io"
            )

        return api_key

    @classmethod
    def get_openrouter_base_url(cls) -> str:
        """
        Get the OpenRouter base URL from environment or return default.

        Returns:
            str: The base URL for OpenRouter API
        """
        cls.load()
        return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    @classmethod
    def get(cls, key: str, default: str | None = None) -> str | None:
        """
        Get an environment variable with optional default.
        
        Args:
            key: Environment variable name
            default: Default value if not found
        
        Returns:
            str | None: The environment variable value or default
        """
        cls.load()
        return os.getenv(key, default)
