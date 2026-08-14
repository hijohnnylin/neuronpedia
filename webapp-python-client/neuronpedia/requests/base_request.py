import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class NPKeyMissingError(Exception):
    """No API key provided. Use context manager, global setting, parameter, or environment variable."""

    pass


class NPUnauthorizedError(Exception):
    """Unauthorized. Check your API key configuration."""

    pass


class NPRateLimitError(Exception):
    """Rate limit exceeded. Please try again later or email support@neuronpedia.org to raise your limit."""

    pass


class NPInvalidResponseError(Exception):
    """The API returned an invalid JSON response. Please contact support@neuronpedia.org."""

    pass


class NPRequest:
    USE_LOCALHOST = os.getenv("USE_LOCALHOST", "false").lower() == "true"
    BASE_URL = (
        "https://neuronpedia.org/api"
        if not USE_LOCALHOST
        else "http://localhost:3000/api"
    )

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
    ):
        if not self.BASE_URL:
            raise ValueError("BASE_URL must be set in a subclass or globally.")
        self.endpoint = endpoint
        self.api_key = self._get_api_key(api_key)

    @staticmethod
    def _get_api_key(api_key: Optional[str] = None):
        """Retrieve and validate the NEURONPEDIA_API_KEY from parameter, context, or environment variable."""
        # 1. Use explicitly passed parameter if provided
        if api_key:
            return api_key

        # 2. Check context variable
        try:
            from neuronpedia import get_api_key

            context_api_key = get_api_key()
            if context_api_key:
                return context_api_key
        except ImportError:
            # Handle case where neuronpedia module isn't fully loaded yet
            pass

        # 3. Fall back to environment variable
        env_api_key = os.getenv("NEURONPEDIA_API_KEY")
        if not env_api_key:
            raise NPKeyMissingError(
                "No API key provided. Use one of these methods:\n"
                "1. Context manager: with neuronpedia.api_key('key'): ...\n"
                "2. Global setting: neuronpedia.set_api_key('key')\n"
                "3. Parameter: FeatureRequest(api_key='key')\n"
                "4. Environment: set NEURONPEDIA_API_KEY variable"
            )
        return env_api_key

    def get_url(
        self,
    ):
        """Construct the full URL for the request."""
        return f"{self.BASE_URL}/{self.endpoint}"

    def send_request(
        self,
        uri: str = "",
        method: str = "POST",
        **kwargs,
    ):
        """Send an HTTP request."""
        headers = kwargs.get(
            "headers",
            {},
        )
        headers["X-Api-Key"] = self.api_key
        headers["Accept-Encoding"] = "gzip"
        url = f"{self.BASE_URL}/{self.endpoint}/{uri}"
        print(f"Sending {method} request to {url}")
        # print(f"Body: {json.dumps(kwargs.get('json', {}), indent=2)}")

        response = requests.request(
            method,
            url,
            headers=headers,
            **kwargs,
        )

        if response.status_code == 401:
            raise NPUnauthorizedError(
                "Unauthorized. Check your API key using one of these methods:\n"
                "1. Context: with neuronpedia.api_key('key'): ...\n"
                "2. Global: neuronpedia.set_api_key('key')\n"
                "3. Parameter: Request(api_key='key')\n"
                "4. Environment: NEURONPEDIA_API_KEY variable"
            )
        elif response.status_code == 404:
            raise requests.exceptions.HTTPError(
                f"Resource not found: {response.status_code}"
            )
        elif response.status_code == 429:
            raise NPRateLimitError(
                "You've exceeded the API rate limit. Try later or email support@neuronpedia.org to raise your limit."
            )
        elif 400 <= response.status_code < 500:
            raise requests.exceptions.HTTPError(
                f"Request failed with status {response.status_code}: {response.json()}"
            )
        elif 500 <= response.status_code < 600:
            raise requests.exceptions.HTTPError(
                f"Server error occurred: {response.status_code}. Try again later or contact support@neuronpedia.org."
            )
        response.raise_for_status()  # Raise an exception for HTTP errors
        try:
            response_json = response.json()
            print("Got a successful response.")
            # print(json.dumps(response_json, indent=2))
            return response_json
        except requests.exceptions.JSONDecodeError:
            print("Error: Status code:", response.status_code)
            print(f"Error: Response text: {response.text[:1024]}")
            raise NPInvalidResponseError(
                "The API returned an invalid JSON response. Please contact support@neuronpedia.org."
            )
