"""The OpenRouter client that the explain and score routes give to sae-auto-interp."""

import json
from asyncio import sleep
from typing import Any

from sae_auto_interp.clients import OpenRouter
from sae_auto_interp.clients.openrouter import Response
from sae_auto_interp.logger import logger

# The parent's limit is 500 tokens. A reasoning model can use all of it before it replies.
DEFAULT_MAX_TOKENS = 4000
REASONING_EFFORT = "low"


class LowReasoningOpenRouter(OpenRouter):
    """sae-auto-interp's OpenRouter client, with low reasoning effort and a larger token limit."""

    def request_body(self, prompt: Any, **kwargs: Any) -> dict[str, Any]:
        """Make the chat completion body. OpenRouter ignores `reasoning` for models without it."""
        return {
            "model": self.model,
            "messages": prompt,
            "max_tokens": kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            "temperature": kwargs.get("temperature", 1.0),
            "reasoning": {"effort": REASONING_EFFORT},
        }

    async def generate(self, prompt: Any, raw: bool = False, max_retries: int = 1, **kwargs: Any) -> Response:
        """Send the request and return the reply text, with the same retries as the parent."""
        data = self.request_body(prompt, **kwargs)
        for attempt in range(max_retries):
            try:
                response = await self.client.post(url=self.url, json=data, headers=self.headers)
                if raw:
                    return response.json()
                return self.postprocess(response)
            except json.JSONDecodeError:
                logger.warning(f"Attempt {attempt + 1}: Invalid JSON response, retrying...")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {e!s}, retrying...")
            await sleep(1)
        logger.error("All retry attempts failed.")
        raise RuntimeError("Failed to generate text after multiple attempts.")
