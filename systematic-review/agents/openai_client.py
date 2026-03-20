"""
OpenAI API client wrapper with retry logic, error handling, and cost tracking.
"""
import json
import re
import time
import logging
from typing import Optional, Dict, Any, Tuple
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from config.settings import settings

logger = logging.getLogger(__name__)

_CTRL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# Models that require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _token_limit_key(model: str) -> str:
    m = model.lower()
    if any(m.startswith(p) for p in _MAX_COMPLETION_PREFIXES):
        return "max_completion_tokens"
    return "max_tokens"


def _sanitize(text: str) -> str:
    """Strip null bytes and control characters that make JSON requests invalid."""
    return _CTRL_CHARS.sub(' ', text)


class OpenAIClient:
    """Thin wrapper around OpenAI client with retry and JSON parsing."""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._usage_log: list = []

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
        response_format: Optional[str] = "json_object",
    ) -> Tuple[str, Dict[str, int]]:
        """
        Send a chat completion request.

        Returns:
            (content_string, usage_dict)
        Raises:
            RuntimeError after max_retries exhausted
        """
        messages = [
            {"role": "system", "content": _sanitize(system_prompt)},
            {"role": "user", "content": _sanitize(user_prompt)},
        ]

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            _token_limit_key(model): max_tokens,
        }

        # Only add response_format for models that support it
        if response_format == "json_object" and "gpt" in model.lower():
            kwargs["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                    "model": model,
                }
                self._usage_log.append(usage)
                return content, usage

            except RateLimitError as e:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limit hit, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                last_error = e

            except APITimeoutError as e:
                wait = 2 ** attempt
                logger.warning(f"Timeout, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                last_error = e

            except APIError as e:
                logger.error(f"API error: {e}")
                last_error = e
                break

        raise RuntimeError(f"OpenAI API failed after {max_retries} attempts: {last_error}")

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Like chat() but parses JSON response.
        Returns (parsed_dict, usage_dict)
        """
        content, usage = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract JSON if wrapped in markdown
        json_str = content.strip()
        if json_str.startswith("```"):
            # Strip markdown code blocks
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            return json.loads(json_str), usage
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nContent: {content[:500]}")
            raise ValueError(f"Model returned invalid JSON: {e}") from e

    def get_usage_summary(self) -> Dict[str, Any]:
        """Return aggregate token usage across all calls."""
        if not self._usage_log:
            return {"total_calls": 0, "total_tokens": 0}

        total_prompt = sum(u["prompt_tokens"] for u in self._usage_log)
        total_completion = sum(u["completion_tokens"] for u in self._usage_log)
        models_used = list(set(u["model"] for u in self._usage_log))

        return {
            "total_calls": len(self._usage_log),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "models_used": models_used,
        }


# Module-level singleton
_client: Optional[OpenAIClient] = None


def get_client() -> OpenAIClient:
    global _client
    if _client is None:
        _client = OpenAIClient()
    return _client
