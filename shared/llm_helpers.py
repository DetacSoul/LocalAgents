"""
Shared LLM utilities for all agents in this repo.
Import from any agent: from shared import invoke_with_retry, load_config, get_llm
"""

import time
import logging
import yaml
import ollama

logger = logging.getLogger(__name__)


class LLMInvocationError(Exception):
    """Raised when LLM invocation fails after all retries."""
    pass


def invoke_with_retry(
    llm,
    prompt: str,
    prompt_name: str = "prompt",
    max_retries: int = 3,
    retry_delay: float = 2.0,
):
    """
    Invoke an LLM with exponential backoff retry logic.

    Args:
        llm: The LangChain-compatible LLM instance.
        prompt: The prompt string to send.
        prompt_name: Label for log messages.
        max_retries: Maximum number of attempts.
        retry_delay: Base delay in seconds (doubles each retry).

    Returns:
        The LLM response object.

    Raises:
        LLMInvocationError: If all retries are exhausted.
    """
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Invoking LLM for {prompt_name} (attempt {attempt}/{max_retries})")
            response = llm.invoke(prompt)
            logger.info(f"Successfully invoked LLM for {prompt_name}")
            return response

        except (ollama.ResponseError, TimeoutError, ConnectionError, OSError) as e:
            last_exception = e
            logger.warning(
                f"LLM invocation failed for {prompt_name} "
                f"(attempt {attempt}/{max_retries}): {type(e).__name__} - {e!s}"
            )

            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.info(f"Retrying {prompt_name} in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed for {prompt_name}")

    raise LLMInvocationError(
        f"LLM invocation failed for {prompt_name} after {max_retries} attempts. "
        f"Last error: {type(last_exception).__name__} - {last_exception!s}"
    )


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return a YAML config file as a dictionary."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_llm(config: dict):
    """
    Instantiate an LLM based on config settings.
    Currently supports Ollama. Extend for OpenAI/Anthropic as needed.
    """
    from langchain_ollama import ChatOllama

    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "ollama")

    if provider == "ollama":
        return ChatOllama(
            model=llm_config.get("model", "qwen2.5:14b"),
            base_url=llm_config.get("base_url", "http://localhost:11434"),
            temperature=llm_config.get("temperature", 0.4),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
