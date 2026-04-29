"""
shared/llm_helpers.py
Reusable LLM factory, retry logic, config loader for the LocalAgents monorepo.
Supports OpenAI, Anthropic (Claude), Google (Gemini), Ollama, and xAI (Grok).
"""

import os
import copy
import logging
import yaml
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# ==================== EXCEPTIONS ====================

class LLMConfigError(Exception):
    """Raised for invalid or missing LLM configuration."""
    pass


class LLMInvocationError(Exception):
    """Raised when LLM invocation fails after all retries."""
    pass


# ==================== CONFIG ====================

_config_cache: Dict[str, Dict[str, Any]] = {}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML config with caching. Returns a deep copy so callers
    can modify it without corrupting the cache.

    Tries the given path first, then falls back to common locations.
    """
    config_path = str(config_path)  # Handle Path objects

    if config_path in _config_cache:
        return copy.deepcopy(_config_cache[config_path])

    # Try the given path, then fallbacks
    search_paths = [config_path, "resume_agent/config.yaml", "config.yaml"]
    resolved_path = None

    for candidate in search_paths:
        if os.path.exists(candidate):
            resolved_path = candidate
            break

    if resolved_path is None:
        raise FileNotFoundError(
            f"Config not found. Searched: {', '.join(search_paths)}"
        )

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    _config_cache[config_path] = config
    logger.debug(f"Loaded config from {resolved_path}")
    return copy.deepcopy(config)


def clear_config_cache():
    """Clear the config cache. Useful for testing or hot-reloading."""
    _config_cache.clear()


# ==================== LLM FACTORY ====================

# Provider-to-module mapping: (module, class_name, default_model)
_PROVIDERS = {
    "ollama":    ("langchain_ollama",         "ChatOllama",              "qwen2.5:14b"),
    "openai":    ("langchain_openai",         "ChatOpenAI",              "gpt-4o"),
    "anthropic": ("langchain_anthropic",      "ChatAnthropic",           "claude-sonnet-4-20250514"),
    "google":    ("langchain_google_genai",   "ChatGoogleGenerativeAI",  "gemini-2.0-flash"),
    "xai":       ("langchain_xai",            "ChatXAI",                 "grok-3"),
    "grok":      ("langchain_xai",            "ChatXAI",                 "grok-3"),
}


def get_llm(
    config: Optional[Dict[str, Any]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> BaseChatModel:
    """
    Factory for multi-provider LLMs.

    Priority: explicit args > config dict > config.yaml defaults.

    Args:
        config: Pre-loaded config dict. If None, loads from config.yaml.
        provider: LLM provider name (ollama, openai, anthropic, google, xai/grok).
        model: Model identifier string.
        temperature: Sampling temperature.
        **kwargs: Additional provider-specific arguments.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        LLMConfigError: If the provider is not supported or import fails.
    """
    if config is None:
        config = load_config()

    llm_config = config.get("llm", {})

    provider = (provider or llm_config.get("provider", "ollama")).lower()
    model = model or llm_config.get("model")
    temperature = temperature if temperature is not None else llm_config.get("temperature", 0.4)

    if provider not in _PROVIDERS:
        supported = ", ".join(sorted(set(_PROVIDERS.keys())))
        raise LLMConfigError(
            f"Unsupported provider: '{provider}'. Supported: {supported}"
        )

    module_name, class_name, default_model = _PROVIDERS[provider]

    # Build kwargs, filtering out None values to avoid provider errors
    llm_kwargs = {
        "model": model or default_model,
        "temperature": temperature,
    }

    # Only include max_tokens if explicitly set
    max_tokens = kwargs.pop("max_tokens", llm_config.get("max_tokens"))
    if max_tokens is not None:
        llm_kwargs["max_tokens"] = max_tokens

    # Ollama-specific: base_url
    if provider == "ollama":
        base_url = kwargs.pop(
            "base_url", llm_config.get("base_url", "http://localhost:11434")
        )
        llm_kwargs["base_url"] = base_url

    # Pass through any remaining kwargs
    llm_kwargs.update(kwargs)

    # Dynamic import — gives a clear error if the package isn't installed
    try:
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
    except ImportError as e:
        raise LLMConfigError(
            f"Cannot import {module_name}. "
            f"Install it: pip install {module_name}\n"
            f"Error: {e}"
        ) from e
    except AttributeError as e:
        raise LLMConfigError(
            f"Class {class_name} not found in {module_name}. "
            f"You may need to update the package.\n"
            f"Error: {e}"
        ) from e

    logger.info(
        f"Initializing {provider} LLM: {llm_kwargs.get('model')} "
        f"(temp={temperature})"
    )
    return cls(**llm_kwargs)


# ==================== RETRY LOGIC ====================

# Transient errors worth retrying — NOT bugs like ValueError or KeyError
_RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    OSError,
)

# Add Ollama-specific errors if the package is installed
try:
    import ollama as _ollama_mod
    _RETRYABLE_EXCEPTIONS = (*_RETRYABLE_EXCEPTIONS, _ollama_mod.ResponseError)
except ImportError:
    pass


def invoke_with_retry(
    llm: BaseChatModel,
    prompt: Any,
    prompt_name: str = "prompt",
    max_retries: int = 3,
    retry_delay: float = 2.0,
    **kwargs,
):
    """
    Invoke an LLM with exponential backoff retry logic.

    Only retries on transient network/timeout errors, not on bugs
    like ValueError or KeyError.

    Args:
        llm: The LangChain-compatible LLM instance.
        prompt: The prompt string or message list to send.
        prompt_name: Label for log messages (e.g., 'resume_bullets').
        max_retries: Maximum number of attempts.
        retry_delay: Base delay in seconds (doubles each retry).
        **kwargs: Additional arguments passed to llm.invoke().

    Returns:
        The LLM response object.

    Raises:
        LLMInvocationError: If all retries are exhausted.
    """

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=retry_delay, min=retry_delay, max=60),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _invoke():
        logger.info(f"Invoking LLM for {prompt_name}")
        return llm.invoke(prompt, **kwargs)

    try:
        response = _invoke()
        logger.info(f"Successfully invoked LLM for {prompt_name}")
        return response
    except _RETRYABLE_EXCEPTIONS as e:
        raise LLMInvocationError(
            f"LLM invocation failed for {prompt_name} after {max_retries} attempts. "
            f"Last error: {type(e).__name__} - {e!s}"
        ) from e


# ==================== STRUCTURED OUTPUT HELPER ====================

def create_structured_chain(
    llm: BaseChatModel,
    output_schema,
    system_prompt: str,
):
    """
    Build a chain that returns structured (Pydantic) output.

    Useful for reviewer nodes that need to return typed data
    instead of free-form text.

    Args:
        llm: The LLM instance.
        output_schema: A Pydantic model class defining the expected output.
        system_prompt: System-level instructions for the LLM.

    Returns:
        A runnable chain: prompt | llm | parser

    Example:
        from pydantic import BaseModel, Field

        class ReviewResult(BaseModel):
            bullets: str = Field(description="Polished resume bullets")
            cover_letter: str = Field(description="Polished cover letter")
            notes: str = Field(description="What was changed and why")

        chain = create_structured_chain(llm, ReviewResult, "You are a reviewer...")
        result = chain.invoke({"input": "Review these materials..."})
        print(result.bullets)
    """
    parser = PydanticOutputParser(pydantic_object=output_schema)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\n{format_instructions}"),
        ("human", "{input}"),
    ])
    return (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
    )
