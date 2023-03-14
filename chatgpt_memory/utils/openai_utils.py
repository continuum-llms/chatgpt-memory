"""Utils for using OpenAI API"""
import logging
import json
from typing import Any, Dict, Union, Tuple
import requests

from transformers import GPT2TokenizerFast

from chatgpt_memory.errors import OpenAIError, OpenAIRateLimitError
from chatgpt_memory.utils.reflection import retry_with_exponential_backoff
from chatgpt_memory.environment import (
    OPENAI_BACKOFF,
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT,
)

logger = logging.getLogger(__name__)


def load_openai_tokenizer(tokenizer_name: str, use_tiktoken: bool):
    """Load either the tokenizer from tiktoken (if the library is available) or
    fallback to the GPT2TokenizerFast from the transformers library.

    :param tokenizer_name: The name of the tokenizer to load.
    :param use_tiktoken: Use tiktoken tokenizer or not.
    """
    tokenizer = None
    if use_tiktoken:
        import tiktoken  # pylint: disable=import-error

        logger.debug("Using tiktoken %s tokenizer", tokenizer_name)
        tokenizer = tiktoken.get_encoding(tokenizer_name)
    else:
        logger.warning(
            "OpenAI tiktoken module is not available for Python < 3.8,Linux ARM64 and "
            "AARCH64. Falling back to GPT2TokenizerFast."
        )

        logger.debug("Using GPT2TokenizerFast tokenizer")
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    return tokenizer


def count_openai_tokens(text: str, tokenizer: Any, use_tiktoken: bool) -> int:
    """Count the number of tokens in `text` based on the provided OpenAI `tokenizer`.

    :param text: A string to be tokenized.
    :param tokenizer: An OpenAI tokenizer.
    :param use_tiktoken: Use tiktoken tokenizer or not.
    """
    if use_tiktoken:
        return len(tokenizer.encode(text))
    else:
        return len(tokenizer.tokenize(text))


@retry_with_exponential_backoff(
    backoff_in_seconds=OPENAI_BACKOFF,
    max_retries=OPENAI_MAX_RETRIES,
    errors=(OpenAIRateLimitError, OpenAIError),
)
def openai_request(
    url: str,
    headers: Dict,
    payload: Dict,
    timeout: Union[float, Tuple[float, float]] = OPENAI_TIMEOUT,
) -> Dict:
    """Make a request to the OpenAI API given a `url`, `headers`, `payload`, and
    `timeout`.

    :param url: The URL of the OpenAI API.
    :param headers: Dictionary of HTTP Headers to send with the :class:`Request`.
    :param payload: The payload to send with the request.
    :param timeout: The timeout length of the request. The default is 30s.
    """
    response = requests.request(
        "POST", url, headers=headers, data=json.dumps(payload), timeout=timeout
    )
    res = json.loads(response.text)

    if response.status_code != 200:
        openai_error: OpenAIError
        if response.status_code == 429:
            openai_error = OpenAIRateLimitError(
                f"API rate limit exceeded: {response.text}"
            )
        else:
            openai_error = OpenAIError(
                f"OpenAI returned an error.\n"
                f"Status code: {response.status_code}\n"
                f"Response body: {response.text}",
                status_code=response.status_code,
            )
        raise openai_error

    return res
