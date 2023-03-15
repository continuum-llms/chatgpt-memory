import logging
from typing import Any, Dict, List, Union

import numpy as np
from tqdm import tqdm

from chatgpt_memory.constants import MAX_ALLOWED_SEQ_LEN_001, MAX_ALLOWED_SEQ_LEN_002
from chatgpt_memory.llm_client.llm_client import LLMClient
from chatgpt_memory.llm_client.openai.embedding.config import EmbeddingConfig, EmbeddingModels
from chatgpt_memory.utils.openai_utils import count_openai_tokens, load_openai_tokenizer, openai_request

logger = logging.getLogger(__name__)


class EmbeddingClient(LLMClient):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config=config)

        self.openai_embedding_config = config
        model_class: str = EmbeddingModels(self.openai_embedding_config.model).name

        tokenizer = self._setup_encoding_models(
            model_class,
            self.openai_embedding_config.model,
            self.openai_embedding_config.max_seq_len,
        )
        self._tokenizer = load_openai_tokenizer(
            tokenizer_name=tokenizer,
            use_tiktoken=self.openai_embedding_config.use_tiktoken,
        )

    def _setup_encoding_models(self, model_class: str, model_name: str, max_seq_len: int):
        """
        Setup the encoding models for the retriever.

        Raises:
            ImportError: When `tiktoken` package is missing.
            To use tiktoken tokenizer install it as follows:
            `pip install tiktoken`
        """

        tokenizer_name = "gpt2"
        # new generation of embedding models (December 2022), specify the full name
        if model_name.endswith("-002"):
            self.query_encoder_model = model_name
            self.doc_encoder_model = model_name
            self.max_seq_len = min(MAX_ALLOWED_SEQ_LEN_002, max_seq_len)
            if self.openai_embedding_config.use_tiktoken:
                try:
                    from tiktoken.model import MODEL_TO_ENCODING

                    tokenizer_name = MODEL_TO_ENCODING.get(model_name, "cl100k_base")
                except ImportError:
                    raise ImportError(
                        "The `tiktoken` package not found.",
                        "To install it use the following:",
                        "`pip install tiktoken`",
                    )
        else:
            self.query_encoder_model = f"text-search-{model_class}-query-001"
            self.doc_encoder_model = f"text-search-{model_class}-doc-001"
            self.max_seq_len = min(MAX_ALLOWED_SEQ_LEN_001, max_seq_len)

        return tokenizer_name

    def _ensure_text_limit(self, text: str) -> str:
        """
         Ensure that length of the text is within the maximum length of the model.
        OpenAI v1 embedding models have a limit of 2046 tokens, and v2 models have
        a limit of 8191 tokens.

        Args:
            text (str):  Text to be checked if it exceeds the max token limit

        Returns:
            text (str): Trimmed text if exceeds the max token limit
        """
        n_tokens = count_openai_tokens(text, self._tokenizer, self.openai_embedding_config.use_tiktoken)
        if n_tokens <= self.max_seq_len:
            return text

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens to fit" "within the max token limit.",
            "Reduce the length of the prompt to prevent it from being cut off.",
            n_tokens,
            self.max_seq_len,
        )

        if self.openai_embedding_config.use_tiktoken:
            tokenized_payload = self._tokenizer.encode(text)
            decoded_string = self._tokenizer.decode(tokenized_payload[: self.max_seq_len])
        else:
            tokenized_payload = self._tokenizer.tokenize(text)
            decoded_string = self._tokenizer.convert_tokens_to_string(tokenized_payload[: self.max_seq_len])

        return decoded_string

    def embed(self, model: str, text: List[str]) -> np.ndarray:
        """
        Embeds the batch of texts using the specified LLM.

        Args:
            model (str): LLM model name for embeddings.
            text (List[str]): List of documents to be embedded.

        Raises:
            ValueError: When the OpenAI API key is missing.

        Returns:
            np.ndarray: embeddings for the input documents.
        """
        if self.api_key is None:
            raise ValueError(
                "OpenAI API key is not set. You can set it via the " "`api_key` parameter of the `LLMClient`."
            )

        generated_embeddings: List[Any] = []

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        payload: Dict[str, Union[List[str], str]] = {"model": model, "input": text}
        headers["Authorization"] = f"Bearer {self.api_key}"

        res = openai_request(
            url=self.openai_embedding_config.url,
            headers=headers,
            payload=payload,
            timeout=self.time_out,
        )

        unordered_embeddings = [(ans["index"], ans["embedding"]) for ans in res["data"]]
        ordered_embeddings = sorted(unordered_embeddings, key=lambda x: x[0])

        generated_embeddings = [emb[1] for emb in ordered_embeddings]

        return np.array(generated_embeddings)

    def embed_batch(self, model: str, text: List[str]) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(
            range(0, len(text), self.openai_embedding_config.batch_size),
            disable=not self.openai_embedding_config.progress_bar,
            desc="Calculating embeddings",
        ):
            batch = text[i : i + self.openai_embedding_config.batch_size]
            batch_limited = [self._ensure_text_limit(content) for content in batch]
            generated_embeddings = self.embed(model, batch_limited)
            all_embeddings.append(generated_embeddings)

        return np.concatenate(all_embeddings)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        return self.embed_batch(self.query_encoder_model, queries)

    def embed_documents(self, docs: List[Dict]) -> np.ndarray:
        return self.embed_batch(self.doc_encoder_model, [d["text"] for d in docs])
