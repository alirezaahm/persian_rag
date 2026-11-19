"""
API manager for the Persian RAG application.

Supports OpenAI-compatible chat completion and embedding APIs.

Important:
This implementation assumes the service follows the OpenAI API format.
A company service using a different format will need a provider adapter.
"""

from openai import OpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError
import numpy as np
from config import APIConfig, validate_cloud_config


class CloudLLMManager:
    #connect to an openai compatible cloud llm api

    def __init__(self, config: APIConfig):
        validate_cloud_config("cloud", config, "LLM")
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=2
        )

    def generate(self,messages,temperature=0.1,max_tokens=700):
        """
        generate an answer from the cloud LLM.
        messages must follow the OpenAI chat format.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if not response.choices:
                raise RuntimeError("The LLM API returned no response choices.")

            choice = response.choices[0]
            content = choice.message.content

            if not content or not content.strip():
                raise RuntimeError(
                    "The LLM API returned empty content. "
                    f"Finish reason: {choice.finish_reason}"
                )

            return content.strip()

        except APITimeoutError as error:
            raise RuntimeError("The LLM API request timed out.") from error

        except APIConnectionError as error:
            raise RuntimeError(
                "Could not connect to the LLM API. "
                "Check the base URL and internet connection.") from error

        except APIStatusError as error:
            raise RuntimeError(
                f"LLM API returned HTTP status {error.status_code}."
            ) from error


class CloudEmbeddingManager:
    #connect to an openai-compatible cloud embedding api

    def __init__(self, config: APIConfig):
        validate_cloud_config("cloud", config, "Embedding")

        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=2
        )

        self.dimension = None

    def embed_texts(self, texts, batch_size=32):
        """
        convert a list of texts into embedding vectors.
        returns a numpy array with shape:
            (number_of_texts, embedding_dimension)

        no e5 prefixes are added here. The embedding model's
        documentation must determine whether prefixes are needed.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []

        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]

                response = self.client.embeddings.create(model=self.config.model,input=batch)
                #sort by index to preserve the original text order.
                ordered_data = sorted(response.data,key=lambda item: item.index)
                batch_embeddings = [item.embedding for item in ordered_data]

                if len(batch_embeddings) != len(batch):
                    raise RuntimeError(
                        "The embedding API returned an unexpected "
                        "number of vectors."
                    )
                all_embeddings.extend(batch_embeddings)

        except APITimeoutError as error:
            raise RuntimeError("The embedding API request timed out.") from error

        except APIConnectionError as error:
            raise RuntimeError(
                "Could not connect to the embedding API. "
                "Check the base URL and internet connection."
            ) from error

        except APIStatusError as error:
            raise RuntimeError(f"Embedding API returned HTTP status {error.status_code}.") from error

        embeddings = np.asarray(all_embeddings,dtype=np.float32)

        if embeddings.ndim != 2:
            raise RuntimeError("The embedding API returned an invalid vector shape.")

        #store the vector dimension for index validation.
        self.dimension = embeddings.shape[1]

        #normalize vectors for cosine similarity using dot product.
        norms = np.linalg.norm(embeddings,axis=1,keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings

    def embed_query(self, text):
        #generate one normalized query embedding."""
        vectors = self.embed_texts([text], batch_size=1)
        return vectors[0]