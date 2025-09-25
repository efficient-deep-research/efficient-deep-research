import base64
import json
import os

import requests

from search.data import Document


class Retriever:
    def __init__(self, default_k: int = 10):
        self.default_k = default_k

    def __call__(self, query: str, k: int | None = None) -> list[Document]:
        raise NotImplementedError


class FineWebRetriever(Retriever):
    endpoint_url = "https://clueweb22.us/fineweb/search"

    def __call__(self, query: str, k: int | None = None) -> list[Document]:
        if k is None:
            k = self.default_k

        params = {"query": query, "k": k}

        response = requests.get(self.endpoint_url, params=params)
        response.raise_for_status()

        response_json = response.json()
        encoded_results = response_json.get("results", [])

        documents = []
        for encoded_result in encoded_results:
            result = base64.b64decode(encoded_result).decode("utf-8")
            result_json = json.loads(result)

            text = result_json.get("text", "")
            url = result_json.get("url", "")
            documents.append(Document(text, url))

        return documents


class ClueWeb22Retriever(Retriever):
    endpoint_url = "https://clueweb22.us/search"

    def __init__(self, default_k: int = 10, use_cw22_a: bool = False):
        super().__init__(default_k=default_k)

        self.use_cw22_a = use_cw22_a
        self.headers = {"X-API-Key": os.getenv("RETRIEVER_API_KEY")}

    def __call__(self, query: str, k: int | None = None) -> list[Document]:
        if k is None:
            k = self.default_k

        params = {"query": query, "k": k}
        if self.use_cw22_a:
            params["cw22_a"] = True

        response = requests.get(self.endpoint_url, params=params, headers=self.headers)
        response.raise_for_status()

        response_json = response.json()
        encoded_results = response_json.get("results", [])

        documents = []
        for encoded_result in encoded_results:
            result = base64.b64decode(encoded_result).decode("utf-8")
            result_json = json.loads(result)

            text = result_json.get("Clean-Text", "")
            url = result_json.get("URL", "").strip()
            documents.append(Document(text, url))

        return documents


def load_retriever(name: str, default_k: int = 10, **kwargs) -> Retriever:
    if name == "clueweb22":
        return ClueWeb22Retriever(default_k=default_k, **kwargs)
    elif name == "clueweb22-a":
        return ClueWeb22Retriever(default_k=default_k, use_cw22_a=True, **kwargs)
    elif name == "clueweb22-b":
        return ClueWeb22Retriever(default_k=default_k, use_cw22_a=False, **kwargs)
    elif name == "fineweb":
        return FineWebRetriever(default_k=default_k, **kwargs)
    else:
        raise ValueError(f"Unknown retriever name: {name}")
