import argparse
import base64
import json
import os
from dataclasses import dataclass

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Document:
    text: str
    url: str


class Retriever:
    def __init__(self, default_k: int = 10):
        self.default_k = default_k

    def __call__(self, query: str, k: int | None = None) -> list[Document]:
        raise NotImplementedError


class FinewWebRetriever(Retriever):
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
        self.headers = {"x-api-key": os.getenv("RETRIEVER_API_KEY")}

    def __call__(self, query: str, k: int | None = None) -> list[Document]:
        if k is None:
            k = self.default_k

        params = {"query": query, "k": k, "cw22_a": self.use_cw22_a}

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


class Reranker:
    def __call__(self, query: str, documents: list[Document]) -> list[Document]:
        raise NotImplementedError


class ContextualAIReranker(Reranker):
    def __init__(self, model_name: str = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b"):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype).to(self.device)
        self.model.eval()

    def format_prompts(self, query: str, documents: list[str], instruction: str | None = None) -> list[str]:
        if instruction is not None:
            query_and_instruction = f"{query} {instruction}"
        else:
            query_and_instruction = query

        prompts = [
            (
                "Check whether a given document contains information helpful to answer the query.\n"
                f"<Document> {document}\n"
                f"<Query> {query_and_instruction} ??"
            )
            for document in documents
        ]
        return prompts

    def __call__(
        self, query: str, documents: list[Document], instruction: str | None = None
    ) -> tuple[list[Document], torch.tensor]:
        prompts = self.format_prompts(query, documents, instruction=instruction)
        batch_encoding = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            model_output = self.model(**batch_encoding)

        scores = model_output.logits[:, -1, 0]
        sorting_idxs = scores.argsort(descending=True).tolist()

        sorted_documents = [documents[i] for i in sorting_idxs]
        sorted_scores = scores[sorting_idxs]

        return sorted_documents, sorted_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    retriever = FinewWebRetriever(default_k=args.k)
    reranker = ContextualAIReranker()

    documents = retriever(args.query, args.k)
    reranked_documents, scores = reranker(args.query, documents, instruction=args.instruction)

    for i, document in enumerate(reranked_documents):
        print(f"Rank {i + 1} (Score: {scores[i]:.4f}):")
        print(f"  URL: {document.url}")
        print(f"  Excerpt: {document.text[:200]}...")
        print("")
