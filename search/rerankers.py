import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from search.data import Document


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
        self,
        query: str,
        documents: list[Document],
        instruction: str | None = None,
        max_length: int | None = None,
        batch_size: int = 1,
    ) -> tuple[list[Document], list[float]]:
        prompts = self.format_prompts(query, documents, instruction=instruction)

        scores_list = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_encoding = self.tokenizer(
                batch_prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            with torch.no_grad():
                batch_output = self.model(**batch_encoding.to(self.device))

            batch_scores = batch_output.logits[:, -1, 0]
            scores_list.append(batch_scores.cpu())

            del batch_encoding
            del batch_output

        scores = torch.cat(scores_list)
        sorting_idxs = scores.argsort(descending=True).tolist()

        sorted_documents = [documents[i] for i in sorting_idxs]
        sorted_scores = scores[sorting_idxs]

        return sorted_documents, sorted_scores.tolist()
