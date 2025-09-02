import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from search.data import Document


class Reranker:
    def __call__(self, query: str, documents: list[Document]) -> list[Document]:
        raise NotImplementedError


class Qwen3Reranker(Reranker):
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def _format_instruction(self, instruction: str | None, query: str, doc: str) -> str:
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )
        return output

    def _process_inputs(self, pairs: list[str], max_length: int = 8192) -> BatchEncoding:
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    @torch.no_grad()
    def _compute_logits(self, inputs: BatchEncoding, **kwargs) -> list[float]:
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def __call__(
        self,
        query: str,
        documents: list[Document],
        instruction: str | None = None,
        max_length: int = 8192,
        batch_size: int = 1,
    ) -> tuple[list[Document], list[float]]:
        pairs = [self._format_instruction(instruction, query, doc) for doc in documents]

        scores: list[float] = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            batch_inputs = self._process_inputs(batch_pairs, max_length=max_length)
            batch_scores = self._compute_logits(batch_inputs)

            scores.extend(batch_scores)

        sorted_idxs, sorted_scores = zip(*sorted(enumerate(scores), key=lambda x: x[1], reverse=True))
        sorted_documents = [documents[i] for i in sorted_idxs]

        return sorted_documents, sorted_scores


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
