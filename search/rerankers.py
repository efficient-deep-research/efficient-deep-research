import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding

from search.data import Document


class Reranker:
    def __init__(self, max_length: int | None = None, batch_size: int = 1):
        self.max_length = max_length
        self.batch_size = batch_size

    def __call__(self, query: str, documents: list[Document]) -> list[Document]:
        raise NotImplementedError


class JinaReranker(Reranker):
    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        max_length: int | None = None,
        batch_size: int = 1,
    ):
        super().__init__(max_length=max_length, batch_size=batch_size)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, dtype="auto", trust_remote_code=True)
            .to(self.device)
            .eval()
        )

    def __call__(self, query: str, documents: list[Document]) -> tuple[list[Document], list[float]]:
        text_pairs = [(query, document.text) for document in documents]

        scores: list[float] = self.model.compute_score(
            text_pairs, batch_size=self.batch_size, max_length=self.max_length
        )

        sorted_idxs, sorted_scores = zip(*sorted(enumerate(scores), key=lambda x: x[1], reverse=True))
        sorted_documents = [documents[i] for i in sorted_idxs]

        return sorted_documents, sorted_scores


class Qwen3Reranker(Reranker):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        instruction: str | None = None,
        max_length: int | None = None,
        batch_size: int = 1,
    ):
        super().__init__(max_length=max_length, batch_size=batch_size)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"

        self.instruction = instruction

    def _format_instruction(self, query: str, doc: str) -> str:
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=self.instruction, query=query, doc=doc
        )
        return output

    def _process_inputs(self, pairs: list[str]) -> BatchEncoding:
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
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

    def __call__(self, query: str, documents: list[Document]) -> tuple[list[Document], list[float]]:
        pairs = [self._format_instruction(query, document.text) for document in documents]

        scores: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            batch_inputs = self._process_inputs(batch_pairs)
            batch_scores = self._compute_logits(batch_inputs)

            scores.extend(batch_scores)

        sorted_idxs, sorted_scores = zip(*sorted(enumerate(scores), key=lambda x: x[1], reverse=True))
        sorted_documents = [documents[i] for i in sorted_idxs]

        return sorted_documents, sorted_scores


class ContextualAIReranker(Reranker):
    def __init__(
        self,
        model_name: str = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b",
        instruction: str = "",
        max_length: int | None = None,
        batch_size: int = 1,
    ):
        super().__init__(max_length=max_length, batch_size=batch_size)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if instruction:
            instruction = f" {instruction}"

        self.instruction = instruction

    def _format_prompts(self, query: str, documents: list[str]) -> list[str]:
        prompts = []
        for doc in documents:
            prompt = f"Check whether a given document contains information helpful to answer the query.\n<Document> {doc}\n<Query> {query}{self.instruction} ??"
            prompts.append(prompt)
        return prompts

    def __call__(self, query: str, documents: list[Document]) -> tuple[list[Document], list[float]]:
        prompts = self._format_prompts(query, [document.text for document in documents])

        scores: list[float] = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_enc = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            )
            batch_input_ids = batch_enc["input_ids"].to(self.device)
            batch_attention_mask = batch_enc["attention_mask"].to(self.device)

            with torch.no_grad():
                batch_out = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

            batch_next_logits = batch_out.logits[:, -1, :]
            batch_scores_bf16 = batch_next_logits[:, 0].to(torch.bfloat16)
            batch_scores = batch_scores_bf16.float().tolist()

            scores.extend(batch_scores)

        sorted_idxs, sorted_scores = zip(*sorted(enumerate(scores), key=lambda x: x[1], reverse=True))
        sorted_documents = [documents[i] for i in sorted_idxs]

        return sorted_documents, sorted_scores


def load_reranker(name: str, max_length: int | None = None, batch_size: int = 1, **kwargs) -> Reranker:
    if name == "contextualai":
        return ContextualAIReranker(max_length=max_length, batch_size=batch_size, **kwargs)
    elif name == "jina":
        return JinaReranker(max_length=max_length, batch_size=batch_size, **kwargs)
    elif name == "qwen3":
        return Qwen3Reranker(max_length=max_length, batch_size=batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown reranker name: {name}")
