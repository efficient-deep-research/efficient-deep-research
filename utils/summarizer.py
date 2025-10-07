import json
import logging
import re

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from search.data import Document


logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(
        self,
        llm: LLM,
        tokenizer: AutoTokenizer,
        model_context_length,
        top_k: int,
        max_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.model_context_length = model_context_length
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def __call__(
        self,
        previous_reasonings: list[str],
        search_queries: list[str],
        documents: list[dict],
        batch_output_records: list[dict] | None = None,
        max_retry: int = 10,
    ) -> list[str]:
        if len(previous_reasonings) == 0:
            logger.info("Performing initial search summarization...")
            user_prompts = [
                self._generate_initial_search_summary_prompt(sq, docs) for sq, docs in zip(search_queries, documents)
            ]
        else:
            logger.info("Performing iterative search summarization...")
            user_prompts = [
                self._generate_summary_prompt(pr, sq, docs)
                for pr, sq, docs in zip(previous_reasonings, search_queries, documents)
            ]

        prompts = [{"role": "user", "content": up} for up in user_prompts]

        summ_sampling_params = SamplingParams(
            max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, stop=None
        )
        raw_outputs = self.llm.chat(
            messages=[[prompt] for prompt in prompts], sampling_params=summ_sampling_params, use_tqdm=True
        )

        results = [self._parse_result(raw.outputs[0].text) for raw in raw_outputs]

        retry_count = 0
        while True:
            invalid_indices = []
            for i, (res, docs) in enumerate(zip(results, documents)):
                valid_ids = list(docs.keys())
                validation = self._validate_citation_format(res, valid_ids)
                if not validation["is_valid"]:
                    invalid_indices.append(i)
                    logger.warning(f"Invalid citation format found in result {i}: {validation['errors']}")

            if len(invalid_indices) == 0:
                logger.info("All outputs have valid citation formats.")
                break

            if retry_count >= max_retry:
                logger.error("Maximum retry attempts reached. Some outputs may still have invalid citation formats.")
                break

            logger.info(f"Retrying {len(invalid_indices)} outputs due to invalid citation formats...")
            retry_prompts = [prompts[i] for i in invalid_indices]
            retry_raw_outputs = self.llm.chat(
                messages=[[prompt] for prompt in retry_prompts], sampling_params=summ_sampling_params, use_tqdm=True
            )
            for idx, raw in zip(invalid_indices, retry_raw_outputs):
                results[idx] = self._parse_result(raw.outputs[0].text)
                raw_outputs[idx] = raw

            retry_count += 1

        if batch_output_records is not None:
            for p, r, e in zip(prompts, raw_outputs, results):
                batch_output_records.append({"prompt": p, "raw_output": r.outputs[0].text, "extracted_info": e})

        return results

    def _validate_citation_format(self, text: str, valid_ids: list[str]) -> dict:
        results = {"is_valid": True, "errors": []}

        citation_pattern = r"\([^)]*#[0-9a-f]{4}[^)]*\)"
        matches = re.finditer(citation_pattern, text)

        for match in matches:
            citation = match.group()
            content = citation[1:-1]

            # Check if it starts with #
            if not content.startswith("#"):
                results["is_valid"] = False
                results["errors"].append(citation)
                continue

            # Check if it matches the allowed pattern
            allowed_pattern = r"^#[0-9a-f]{4}(,#[0-9a-f]{4})*$"
            if not re.match(allowed_pattern, content):
                results["is_valid"] = False
                results["errors"].append(citation)
                continue

            # Check if each ID is in valid_ids
            ids = content.split(",")
            for id_str in ids:
                if id_str not in valid_ids:
                    results["is_valid"] = False
                    results["errors"].append(citation)
                    break

        return results

    @staticmethod
    def _parse_result(output: str) -> str:
        split_str = "**Final Information**"
        if split_str in output:
            extracted_text = output.split(split_str)[-1].replace("\n", "").strip("```").strip()
        else:
            logger.warning(f"The output does not contain the expected '**Final Information**' tag: {output}")
            extracted_text = output

        return extracted_text

    def _prepare_documents_str(self, documents: dict) -> str:
        documents_str = ""
        valid_len_per_doc = int(self.model_context_length * 0.8 / self.top_k)
        for i, (ref_id, data) in enumerate(documents.items()):
            if i >= self.top_k:
                break

            # token length check
            tokenized_doc = self.tokenizer(data["text"])["input_ids"]
            if len(tokenized_doc) > valid_len_per_doc:
                tokenized_doc = tokenized_doc[:valid_len_per_doc]  # truncate
                doc_text = self.tokenizer.decode(tokenized_doc, skip_special_tokens=True)
                logger.info(f"Document {ref_id} is truncated to fit the token limit.")
            else:
                doc_text = data["text"]
            documents_str += f"Webpage ID: {ref_id}\n"
            document_data = {"context": doc_text, "url": data["url"]}
            documents_str += json.dumps(document_data, ensure_ascii=False, indent=2) + "\n"

        return documents_str

    def _generate_summary_prompt(self, prev_reasoning: str, search_query: str, documents: dict) -> str:
        documents_str = self._prepare_documents_str(documents)

        return f"""**Role**
- You are an expert at extracting content relevant to a question from multiple ===Web Pages=== and integrating it after understanding the contents of ===Previous Reasoning Steps===.
**Instructions**
- Carefully read the ===Web Pages=== provided in Inputs and, following the **Webpage ID Guidelines** and **Output Format** below, extract the content relevant to the ===Query===.
- Read and fully understand ===Previous Reasoning Steps===, then integrate the extracted content with it.
- Let's think this out in a step by step way to be sure we have the right answer.
**Webpage ID Guidelines**
- ===Web Pages=== are presented in the following format: "Webpage ID: #xxxx (x = alphanumeric)\n"context": data["text"], "url": data["url"]"
- When using sentences from the ===Web Pages=== that are relevant to the ===Query===, you **MUST** record the Webpage ID in the format (#+ alphanumerics) exactly as shown in the **Webpage ID Examples** below.
- A Webpage ID is the identifier of the web page and begins with a leading "#" followed by alphanumeric characters.
- Because the Webpage ID is an identifier, do not include any text other than the identifier inside the parentheses.
- If you rely on multiple sources, output multiple Webpage IDs in a single set of parentheses separated by commas, like (#ab12,#cd34).
**Webpage ID Examples**
	- Single source: "Compared with pre-industrial times, the global average temperature has increased by 1.1°C (#ab12)"
	- Multiple sources: "In recent years, the adoption of renewable energy has accelerated (#ab12,#cd34)"
**Output Format**
- You **MUST** begin with `**Final Information**`.
- Include the correct Webpage ID(s) in parentheses (#+ alphanumerics) in the extracted sentences.
**Inputs**
- ===Query===
{search_query}
- ===Web Pages===
{documents_str}
- ===Previous Reasoning Steps===
{prev_reasoning}

Go ahead—confidently extract the information for the question and integrate it into the Previous Reasoning Steps."""

    def _generate_initial_search_summary_prompt(self, search_query: str, documents: dict) -> str:
        documents_str = self._prepare_documents_str(documents)

        return f"""**Role**
- You are an expert at extracting content relevant to a question from multiple ===Web Pages===.
**Instructions**
- Carefully read the ===Web Pages=== provided in Inputs and, following the **Webpage ID Guidelines** and **Output Format** below, extract the content relevant to the ===Query===.
- Let's think this out in a step by step way to be sure we have the right answer.
**Webpage ID Guidelines**
- ===Web Pages=== are presented in the following format: "Webpage ID: #xxxx (x = alphanumeric)\n"context": data["text"], "url": data["url"]"
- When using sentences from the ===Web Pages=== that are relevant to the ===Query===, you **MUST** record the Webpage ID in the format (#+ alphanumerics) exactly as shown in the **Webpage ID Examples** below.
- A Webpage ID is the identifier of the web page and begins with a leading "#" followed by alphanumeric characters.
- Because the Webpage ID is an identifier, do not include any text other than the identifier inside the parentheses.
- If you rely on multiple sources, output multiple Webpage IDs in a single set of parentheses separated by commas, like (#ab12,#cd34)
**Webpage ID Examples**
	- Single source: "Compared with pre-industrial times, the global average temperature has increased by 1.1°C (#ab12)"
	- Multiple sources: "In recent years, the adoption of renewable energy has accelerated (#ab12,#cd34)"
**Output Format**
- You **MUST** begin with `**Final Information**`.
- Include the correct Webpage ID(s) in parentheses (#+ alphanumerics) in the extracted sentences.
**Inputs**
- ===Query===
{search_query}
- ===Web Pages===
{documents_str}

Go ahead—you've got this; extract the information step by step."""
