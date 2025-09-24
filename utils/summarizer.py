import json
import logging

from vllm import LLM, SamplingParams

from search.data import Document


logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(self, llm: LLM, top_k: int, max_tokens: int = 8192, temperature: float = 0.6, top_p: float = 0.95):
        self.llm = llm
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def __call__(
        self,
        previous_reasonings: list[str],
        search_queries: list[str],
        documents: list[list[Document]],
        batch_output_records: list[dict] | None = None,
    ) -> list[str]:
        user_prompts = [
            self._generate_prompt(pr, sq, docs) for pr, sq, docs in zip(previous_reasonings, search_queries, documents)
        ]

        prompts = [{"role": "user", "content": up} for up in user_prompts]
        logger.info(f"Summarizer prompts[0]: {prompts[0]}")

        summ_sampling_params = SamplingParams(
            max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, stop=None
        )
        raw_outputs = self.llm.chat(
            messages=[[prompt] for prompt in prompts], sampling_params=summ_sampling_params, use_tqdm=True
        )

        results = [self._parse_result(raw.outputs[0].text) for raw in raw_outputs]

        if batch_output_records is not None:
            for p, r, e in zip(prompts, raw_outputs, results):
                batch_output_records.append({"prompt": p, "raw_output": r.outputs[0].text, "extracted_info": e})

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

    def _generate_prompt(self, prev_reasoning: str, search_query: str, documents: list[Document]) -> str:
        documents_str = ""
        for i, document in enumerate(documents[: self.top_k]):
            documents_str += f"**Web Page {i + 1}:**\n"
            document_data = {"context": document.text, "url": document.url}
            documents_str += json.dumps(document_data, ensure_ascii=False, indent=2) + "\n"

        prompt = f"""**Task Instruction:**

    You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

    **Guidelines:**

    1. **Analyze the Searched Web Pages:**
    - Carefully review the content of each searched web page.
    - Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

    2. **Extract Relevant Information:**
    - Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
    - Ensure that the extracted information is accurate and relevant.

    3. **Output Format:**
    - Present the helpful information for current search query: beginning with `**Final Information**` as shown below.
    **Final Information**

    [Helpful information]

    **Inputs:**
    - **Previous Reasoning Steps:**  
    {prev_reasoning}

    - **Current Search Query:**  
    {search_query}

    - **Searched Web Pages:**  
    {documents_str}

    Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
    """

        return prompt
