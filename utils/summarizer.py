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
        documents: list[dict],
        batch_output_records: list[dict] | None = None,
    ) -> list[str]:

        if len(previous_reasonings) == 0:
            print("Performing initial search summarization...")
            user_prompts = [
                self._generate_initial_search_summary_prompt(sq, docs) for sq, docs in zip(search_queries, documents)
            ]
        else:
            print("Performing iterative search summarization...")
            user_prompts = [
                self._generate_summary_prompt(pr, sq, docs) for pr, sq, docs in zip(previous_reasonings, search_queries, documents)
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

    def _prepare_documents_str(self, documents: dict) -> str:
        documents_str = ""
        for i, (ref_id, data) in enumerate(documents.items()):
            if i < self.top_k:
                documents_str += f"Webpage ID: {ref_id}\n"
                document_data = {"context": data["text"], "url": data["url"]}
                documents_str += json.dumps(document_data, ensure_ascii=False, indent=2) + "\n"

        return documents_str
    
    def _generate_summary_prompt(self, prev_reasoning: str, search_query: str, documents: dict) -> str:
        documents_str = self._prepare_documents_str(documents)

        return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Citation Requirements:**
- You MUST cite the source web page for every piece of information you extract.
- Always cite the most relevant web page that supports each statement.
- Each web page is identified by its **Webpage ID** (e.g., "ab12", "cd34").
- Use the citation format: (#WEBPAGE_ID) before the period or punctuation at the end of each sentence or statement.
- For information supported by multiple sources, use: (#WEBPAGE_ID1)(#WEBPAGE_ID2)
- **Citation Format Examples:**
    * Single source: "The global temperature has increased by 1.1°C since pre-industrial times (#ab12)."
    * Multiple sources: "Renewable energy adoption has accelerated in recent years (#ab12)(#cd34)."

4. **Output Format:**
- Present the helpful information for current search query: beginning with `**Final Information**` as shown below.
- Ensure all statements include proper citations.

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
