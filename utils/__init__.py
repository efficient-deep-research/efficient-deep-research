import logging
import re

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.constants import END_SEARCH_QUERY


logger = logging.getLogger(__name__)


def extract_final_information(output: str) -> str:
    split_str = "**Final Information**"
    if split_str in output:
        extracted_text = output.split(split_str)[-1].replace("\n", "").strip("```").strip()
    else:
        logger.warning(f"The output does not contain the expected '**Final Information**' tag: {output[:100]}")
        extracted_text = None

    return extracted_text

def delete_invalid_spaces_from_citation(text):
    citation_pattern = r"\([^)]*#[0-9a-f]{4}[^)]*\)"
    
    if text is None:
        return None

    def normalize_citation(match):
        citation = match.group()
        content = citation[1:-1]
        normalized_content = content.strip().replace(" ", "")
        return f"({normalized_content})"

    normalized_text = re.sub(citation_pattern, normalize_citation, text)
    return normalized_text


def validate_citation_format(text: str, valid_ids: list[str]) -> dict:
    results = {"is_valid": True, "errors": []}
    
    if text is None:
        results["is_valid"] = False
        results["errors"].append("Final answer format error")
        return results

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

def extract_between_tags(text: str, start_tag: str, end_tag: str) -> str | None:
    escaped_start = re.escape(start_tag)
    escaped_end = re.escape(end_tag)
    pattern = escaped_start + r"((?:(?!" + escaped_start + r").)*?)" + escaped_end
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

def run_generation(
    prompts: list[str],
    llm: LLM,
    tokenizer: AutoTokenizer,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k_sampling: int,
    stop: list[str],
) -> list:
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k_sampling,
        stop=[END_SEARCH_QUERY, tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    output_list = llm.generate(prompts, sampling_params=sampling_params)
    print(f"run_generation completed {len(output_list)}")
    return output_list


def load_tokenizer(model_path: str, trust_remote_code: bool = True, padding_side: str = "left") -> AutoTokenizer:
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

    return tokenizer


def load_vllm_model(
    model_path: str, tensor_parallel_size: int = torch.cuda.device_count(), gpu_memory_utilization: float = 0.95
) -> LLM:
    logger.info(f"Loading model from {model_path} (device_count: {torch.cuda.device_count()})")
    llm = LLM(
        model=model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization
    )

    return llm
