import logging
import re

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.constants import END_SEARCH_QUERY


logger = logging.getLogger(__name__)


def extract_final_information(output: str) -> str:
    pattern_info = "**Final Information**"
    if pattern_info in output:
        extracted_text = output.split(pattern_info)[-1].replace("\n", "").strip("```").strip()
    else:
        extracted_text = output

    return extracted_text


def extract_between_tags(text: str, start_tag: str, end_tag: str) -> str | None:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
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
