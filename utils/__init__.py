import logging
import re

import torch
from transformers import AutoTokenizer
from vllm import LLM


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
