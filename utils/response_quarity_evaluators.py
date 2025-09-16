import argparse
import json
from pathlib import Path
from typing import Literal, List, Dict, Any
from pydantic import BaseModel
import os
import re
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import torch
from collections import defaultdict
from prompts import (
    create_kpr_prompt,
    create_eval_criteria_prompt
)


def extract_final_answer(answer: str) -> str:

    pattern = r'\\boxed\{\\text{(.*?)\}\}'
    match = re.search(pattern, answer, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        # If no match is found, return the original answer trimmed
        print("No final answer found in the expected format.")
        return answer.strip()

def create_chat_pattern(prompt: str):
    chat_pattern = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return chat_pattern


class KeyPointRecall(BaseModel):
    label: Literal["Supported", "Omitted", "Contradicted"]
    justification: str


class CriterionEvaluation(BaseModel):
    rating: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    justification: str


def evaluate_with_llm_judge(
    messages: List[List[Dict]],
    schema_class: Any,
    llm: LLM,
    temperature: float = 0,
    max_tokens: int = 512,
) -> List[Dict]:
    
    guided_params = GuidedDecodingParams(
        json=schema_class.model_json_schema()
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        guided_decoding=guided_params,
    )
    outputs = llm.chat(
        messages,
        sampling_params=sampling_params,
    )
    judges = [json.loads(output.outputs[0].text) for output in outputs]
    
    return judges


def evaluate_kpr_batch(
    key_points_collection: list, 
    answers: list,
    llm: LLM = None,
):
    key_point_answer_pairs = []
    for kp_group_id, (key_points, answer) in enumerate(zip(key_points_collection, answers)):
        for key_point in key_points:
            key_point_answer_pairs.append((kp_group_id, key_point, answer))

    prompts = [create_kpr_prompt(key_point["point_content"], answer) for _, key_point, answer in key_point_answer_pairs]
    messages = [create_chat_pattern(prompt) for prompt in prompts]

    # llm-as-a-judge with structured output
    judges = evaluate_with_llm_judge(
        messages,
        KeyPointRecall,
        llm,
        temperature=0,
        max_tokens=512,
    )
    
    # reconstruct the results
    kpr_results = []
    for _ in range(len(key_points_collection)):
        kpr_results.append({"judges": {}, "supported_rate": 0.0, "omitted_rate": 0.0, "contradicted_rate": 0.0})
    for i, (kp_group_id, key_point, answer) in enumerate(key_point_answer_pairs):
        kpr_results[kp_group_id]["judges"][key_point["point_number"]] = judges[i]

    # compute KPR
    for result in kpr_results:
        labels = result["judges"].values()
        if len(labels) == 0:
            result["supported_rate"] = 0.0
            result["omitted_rate"] = 0.0
            result["contradicted_rate"] = 0.0
            continue
        total_points = len(labels)
        supported_count = sum(1 for label in labels if label["label"] == "Supported")
        omitted_count = sum(1 for label in labels if label["label"] == "Omitted")
        contradicted_count = sum(1 for label in labels if label["label"] == "Contradicted")
        result["supported_rate"] = supported_count / total_points * 100
        result["omitted_rate"] = omitted_count / total_points * 100
        result["contradicted_rate"] = contradicted_count / total_points * 100

    return kpr_results


def evaluate_criteria_batch(
    questions: List[str],
    answers: List[str],
    eval_criteria: List[str],
    llm: LLM = None,
):
    criterion_answer_pairs = []
    for qa_group_id, (question, answer) in enumerate(zip(questions, answers)):
        for criterion in eval_criteria:
            criterion_answer_pairs.append((qa_group_id, criterion, question, answer))

    prompts = [create_eval_criteria_prompt(criterion, question, answer) for _, criterion, question, answer in criterion_answer_pairs]
    messages = [create_chat_pattern(prompt) for prompt in prompts]

    # llm-as-a-judge with structured output
    judges = evaluate_with_llm_judge(
        messages,
        CriterionEvaluation,
        llm,
        temperature=0,
        max_tokens=512,
    )

    # reconstruct the results
    eval_criteria_results = []
    for _ in range(len(questions)):
        for criterion in eval_criteria:
            eval_criteria_results.append({criterion: {}})
    for i, (qa_group_id, criterion, question, answer) in enumerate(criterion_answer_pairs):
        eval_criteria_results[qa_group_id][criterion] = judges[i]

    return eval_criteria_results



def evaluate_reasoning_quality(
    input_data: dict,
):
    
    llm = LLM(
        model="Qwen/Qwen3-30B-A3B-Thinking-2507",
        guided_decoding_backend="xgrammar",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
    )

    # flatten the input data
    flat_questions = []
    flat_final_answers = []
    flat_key_points_collection = []
    reconstruction_map = [] # (query, rollout_idx)
    for query, rollouts in input_data.items():
        for rollout_idx, rollout in enumerate(rollouts):
            flat_questions.append(query)
            flat_final_answers.append(extract_final_answer(rollout["output"]))
            flat_key_points_collection.append(rollout["item"]["key_points"])
            reconstruction_map.append((query, rollout_idx))

    # evaluate KPR
    eval_kpr_results = evaluate_kpr_batch(
        flat_key_points_collection, 
        flat_final_answers, 
        llm=llm
    )

    # evaluate clarity and insightfulness
    eval_criteria = ["Clarity", "Insightfulness"]
    eval_criteria_results = evaluate_criteria_batch(
        flat_questions,
        flat_final_answers,
        eval_criteria,
        llm=llm
    )

    # reconstruct the results and add quality metrics
    results = {}
    for i, (query, rollout_idx) in enumerate(reconstruction_map):
        if query not in results:
            results[query] = []
        while len(results[query]) <= rollout_idx:
            results[query].append({})

        original_rollout = input_data[query][rollout_idx]
        results[query][rollout_idx] = original_rollout.copy()
        results[query][rollout_idx]["kpr"] = eval_kpr_results[i]
        results[query][rollout_idx]["criteria_evaluation"] = eval_criteria_results[i]
    
    with open("test_reasoning_quality_evaluation_1sample.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    args = parser.parse_args()
    
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    
    evaluate_reasoning_quality(input_data)
