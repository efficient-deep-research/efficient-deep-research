import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from search.rerankers import JinaReranker
from search.retrievers import ClueWeb22Retriever
from utils import extract_answer
from utils.prompts import (
    get_qa_instruction,
    get_task_instruction,
    get_webpage_to_reasonchain_instruction,
)
from utils.stage_wise_analysis import stage_wise_analysis


# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"


def load_reasoning_model(model_path: str) -> Tuple[LLM, AutoTokenizer]:
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Tokenizer loaded successfully.")

    print(f"Loading model from {model_path}...")
    print(f"device_count: {torch.cuda.device_count()}")

    # Initialize the LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
    )
    print("Model loaded successfully.")
    return llm, tokenizer


def make_output_dir(output_dir_base: str, dataset_name: str, rollout_id: int) -> str:
    # Define output directory based on the dataset
    output_dir = os.path.join(output_dir_base, dataset_name, f"rollout_{rollout_id}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_data(data_path: str) -> List[Dict]:
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as json_file:
        filtered_data = json.load(json_file)
    print(f"Data loaded successfully. Total examples: {len(filtered_data)}")
    return filtered_data


def extract_relevant_info(search_results):
    useful_info = []
    for doc in search_results:
        info = {
            "context": doc.text,
            "url": doc.url,
        }
        useful_info.append(info)

    return useful_info


def generate_webpage_to_reasonchain_batch(
    original_questions: List[str],
    prev_reasonings: List[str],
    search_queries: List[str],
    documents: List[str],
    dataset_name: str,
    batch_output_records: List[Dict],  # New parameter to collect outputs
    llm: LLM,
    coherent: bool = False,
) -> List[str]:

    user_prompts = [
        get_webpage_to_reasonchain_instruction(r, sq, doc)
        for r, sq, doc in zip(prev_reasonings, search_queries, documents)
    ]

    prompts = [{"role": "user", "content": up} for up in user_prompts]
    print("webpage ana prompts[0]")
    print(prompts[0])

    summ_sampling_params = SamplingParams(
        max_tokens=8192, temperature=0.6, top_p=0.95, stop=None
    )
    raw_outputs = llm.chat(
        messages=[[prompt] for prompt in prompts],
        sampling_params=summ_sampling_params,
        use_tqdm=True,
    )

    extracted_infos = [extract_answer(raw.outputs[0].text) for raw in raw_outputs]

    for i, (p, r, e) in enumerate(zip(prompts, raw_outputs, extracted_infos)):
        batch_output_records.append(
            {"prompt": p, "raw_output": r.outputs[0].text, "extracted_info": e}
        )

    return extracted_infos


def run_generation(
    sequences: List[Dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k_sampling: int,
    llm: LLM,
    tokenizer: AutoTokenizer,
) -> List:
    prompts = [s["prompt"] for s in sequences]

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


def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def prepare_input_prompts(
    filtered_data: List[Dict],
    max_search_limit: int,
    tokenizer: AutoTokenizer,
    subset_num: int,
) -> None:
    input_list = []
    for item in filtered_data:
        question = item["Question"]

        instruction = get_qa_instruction(max_search_limit)
        user_prompt = get_task_instruction(question)

        prompt = [{"role": "user", "content": instruction + user_prompt}]
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        input_list.append(prompt)

    if subset_num != -1:
        input_list = input_list[:subset_num]
        filtered_data = filtered_data[:subset_num]

    # Initialize active sequences
    active_sequences = [
        {
            "item": item,
            "prompt": prompt,
            "output": "",
            "finished": False,
            "history": [],
            "search_count": 0,
            "executed_search_queries": set(),
            "all_info": [],
        }
        for item, prompt in zip(filtered_data, input_list)
    ]

    return active_sequences, input_list


def parse_steps(text: str) -> dict:
    """
    Parses the reasoning steps from a given text.

    Parameters:
    - text (str): The text containing reasoning steps.

    Returns:
    - dict: A dictionary mapping step numbers to their content.
    """
    step_pattern = re.compile(r"Step\s+(\d+):\s*")
    steps = {}
    current_step_num = None
    current_content = []

    for line in text.splitlines():
        step_match = step_pattern.match(line)
        if step_match:
            # If there's an ongoing step, save its content
            if current_step_num is not None:
                steps[current_step_num] = "\n".join(current_content).strip()
            current_step_num = int(step_match.group(1))
            content = line[step_match.end() :].strip()
            current_content = [content] if content else []
        else:
            if current_step_num is not None:
                current_content.append(line)

    # Save the last step if any
    if current_step_num is not None:
        steps[current_step_num] = "\n".join(current_content).strip()

    return steps


def replace_recent_steps(origin_str: str, replace_str: str) -> str:
    """
    Replaces specific steps in the original reasoning steps with new steps.
    If a replacement step contains "DELETE THIS STEP", that step is removed.

    Parameters:
    - origin_str (str): The original reasoning steps.
    - replace_str (str): The steps to replace or delete.

    Returns:
    - str: The updated reasoning steps after applying replacements.
    """
    # Parse the original and replacement steps
    origin_steps = parse_steps(origin_str)
    replace_steps = parse_steps(replace_str)

    # Apply replacements
    for step_num, content in replace_steps.items():
        if "DELETE THIS STEP" in content:
            # Remove the step if it exists
            if step_num in origin_steps:
                del origin_steps[step_num]
        else:
            # Replace or add the step
            origin_steps[step_num] = content

    # Sort the steps by step number
    sorted_steps = sorted(origin_steps.items())

    # Reconstruct the reasoning steps as a single string
    new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

    return new_reasoning_steps


def main(args: argparse.Namespace):
    data_path = args.data_path
    subset_num = args.subset_num
    max_search_limit = args.max_search_limit
    max_turn = args.max_turn
    top_k = args.top_k
    max_doc_len = args.max_doc_len
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    max_tokens = args.max_tokens
    output_dir_base = args.output_dir_base
    rollout_num = args.rollout_num

    print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    dataset_name = data_path.split("/")[-1].split(".")[0]

    print("-----------------------")
    print(f"Using {dataset_name} set.")
    print("-----------------------")

    # Reasoning Model Loading
    llm, tokenizer = load_reasoning_model(model_path)

    # Retriever Setup
    # retriever = FinewWebRetriever(default_k=10)
    retriever = ClueWeb22Retriever(default_k=10, use_cw22_a=False)

    # Reranker Setup
    reranker = JinaReranker()
    # reranker = Qwen3Reranker()

    # Rollout
    for rollout_id in tqdm(range(rollout_num), desc="rollouts"):
        print(
            f"\n===================Rollout {rollout_id + 1} of {rollout_num}==================="
        )

        output_dir = make_output_dir(output_dir_base, dataset_name, rollout_id)
        filtered_data = load_data(data_path)

        # Prepare input prompts
        active_sequences, input_list = prepare_input_prompts(
            filtered_data,
            max_search_limit,
            tokenizer,
            subset_num,
        )

        # Initialize collection structure
        batch_output_records = []
        start_time = time.time()
        turn = 0
        unfinished = True

        # Start the interaction loop
        while turn < max_turn and unfinished:
            sequences_needing_generation = [
                seq for seq in active_sequences if not seq["finished"]
            ]

            if sequences_needing_generation:
                turn += 1
                print(f"\n-------------- Turn {turn} --------------")
                print(
                    f"We have {len(sequences_needing_generation)} sequences needing generation..."
                )

                outputs = run_generation(
                    sequences_needing_generation,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k_sampling,
                    llm,
                    tokenizer,
                )
                print("Generation completed, processing outputs...")

                # Initialize batch variables
                batch_relevant_info = []
                batch_original_questions = []
                batch_prev_reasonings = []
                batch_search_queries = []
                batch_documents = []
                batch_sequences = []

                start_search_time = time.time()
                for seq, out in zip(sequences_needing_generation, outputs):
                    text = out.outputs[0].text
                    seq["history"].append(text)
                    seq["prompt"] += text
                    seq["output"] += text
                    seq["all_info"].append({f"turn_{turn}_reason": text})

                    # Extract search query
                    search_query = extract_between(
                        text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY
                    )

                    # If a search query is present and the needs to be executed
                    if search_query and seq["output"].rstrip().endswith(
                        END_SEARCH_QUERY
                    ):
                        if (
                            seq["search_count"] < max_search_limit
                            and search_query not in seq["executed_search_queries"]
                        ):
                            try:
                                print(f'Executing search for query: "{search_query}"')
                                search_results = retriever(search_query)
                                rerankered_results, _ = reranker(
                                    search_query, search_results
                                )
                            except Exception as e:
                                print(f'Search failed for query "{search_query}": {e}')
                                search_results = []
                                rerankered_results = []
                            relevant_info = extract_relevant_info(
                                rerankered_results[:top_k]
                            )
                            seq["relevant_info"] = relevant_info

                            all_reasoning_steps = seq["output"]
                            all_reasoning_steps = all_reasoning_steps.replace(
                                "\n\n", "\n"
                            ).split("\n")

                            truncated_prev_reasoning = ""
                            for i, step in enumerate(all_reasoning_steps):
                                truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

                            prev_steps = truncated_prev_reasoning.split("\n\n")
                            if len(prev_steps) <= 5:
                                truncated_prev_reasoning = "\n\n".join(prev_steps)
                            else:
                                truncated_prev_reasoning = ""
                                for i, step in enumerate(prev_steps):
                                    if (
                                        i == 0
                                        or i >= len(prev_steps) - 4
                                        or BEGIN_SEARCH_QUERY in step
                                        or BEGIN_SEARCH_RESULT in step
                                    ):
                                        truncated_prev_reasoning += step + "\n\n"
                                    else:
                                        if (
                                            truncated_prev_reasoning[
                                                -len("\n\n...\n\n") :
                                            ]
                                            != "\n\n...\n\n"
                                        ):
                                            truncated_prev_reasoning += "...\n\n"
                            truncated_prev_reasoning = truncated_prev_reasoning.strip(
                                "\n"
                            )

                            # Collect parameters for batch processing
                            batch_relevant_info.append(relevant_info)
                            batch_original_questions.append(seq["item"]["Question"])
                            batch_prev_reasonings.append(truncated_prev_reasoning)
                            batch_search_queries.append(search_query)
                            batch_sequences.append(seq)

                            # Update search count and executed queries
                            seq["search_count"] += 1
                            seq["executed_search_queries"].add(search_query)

                        elif seq["search_count"] >= max_search_limit:
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                            seq["prompt"] += limit_message
                            seq["output"] += limit_message
                            seq["history"].append(limit_message)
                            seq["all_info"].append(
                                {f"turn_{turn}_search_limited": limit_message}
                            )
                            print(f'Search limit reached for query: "{search_query}"')

                        elif search_query in seq["executed_search_queries"]:
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                            seq["prompt"] += limit_message
                            seq["output"] += limit_message
                            seq["history"].append(limit_message)
                            seq["all_info"].append(
                                {f"turn_{turn}_search_limited": limit_message}
                            )
                            print(f'Repeated search for query: "{search_query}"')
                    else:
                        # If no search query needs to be executed, mark the sequence as finished
                        seq["finished"] = True
                        print("Sequence marked as complete.")

                print(f"get search time taken: {time.time() - start_search_time}")

                for relevant_info in batch_relevant_info:
                    formatted_documents = ""
                    for i, doc_info in enumerate(relevant_info):
                        formatted_documents += f"**Web Page {i + 1}:**\n"
                        formatted_documents += (
                            json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
                        )
                    print(f"formatted_webpage_documents: {len(formatted_documents)}")
                    batch_documents.append(formatted_documents)

                if batch_sequences:
                    print(
                        f"Batch processing {len(batch_sequences)} sequences with generate_webpage_to_reasonchain_batch..."
                    )
                    webpage_analyses = generate_webpage_to_reasonchain_batch(
                        original_questions=batch_original_questions,
                        prev_reasonings=batch_prev_reasonings,
                        search_queries=batch_search_queries,
                        documents=batch_documents,
                        dataset_name=dataset_name,
                        batch_output_records=batch_output_records,  # Pass the collection list
                        llm=llm,
                    )
                    print(
                        "Batch generation completed, assigning outputs to sequences..."
                    )

                    for seq, analysis, doc in zip(
                        batch_sequences, webpage_analyses, batch_documents
                    ):
                        if isinstance(analysis, str):
                            append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                            seq["prompt"] += append_text
                            seq["output"] += append_text
                            seq["history"].append(append_text)
                            seq["all_info"].extend(
                                [
                                    {f"turn_{turn}_search": doc},
                                    {f"turn_{turn}_webpage_analyses": analysis},
                                ]
                            )
                        else:
                            append_text = replace_recent_steps(seq["output"], analysis)
                            seq["prompt"] += append_text
                            seq["output"] += append_text
                            seq["history"].append(append_text)
                            seq["all_info"].extend(
                                [
                                    {f"turn_{turn}_search": doc},
                                    {f"turn_{turn}_webpage_analyses": analysis},
                                ]
                            )

            # Check if all sequences are finished
            active_sequences_part = [
                {
                    "item": ele["item"],
                    "prompt": ele["prompt"],
                    "output": ele["output"],
                    "finished": ele["finished"],
                    "history": ele["history"],
                    "search_count": ele["search_count"],
                    "all_info": ele["all_info"],
                }
                for ele in active_sequences
            ]
            with open(
                os.path.join(output_dir, f"turn_{turn}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(active_sequences_part, f, ensure_ascii=False, indent=2)
            unfinished = [seq for seq in active_sequences if not seq["finished"]]

        total_time = time.time() - start_time
        print(f"Total time taken: {total_time} seconds")

        # ---------------------- Save Batch Output Records to JSON File ----------------------
        # Define output JSON file path
        t = time.localtime()
        batch_output_file = os.path.join(
            output_dir,
            f"test.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.info_extract.json",
        )

        # Save batch_output_records to JSON file
        with open(batch_output_file, "w", encoding="utf-8") as f:
            json.dump(batch_output_records, f, ensure_ascii=False, indent=2)

        print(f"Batch outputs saved to {batch_output_file}")

        # Prepare output list for evaluation
        output_list = [seq["output"] for seq in active_sequences]

        # Run evaluation for factoid QAs
        # if dataset_name in ["eval", "gaia"]:
        #     run_evaluation_for_eval(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, 'test')
        # else:
        #     run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, 'test')

        # ---------------------- Stage-wise Analysis ----------------------
        turn_files = os.listdir(output_dir)
        turn_files = [file for file in turn_files if file.startswith("turn_")]
        max_turn_file = max(
            turn_files, key=lambda x: int(re.search(r"turn_(\d+)", x).group(1))
        )

        max_turn_file_path = os.path.join(output_dir, max_turn_file)
        print(f"max_turn_file_path: {max_turn_file_path}")
        stage_wise_analysis(model_path, max_turn_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SimpleDeepsearcer for various datasets."
    )

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset to use."
    )

    parser.add_argument(
        "--subset_num",
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified.",
    )

    # Search and document retrieval configuration
    parser.add_argument(
        "--max_search_limit",
        type=int,
        default=10,
        help="Maximum number of searches per question.",
    )

    parser.add_argument(
        "--max_turn", type=int, default=15, help="Maximum number of turns."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Maximum number of search documents to return.",
    )

    parser.add_argument(
        "--max_doc_len",
        type=int,
        default=3000,
        help="Maximum length of each searched document.",
    )

    # Model configuration
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the reasoning model."
    )

    # Sampling parameters
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature."
    )

    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p sampling parameter."
    )

    parser.add_argument(
        "--top_k_sampling", type=int, default=40, help="Top-k sampling parameter."
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=20480,
        help="Maximum number of tokens to generate.",
    )

    parser.add_argument("--cache_dir_base", type=str, required=True, help="cache path.")

    parser.add_argument("--output_dir_base", type=str, required=True, help="output_dir")

    parser.add_argument(
        "--is_exclude_urls", action="store_true", help="is_exclude_urls"
    )

    parser.add_argument(
        "--rollout_num", type=int, default=1, help="The number of rollout per question"
    )

    args = parser.parse_args()

    main(args)
