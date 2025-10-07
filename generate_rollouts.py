import argparse
import hashlib
import json
import logging
import os
import re
import time
import types

from transformers import AutoTokenizer, AutoConfig

from search.rerankers import load_reranker
from search.retrievers import load_retriever
from utils import extract_between_tags, load_tokenizer, load_vllm_model, run_generation
from utils.constants import BEGIN_SEARCH_QUERY, BEGIN_SEARCH_RESULT, END_SEARCH_QUERY, END_SEARCH_RESULT
from utils.prompts import get_qa_instruction
from utils.stage_wise_analysis import stage_wise_analysis
from utils.summarizer import Summarizer


logger = logging.getLogger(__name__)


def make_output_dir(output_dir_base: str, dataset_name: str, rollout_id: int) -> str:
    # Define output directory based on the dataset
    output_dir = os.path.join(output_dir_base, dataset_name, f"rollout_{rollout_id}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_data(data_path: str) -> list[dict]:
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as json_file:
        filtered_data = json.load(json_file)
    print(f"Data loaded successfully. Total examples: {len(filtered_data)}")
    return filtered_data


def prepare_input_prompts(
    filtered_data: list[dict],
    max_search_limit: int,
    tokenizer: AutoTokenizer,
    subset_num: int,
    initial_search_documents: list[str],
    initial_search_summaries: list[str],
) -> tuple[list[dict], list[dict]]:
    input_list = []
    for item, initial_summary in zip(filtered_data, initial_search_summaries):
        question = item["Question"]

        instruction = get_qa_instruction(max_search_limit, question, initial_summary)

        prompt = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
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
            "executed_search_queries": [],
            "executed_search_urls": {ref_id: data["url"] for ref_id, data in initial_docs.items()},
            "all_info": [
                {"initial_search": {ref_id: data["text"] for ref_id, data in initial_docs.items()}},
                {"initial_search_webpage_analysis": initial_summary},
            ],
        }
        for item, prompt, initial_docs, initial_summary in zip(
            filtered_data, input_list, initial_search_documents, initial_search_summaries
        )
    ]

    return active_sequences, input_list


def save_checkpoint(output_dir: str, rollout_id: int, turn: int, active_sequences: list, batch_output_records: list):
    checkpoint_data = {
        "rollout_id": rollout_id,
        "turn": turn,
        "active_sequences": active_sequences,
        "batch_output_records": batch_output_records,
        "timestamp": time.time(),
    }
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(output_dir: str) -> dict | None:
    # Load checkpoint data from a JSON file if it exists
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)

        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint_data
    return None


def find_last_rollout(output_dir_base: str, dataset_name: str) -> int:
    # Check for existing rollout directories
    dataset_dir = os.path.join(output_dir_base, dataset_name)
    if not os.path.exists(dataset_dir):
        return -1

    rollout_dirs = [d for d in os.listdir(dataset_dir) if d.startswith("rollout_")]
    if not rollout_dirs:
        return -1

    rollout_numbers = [int(d.split("_")[1]) for d in rollout_dirs]
    return max(rollout_numbers)


def generate_ref_id(existing_ids: set, reranked_webpages: list[str]) -> str:
    # Generate a unique hash ID based on the reranked webpages
    result = {}

    for webpage in reranked_webpages:
        hash_object = hashlib.md5(webpage.text.encode()).hexdigest()
        for i in range(len(hash_object) - 3):
            ref_id = "#" + hash_object[i : i + 4]
            if ref_id not in existing_ids and ref_id not in result.keys():
                result[ref_id] = {"text": webpage.text, "url": webpage.url}
                break

    return result


def with_429_retry(func, max_retries=5, initial_wait=5):
    def wrapper(*args, **kwargs):
        wait_time = initial_wait
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e):
                    print(f"Received 429 error. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    raise e
        raise Exception("Max retries exceeded.")

    return wrapper


def main(args: argparse.Namespace):
    data_path = args.data_path
    subset_num = args.subset_num
    max_search_limit = args.max_search_limit
    max_turn = args.max_turn
    model_path = args.model_path
    model_context_length = args.model_context_length
    output_dir_base = args.output_dir_base
    rollout_num = args.rollout_num

    print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    dataset_name = data_path.split("/")[-1].split(".")[0]

    print("-----------------------")
    print(f"Using {dataset_name} set.")
    print("-----------------------")

    # Load model and tokenizer
    llm = load_vllm_model(model_path, gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = load_tokenizer(model_path)
    # model_context_length = AutoConfig.from_pretrained(model_path).max_position_embeddings

    # Initialize retriever
    retriever = load_retriever(args.retriever, default_k=args.retriever_top_k, **json.loads(args.retriever_kwargs))
    retriever.__call__ = types.MethodType(with_429_retry(retriever.__call__), retriever)

    # Initialize reranker
    reranker = None
    if args.reranker is not None:
        reranker = load_reranker(
            args.reranker,
            max_length=args.reranker_max_tokens,
            batch_size=args.reranker_batch_size,
            **json.loads(args.reranker_kwargs),
        )

    # Initialize summarizer
    summarizer = Summarizer(
        llm=llm,
        tokenizer=tokenizer,
        model_context_length=model_context_length,
        top_k=args.summarizer_top_k,
        max_tokens=args.summarizer_max_tokens,
        temperature=args.summarizer_temperature,
        top_p=args.summarizer_top_p,
    )

    # load last rollout
    last_rollout = find_last_rollout(output_dir_base, dataset_name)

    is_rollout_initialized = False  # True if the initialization when starting the rollout is done

    if args.auto_resume and last_rollout >= 0:
        rollout_id = last_rollout
        output_dir = make_output_dir(output_dir_base, dataset_name, rollout_id)
        checkpoint = load_checkpoint(output_dir)

        assert rollout_num > rollout_id, (
            f"rollout_num ({rollout_num}) must be greater than current rollout_id ({rollout_id}) "
            f"when using auto_resume. Current rollout: {rollout_id}, Target rollouts: {rollout_num}"
        )

        if checkpoint:
            print(f"Resuming from rollout {rollout_id}, turn {checkpoint['turn']}")
            # restore state from checkpoint
            active_sequences = checkpoint["active_sequences"]
            batch_output_records = checkpoint["batch_output_records"]
            start_turn = checkpoint["turn"]
            is_rollout_initialized = True
        else:
            print("No valid checkpoint found, starting a new rollout.")
            rollout_id = last_rollout + 1
    else:
        rollout_id = 0

    # Rollout
    while rollout_id < rollout_num:
        print(f"\n===================Rollout {rollout_id + 1} of {rollout_num}===================")

        if not is_rollout_initialized:
            output_dir = make_output_dir(output_dir_base, dataset_name, rollout_id)
            initial_active_sequences_path = os.path.join(
                output_dir_base, dataset_name, "initial_active_sequences.json"
            )
            initial_batch_output_records_path = os.path.join(
                output_dir_base, dataset_name, "initial_batch_output_records.json"
            )
            start_turn = 0

            if os.path.exists(initial_active_sequences_path) and os.path.exists(initial_batch_output_records_path):
                print(
                    f"Loading initial search status from {initial_active_sequences_path}, {initial_batch_output_records_path}"
                )
                with open(initial_active_sequences_path, "r", encoding="utf-8") as f:
                    active_sequences = json.load(f)
                with open(initial_batch_output_records_path, "r", encoding="utf-8") as f:
                    batch_output_records = json.load(f)
            else:
                batch_output_records = []
                data = load_data(data_path)
                questions = [item["Question"] for item in data]

                # perform initial search for all questions
                batch_initial_search_documents_path = os.path.join(
                    output_dir_base, dataset_name, f"batch_initial_search_documents.json"
                )
                if os.path.exists(batch_initial_search_documents_path):
                    print(f"Loading initial search documents from {batch_initial_search_documents_path}")
                    with open(batch_initial_search_documents_path, "r", encoding="utf-8") as f:
                        batch_initial_search_documents = json.load(f)
                else:
                    batch_initial_search_documents = []
                    for question in questions:
                        try:
                            print(f'Executing search for query: "{question}"')
                            search_results = retriever(question)
                        except Exception as e:
                            print(f'Search failed for query "{question}": {e}')
                            search_results = []

                        if reranker is not None and len(search_results) > 0:
                            print("Reranking search results")
                            reranked_results, _ = reranker(question, search_results)
                        else:
                            reranked_results = search_results

                        # attend unique hash ids to each webpage
                        initial_search_documents = generate_ref_id(set(), reranked_results)

                        batch_initial_search_documents.append(initial_search_documents)

                    with open(batch_initial_search_documents_path, "w", encoding="utf-8") as f:
                        json.dump(batch_initial_search_documents, f, ensure_ascii=False, indent=2)

                initial_search_summaries = summarizer(
                    previous_reasonings=[],  # empty list for initial search
                    search_queries=questions,
                    documents=batch_initial_search_documents,
                    batch_output_records=batch_output_records,  # Pass the collection list
                    max_retry=20,
                )

                active_sequences, input_list = prepare_input_prompts(
                    data,
                    max_search_limit,
                    tokenizer,
                    subset_num,
                    batch_initial_search_documents,
                    initial_search_summaries,
                )

                # save initial active sequences for future use
                with open(initial_active_sequences_path, "w", encoding="utf-8") as f:
                    json.dump(active_sequences, f, ensure_ascii=False, indent=2)
                with open(initial_batch_output_records_path, "w", encoding="utf-8") as f:
                    json.dump(batch_output_records, f, ensure_ascii=False, indent=2)

        is_rollout_initialized = False

        # Initialize collection structure
        start_time = time.time()
        turn = start_turn
        unfinished = True

        # Start the interaction loop
        while turn < max_turn and unfinished:
            sequences_needing_generation = [seq for seq in active_sequences if not seq["finished"]]

            if sequences_needing_generation:
                turn += 1
                print(f"\n-------------- Turn {turn} --------------")
                print(f"We have {len(sequences_needing_generation)} sequences needing generation...")

                outputs = run_generation(
                    prompts=[s["prompt"] for s in sequences_needing_generation],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k_sampling=args.top_k_sampling,
                    llm=llm,
                    tokenizer=tokenizer,
                    stop=[END_SEARCH_QUERY, tokenizer.eos_token],
                )
                print("Generation completed, processing outputs...")

                # Initialize batch variables
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
                    search_query = extract_between_tags(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

                    # If a search query is present and the needs to be executed
                    if search_query and seq["output"].rstrip().endswith(END_SEARCH_QUERY):
                        if seq["search_count"] < max_search_limit and search_query not in set(
                            seq["executed_search_queries"]
                        ):
                            try:
                                print(f'Executing search for query: "{search_query}"')
                                search_results = retriever(search_query)
                            except Exception as e:
                                print(f'Search failed for query "{search_query}": {e}')
                                search_results = []

                            if reranker is not None and len(search_results) > 0:
                                print("Reranking search results")
                                reranked_results, _ = reranker(search_query, search_results)
                            else:
                                reranked_results = search_results

                            # attend unique hash ids to each webpage
                            existing_ids = seq["executed_search_urls"].keys()
                            search_documents = generate_ref_id(existing_ids, reranked_results)
                            seq["executed_search_urls"].update(
                                {ref_id: data["url"] for ref_id, data in search_documents.items()}
                            )

                            all_reasoning_steps = seq["output"]
                            all_reasoning_steps = all_reasoning_steps.replace("\n\n", "\n").split("\n")

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
                                        if truncated_prev_reasoning[-len("\n\n...\n\n") :] != "\n\n...\n\n":
                                            truncated_prev_reasoning += "...\n\n"
                            truncated_prev_reasoning = truncated_prev_reasoning.strip("\n")

                            # Collect parameters for batch processing
                            batch_original_questions.append(seq["item"]["Question"])
                            batch_prev_reasonings.append(truncated_prev_reasoning)
                            batch_search_queries.append(search_query)
                            batch_documents.append(search_documents)
                            batch_sequences.append(seq)

                            # Update search count and executed queries
                            seq["search_count"] += 1
                            seq["executed_search_queries"].append(search_query)

                        elif seq["search_count"] >= max_search_limit:
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                            seq["prompt"] += limit_message
                            seq["output"] += limit_message
                            seq["history"].append(limit_message)
                            seq["all_info"].append({f"turn_{turn}_search_limited": limit_message})
                            print(f'Search limit reached for query: "{search_query}"')

                        elif search_query in set(seq["executed_search_queries"]):
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                            seq["prompt"] += limit_message
                            seq["output"] += limit_message
                            seq["history"].append(limit_message)
                            seq["all_info"].append({f"turn_{turn}_search_limited": limit_message})
                            print(f'Repeated search for query: "{search_query}"')
                    else:
                        # If no search query needs to be executed, mark the sequence as finished
                        seq["finished"] = True
                        print("Sequence marked as complete.")

                print(f"get search time taken: {time.time() - start_search_time}")

                if batch_sequences:
                    print(f"Batch processing {len(batch_sequences)} sequences with summarizer...")
                    webpage_summaries = summarizer(
                        previous_reasonings=batch_prev_reasonings,
                        search_queries=batch_search_queries,
                        documents=batch_documents,
                        batch_output_records=batch_output_records,  # Pass the collection list
                    )

                    print("Batch generation completed, assigning outputs to sequences...")

                    for seq, analysis, documents in zip(batch_sequences, webpage_summaries, batch_documents):
                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                        seq["prompt"] += append_text
                        seq["output"] += append_text
                        seq["history"].append(append_text)
                        seq["all_info"].extend(
                            [
                                {f"turn_{turn}_search": {ref_id: data["text"] for ref_id, data in documents.items()}},
                                {f"turn_{turn}_webpage_analysis": analysis},
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
                    "executed_search_urls": ele["executed_search_urls"],
                    "all_info": ele["all_info"],
                }
                for ele in active_sequences
            ]
            with open(os.path.join(output_dir, f"turn_{turn}.json"), "w", encoding="utf-8") as f:
                json.dump(active_sequences_part, f, ensure_ascii=False, indent=2)

            save_checkpoint(output_dir, rollout_id, turn, active_sequences, batch_output_records)
            unfinished = [seq for seq in active_sequences if not seq["finished"]]

        total_time = time.time() - start_time
        print(f"Total time taken: {total_time} seconds")

        # ---------------------- Save Batch Output Records to JSON File ----------------------
        # Define output JSON file path
        t = time.localtime()
        batch_output_file = os.path.join(
            output_dir, f"test.{t.tm_mon}.{t.tm_mday},{t.tm_hour}{t.tm_min}.info_extract.json"
        )

        # Save batch_output_records to JSON file
        with open(batch_output_file, "w", encoding="utf-8") as f:
            json.dump(batch_output_records, f, ensure_ascii=False, indent=2)

        print(f"Batch outputs saved to {batch_output_file}")

        # ---------------------- Stage-wise Analysis ----------------------
        turn_files = os.listdir(output_dir)
        turn_files = [file for file in turn_files if file.startswith("turn_")]
        max_turn_file = max(turn_files, key=lambda x: int(re.search(r"turn_(\d+)", x).group(1)))

        max_turn_file_path = os.path.join(output_dir, max_turn_file)
        print(f"max_turn_file_path: {max_turn_file_path}")
        stage_wise_analysis(model_path, max_turn_file_path)

        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"Checkpoint removed: {checkpoint_path}")

        rollout_id += 1

    # Save the hyperparameters to a JSON file
    hparams = vars(args)
    hparams_file = os.path.join(output_dir_base, "hparams.json")
    with open(hparams_file, "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)

    print(f"Hyperparameters saved to {hparams_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SimpleDeepsearcher for various datasets.")

    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset to use.")

    parser.add_argument(
        "--subset_num", type=int, default=-1, help="Number of examples to process. Defaults to all if not specified."
    )

    # Search and document retrieval configuration
    parser.add_argument("--max_search_limit", type=int, default=10, help="Maximum number of searches per question.")

    parser.add_argument("--max_turn", type=int, default=15, help="Maximum number of turns.")

    parser.add_argument("--summarizer_top_k", type=int, default=10, help="Maximum number of search documents to use.")

    parser.add_argument("--max_doc_len", type=int, default=3000, help="Maximum length of each searched document.")

    # Model configuration
    parser.add_argument("--model_path", type=str, required=True, help="Path to the reasoning model.")

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75, help="GPU memory utilization for vLLM.")

    parser.add_argument("--model_context_length", type=int, default=262144, help="Max context length of the model specified by --model_path.")

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")

    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter.")

    parser.add_argument("--top_k_sampling", type=int, default=40, help="Top-k sampling parameter.")

    parser.add_argument("--max_tokens", type=int, default=20480, help="Maximum number of tokens to generate.")

    parser.add_argument("--output_dir_base", type=str, required=True, help="output_dir")

    parser.add_argument("--is_exclude_urls", action="store_true", help="is_exclude_urls")

    parser.add_argument("--rollout_num", type=int, default=1, help="The number of rollout per question")

    parser.add_argument(
        "--auto_resume", action="store_true", help="Automatically resume from the last checkpoint if available"
    )

    # Retriever configuration
    parser.add_argument("--retriever", type=str, required=True, help="Retriever to use")
    parser.add_argument(
        "--retriever_top_k", type=int, default=10, help="Top-k documents to retrieve from the retriever"
    )
    parser.add_argument(
        "--retriever_kwargs", type=str, default="{}", help="Additional kwargs for the retriever in JSON format"
    )

    # Reranker configuration
    parser.add_argument("--reranker", type=str, help="Reranker to use")
    parser.add_argument(
        "--reranker_max_tokens", type=int, default=1024, help="Maximum number of tokens for the reranker"
    )
    parser.add_argument("--reranker_batch_size", type=int, default=1, help="Batch size for the reranker")
    parser.add_argument(
        "--reranker_kwargs", type=str, default="{}", help="Additional kwargs for the reranker in JSON format"
    )

    parser.add_argument(
        "--summarizer_max_tokens", type=int, default=8192, help="Maximum number of tokens for the summarizer"
    )
    parser.add_argument(
        "--summarizer_temperature", type=float, default=0.6, help="Sampling temperature for the summarizer"
    )
    parser.add_argument(
        "--summarizer_top_p", type=float, default=0.95, help="Top-p sampling parameter for the summarizer"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
