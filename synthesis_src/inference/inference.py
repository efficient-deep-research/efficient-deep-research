# Adapted from https://github.com/sunnynexus/Search-o1/blob/main/scripts/run_search_o1.py 
import os
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict
import argparse
from functools import partial

import multiprocessing
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from google_search import (
    google_web_search, 
    extract_relevant_info, 
    fetch_page_content, 
    extract_snippet_with_context
)
from evaluate import (
    run_evaluation, 
    run_evaluation_for_eval,
    extract_answer
)
from prompts import (
    get_multiqa_instruction,
    get_math_instruction,
    get_task_instruction_openqa,
    get_task_instruction_math,
    get_webpage_to_reasonchain_instruction
)
from functools import partial
from openai import OpenAI

from stage_wise_analysis import stage_wise_analysis

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"


def parse_args():
    parser = argparse.ArgumentParser(description="Run SimpleDeepsearcer for various datasets.")


    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--subset_num',
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified."
    )

    # Search and document retrieval configuration
    parser.add_argument(
        '--max_search_limit',
        type=int,
        default=10,
        help="Maximum number of searches per question."
    )

    parser.add_argument(
        '--max_turn',
        type=int,
        default=15,
        help="Maximum number of turns."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Maximum number of search documents to return."
    )

    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )

    # Model configuration
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the reasoning model."
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k_sampling',
        type=int,
        default=40,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=20480,
        help="Maximum number of tokens to generate."
    )

    # Bing API Configuration
    parser.add_argument(
        '--google_subscription_key',
        type=str,
        required=True,
        help="Google Search API subscription key."
    )

    parser.add_argument(
        '--google_endpoint',
        type=str,
        default="https://google.serper.dev/search",
        help="Google Search API endpoint."
    )

    parser.add_argument(
        '--cache_dir_base',
        type=str,
        required=True,
        help="cache path."
    )

    parser.add_argument(
        '--output_dir_base',
        type=str,
        required=True,
        help="output_dir"
    )

    parser.add_argument(
        '--is_exclude_urls',
        action="store_true",
        help="is_exclude_urls"
    )


    parser.add_argument(
        '--summarization_model_path',
        type=str,
        required=True,
        help="Path to the summary model."
    )


    parser.add_argument(
        '--summarization_model_url',
        type=str,
        required=True,
        help="Base url of the summarization model."
    )

    return parser.parse_args()



def webpage_analysis_single(summ_model_url, summ_model_path, prompt) -> str:
    client_summ_model = OpenAI(
        base_url=summ_model_url,
        api_key="EMPTY"
    )
    for i in range(10): # max retry 10 times
        try:
            completion = client_summ_model.chat.completions.create(
                model=summ_model_path,
                max_tokens=8192,
                temperature=0.6,
                top_p=0.95,
                messages=[prompt],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(1)
            continue
    return "None"

def main():
    args = parse_args()
    # Extract arguments
    dataset_name = args.dataset_name
    subset_num = args.subset_num
    MAX_SEARCH_LIMIT = args.max_search_limit
    MAX_TURN = args.max_turn
    top_k = args.top_k
    max_doc_len = args.max_doc_len
    model_path = args.model_path
    summ_model_path = args.summarization_model_path
    summ_model_url = args.summarization_model_url
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    max_tokens = args.max_tokens
    google_subscription_key = args.google_subscription_key
    google_endpoint = args.google_endpoint
    cache_dir_base = args.cache_dir_base
    output_dir_base = args.output_dir_base
    is_exclude_urls = args.is_exclude_urls

    print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
   
    # Data paths based on dataset
    data_path = f"./data/{dataset_name}.json"

    print('-----------------------')
    print(f'Using {dataset_name} set.')
    print('-----------------------')

    # ---------------------- Caching Mechanism ----------------------
    # Define cache directories and file paths
    model_name = model_path.split('/')[-1].replace('-instruct', '')
    cache_dir = cache_dir_base
    search_cache_path = os.path.join(cache_dir, 'search_cache.json')
    url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        try: 
            with open(search_cache_path, 'r', encoding='utf-8') as f:
                search_cache = json.load(f)
        except Exception as e:
            print(f"load search_cache.json error: {e}")
            search_cache = {}

    else:
        search_cache = {}

    if os.path.exists(url_cache_path):
        try:
            with open(url_cache_path, 'r', encoding='utf-8') as f:
                url_cache = json.load(f)
        except Exception as e:
            print(f"load url_cache.json error: {e}")
            url_cache = {}

    else:
        url_cache = {}

    # Function to save caches
    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    # ---------------------- Reasoning Model Loading ----------------------
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    print("Tokenizer loaded successfully.")

    # Define output directory based on the dataset
    output_dir = os.path.join(output_dir_base, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    print(f"device_count: {torch.cuda.device_count()}")
    
    # Initialize the LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,

    )
    print("Model loaded successfully.")

    # ---------------------- Data Loading ----------------------
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)
    print(f"Data loaded successfully. Total examples: {len(filtered_data)}")

    # ---------------------- Batch Generation Function ----------------------
    def generate_webpage_to_reasonchain_batch(
        original_questions: List[str],
        prev_reasonings: List[str],
        search_queries: List[str],
        documents: List[str],
        dataset_name: str,
        summ_model_url: OpenAI,
        summ_model_path: str,
        batch_output_records: List[Dict],  # New parameter to collect outputs
        coherent: bool = False,
    ) -> List[str]:

        user_prompts = [
            get_webpage_to_reasonchain_instruction(r, sq, doc)
            for r, sq, doc in zip(prev_reasonings, search_queries, documents)
        ]


        prompts = [{"role": "user", "content": up} for up in user_prompts]
        print("webpage ana prompts[0]")
        print(prompts[0])

        webpage_analysis_single_to_map = partial(webpage_analysis_single, summ_model_url, summ_model_path)
        with multiprocessing.Pool(processes=50) as pool:
            raw_outputs = list(tqdm(pool.imap(webpage_analysis_single_to_map, prompts), total=len(prompts), desc="generate webpage analyses"))


        # Count the number of summarization errors
        sum_error = 0
        for output in raw_outputs:
            if output is None or output == "None" or output == "":
                sum_error += 1
        print(f"summarization_error: {sum_error}, ratios: {sum_error / len(raw_outputs)}")
     
        extracted_infos = [extract_answer(raw, mode='infogen') for raw in raw_outputs]

        for i, (p, r, e) in enumerate(zip(prompts, raw_outputs, extracted_infos)):
            batch_output_records.append({
                'prompt': p,
                'raw_output': r,
                'extracted_info': e
            })

        return extracted_infos

    # ---------------------- Preparation of Input Prompts ----------------------
    input_list = []
    for item in filtered_data:
        question = item['Question']

        if dataset_name in ['aime']:
            instruction = get_multiqa_instruction(MAX_SEARCH_LIMIT)
            user_prompt = get_task_instruction_math(question)

        else:
            instruction = get_multiqa_instruction(MAX_SEARCH_LIMIT)
            user_prompt = get_task_instruction_openqa(question)
            
        
        prompt = [{"role": "user", "content": instruction + user_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)

    if subset_num != -1:
        input_list = input_list[:subset_num]
        filtered_data = filtered_data[:subset_num]

    # Initialize active sequences
    active_sequences = [{
        'item': item,
        'prompt': prompt,
        'output': '',
        'finished': False,
        'history': [],
        'search_count': 0,
        'executed_search_queries': set(),
        'all_info': [],
    } for item, prompt in zip(filtered_data, input_list)]

    # ---------------------- Generation Function ----------------------
    def run_generation(sequences: List[Dict], max_tokens: int) -> List:
        prompts = [s['prompt'] for s in sequences]

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

    # Function to extract text between two tags
    def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def replace_recent_steps(origin_str, replace_str):
        """
        Replaces specific steps in the original reasoning steps with new steps.
        If a replacement step contains "DELETE THIS STEP", that step is removed.

        Parameters:
        - origin_str (str): The original reasoning steps.
        - replace_str (str): The steps to replace or delete.

        Returns:
        - str: The updated reasoning steps after applying replacements.
        """

        def parse_steps(text):
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
                    content = line[step_match.end():].strip()
                    current_content = [content] if content else []
                else:
                    if current_step_num is not None:
                        current_content.append(line)
            
            # Save the last step if any
            if current_step_num is not None:
                steps[current_step_num] = "\n".join(current_content).strip()
            
            return steps

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

    # ---------------------- Initialize Collection Structure ----------------------
    # Initialize a list to collect batch outputs
    batch_output_records = []

    start_time = time.time()
    turn = 0

    # Main loop until all sequences are finished or maximum turns reached
    while True:
        # Identify sequences that need generation
        sequences_needing_generation = [seq for seq in active_sequences if not seq['finished']]

        if sequences_needing_generation:
            turn += 1
            print(f'\n-------------- Turn {turn} --------------')
            print(f"We have {len(sequences_needing_generation)} sequences needing generation...")
            outputs = run_generation(sequences_needing_generation, max_tokens) 
            print("Generation completed, processing outputs...")

            # Initialize batch variables
            batch_relevant_info = []
            batch_original_questions = []
            batch_prev_reasonings = []
            batch_search_queries = []
            batch_documents = []
            batch_sequences = []

            # Collect URLs to fetch across all sequences
            all_urls_to_fetch = set()
            url_snippets = {}
            url_sequence_map = {}  # Map URL to list of sequences needing it

            start_search_time = time.time()
            # Process each sequence and collect URLs
            for seq, out in zip(sequences_needing_generation, outputs):
                text = out.outputs[0].text
                seq['history'].append(text)
                # Append generated text to prompt and output
                seq['prompt'] += text
                seq['output'] += text
                seq['all_info'].append({f"turn_{turn}_reason": text})
                # Extract search query
                search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

                # If a search query is present and needs to be executed
                if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
                    if seq['search_count'] < MAX_SEARCH_LIMIT and search_query not in seq['executed_search_queries']:
                        # Execute search, use cache if available
                        if search_query in search_cache:
                            results = search_cache[search_query]
                            print(f"Using cached search results for query: \"{search_query}\"")
                        else:
                            try:
                                if is_exclude_urls and "urls" in seq["item"]["metadata"]: 
                                    print(f"is_exclude_urls: {is_exclude_urls}")
                                    exclude_urls = seq["item"]["metadata"]["urls"]
                                else:
                                    exclude_urls = []

                                print(f"Execute and cache search for query: \"{search_query}\"")
                                results = google_web_search(search_query, google_subscription_key, google_endpoint, market='en-US', language='en', exclude_urls=exclude_urls) # 执行搜索
                                search_cache[search_query] = results
                                print(f"Executed and cached search for query: \"{search_query}\"")
                            except Exception as e:
                                print(f"Error during search query '{search_query}': {e}")
                                search_cache[search_query] = {}
                                results = {}

                        # Extract relevant information from Bing search results
                        relevant_info = extract_relevant_info(results)[:top_k]
                        seq['relevant_info'] = relevant_info

                        # Extract URLs and snippets
                        urls_to_fetch = [it['url'] for it in relevant_info]
                        snippets = {info['url']: info['snippet'] for info in relevant_info if 'snippet' in info}

                        # Filter URLs that are not cached
                        urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]
                        cached_urls = [u for u in urls_to_fetch if u in url_cache]

                        # Store info for all_urls_to_fetch and url_snippets
                        for url in urls_to_fetch_filtered:
                            all_urls_to_fetch.add(url)
                            url_snippets[url] = snippets.get(url, "") 

                        all_reasoning_steps = seq['output']
                        all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n")

                        truncated_prev_reasoning = ""
                        for i, step in enumerate(all_reasoning_steps):
                            truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

                        prev_steps = truncated_prev_reasoning.split('\n\n')
                        if len(prev_steps) <= 5:
                            truncated_prev_reasoning = '\n\n'.join(prev_steps)
                        else:
                            truncated_prev_reasoning = ''
                            for i, step in enumerate(prev_steps):
                                if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                                    truncated_prev_reasoning += step + '\n\n'
                                else:
                                    if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                        truncated_prev_reasoning += '...\n\n'
                        truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')

                        # Collect parameters for batch processing
                        batch_relevant_info.append(relevant_info)
                        batch_original_questions.append(seq['item']['Question'])
                        batch_prev_reasonings.append(truncated_prev_reasoning)
                        batch_search_queries.append(search_query)
                        batch_sequences.append(seq)

                        # Update search count and executed queries
                        seq['search_count'] += 1
                        seq['executed_search_queries'].add(search_query)

                    elif seq['search_count'] >= MAX_SEARCH_LIMIT:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        seq["all_info"].append({f"turn_{turn}_search_limited": limit_message})
                        print(f"Search limit reached for query: \"{search_query}\"")

                    elif search_query in seq['executed_search_queries']:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        seq["all_info"].append({f"turn_{turn}_search_limited": limit_message})
                        print(f"Repeated search for query: \"{search_query}\"")
                        

                else:
                    # If no search query needs to be executed, mark the sequence as finished
                    seq['finished'] = True
                    print("Sequence marked as complete.")

            print(f"get search time taken: {time.time() - start_search_time}")
            print(f"all_urls_to_fetch len: {len(all_urls_to_fetch)}, url_cache len: {len(url_cache)}")
            print(f"all_urls_to_fetch: {all_urls_to_fetch}")
            # Batch fetch all URLs at once to optimize speed
            
            if all_urls_to_fetch:
                print(f"Fetching {len(all_urls_to_fetch)} URLs...")
                try:
                    fetched_contents = fetch_page_content(
                        list(all_urls_to_fetch),
                        use_jina=False,
                        jina_api_key=None,
                        # snippets=url_snippets  # Do not pass snippets when updating url_cache directly
                    )
                    print(f"Fetched {len(fetched_contents)} URLs successfully.")
                except Exception as e:
                    print(f"Error during batch URL fetching: {e}")
                    fetched_contents = {url: f"Error fetching URL: {e}" for url in all_urls_to_fetch}
                # Update cache with fetched contents
                for url, content in fetched_contents.items():
                    url_cache[url] = content

            # After fetching, prepare formatted documents for batch processing
            for relevant_info in batch_relevant_info:
                formatted_documents = ""
                for i, doc_info in enumerate(relevant_info):
                    url = doc_info['url']
                    raw_context = url_cache.get(url, "")
                    doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')            
                    success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=max_doc_len)
                    if success:
                        print("extract_snippet_with_context")
                        context = filtered_context
                    else: 
                        print(f"use raw_webpage_context, {len(raw_context)}")
                        context = raw_context[:max_doc_len*2]

                    doc_info['context'] = context
                    formatted_documents += f"**Web Page {i + 1}:**\n"
                    formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
                print(f'formatted_webpage_documents: {len(formatted_documents)}')
                batch_documents.append(formatted_documents)

            # After fetching, prepare for batch processing if there are any
            if batch_sequences:
                print(f"Batch processing {len(batch_sequences)} sequences with generate_webpage_to_reasonchain_batch...")
                webpage_analyses = generate_webpage_to_reasonchain_batch(
                    original_questions=batch_original_questions,
                    prev_reasonings=batch_prev_reasonings,
                    search_queries=batch_search_queries,
                    documents=batch_documents,
                    dataset_name=dataset_name,
                    summ_model_url=summ_model_url,
                    summ_model_path=summ_model_path,
                    batch_output_records=batch_output_records,  # Pass the collection list
                )
                print("Batch generation completed, assigning outputs to sequences...")

                for seq, analysis,doc in zip(batch_sequences, webpage_analyses, batch_documents): 
                    if isinstance(analysis, str):
                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text) 
                        seq['all_info'].extend([{f"turn_{turn}_search": doc}, {f"turn_{turn}_webpage_analyses": analysis}])
                    else:
                        append_text = replace_recent_steps(seq['output'], analysis)
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)
                        seq['all_info'].extend([{f"turn_{turn}_search": doc}, {f"turn_{turn}_webpage_analyses": analysis}])

        # Check if all sequences are finished
        active_sequences_part = [{
            'item': ele["item"],
            'prompt': ele['prompt'],
            'output': ele["output"],
            'finished': ele["finished"],
            'history':ele["history"],
            'search_count': ele["search_count"],
            'all_info': ele['all_info']
        } for ele in active_sequences]
        with open(os.path.join(output_dir, f"turn_{turn}.json"), 'w', encoding='utf-8') as f:
            json.dump(active_sequences_part, f, ensure_ascii=False, indent=2)
        unfinished = [seq for seq in active_sequences if not seq['finished']]
        if not unfinished:
            break
        else:
            if turn >= MAX_TURN:
                print(f"Maximum number of turns ({MAX_TURN}) reached, stopping.")
                break

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time} seconds")

    # ---------------------- Save Batch Output Records to JSON File ----------------------
    # Define output JSON file path
    t = time.localtime()
    batch_output_file = os.path.join(output_dir, f'test.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.info_extract.json')

    # Save batch_output_records to JSON file
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_output_records, f, ensure_ascii=False, indent=2)

    print(f"Batch outputs saved to {batch_output_file}")

    # Prepare output list for evaluation
    output_list = [seq['output'] for seq in active_sequences]

    # Run evaluation
    if dataset_name in ["eval", "gaia"]:
        run_evaluation_for_eval(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, 'test')
    else:
        run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, 'test')

    # ---------------------- Stage-wise Analysis ----------------------
    turn_files = os.listdir(output_dir)
    turn_files = [file for file in turn_files if file.startswith("turn_")]
    max_turn_file = max(turn_files, key=lambda x: int(re.search(r'turn_(\d+)', x).group(1)))

    max_turn_file_path = os.path.join(output_dir, max_turn_file)
    print(f"max_turn_file_path: {max_turn_file_path}")
    stage_wise_analysis(model_path, max_turn_file_path)

    # ---------------------- Update Search and URL Cache ----------------------
    print('Updating Search and URL Cache...')
    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        try:
            with open(search_cache_path, 'r', encoding='utf-8') as f:
                search_cache_new = json.load(f)
        except Exception as e:
            print(f"Error loading search cache: {e}")
            search_cache_new = {}
    else:
        search_cache_new = {}

    if os.path.exists(url_cache_path):
        try:
            with open(url_cache_path, 'r', encoding='utf-8') as f:
                url_cache_new = json.load(f)
        except Exception as e:
            print(f"Error loading url cache: {e}")
            url_cache_new = {}
    else:
        url_cache_new = {}

    search_cache.update(search_cache_new)
    url_cache.update(url_cache_new)

    save_caches()

    print("Process completed.")

if __name__ == "__main__":
    main()
