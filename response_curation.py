import copy
import os
import json
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import argparse

def plot_distribution(data, title="Distribution of First Reason Length", bins=40, name="url", xlabel="First Reason Length", ylabel="Frequency", output_path=""):
    # Set plot style
    sns.set(style="whitegrid")
    
    # Create histogram and kernel density estimation plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=bins, color="skyblue", edgecolor="black")
    
    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Display the plot
    plt.show()
    # plt.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_path, f"{name}.png"), dpi=300, bbox_inches="tight")

    quantiles = np.arange(0.8, 1.0, 0.03)  # Quantiles from 0.8 to 0.9
    quantile_values = np.quantile(data, quantiles)  # Values corresponding to quantiles
    total_count = len(data)  # Total data count
    
    print(f"NAME: {name}")
    print("Quantile statistics:")
    for q, value in zip(quantiles, quantile_values):
        count_below = np.sum(np.array(data) <= value)  # Count of values less than or equal to current quantile
        percentage = count_below / total_count * 100  # Percentage
        print(f"Quantile {q:.2f}: "
              f"Value = {value:.2f}, "
              f"Count = {count_below}, "
              f"Percentage = {percentage:.2f}%")


def extract_incorrect_markers(input_string):
    """
    Extract incorrect markers from the input string. Correct markers are in the form of <|keyword|>.
    
    Args:
        input_string (str): Input string.
    Returns:
        List[str]: List of incorrect markers found in the input string.
    """
    
    standard_keywords = [
        'begin_search_query',
        'end_search_query',
        'begin_search_result',
        'end_search_result'
    ]
    
    # Create regex pattern to match standard keywords
    pattern = re.compile('(' + '|'.join(map(re.escape, standard_keywords)) + ')')
    
    incorrect_markers = []
    
    for match in pattern.finditer(input_string):
        keyword = match.group()
        start, end = match.start(), match.end()

        # Check if the preceding two characters are <|
        prefix_ok = (start >= 2 and input_string[start-2:start] == '<|')
        # Check if the following two characters are |>
        suffix_ok = (end + 2 <= len(input_string) and input_string[end:end+2] == '|>')
        if not (prefix_ok and suffix_ok):
            keyword = input_string[start-2:end+2]
            incorrect_markers.append(keyword)
    
    return incorrect_markers

def detect_variant_markers(input_string):
    """
    Detect whether variant forms of markers exist in the input string.
    
    Args:
        input_string (str): Input string.
    
    Returns:
        bool: Whether variant markers exist or think tags are used excessively. Returns True if issues are detected, otherwise False.
    """
    
    variant_markers = extract_incorrect_markers(input_string)

    think_count = input_string.count("</think>")
    has_think_issue = think_count > 2
    
    return (len(variant_markers) > 0 ) or has_think_issue


def merge_questions(root_path):
    """
    Merge Question data from all qualifying JSON files under the specified path.

    Args:
        root_path (str): Root directory path.

    Returns:
        dict: Merged dictionary with Question as key and list as value.
    """
    merged_dict = defaultdict(list)
        
    if os.path.isdir(root_path):
        
        for rollout_folder in tqdm(sorted(os.listdir(root_path)), total=len(os.listdir(root_path)), desc="rollout_folder"):
            if not rollout_folder.startswith("rollout_"):
                continue
            rollout_path = os.path.join(root_path, rollout_folder)
            if not os.path.isdir(rollout_path) or not rollout_folder.startswith("rollout_"):
                continue
            pattern = re.compile(r"^turn_(\d+)\.json$")
            max_num = None
            
            for file_name in sorted(os.listdir(rollout_path)):
                match = pattern.match(file_name)
                if match:
                    num = int(match.group(1))                    
                    if max_num is None or num > max_num:
                        max_num = num

                elif file_name.startswith("test") and file_name.endswith(".json") and len(file_name.split('.')) == 4:
                    file_path = os.path.join(rollout_path, file_name)
                    metrics = json.load(open(os.path.join(rollout_path, file_name), 'r', encoding='utf-8'))

                    for sample in metrics:
                        question = sample["Question"].split("\\boxed{YOUR_ANSWER}.\n\nQuestion:\n")[-1]
                        question = question.split("\n\n<|im_end|>\n<|im_start|>assistant")[0]
                        question = question.strip()
                        # q2m[question] = sample["Metrics"]

            if max_num is not None:
                file_name = f"turn_{max_num}.json"
                file_path = os.path.join(rollout_path, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for sample in data:
                            question = sample["item"]["Question"].strip()
                            merged_dict[question].append(sample)
                                
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    import pdb
                    pdb.set_trace()
                    
    return dict(merged_dict)


def detect_repeat(response, n=3, threshold=0.3):
    """
    Detect whether there are repetitive patterns in text generated by large language models.
    
    Args:
        response (str): Text generated by the large language model.
        n (int): Size of n-gram, default is 3.
        threshold (float): Threshold for determining repetition, default is 0.3.
                          If the frequency of an n-gram exceeds this threshold, 
                          repetitive behavior is considered to exist.
    
    Returns:
        bool: Whether repetitive patterns are detected. Returns True if detected, otherwise False.
        dict: Dictionary containing each n-gram and its occurrence count.
    """

    words = response.split()
    
    # If text is too short to generate valid n-grams, return False directly
    if len(words) < n:
        return False, {}
    
    # Count the frequency of n-grams
    ngram_counts = defaultdict(int)
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])  # Use tuple as key
        ngram_counts[ngram] += 1

    # Count total n-grams
    total_ngrams = sum(ngram_counts.values())

    # Check if any n-gram frequency exceeds threshold
    repeated_ngrams = {"".join(ngram): count / total_ngrams for ngram, count in ngram_counts.items() if count / total_ngrams > threshold}
    
    # If repeated n-grams exist, return True and repeated n-gram information
    if repeated_ngrams:
        print(f"repeated_ngrams: {repeated_ngrams}")
        return True, repeated_ngrams

    # Otherwise return False
    return False, {}

def is_valid_history(history):
    """
    Detect whether the history list conforms to valid cases.
    
    Args:
        history (list): Input history list.
    
    Returns:
        bool: Returns True if it conforms to valid cases, otherwise returns False.
    """
    
    SPECIAL_TOKENS = {
        "<|begin_search_query|>",
        "<|end_search_query|>",
        "<|begin_search_result|>",
        "<|end_search_result|>",
        "</think>"
    }
    
    for hid, his in enumerate(history):
        
        full_text = his.strip()
    
        # Case 1: Only one </think> and it's the last element
        if "</think>" in his:
            if hid == len(history) - 1 and history[-1].count("</think>") == 1:
                for token in SPECIAL_TOKENS:
                    if token != "</think>" and token in full_text:
                        return False
            else:
                return False

        # Case 2: Starts with <|begin_search_result|> and ends with <|end_search_result|>, no other special markers in between
        elif (
            full_text.startswith("<|begin_search_result|>") and
            full_text.endswith("<|end_search_result|>")
        ):
            middle_content = full_text[len("<|begin_search_result|>"):-len("<|end_search_result|>")].strip()
            if any(token in middle_content for token in SPECIAL_TOKENS):
                return False
        
        # Case 3: Ends with <|end_search_query|>, contains exactly one <|begin_search_query|> in the middle, no other special tokens
        elif full_text.endswith("<|end_search_query|>"):
            middle_content = full_text[:-len("<|end_search_query|>")].strip()
            if middle_content.count("<|begin_search_query|>") == 1:
                # Remove <|begin_search_query|> and check for other special markers
                middle_without_begin = middle_content.replace("<|begin_search_query|>", "").strip()
                if any(token in middle_without_begin for token in SPECIAL_TOKENS):
                    return False
            else:
                return False

        else:
            return False
    return True

def parse_web_pages(web_page_str):
    """
    Parse a string containing multiple Web Pages into a List[Dict].
    
    Args:
        web_page_str (str): String containing multiple Web Pages.
        
    Returns:
        List[Dict]: List of dictionaries, each corresponding to a Web Page.
    """
    result = []
    
    # Split the string by "**Web Page X:**" segments
    segments = web_page_str.split("**Web Page ")[1:]  # skip the first empty segment

    for segment in segments:
        try:
            # Extract the JSON part
            json_part = segment.split("\n", 1)[1].strip()  # Remove the number and newline

            parsed_dict = json.loads(json_part)
            parsed_dict = {
                'title': parsed_dict['title'] if 'title' in parsed_dict else "",
                "context": parsed_dict['context'] if 'context' in parsed_dict else "",
                "url": parsed_dict['url'] if 'url' in parsed_dict else "",
            }
            result.append(parsed_dict)
        except (IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing segment: {segment[:50]}... Error: {e}")
    
    return result

def sort_solution(data):
    """
    Sort a list containing dictionaries.
    Primary key: ["is_valid_solution"], True comes first.
    Secondary key: ["search_count"], *larger* values come first.
    """
    
    sorted_data = sorted(
        data,
        key=lambda x: (not x["is_valid_solution"], -x["search_count"])  # True comes first, then larger search_count
    )
    return sorted_data

def sort_query(data):
    """
    Sort a list containing dictionaries.
    """
    sorted_data = data
    # sorted_data = sorted(
    #     data,
    #     # key=lambda x: (-x["solutions"][0]["max_reason_length"])
    #     key=lambda x: (x["ratios"], -x["min_search"])
    # )
    return sorted_data

def curate_and_rank_solutions(data, output_path):
    first_reason_length = []
    url_per_search = []
    reason_length = []
    max_reason_length = []
    max_alt = []
    max_hmm = []
    max_wait = []
    """
    Curate and rank solution candidates for each question based on quality metrics.
    
    Args:
        data (Dict[str, List[Dict]]): Questions mapped to solution attempts with metadata.
        output_path (str): Directory for saving distribution plots.
    
    Returns:
        List[Dict]: Questions with ranked solutions (best first) and computed metrics.
    """
    
    result = []
    
    for question, entries in tqdm(data.items(), desc="calculate acc ratios"):
        for entry in entries:
            reason_1 = ""
            reason_values = []
            search_results = []
            # Collect reason_1, reason_values and search_results from all_info
            for info in entry["all_info"]:
                if "turn_1_reason" in info.keys():
                    reason_1 = info["turn_1_reason"]
                for key, value in info.items():
                    if 'reason' in key:
                        reason_values.append(value)
                    elif key.endswith("_search"):
                        for web in parse_web_pages(value):
                            if not "url" in web.keys():
                                if "title" in web.keys():
                                    search_results.append(web["title"])
                                else:
                                    search_results.append(json.dumps(web))
                            else:
                                search_results.append(web["url"])

            # Compute quality metrics
            entry["first_reason_length"] = len(reason_1.split(' '))
            entry["has_repeat"], entry["repeated_n_grams"] = detect_repeat(entry["output"])
            entry["error_special_token"], entry["is_valid_history"] = detect_variant_markers(entry["output"]), is_valid_history(entry["history"])
            urls = set(search_results)
            entry["url_per_search"] = len(urls) / entry["search_count"] if entry["search_count"] else 0
            if reason_values:
                word_counts = [len(value.split()) for value in reason_values]
                entry["reason_length"] = sum(word_counts) / len(word_counts)
                entry["max_reason_length"] = max(word_counts)
                entry["max_alt"] = max([value.count("Alternatively") for value in reason_values])
                entry["max_hmm"] = max([value.count("hmm") for value in reason_values])
                entry["max_wait"] = max([value.count("wait") for value in reason_values])
            
            # Append to lists for distribution plotting
            first_reason_length.append(entry["first_reason_length"])
            url_per_search.append(entry["url_per_search"])
            reason_length.append(entry["reason_length"])
            max_reason_length.append(entry["max_reason_length"])
            max_alt.append(entry["max_alt"])
            max_hmm.append(entry["max_hmm"])
            max_wait.append(entry["max_wait"])
            
            # TODO: add conditions for is_valid_solution
            entry["is_valid_solution"] = True

        result.append({
            "question": question,
            "solutions": sort_solution(entries),
        })

    # Plot distributions
    plot_distribution(first_reason_length, title= "first_reason_length", name="reason", xlabel="first_reason_length", ylabel="count", output_path=output_path)
    plot_distribution(url_per_search, title="url_per_search", name="url", xlabel="url_per_search", ylabel="count", output_path=output_path)
    plot_distribution(reason_length, title="reason_length", name="average_reason", xlabel="reason_length", ylabel="count", output_path=output_path)
    plot_distribution(max_reason_length, title="max_reason_length", name="max_reason", xlabel="max_reason_length", ylabel="count", output_path=output_path)
    plot_distribution(max_alt, title="max_alt", name="max_alt", xlabel="max_alt", ylabel="count", output_path=output_path)
    plot_distribution(max_hmm, title="max_hmm", name="max_hmm", xlabel="max_hmm", ylabel="count", output_path=output_path)
    plot_distribution(max_wait, title="max_wait", name="max_wait", xlabel="max_wait", ylabel="count", output_path=output_path)
    
    # Sort samples. TODO: refine sorting criteria
    result = sort_query(result)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and process questions from given root path.")
    parser.add_argument("--root_path", type=str, required=True, help="Path to the directory containing question data.")
    parser.add_argument("--k", type=int, default=100000, help="Number of top valid data to retain.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the directory to save the selected data.")
    args = parser.parse_args()

    root_path = args.root_path
    k = args.k
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    merge_qa_pairs_path = os.path.join(output_path, "merged_qa_pairs.json")
    selected_data_path = os.path.join(output_path, "selected_data.json")

    print("Merge question-answer pairs...")
    result = merge_questions(root_path)
    print(f"len of all question-answer pairs: {len(result)}")

    print("Select valid solutions...")
    metrics_distribution_path = os.path.join(output_path, "metrics_distribution")
    os.makedirs(metrics_distribution_path, exist_ok=True)
    result = curate_and_rank_solutions(result, output_path=metrics_distribution_path)

    # Extract top k valid solutions
    valid = []
    for sample in result:
        if sample["solutions"][0]["is_valid_solution"]:
            sample = copy.deepcopy(sample)
            sample.update(sample["solutions"][0])
            sample.pop("solutions")
            valid.append(sample)
    print(f"len of pairs before selection: {len(valid)}")
    valid = valid[: k]

    with open(merge_qa_pairs_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Saving merged question-answer pairs to {merge_qa_pairs_path}")

    with open(selected_data_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=4)
    print(f"Saving selected data to {selected_data_path}")