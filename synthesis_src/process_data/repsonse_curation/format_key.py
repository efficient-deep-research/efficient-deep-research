import json
from tqdm import tqdm
import os
from typing import Optional, Tuple, List, Dict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import re
# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"Loaded {len(data)} items from {file_path}")
    return data

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Saved {len(data)} items to {file_path}")

def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

def process_all_info(all_info):
    processd_info = []
    error_cnt = 0 # 统计单条回复中出现的格式错误的次数

    for info in all_info:
        keys = list(info.keys())
        assert len(keys) == 1, f"Expecting only one key in the info dict, but got {keys}"
        if keys[0].endswith("_reason"):
            if "<|begin_search_result|>" in info[keys[0]] or "<|end_search_result|>" in info[keys[0]]: # 捏造搜索结果
                # print(f"error: {info[keys[0]]}")
                error_cnt += 1
            processd_info.append({"gen": info[keys[0]]})
            search_query = extract_between(info[keys[0]], BEGIN_SEARCH_QUERY, END_SEARCH_QUERY) # 提取搜索的query

            if search_query:
                processd_info.append({"search_query": search_query})

        elif keys[0].endswith("_search"):
            processd_info.append({"doc": info[keys[0]]})

        elif keys[0].endswith("_webpage_analyses"):
            if "<|begin_search_result|>" in info[keys[0]] or "<|end_search_result|>" in info[keys[0]]:
                error_cnt += 1

            doc_gen = f"\n\n{BEGIN_SEARCH_RESULT}{info[keys[0]]}{END_SEARCH_RESULT}\n\n" # 封装处理结果，添加到序列的历史记录、提示和输出中
            processd_info.append({"doc_gen": doc_gen})

        elif keys[0].endswith("_search_limited"):
            processd_info.append({"doc_gen": info[keys[0]]})
            
        else:
            print(f"Unknown key {keys[0]} in the info dict, skipping...")
            # processd_info.append(info[keys[0]])
    # assert error_cnt == 0, f"Found {error_cnt} errors, please check the data"
    # if error_cnt > 0:
    #     return -1
    return processd_info, error_cnt


def get_output(all_info): # 拼接得到完整的输出
    output_text = ""
    for info in all_info:
        for key, value in info.items():
            if key == "gen":
                output_text += value
            elif key == "doc_gen":
                output_text += value
    return output_text

def format_key(data):
    new_data = []
    for item in tqdm(data):
        processed_info, error_cnt = process_all_info(item["all_info"])
        if error_cnt > 0:
            print(f"Error processing item: {item['item']['Question']}")
            # continue
        
        output_text = get_output(processed_info)
        assert output_text == item["output"], f"Output mismatch for item: {item['item']['Question']}"
        new_item = {
            "question": item["item"]["Question"],
            "answer": item["item"]["answer"],
            "input": item["input"],
            "output": processed_info,
            # "output_text": item["output"],
            "metric": item["Metrics"],
            "search_count": item["search_count"],
            "min_search": item["min_search"]
        }
        
        new_data.append(new_item)

    return new_data


if __name__ == "__main__":
    input_file = ""
    data = load_json(input_file)
    formatted_data = format_key(data)
    output_file = input_file.replace(".json", "_formatted.json")
    save_json(formatted_data, output_file)