import json
import os
import torch
from tqdm import tqdm
import ast
import re
import argparse

keywords = ["who", "whom", "whose","where", "what", "when", "how", "why", "which", "whether", "is", "are", "do", "does", "was", "were"]

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"save to {filename}, data len: {len(data)}")
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"load from {file_path}, data len: {len(data)}")
    return data


def count_keywords(text, keywords):
    """
    统计字符串中指定关键词的出现次数。
    
    参数:
        text (str): 输入的字符串。
        keywords (list): 需要统计的关键词列表。
    
    返回:
        dict: 每个关键词及其对应的出现次数。
    """
    # 将文本转换为小写以忽略大小写
    text = text.lower()
    
    # 创建一个字典存储每个关键词的计数结果
    keyword_counts = {}
    total = 0
    special_total = 0
    for keyword in keywords:
        # 使用正则表达式匹配独立的单词
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.findall(pattern, text)
        keyword_counts[keyword] = len(matches)
        if keyword not in ["is", "are", "do", "does", "was", "were"]:
            special_total += len(matches)
        total += len(matches)
    keyword_counts["total"] = total
    keyword_counts["special_total"] = special_total
    return keyword_counts



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count unit frequencies in JSON data.")
    parser.add_argument("--input_file_path", type=str, required=True, help="Path to the input JSON file.")
    args = parser.parse_args()

    input_file_path = args.input_file_path
    output_file = input_file_path.replace(".json", "_units_count.json")
    data = load_json(input_file_path)
    for i in tqdm(range(len(data))):
        text = data[i]["Question"]
        data[i]["keywords_count"] = count_keywords(text, keywords)
    save_to_json(data, output_file)