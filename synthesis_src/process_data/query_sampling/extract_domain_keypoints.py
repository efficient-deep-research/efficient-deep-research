import json
import os
import torch
from tqdm import tqdm
import argparse
import re

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"save to {filename}, data len: {len(data)}")

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"load from {file_path}, data len: {len(data)}")
    return data

def extract_brace_content(s):
    matches = re.findall(r'\{[^}]*\}', s)
    return matches[-1] if matches else ''

def extract_domain_keypoints(input_file_path, output_file_path):
    data = load_json(input_file_path)

    error_count = 0
    error_two_count = 0

    new_data = []
    for item in tqdm(data, desc="Extracting domain and key points"):
        domain_keypoints = item["tag_output"].split("\n</think>\n\n")[-1]
        
        try:  # 直接解析
            domain_keypoints_formatted = json.loads(domain_keypoints)
            item["domain_keypoints"] = domain_keypoints_formatted
        except:
            error_count += 1
            try:  # 二次尝试解析
                domain_keypoints = domain_keypoints.split("\n\n")[0]
                item["domain_keypoints"] = json.loads(domain_keypoints)
            except:
                error_two_count += 1
                print(f"error_two_count: {[domain_keypoints]}")
                item["domain_keypoints"] = domain_keypoints

        if isinstance(item["domain_keypoints"], dict):
            new_data.append(item)

    print(f"error_count: {error_count}")
    print(f"error_two_count: {error_two_count}")
    print(f"len data: {len(data)}")
    print(f"len new_data: {len(new_data)}")

    save_to_json(new_data, output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract domain and key points from LLM output.")
    parser.add_argument("--input_file_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to save the processed output JSON.")
    args = parser.parse_args()

    extract_domain_keypoints(args.input_file_path, args.output_file_path)
