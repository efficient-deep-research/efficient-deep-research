import json
import argparse

from utils.format_key import format_key
from utils.format_prompt import format_prompt
from utils.format_ans import format_ans


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"Loaded {len(data)} items from {file_path}")
    return data


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Saved {len(data)} items to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format JSON data with prompt, key, and answer formatting.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    args = parser.parse_args()

    input_file = args.input_file
    data = load_json(input_file)
    print(f"Initial data: {len(data)} items")

    data = format_prompt(data, MAX_SEARCH_LIMIT=10)

    data = format_key(data)

    data = format_ans(data)

    output_path = input_file.replace(".json", f"_formatted_{len(data)}.json")
    save_json(data, output_path)
