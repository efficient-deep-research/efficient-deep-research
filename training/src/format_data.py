import argparse
import json


def arg_parse() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to load")
    parser.add_argument("--output_file", type=str, default="formatted_data.jsonl", help="Output file name")
    args = parser.parse_args()
    return args.dataset_name, args.output_file


def main():
    dataset_name, output_file = arg_parse()
    formatted_data = []
    with open(dataset_name, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            formatted_data.append({
                "messages": [
                    data["prompt"][0],
                    {"role": "assistant", "content": "<think>\n" + data["chosen"][0]["content"]},
                ],
                "rejected_response": "<think>\n" + data["rejected"][0]["content"]
            })

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
