import argparse
import os
import json
from pathlib import Path

def process_json_files(input_json_path:str) -> dict:
    with open(input_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    question = data.get("question", [])
    key_points = data.get("key_points", [])
    
    id_number = input_json_path.name.split("_")[0]
    id = f"deepresearchgym_{id_number}"
    source = "researchy_queries"
    
    return {
        "Question": question,
        "source": source,        
        "id": id,
        "key_points": key_points
    }

def process_deepresearchgym(input_path, output_path):
    name_pattern="*aggregated.json"
    
    data=[]
    
    input_path = Path(input_path)
    aggregated_json_files = list(input_path.rglob(name_pattern))
    for json_file in aggregated_json_files:
        data.append(process_json_files(json_file))

    output_path = Path(f"{output_path}/deepresearchgym.json")
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(data, out_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON data")
    parser.add_argument("--input_path", help="Path to the input directory")
    parser.add_argument("--output_path", help="Path to the output directory")
    args = parser.parse_args()

    process_deepresearchgym(args.input_path, args.output_path)