import os
import argparse
import json
from vllm import LLM, SamplingParams
from datasets import Dataset
from transformers import AutoTokenizer
import random
from tqdm import tqdm
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

user_prompt = f"""You are an advanced semantic analyzer. For the given question, perform the following tasks step-by-step:

1. **Domain Identification**:
- Determine the broad subject category (domain) this question belongs to.
- Examples: film, history, biology, geography, politics, technology, etc (or any other suitable domain)

2. **Key Point Extraction**:
- Identify 2-4 core semantic components that are crucial for answering
- Include:
• Key entities (e.g., films, people, locations)
• Critical attributes (e.g., age, duration, population)
• Core relationships (e.g., comparison, causality)
• Measurement dimensions (e.g., time, quantity)
- Exclude filler words and non-essential descriptors

**Output Requirements**:
- Use JSON format: {{"domain": "...", "key_points": [...]}}
- Keep key_points concise (1-2 words each)
- Use lowercase for all outputs
- Separate multiple key_points with commas

**Examples**:
Question: "Which film whose director is younger, Charge It To Me or Danger: Diabolik?"
Output: {{"domain": "film", "key_points": ["director", "age"]}}

**Now process this question:**
{{Question}}"""

def main(input_file_path, output_path):
    
    data = []
    with open(input_file_path, "r", encoding="utf-8") as file:
        # for line in file:
        #     data.append(json.loads(line))
        data = json.load(file)

    
    prompts = []
    new_data = []

    for item in tqdm(data, desc="gen prompts"):
        prompt = user_prompt.replace('{Question}', item["Question"])
        messages = [{"role": "user", "content": prompt}]
        
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompts.append(text)
        item["tag_prompt"] = text
    print(prompts[0])
    outputs = llm.generate(prompts, sampling_params)

    generated_texts = []
    for i, output in tqdm(enumerate(outputs), desc="process outputs"):
        generated_text = output.outputs[0].text.strip()
        data[i]["tag_output"] = generated_text

        generated_texts.append(data[i])

    
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(generated_texts, json_file, ensure_ascii=False, indent=4)
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     for item in generated_texts:
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"save to {output_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input file and CUDA devices.")
    parser.add_argument("--input_file_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--cuda_visible_devices", type=str, required=True, help="CUDA devices to use (e.g., '0,1').")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM model directory.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    input_file_path = args.input_file_path
    model_path = args.model_path
    
    file_name = os.path.basename(input_file_path).split(".")[0]
    base_path = os.path.dirname(input_file_path)
    output_dir = os.path.join(base_path, "tagged_domain_keypoints")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_name}_tagged.json")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    sampling_params = SamplingParams(n=1, temperature=0.6, max_tokens=10000, top_p=0.95, top_k=40)
    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.95,trust_remote_code=True)
    
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"model path: {model_path}")
    print(f"input_file_path: {input_file_path}")
    print(f"output_path: {output_path}")

    main(input_file_path, output_path)

