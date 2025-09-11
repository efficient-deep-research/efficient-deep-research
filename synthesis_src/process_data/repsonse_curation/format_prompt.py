import json

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"Loaded {len(data)} items from {file_path}")
    return data

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Saved {len(data)} items to {file_path}")


def get_task_instruction_openqa(question):
    return (
        'Please answer the following question. '
        'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
        f'Question:\n{question}\n\n'
    )

def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT): # 给出的样例是进行了两次搜索
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"Whenever you encounter a topic, fact, or piece of information you are uncertain about or need further details on, please perform a search to gather more accurate, up-to-date, or specific information. You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n"
        "- Do not generate <|begin_search_result|> and <|end_search_result|> tags yourself.\n\n"
    )

def format_prompt(data, MAX_SEARCH_LIMIT=10):
    for item in data: # 生成prompts
        question = item['question']
        instruction = get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT)
        user_prompt = get_task_instruction_openqa(question)

        prompt = instruction + user_prompt

        item["input"] = prompt
    return data


if __name__ == "__main__":
    file_path = ""
    output_path = ""
    data = load_json(file_path)

    data = format_prompt(data, MAX_SEARCH_LIMIT=10)
    save_json(data, output_path)
