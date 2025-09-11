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

def process_string(item, input_str): # boxed前字符串的删除

    # 找到 </think> 的位置
    think_end_index = input_str.find("</think>")
    
    if think_end_index == -1:
        print(f"error not found </think>: {item['id']}, {input_str}")
        # 如果没有找到 </think>，直接返回原字符串
        return input_str
    
    # 从 </think> 后开始查找 boxed{
    boxed_start_index = input_str.find("boxed{", think_end_index)
    
    if boxed_start_index == -1:
        print(f"error not found boxed{{: {item['id'], input_str}")
        return input_str
        
    return input_str[:think_end_index+10] + input_str[boxed_start_index-1:]


def process_string_post(item, input_str): # boxed后字符串的删除

    # assert "gen" in item['output'][-1], "gen not found in output"
    # input_str = item['output'][-1]["gen"]
    input_str = input_str.strip()

    # 找到 </think> 的位置
    think_end_index = input_str.find("</think>")
    
    if think_end_index == -1:
        print(f"error not found </think>: {item['id']}, {input_str}")
        # 如果没有找到 </think>，直接返回原字符串
        return input_str
    
    # 从 </think> 后开始查找 boxed{
    boxed_start_index = input_str.find("boxed{", think_end_index)
    
    if boxed_start_index == -1:
        print(f"error not found boxed{{: {item['id'], input_str}")
        # 如果在 </think> 后没有找到 boxed{，直接返回原字符串
        return input_str
    
    # 找到 boxed{ 的结束位置 }
    boxed_end_index = input_str.find("}", boxed_start_index)
    
    if boxed_end_index == -1:
        print(f"error not found }}: {item['id']}, {input_str}")
        # 如果找不到 }，说明格式有问题，直接返回原字符串
        return input_str

    result = input_str[:boxed_end_index + 1]
    
    return result



def format_ans(data):
    for id, item in enumerate(data):
        item["id"] = id
        for idx, turn in enumerate(item["output"]):
            for key in turn:
                if key in ["gen", "doc_gen"]:
                    if idx == len(item["output"]) -1:
                        turn[key] = process_string_post(item, process_string(item, turn[key]))
                        if not turn[key].endswith("}"):
                            print(item["id"])
    return data

if __name__ == "__main__":

    input_file = ""
    data = load_json(input_file)
    data = format_ans(data)
    output_file = input_file.replace(".json", f"_final_{len(data)}.json")
    save_json(data, output_file)