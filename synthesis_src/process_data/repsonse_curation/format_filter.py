import json
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
from collections import Counter

# format errorを検出したときに保存するデータをGlobalに定義
format_error_data = []
error_ids = []


def process_and_visualize(data_list, filename, output_path):
    """
    统计列表中各内容的出现次数，保存为 JSON 文件，并绘制直方图。

    参数:
        data_list (list): 输入的列表数据。
        filename (str): 输出 JSON 文件的文件名（不带扩展名）。
    
    返回:
        None
    """
    # 1. 统计每个内容的出现次数
    counts = Counter(data_list)
    
    # 2. 按照次数降序排序
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    
    # 3. 保存为 JSON 文件
    json_filename = os.path.join(output_path, f"{filename}.json")
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(sorted_counts, json_file, ensure_ascii=False, indent=4)
    print(f"统计结果已保存到 {json_filename}")
    
    # 4. 绘制直方图
    labels, values = zip(*sorted_counts.items())
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('number')
    plt.ylabel('count')
    plt.title(filename)
    plt.xticks(rotation=45, ha='right')  # 旋转标签以便显示更清晰
    plt.tight_layout()  # 自动调整布局以避免重叠
    
    # 5. 保存直方图为图片文件
    plot_filename = os.path.join(output_path, f"{filename}.png")
    plt.savefig(plot_filename)
    print(f"直方图已保存到 {plot_filename}")
    
    # 6. 显示直方图（可选）
    plt.show()

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"Loaded {len(data)} items from {file_path}")
    return data

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Saved {len(data)} items to {file_path}")


def find_len(id, input_str):
    input_str = input_str.split(" ") 
    if len(input_str) > 3000:
        print(f"ID: {id}, len(input_str): {len(input_str)}")
        if id not in error_ids:
            error_ids.append(id)
            format_error_data.append({
                'id': id,
                'len_error': len(input_str)
            })
        else:
            for error in format_error_data:
                if error['id'] == id:
                    error['len_error'] = len(input_str)
                    break
    return len(input_str)

def find_boxed(id, input_str):
    boxed_str = "boxed{"
    input_str = input_str.split("</think>")[0]
    cnt_boxed = input_str.count(boxed_str)

    if cnt_boxed > 0:
        # if id not in format_error_data:
        #     format_error_data[id] = {}
        
        # format_error_data[id]['boxed_error'] = cnt_boxed
        if id not in error_ids:
            error_ids.append(id)
            format_error_data.append({
                'id': id,
                'boxed_error': cnt_boxed
            })
        else:
            for error in format_error_data:
                if error['id'] == id:
                    error['boxed_error'] = cnt_boxed
                    break


def find_special_words(id, input_str):
    words = "alternatively"
    input_str = input_str.lower()
    cnt_words = input_str.count(words)

    if cnt_words > 3:
        # if id not in format_error_data:
        #     format_error_data[id] = {}
        
        # format_error_data[id]['words_error'] = cnt_words
        if id not in error_ids:
            error_ids.append(id)
            format_error_data.append({
                'id': id,
                'words_error': cnt_words
            })  
        else:
            for error in format_error_data:
                if error['id'] == id:
                    error['words_error'] = cnt_words
                    break


def find_lang(id, input_str):
    # have_chinese = ['\u4e00' <= char <= '\u9fff' for char in input_str]
    chinese_chars = [char for char in input_str if '\u4e00' <= char <= '\u9fff']
    if chinese_chars:
        # if id not in format_error_data:
        #     format_error_data[id] = {}
        
        # format_error_data[id]['lang_error'] = True
        if id not in error_ids:
            error_ids.append(id)
            format_error_data.append({
                'id': id,
                'lang_error': {
                    'chinese_cnt': len(chinese_chars),
                    'chinese_chars': chinese_chars
                }
            })
        else:
            for error in format_error_data:
                if error['id'] == id:
                    error['lang_error'] = {
                        'chinese_cnt': len(chinese_chars),
                        'chinese_chars': chinese_chars
                    }
                    break

def find_ans(id, item):
    if item['metric']['acc'] == 1 and item['metric']['em'] == 0:
        # if id not in format_error_data:
        #     format_error_data[id] = {}
        
        # format_error_data[id]['ans_em_error'] = 1
        if id not in error_ids:
            error_ids.append(id)
            format_error_data.append({
                'id': id,
                'ans_em_error': 1
            })
        else:
            for error in format_error_data:
                if error['id'] == id:
                    error['ans_em_error'] = 1
                    break






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format error detection and statistics")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file')
    args = parser.parse_args()
    input_file = args.input_file

    base_dir = os.path.join(os.path.dirname(input_file), "filter_process")
    os.makedirs(base_dir, exist_ok=True)

    data = load_json(input_file)

    for idx, item in tqdm(enumerate(data)): # 遍历数据，标记每个数据中的错误
        item['id'] = idx
        output_text = ""
        for turn in item["output"]:
            for key, value in turn.items():
                if key in ["gen", "doc_gen"]:
                    output_text += value
            item["output_text"] = output_text
        find_boxed(item["id"], output_text)
        find_special_words(item["id"], output_text)
        find_lang(item["id"], output_text)
        find_ans(item["id"], item)
        find_len(item["id"], output_text)
    
    # 遍历format_error_data，分别统计每个错误的数量
    error_count = {}
    boxed_error_data = []
    boxed_error_cnt = []
    words_error_data = []
    words_error_cnt = []
    lang_error_data = []
    lang_error_cnt = []
    ans_em_error_data = []
    len_error_data = []
    len_error_cnt = []

    breakpoint()
    format_error_items = []
    for error in format_error_data:
        print("hello")
        item = data[error['id']]
        item['ans'] = item['answer']
        item["id_re"] = item['id']
        item['question_re'] = item['question']
        del item['output_text']

        assert item['id'] == error['id'], f"item['id']: {item['id']}, error['id']: {error['id']}"
        
        # 在item中添加错误信息
        if 'boxed_error' in error:
            item["boxed_error"] = error['boxed_error']
            # boxed_error_data.append(item)
            boxed_error_cnt.append(error['boxed_error'])

        if 'words_error' in error:
            # item = data[error['id']]
            # assert item['id'] == error['id'], f"item['id']: {item['id']}, error['id']: {error['id']}"
            item["words_error"] = error['words_error']
            # words_error_data.append(item)
            words_error_cnt.append(error['words_error'])

        if 'lang_error' in error:
            # item = data[error['id']]
            # assert item['id'] == error['id'], f"item['id']: {item['id']}, error['id']: {error['id']}"
            item["lang_error"] = error['lang_error']
            # lang_error_data.append(item)
            lang_error_cnt.append(error['lang_error']['chinese_cnt'])

        if 'ans_em_error' in error:
            # item = data[error['id']]
            # assert item['id'] == error['id'], f"item['id']: {item['id']}, error['id']: {error['id']}"
            item["ans_em_error"] = error['ans_em_error']
            # ans_em_error_data.append(item)

        if 'len_error' in error:
            item['len_error'] = error['len_error']
            # len_error_data.append(item)
            len_error_cnt.append(error['len_error'])

        # 分别收集各个error的数据
        if 'boxed_error' in error:
            boxed_error_data.append(item)
        if 'words_error' in error:
            words_error_data.append(item)
        if 'lang_error' in error:
            lang_error_data.append(item)
        if 'ans_em_error' in error:
            ans_em_error_data.append(item)
        if 'len_error' in error:
            len_error_data.append(item)
        format_error_items.append(item)



        for key, value in error.items(): # 累加该条数据中每个错误的数量
            if key not in ['id']:
                if key not in error_count:
                    error_count[key] = 0
                error_count[key] += 1
    print(f"error_count: {error_count}")

    
    # 调用函数统计直方图
    print("开始统计直方图")
    print(f"boxed_error_cnt: {boxed_error_cnt}")
    if len(boxed_error_cnt) > 0:
        process_and_visualize(boxed_error_cnt, "boxed_error_stats", base_dir)
    if len(words_error_cnt) > 0:
        process_and_visualize(words_error_cnt, "words_error_stats", base_dir)
    if len(lang_error_cnt) > 0: 
        process_and_visualize(lang_error_cnt, "lang_error_stats", base_dir)
    if len(len_error_cnt) > 0:
        process_and_visualize(len_error_cnt, "len_error_stats", base_dir)
    # process_and_visualize(ans_em_error_data, "boxed_error_stats")

    # 完整的数据
    output_file = os.path.join(base_dir, "find_format_error.json")
    save_json(format_error_data, output_file)

    # 保存每个错误文件
    error_file_dir = os.path.join(base_dir, "error_file")
    os.makedirs(error_file_dir, exist_ok=True)
    boxed_error_file = os.path.join(error_file_dir, f"boxed_error_{len(boxed_error_data)}.json")
    words_error_file = os.path.join(error_file_dir, f"words_error_{len(words_error_data)}.json")
    lang_error_file = os.path.join(error_file_dir, f"lang_error_{len(lang_error_data)}.json")
    ans_em_error_file = os.path.join(error_file_dir, f"ans_em_error_{len(ans_em_error_data)}.json")
    len_error_file = os.path.join(error_file_dir, f"len_error_{len(len_error_data)}.json")
    output_file_error_items = os.path.join(error_file_dir, f"all_error_items_{len(format_error_items)}.json")
    if len(format_error_items) > 0:
        save_json(format_error_items, output_file_error_items)
    if len(boxed_error_data) > 0:
        save_json(boxed_error_data, boxed_error_file)
    if len(words_error_data) > 0:
        save_json(words_error_data, words_error_file)
    if len(lang_error_data) > 0:
        save_json(lang_error_data, lang_error_file)
    if len(len_error_data) > 0:
        save_json(len_error_data, len_error_file)
    if len(ans_em_error_data) > 0:
        save_json(ans_em_error_data, ans_em_error_file)

    # 保存没有问题的数据
    no_error_data = []
    for item in data:
        if item['id'] not in error_ids:
            no_error_data.append(item)
    no_error_file = os.path.join(base_dir, f"filtered_data_{len(no_error_data)}.json")
    save_json(no_error_data, no_error_file)

    print(f"error_ids: {len(error_ids)}")