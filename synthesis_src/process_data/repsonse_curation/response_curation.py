import copy
import os
import json
from collections import defaultdict
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import argparse
random.seed(42)

def plot_distribution(data, title="Distribution of First Reason Length", bins=40, name="url", xlabel="First Reason Length", ylabel="Frequency", output_path=""):
    # 设置绘图风格
    sns.set(style="whitegrid")
    
    # 创建直方图和核密度估计图
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=bins, color="skyblue", edgecolor="black")
    
    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # 显示图形
    plt.show()
    # plt.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_path, f"{name}.png"), dpi=300, bbox_inches="tight")

    quantiles = np.arange(0.8, 1.0, 0.03)  # 0.1 到 0.9 的分位点
    quantile_values = np.quantile(data, quantiles)  # 分位点对应的值
    total_count = len(data)  # 数据总数
    
    print(f"NAME: {name}")
    print("分位点统计:")
    for q, value in zip(quantiles, quantile_values):
        count_below = np.sum(np.array(data) <= value)  # 小于等于当前分位点的数量
        percentage = count_below / total_count * 100  # 占比
        print(f"分位点 {q:.2f}: "
              f"值 = {value:.2f}, "
              f"数量 = {count_below}, "
              f"占比 = {percentage:.2f}%")


def extract_incorrect_markers(input_string):
    standard_keywords = [
        'begin_search_query',
        'end_search_query',
        'begin_search_result',
        'end_search_result'
    ]
    
    # 创建匹配标准关键字的正则表达式模式
    # pattern = re.compile(r'\b(' + '|'.join(map(re.escape, standard_keywords)) + r')\b')
    pattern = re.compile('(' + '|'.join(map(re.escape, standard_keywords)) + ')')
    
    incorrect_markers = []
    
    for match in pattern.finditer(input_string):
        keyword = match.group()
        # print(f"keyword: {keyword}")
        start, end = match.start(), match.end()
        
        # 检查前两个字符是否是<|
        prefix_ok = (start >= 2 and input_string[start-2:start] == '<|')
        # 检查后两个字符是否是|>
        suffix_ok = (end + 2 <= len(input_string) and input_string[end:end+2] == '|>')
        
        if not (prefix_ok and suffix_ok):
            keyword = input_string[start-2:end+2]
            incorrect_markers.append(keyword)
    
    return incorrect_markers

def detect_variant_markers(input_string):
    """
    检测输入字符串中是否存在标记的变体形式。
    
    Args:
        input_string (str): 输入字符串。
    
    Returns:
        tuple: (bool, list) 是否存在变体标记，以及变体标记列表。
    """
    # 定义标准标记
    # standard_markers = [
    #     "<|begin_search_query|>",
    #     "<|end_search_query|>",
    #     "<|begin_search_result|>",
    #     "<|end_search_result|>"
    # ]
    
    # # 定义正则表达式匹配可能的变体标记
    # variant_pattern = re.compile(
    #     r"<[\\|]?begin_search_(query|result)[\\|]?>|"
    #     r"<[\\|]?end_search_(query|result)[\\|]?>"
    # )
    
    # # 提取所有可能的标记
    # all_markers = variant_pattern.findall(input_string)
    # all_markers = [f"<{match}>" for match in all_markers]
    
    # # 筛选出变体标记
    # variant_markers = []
    # for marker in all_markers:
    #     if marker not in standard_markers:
    #         import pdb
    #         pdb.set_trace()
    #         variant_markers.append(marker)

    variant_markers = extract_incorrect_markers(input_string)

    think_count = input_string.count("</think>")
    has_think_issue = think_count > 2
    # 返回结果
    return (len(variant_markers) > 0 ) or has_think_issue


def merge_questions(root_path): # 获取所有Question，及其所有的response
    """
    合并指定路径下所有符合条件的 JSON 文件中的 Question 数据。
    
    Args:
        root_path (str): 根目录路径。
    
    Returns:
        dict: 合并后的字典，键为 Question，值为列表。
    """
    # 用于存储最终结果的大字典
    merged_dict = defaultdict(list)
        
    if os.path.isdir(root_path):

        # if not os.path.isdir(outputs_path) or not outputs_folder.startswith("hotpotqa_2k"):
        #     print(f"skip {outputs_folder}")
        #     continue

        for rollout_folder in tqdm(sorted(os.listdir(root_path)), total=len(os.listdir(root_path)), desc="rollout_folder"):
            if not rollout_folder.startswith("rollout_"):
                continue
            rollout_path = os.path.join(root_path, rollout_folder)
            
            # 确保是 rollout_{num} 文件夹
            if not os.path.isdir(rollout_path) or not rollout_folder.startswith("rollout_"):
                continue
            pattern = re.compile(r"^turn_(\d+)\.json$")
            max_num = None
            q2m = {}
            # 遍历 rollout_{num} 文件夹中的 JSON 文件
            # for file_name in os.listdir(rollout_path):
            for file_name in sorted(os.listdir(rollout_path)):
                match = pattern.match(file_name)
                if match:
                    # 提取 num 值并转换为整数
                    num = int(match.group(1))
                    
                    # 更新最大值
                    if max_num is None or num > max_num:
                        max_num = num
                # print(file_name, len(file_name.split('.')))
                elif file_name.startswith("test") and file_name.endswith(".json") and len(file_name.split('.')) == 4:
                    file_path = os.path.join(rollout_path, file_name)
                    metrics = json.load(open(os.path.join(rollout_path, file_name), 'r', encoding='utf-8'))

                    for sample in metrics:
                        question = sample["Question"].split("\\boxed{YOUR_ANSWER}.\n\nQuestion:\n")[-1]
                        question = question.split("\n\n<|im_end|>\n<|im_start|>assistant")[0]
                        question = question.strip()
                        # print(question)
                        q2m[question] = sample["Metrics"]
            if max_num is not None:
                file_name = f"turn_{max_num}.json"
                file_path = os.path.join(rollout_path, file_name)
                # 读取 JSON 文件内容
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                        # 遍历 JSON 文件中的每个键值对
                        for sample in data:
                            # print("11111")
                            question = sample["item"]["Question"].strip()
                            # print(q2m.keys())
                            sample["Metrics"] = q2m[question]
                            merged_dict[question].append(sample)
                                
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    import pdb
                    pdb.set_trace()
                    
    
    return dict(merged_dict)


def detect_repeat(response, n=3, threshold=0.3):
    """
    检测大语言模型生成的文本中是否存在复读机现象。
    
    参数:
        response (str): 大语言模型生成的文本。
        n (int): n-gram 的大小，默认为 3。
        threshold (float): 判断是否重复的阈值，默认为 0.3。
                           如果某个 n-gram 的频率超过此阈值，则认为存在复读现象。
    
    返回:
        bool: 是否检测到复读现象。
        dict: 包含每个 n-gram 及其出现次数的字典。
    """
    # 将文本按空格分割为单词列表
    words = response.split()
    
    # 如果文本太短，无法生成有效的 n-gram，直接返回 False
    if len(words) < n:
        return False, {}
    
    # 统计 n-gram 出现的频率
    ngram_counts = defaultdict(int)
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])  # 使用元组作为键
        ngram_counts[ngram] += 1
    
    # 计算总 n-gram 数量
    total_ngrams = sum(ngram_counts.values())
    
    # 检查是否有 n-gram 频率超过阈值
    # repeated_ngrams = {ngram: count / total_ngrams for ngram, count in ngram_counts.items() if count / total_ngrams > threshold}
    repeated_ngrams = {"".join(ngram): count / total_ngrams for ngram, count in ngram_counts.items() if count / total_ngrams > threshold} # json不支持元组作为key
    
    # 如果存在重复的 n-gram，返回 True 和重复的 n-gram 信息
    if repeated_ngrams:
        # print(f"word: {words}")
        print(f"repeated_ngrams: {repeated_ngrams}")
        return True, repeated_ngrams
    
    # 否则返回 False
    return False, {}

def is_valid_history(history):
    """
    检测 history 列表是否符合合法情况。
    
    Args:
        history (list): 输入的 history 列表。
    
    Returns:
        bool: 如果符合合法情况返回 True，否则返回 False。
    """
    # 定义特殊标记
    SPECIAL_TOKENS = {
        "<|begin_search_query|>",
        "<|end_search_query|>",
        "<|begin_search_result|>",
        "<|end_search_result|>",
        "</think>"
    }
    
    for hid, his in enumerate(history):
        
        full_text = his.strip()
    
        # 情况 1: 只有一个 </think> 且它是最后一个元素
        if "</think>" in his:
            if hid == len(history) - 1 and history[-1].count("</think>") == 1:
                for token in SPECIAL_TOKENS:
                    if token != "</think>" and token in full_text:
                        return False
            else:
                return False
        
        # 情况 2: 以 <|begin_search_result|> 开头，以 <|end_search_result|> 结尾，中间无其他特殊标记
        elif (
            full_text.startswith("<|begin_search_result|>") and
            full_text.endswith("<|end_search_result|>")
        ):
            middle_content = full_text[len("<|begin_search_result|>"):-len("<|end_search_result|>")].strip()
            if any(token in middle_content for token in SPECIAL_TOKENS):
                return False
        
        # 情况 3: 以 <|end_search_query|> 结尾，中间仅出现一次 <|begin_search_query|>，无其他特殊标记
        elif full_text.endswith("<|end_search_query|>"):
            middle_content = full_text[:-len("<|end_search_query|>")].strip()
            if middle_content.count("<|begin_search_query|>") == 1:
                # 移除 <|begin_search_query|> 后检查是否还有其他特殊标记
                middle_without_begin = middle_content.replace("<|begin_search_query|>", "").strip()
                if any(token in middle_without_begin for token in SPECIAL_TOKENS):
                    return False
            else:
                return False
                    
        
        # 如果不符合任何一种情况，则视为异常
        else:
            return False
    return True

def parse_web_pages(web_page_str):
    """
    将包含多个 Web Page 的字符串解析为 List[Dict]。
    
    参数:
        web_page_str (str): 包含多个 Web Page 的字符串。
        
    返回:
        List[Dict]: 每个 Web Page 对应的字典组成的列表。
    """
    # 初始化结果列表
    result = []
    
    # 分割字符串，按 "**Web Page X:**" 分段
    segments = web_page_str.split("**Web Page ")[1:]  # 跳过第一个空段
    
    for segment in segments:
        try:
            # 提取 JSON 部分
            json_part = segment.split("\n", 1)[1].strip()  # 去掉序号和换行符
            
            # 使用 json.loads 将 JSON 字符串解析为字典
            parsed_dict = json.loads(json_part)
            parsed_dict = {
                'title': parsed_dict['title'],
                "context": parsed_dict['context'],
            }
            # 添加到结果列表
            result.append(parsed_dict)
        except (IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing segment: {segment[:50]}... Error: {e}")
    
    return result

def calculate_acc_ratios(data, output_path):
    first_reason_length = []
    url_per_search = []
    acc_list = []
    reason_length = []
    max_reason_length = []
    max_alt = []
    max_hmm = []
    max_wait = []
    """
    计算每个 question 的 acc 值比例。
    
    Args:
        data (Dict[str, List[Dict]]): 输入数据。
    
    Returns:
        Dict[str, Dict[float, float]]: 每个 question 的 acc 值及其比例。
    """
    result = []
    
    for question, entries in tqdm(data.items(), desc="calculate acc ratios"):
        # 统计每个 acc 值的出现次数
        acc_counts = 0
        total_count = 0
        
        for entry in entries:

            
            entry["error_special_token"], entry["is_valid_history"] = detect_variant_markers(entry["output"]), is_valid_history(entry["history"])
            # entry["search_count"] = count_search_queries(entry["history"])

            reason_1 = ""
            reason_values = []
            search_results = []
            for info in entry["all_info"]:
                if "turn_1_reason" in info.keys():
                    reason_1 = info["turn_1_reason"]
                for key, value in info.items():
                    if 'reason' in key:
                        reason_values.append(value)
                    elif key.endswith("_search"):
                        for web in parse_web_pages(value):
                            if not "url" in web.keys():
                                if "title" in web.keys():
                                    search_results.append(web["title"])
                                else:
                                    search_results.append(json.dumps(web))
                            else:
                                search_results.append(web["url"])
            entry["first_reason_length"] = len(reason_1.split(' '))
            first_reason_length.append(entry["first_reason_length"])
            entry["has_repeat"], entry["repeated_n_grams"] = detect_repeat(entry["output"])
            # reason_values = reason_values[:-1]
            if not reason_values:
                entry["reason_length"] = 0
                entry["max_reason_length"] = 0
                entry["max_alt"] = 0
                entry["max_hmm"] = 0
                entry["max_wait"] = 0
            else:
                word_counts = [len(value.split()) for value in reason_values]
                average_word_count = sum(word_counts) / len(word_counts)
                entry["reason_length"] = average_word_count
                entry["max_reason_length"] = max(word_counts)
                entry["max_alt"] = max([value.count("Alternatively") for value in reason_values])
                entry["max_hmm"] = max([value.count("hmm") for value in reason_values])
                entry["max_wait"] = max([value.count("wait") for value in reason_values])
            
            urls = set(search_results)
            entry["url_per_search"] = len(urls) / entry["search_count"] if entry["search_count"] else 0
            url_per_search.append(entry["url_per_search"])
            acc_counts += entry["Metrics"]["acc"]
            total_count += 1
            reason_length.append(entry["reason_length"])
            max_reason_length.append(entry["max_reason_length"])
            max_alt.append(entry["max_alt"])
            max_hmm.append(entry["max_hmm"])
            max_wait.append(entry["max_wait"])
            
            # 根据多个条件判断当前条目是否为有效的解（is_valid_solution），并将其存入条目字段。
            entry["is_valid_solution"] = \
                (not entry["error_special_token"]) and \
                entry["Metrics"]["acc"] and \
                entry["is_valid_history"] and \
                (entry["first_reason_length"] < 300) and \
                (entry["max_alt"] <= 3) and \
                (entry["max_hmm"] <= 3) and \
                (entry["max_wait"] <= 3) and \
                (entry["max_reason_length"] < 300) and \
                (not entry["has_repeat"]) and \
                (entry["url_per_search"] > 9.5)
                
        def sort_query(data):
            """
            对包含字典的列表进行排序。
            第一关键字: ["ratios"]，升序。
            第二关键字: ["min_search"]，降序。
            """
            # 使用 sorted() 和 lambda 表达式进行排序
            sorted_data = sorted(
                data,
                # key=lambda x: (-x["solutions"][0]["max_reason_length"])
                key=lambda x: (x["ratios"], -x["min_search"])  # 第一关键字升序，第二关键字降序（取反）
            )
            return sorted_data
                
        def sort_solution(data):
            """
            对包含字典的列表进行排序。
            第一关键字: ["is_valid_solution"]，True 在前。
            第二关键字: ["search_count"]，小的在前。
            """
            # 使用 sorted() 和 lambda 表达式进行排序
            sorted_data = sorted(
                data,
                key=lambda x: (not x["is_valid_solution"], x["search_count"])  # 第一关键字取反，第二关键字直接比较
            )
            return sorted_data
        data[question] = {
            "solutions": sort_solution(entries)
        }
        # 计算比例
        acc_ratios = acc_counts / total_count
        acc_list.append(acc_ratios)
        result.append({
            "question": question,
            "solutions": sort_solution(entries),
            "ratios": acc_ratios,
            "count": total_count,
            "min_search": min([entry["search_count"] for entry in entries if entry["Metrics"]["acc"]]) if acc_ratios > 0.00001 else 100, 
        })
        # 将结果存入最终字典
        data[question]["ratios"] = acc_ratios if acc_ratios > 1e-5 else 9999
        if data[question]["ratios"] > 1 - 1e-5 and random.random() < 0.8:  # 80%的概率丢弃全对的query
            result[-1]["solutions"][0]["is_valid_solution"] = False
            # print("set is_valid_solution to False")
        data[question]["count"] = total_count
        data[question]["min_search"] = min([entry["search_count"] for entry in entries if entry["Metrics"]["acc"]]) if acc_ratios > 0.00001 else 100
    plot_distribution(first_reason_length, title= "first_reason_length", name="reason", xlabel="first_reason_length", ylabel="count", output_path=output_path)
    plot_distribution(url_per_search, title="url_per_search", name="url", xlabel="url_per_search", ylabel="count", output_path=output_path)
    plot_distribution(acc_list, title="acc", name="acc", xlabel="acc", ylabel="count", output_path=output_path)
    plot_distribution(reason_length, title="reason_length", name="average_reason", xlabel="reason_length", ylabel="count", output_path=output_path)
    plot_distribution(max_reason_length, title="max_reason_length", name="max_reason", xlabel="max_reason_length", ylabel="count", output_path=output_path)
    plot_distribution(max_alt, title="max_alt", name="max_alt", xlabel="max_alt", ylabel="count", output_path=output_path)
    plot_distribution(max_hmm, title="max_hmm", name="max_hmm", xlabel="max_hmm", ylabel="count", output_path=output_path)
    plot_distribution(max_wait, title="max_wait", name="max_wait", xlabel="max_wait", ylabel="count", output_path=output_path)
    result = sort_query(result)
    return result

# 示例调用
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and process questions from given root path.")
    parser.add_argument("--root_path", type=str, required=True, help="Path to the directory containing question data.")
    parser.add_argument("--k", type=int, default=100000, help="Number of top valid data to retain.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the directory to save the selected data.")
    args = parser.parse_args()

    root_path = args.root_path
    k = args.k
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    merge_qa_pairs_path = os.path.join(output_path, "merged_qa_pairs.json")
    selected_data_path = os.path.join(output_path, "selected_data.json")

    print("Merge question-answer pairs...")
    result = merge_questions(root_path)
    print(f"len of all question-answer pairs: {len(result)}")

    print("Select valid solutions...")
    result = calculate_acc_ratios(result, output_path)
    valid = []
    for sample in result:
        if sample["solutions"][0]["is_valid_solution"]:
            sample = copy.deepcopy(sample)
            sample.update(sample["solutions"][0])
            sample.pop("solutions")
            valid.append(sample)

    print(f"len of pairs before selection: {len(valid)}")
    # print(f"select: {k}")
    valid = valid[: k] 

    with open(merge_qa_pairs_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Saving merged question-answer pairs to {merge_qa_pairs_path}")

    with open(selected_data_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=4)
    print(f"Saving selected data to {selected_data_path}")