import json
import os
import argparse
from collections import defaultdict, OrderedDict, Counter
from tqdm import tqdm
import random
import matplotlib.pyplot as plt



def analyze_and_visualize_sources(final_dataset, output_dir):
    """
    统计 final_dataset 中 source 的分布，并绘制饼图和保存为 JSON 文件。
    
    :param final_dataset: 输入的数据列表，每个元素是一个字典
    :param output_image_path: 饼图保存路径（默认为 "source_distribution.png"）
    :param output_json_path: JSON 文件保存路径（默认为 "source_counts.json"）
    """
    # 提取所有包含 source 的项
    sources = [item["source"] for item in final_dataset if "source" in item]
    
    # 统计 source 的分布
    source_counts = Counter(sources)
    
    # 将统计结果保存为 JSON 文件
    output_json_path = os.path.join(output_dir, "source_counts.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(source_counts, f, ensure_ascii=False, indent=4)
    print(f"Source 分布已保存到 {output_json_path}")
    
    # 绘制饼图
    labels = list(source_counts.keys())
    counts = list(source_counts.values())
    
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Source Distribution")
    plt.axis('equal')  # 确保饼图为正圆
    
    # 保存饼图为图片文件
    output_image_path = os.path.join(output_dir, 'source_distribution_selected_data.png')
    plt.savefig(output_image_path)
    plt.close()
    print(f"Source 分布饼图已保存到 {output_image_path}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Sample queries from labeled dataset with domain and keypoint constraints.")
    parser.add_argument("--input_file_path", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--total_samples", type=int, required=True, help="Target number of sampled queries")
    args = parser.parse_args()

    input_file_path = args.input_file_path
    total_samples = args.total_samples

    output_dir = os.path.join(os.path.dirname(args.input_file_path), "sampled_query")
    os.makedirs(output_dir, exist_ok=True)

    # Step 0 读取数据
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, item in enumerate(data): # 增加Question key

        if "question" in item:
            item["Question"] = item["question"]
        

    # step 1 预处理 Domain 分类
    # 统计原始 domain 分布
    print("step 1: preprocss domain data")
    domain_counter = defaultdict(int)
    for item in data:
        domain = item["domain_keypoints"]["domain"]
        domain_counter[domain] += 1

    # 确定有效 domain（>=500 条）
    # valid_domains = {d for d, cnt in domain_counter.items() if cnt >= 100}
    # other_domains = [d for d, cnt in domain_counter.items() if cnt < 100]
    valid_threshold = 500
    valid_domains = {d for d, cnt in domain_counter.items() if cnt >= valid_threshold}
    other_domains = [d for d, cnt in domain_counter.items() if cnt < valid_threshold]
    print(f"valid_domains: {len(valid_domains)}")
    print(f"other_domains: {len(other_domains)}")

    # 重构数据集（合并 small domains 到 other）
    processed_data = []
    for item in data:
        original_domain = item["domain_keypoints"]["domain"]
        if original_domain in valid_domains:
            new_domain = original_domain
        else:
            new_domain = "other"
        
        new_item = {
            **item,
            "domain_keypoints": {
                **item["domain_keypoints"],
                "domain": new_domain
            }
        }
        processed_data.append(new_item)



    # step 2 分配样本配额
    # 计算新 domain 分布
    print("step 2: calculate domain distribution")
    new_domain_counts = defaultdict(int)
    for item in processed_data:
        domain = item["domain_keypoints"]["domain"]
        new_domain_counts[domain] += 1


    # 确定最终 domain 列表
    final_domains = list(valid_domains)
    if new_domain_counts.get("other", 0) > 0:
        final_domains.append("other")

    print(f"new_domain_counts: {len(new_domain_counts)}")
    print(f"final_domains: {len(final_domains)}")

    # 分配样本配额
    num_domains = len(final_domains)
    base_quota = total_samples // num_domains
    remainder = total_samples % num_domains
    print(f"base_quota: {base_quota}")
    print(f"remainder: {remainder}")

    domain_quotas = {}

    for idx, domain in enumerate(final_domains):
        domain_quotas[domain] = base_quota + (1 if idx < remainder else 0)



    # step 3 统计 Keypoints 分布
    print("step 3 calculating keypoints distribution")
    domain_keypoint_dist = defaultdict(lambda: defaultdict(int))

    for item in tqdm(processed_data, desc="Calculating keypoints distribution"):
        domain = item["domain_keypoints"]["domain"]
        keypoints = item["domain_keypoints"]["key_points"]
        for kp in keypoints:
            domain_keypoint_dist[domain][kp] += 1

    # domain内部按照key points降序排列
    for domain, keypoint_counts in domain_keypoint_dist.items():
        sorted_keypoints = sorted(keypoint_counts.items(), key=lambda x: x[1], reverse=True)
        domain_keypoint_dist[domain] = OrderedDict(sorted_keypoints)


    # 保存分布文件
    with open(os.path.join(output_dir, "domain_keypoints_distribution_without_remove_dup.json"), "w", encoding="utf-8") as f:
        json.dump(domain_keypoint_dist, f, indent=4, ensure_ascii=False)


    # step 4 分层抽样与去重
    print("step 4: select data")
    final_dataset = []
    question_set = set()

    real_selected_cnt = 0
    for domain in tqdm(final_domains, desc="Processing domains"):
        # 获取当前 domain 的所有数据
        domain_data = [item for item in processed_data if item["domain_keypoints"]["domain"] == domain]
        print(f"-------------- process {domain}")
        print(f"data: {len(domain_data)}")
        # 计算 keypoint 配额
        keypoints = list(domain_keypoint_dist[domain].keys())
        # random.shuffle(keypoints) # 随机打乱顺序

        print(f"key points {len(keypoints)}")
        print(f"domain_quotas: {domain_quotas[domain]}")

        # 因为domain_quotas远远小于keypoint的数目，改为将所有的数据按照total降序排序，然后从上往下选，重复上面的操作，直到选完
        
        sorted_domain_data = sorted(domain_data, key=lambda x: x["keywords_count"]["total"], reverse=True)

        selected_cnt = 0
        cycle_cnt = 0
        selected_index = [] # 已选择的index
        while selected_cnt < domain_quotas[domain]: # 当前选择的数目小于domain_quotas[domain]
            cycle_cnt += 1
            if cycle_cnt > 100000:
                raise RuntimeError(f"valid_threshold is too small, causing the program to fall into an infinite loop. Please increase valid_threshold")
            # print(f"cycle_cnt: {cycle_cnt}")
            selected_keypoints = set() # 每次循环都清零
            for idx, item in enumerate(sorted_domain_data): # 遍历所有的数据
                if selected_cnt >= domain_quotas[domain]:
                    break
                if idx not in selected_index: # 当前数据未被选择
                    dup_keypoints = False
                    for key_point in item["domain_keypoints"]["key_points"]: # 确保数据的key point不包含在已经选择的key point里
                        if key_point in selected_keypoints:
                            dup_keypoints = True
                            break
                    if dup_keypoints: # 重复key points
                        continue
                    if item["Question"] not in question_set:
                        final_dataset.append(item)
                        question_set.add(item["Question"])
                        selected_keypoints.update(item["domain_keypoints"]["key_points"])
                        selected_index.append(idx)
                        selected_cnt += 1
        real_selected_cnt  += selected_cnt
        print(f"cycle_cnt: {cycle_cnt}")

    print(f"Real sampled queries count: {real_selected_cnt} / Target: {total_samples}")         


    print(f"final_dataset len: {len(final_dataset)}")

    analyze_and_visualize_sources(final_dataset, output_dir)


    # step 5 保存结果
    with open(os.path.join(output_dir, "final_selected_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=4, ensure_ascii=False)