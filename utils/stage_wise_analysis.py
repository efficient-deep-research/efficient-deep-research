import json
import re
# from utils import has_answer, EM_compute, F1_compute, AC_compute

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

num2alpha = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
    "30": "thirty",
    "40": "forty",
    "50": "fifty",
    "60": "sixty",
    "70": "seventy",
    "80": "eighty",
    "90": "ninety",
    "100": "hundred",
}
import argparse
import collections
import json
import copy
import os
import re
import logging
import string
from typing import List
import regex
import unicodedata
from tqdm import tqdm


logger = logging.getLogger()


class Tokens(object):
    """A class to represent a list of tokenized text."""

    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i:j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return "".join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if "pos" not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if "lemma" not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if "ner" not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
            True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [
            (s, e + 1)
            for s in range(len(words))
            for e in range(s, min(s + n, len(words)))
            if not _skip(words[s : e + 1])
        ]

        # Concatenate into strings
        if as_strings:
            ngrams = ["{}".format(" ".join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get("non_ent", "O")
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS), flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get("annotators", {})) > 0:
            logger.warning(
                "%s only tokenizes! Skipping annotators: %s" % (type(self).__name__, kwargs.get("annotators"))
            )
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((token, text[start_ws:end_ws], span))
        return Tokens(data, self.annotators)


tokenizer = SimpleTokenizer()


def normalize_span(text):
    text = unicodedata.normalize("NFD", text)
    text = tokenizer.tokenize(text).words(uncased=False)
    return " ".join(text), len(text)


def has_answer(answers, text, match_type="string"):
    # print(answers, text)
    # input()

    # 如果text为list
    if isinstance(text, list):
        text = " ".join(text)

    text = unicodedata.normalize("NFD", text)
    if match_type == "string":
        text = tokenizer.tokenize(text).words(uncased=True)
        for single_answer in answers:
            single_answer = unicodedata.normalize("NFD", single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    return 1
    return 0


import unicodedata


def fake_answer(answers, text, fake_ans, match_type="string"):
    answers = might_right_answers(answers) + expand_answers(answers)
    # Normalize the input text
    text = unicodedata.normalize("NFD", text)
    if match_type == "string":
        otext = tokenizer.tokenize(text).words(uncased=False)
        oo = " ".join(otext)
        text = tokenizer.tokenize(text).words(uncased=True)
        for single_answer in answers:
            single_answer = unicodedata.normalize("NFD", single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    ss = " ".join(otext[i : i + len(single_answer)])

                    oo = oo.replace(ss, fake_ans)
    return clean_text(oo)


def clean_text(text):
    # 定义一个正则表达式模式，用于去除标点符号后面的多余空格
    # 这里定义了一些常见的英文标点符号
    pattern_remove_trailing_spaces = r"([,.!?;:\(\)\[\]\{\}—–—])\s+"

    # 定义一个正则表达式模式，用于去除标点符号前面的多余空格
    pattern_remove_leading_spaces = r"\s+([,.!?;:\(\)\[\]\{\}—–—])"

    # 定义一个正则表达式模式，确保标点符号前后至少保留一个空格
    pattern_preserve_single_space = r"(\s*)([,.!?;:\(\)\[\]\{\}—–—])(\s*)"

    # 去除标点符号后面的多余空格
    cleaned_text = re.sub(pattern_remove_trailing_spaces, r"\1 ", text)

    # 去除标点符号前面的多余空格
    cleaned_text = re.sub(pattern_remove_leading_spaces, r" \1", cleaned_text)

    # 确保标点符号前后至少保留一个空格
    cleaned_text = re.sub(pattern_preserve_single_space, r" \2 ", cleaned_text)

    # 去除首尾空白
    cleaned_text = cleaned_text.strip()

    # 最终去除连续的空格
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    return cleaned_text


def expand_answers(answers: List[str]):
    copy_answers = answers.copy()
    res = set(answers)
    for single_answer in answers:
        if normalize_answer(single_answer) != "":
            res.add(normalize_answer(single_answer))
        original_answer = single_answer
        single_answer = unicodedata.normalize("NFD", single_answer)
        single_answer = tokenizer.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)
        for idx, word in enumerate(single_answer):
            if word in num2alpha.keys():
                cnt = 0
                for word_before in single_answer[:idx]:
                    if word in word_before:
                        cnt += 1
                pos = 0
                while pos < len(original_answer) - len(word) + 1:
                    if original_answer[pos:].startswith(word):
                        if cnt == 0:
                            res.add(original_answer[:pos] + num2alpha[word] + original_answer[pos + len(word) :])
                            break
                        pos += len(word)
                        cnt -= 1
                    else:
                        pos += 1
    for i in res:
        if i.lower() not in [c.lower() for c in copy_answers] and i != "":
            copy_answers.append(i)
    return copy_answers


def might_right_answers(answers):
    ans = set(answers)
    res = set()
    for single_answer in answers:
        original_answer = single_answer
        single_answer = unicodedata.normalize("NFD", single_answer)
        single_answer = tokenizer.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)
        for idx, word in enumerate(single_answer):
            for spand_len in range(1, len(single_answer)):
                cand_fake_ans = " ".join(single_answer[:idx] + single_answer[idx + spand_len :])
                if _remove_proj(normalize_answer(cand_fake_ans)).replace(" ", "") != "":
                    res.add(cand_fake_ans)
    return list(res - ans)


def _remove_proj(text):
    text = re.sub(r"\b(in|on|at|by|with|for|of|to)\b", " ", text)
    return text


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM_compute(answer_list, prediction):
    return max([int(normalize_answer(prediction) == normalize_answer(ground_truth)) for ground_truth in answer_list])


def AC_compute(answer_list, prediction):
    pred = normalize_answer(prediction)
    for answer in answer_list:
        if normalize_answer(answer) in pred:
            return 1
    return 0


def F1_compute(answers, pred):
    def get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    return max([compute_f1(x, pred) for x in answers])


def deal_judge(pred):
    if pred is None:
        return True
    if has_answer(
        [
            "unknown",
            "no specific answer",
            "not provide",
            "cannot answer",
            "no information provided",
            "no answer",
            "not contain",
            "no definitive answer",
        ],
        pred,
    ):
        return True
    return False


def deal_answer(pred, answers):
    if pred is None:
        return 0, 0
    if pred.lower().startswith("answer:"):
        pred = pred[7:]
    return EM_compute(answers, pred), F1_compute(answers, pred)


def deal_post(pred):
    giveup, istrue = True, None
    if pred is None:
        return giveup, istrue
    if has_answer(
        [
            "unclear",
            "not clear",
            "unknown",
            "partially correct",
            "partially incorrect",
            "not correct",
            "cannot determine",
            "cannot answer",
            "not incorrect",
            "incomplete",
        ],
        pred,
    ):
        giveup = True
    elif has_answer(["correct", "true"], pred):
        giveup, istrue = False, True
    elif has_answer(["incorrect", "false"], pred):
        giveup, istrue = False, False
    else:
        giveup = True
    return giveup, istrue


def str2paras(s):
    if s is None:
        return None
    paras = []
    for text in s.split("\n"):
        if text.strip() != "":
            paras.append(": " + text)
    return paras


def load_source(file):
    data = []
    f = open(file, "r", encoding="utf-8")
    for line in f.readlines():
        data.append(json.loads(line))
    f.close()
    return data


def remove_punctuation(s):
    punctuation_pattern = r"^[^\w\s]+|[^\w\s]+$"
    return re.sub(punctuation_pattern, "", s)


def save_file(args, results, add="res"):
    save_dir = os.path.dirname(args.data)
    model_base_file = os.path.basename(args.model) + "." + os.path.basename(args.data)[: -len(".json")]
    if args.splits:
        model_base_file += f".{args.worker}-{args.splits}"
    with open(os.path.join(save_dir, f"{model_base_file}.{add}.json"), "w") as f:
        json.dump(results, f, indent=4)


def calculate_statistics(data):
    if len(data) == 0:
        return {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0, "25th_percentile": 0, "75th_percentile": 0}

    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "median": np.median(data),
        "min": np.min(data),
        "max": np.max(data),
        "25th_percentile": np.percentile(data, 25),
        "75th_percentile": np.percentile(data, 75),
    }


def analyse_len(all_outputs_len, retrieval_outputs_len, no_retrieval_outputs_len, output_dir, output_stats_file):
    all_outputs_len_stats = calculate_statistics(all_outputs_len)
    retrieval_outputs_len_stats = calculate_statistics(retrieval_outputs_len)
    no_retrieval_outputs_len_stats = calculate_statistics(no_retrieval_outputs_len)

    # # 打印统计数据
    # print("All outputs length statistics:", all_outputs_len_stats)
    # print("Retrieval outputs length statistics:", retrieval_outputs_len_stats)
    # print("No retrieval outputs length statistics:", no_retrieval_outputs_len_stats)

    with open(output_stats_file, "a") as f:  # 使用 "a" 模式追加写入
        f.write("All outputs length statistics:\n")
        for key, value in all_outputs_len_stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Retrieval outputs length statistics:\n")
        for key, value in retrieval_outputs_len_stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("No retrieval outputs length statistics:\n")
        for key, value in no_retrieval_outputs_len_stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
    # # 创建保存结果的目录
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # 绘制直方图并保存图像
    plt.figure(figsize=(12, 8))

    # 绘制所有输出长度的直方图
    plt.subplot(2, 2, 1)
    sns.histplot(all_outputs_len, kde=True, bins=30, color="blue", label="All Outputs", stat="density")
    plt.title("Distribution of All Outputs Length")
    plt.xlabel("Length")
    plt.ylabel("Density")
    # plt.savefig(os.path.join(output_dir, 'all_outputs_length_distribution.png'))

    # 绘制检索输出长度的直方图
    plt.subplot(2, 2, 2)
    sns.histplot(retrieval_outputs_len, kde=True, bins=30, color="green", label="Retrieval Outputs", stat="density")
    plt.title("Distribution of Retrieval Outputs Length")
    plt.xlabel("Length")
    plt.ylabel("Density")
    # plt.savefig(os.path.join(output_dir, 'retrieval_outputs_length_distribution.png'))

    # 绘制没有检索输出长度的直方图
    plt.subplot(2, 2, 3)
    sns.histplot(
        no_retrieval_outputs_len, kde=True, bins=30, color="red", label="No Retrieval Outputs", stat="density"
    )
    plt.title("Distribution of No Retrieval Outputs Length")
    plt.xlabel("Length")
    plt.ylabel("Density")
    # plt.savefig(os.path.join(output_dir, 'no_retrieval_outputs_length_distribution.png'))

    # 总体输出长度分布
    plt.subplot(2, 2, 4)
    sns.histplot(all_outputs_len, kde=True, bins=30, color="blue", label="All Outputs", stat="density", alpha=0.5)
    sns.histplot(
        retrieval_outputs_len, kde=True, bins=30, color="green", label="Retrieval Outputs", stat="density", alpha=0.5
    )
    sns.histplot(
        no_retrieval_outputs_len,
        kde=True,
        bins=30,
        color="red",
        label="No Retrieval Outputs",
        stat="density",
        alpha=0.5,
    )
    plt.title("Overall Distribution of Outputs Length")
    plt.xlabel("Length")
    plt.ylabel("Density")
    plt.legend()
    # plt.savefig(os.path.join(output_dir, 'overall_output_length_distribution.png'))

    # 保存所有图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_output_length_distribution.png"))

    plt.show()


def has_run_retrieve(sample):
    return bool(sample["search_count"])


def cal_has_answer(sample):
    reason_has, search_has, analyses_has = 0, 0, 0
    for info in sample["all_info"]:
        for k, v in info.items():
            if "reason" in k:
                reason_has = max(reason_has, has_answer(sample["answer"], v))
            elif "search" in k:
                search_has = max(search_has, has_answer(sample["answer"], v))
            elif "analyses" in k:
                analyses_has = max(analyses_has, has_answer(sample["answer"], v))
    return {"reason": reason_has, "search": search_has, "analyse": analyses_has}


def extract_answer(sample):
    output = sample.get("output", "")
    match = re.search(r"\\boxed\{(.*?)\}", output)
    if match:
        return match.group(1)
    return output.rsplit("\n", 1)[-1]


def cal_metrics(sample):
    res = {}
    pred = extract_answer(sample)
    # for m, func in {
    #     'em': EM_compute,
    #     'ac': AC_compute,
    #     'f1': F1_compute,
    # }.items():
    #     res[m] = func(sample['answer'], pred)
    # res.update(cal_has_answer(sample))
    res["search_count"] = sample["search_count"]
    return res


def stage_wise_analysis(model_path, data_path):
    output_dir = os.path.dirname(data_path)
    output_stats_file = os.path.join(output_dir, "output_stats.txt")
    # model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(data_path, encoding="utf-8") as f:
        results = json.load(f)

    # 初始化累加器
    total_metrics = {}
    retrieval_true_metrics = {}
    retrieval_false_metrics = {}
    count_total = 0
    count_retrieval_true = 0
    count_retrieval_false = 0

    # 计算平均长度
    all_outputs_len = []
    retrieval_outputs_len = []
    no_retrieval_outputs_len = []

    # 遍历每个样本并计算指标
    for sample in results:
        sample.update(sample["item"])
        metrics = cal_metrics(sample)

        output_ids = tokenizer(sample["output"], add_special_tokens=False)["input_ids"]
        all_outputs_len.append(len(output_ids))

        # 累加总的指标
        for key, value in metrics.items():
            total_metrics[key] = total_metrics.get(key, 0) + value

        # 根据是否跑了检索进行分类累加
        if has_run_retrieve(sample):
            retrieval_outputs_len.append(len(output_ids))

            for key, value in metrics.items():
                retrieval_true_metrics[key] = retrieval_true_metrics.get(key, 0) + value
            count_retrieval_true += 1
        else:
            no_retrieval_outputs_len.append(len(output_ids))
            for key, value in metrics.items():
                retrieval_false_metrics[key] = retrieval_false_metrics.get(key, 0) + value
            count_retrieval_false += 1

        count_total += 1

    # 计算均值
    mean_metrics = {key: value / count_total for key, value in total_metrics.items()}
    mean_retrieval_true_metrics = {key: value / count_retrieval_true for key, value in retrieval_true_metrics.items()}
    mean_retrieval_false_metrics = {
        key: value / count_retrieval_false for key, value in retrieval_false_metrics.items()
    }

    mean_all_output_len = sum(all_outputs_len) / len(all_outputs_len) if len(all_outputs_len) != 0 else 0
    mean_retrieval_outputs_len = (
        sum(retrieval_outputs_len) / len(retrieval_outputs_len) if len(retrieval_outputs_len) != 0 else 0
    )
    mean_no_retrieval_outputs_len = (
        sum(no_retrieval_outputs_len) / len(no_retrieval_outputs_len) if len(no_retrieval_outputs_len) != 0 else 0
    )

    analyse_len(all_outputs_len, retrieval_outputs_len, no_retrieval_outputs_len, output_dir, output_stats_file)

    # print(count_retrieval_false/count_total)
    # print(count_retrieval_true/count_total)

    # # 打印结果
    # print(f"model_path: {model_path}")
    # print(f"data_path: {data_path}")
    # print("Overall Mean Metrics:")
    # for key, value in mean_metrics.items():
    #     print(f"{key}: {value}")
    # print(f"output_len: {mean_all_output_len}")

    # print("\nMean Metrics for Samples with Retrieval:")
    # for key, value in mean_retrieval_true_metrics.items():
    #     print(f"{key}: {value}")
    # print(f"output_len: {mean_retrieval_outputs_len}")

    # print("\nMean Metrics for Samples without Retrieval:")
    # for key, value in mean_retrieval_false_metrics.items():
    #     print(f"{key}: {value}")
    # print(f"output_len: {mean_no_retrieval_outputs_len}")
    with open(output_stats_file, "a") as f:
        f.write(f"\nProportion of samples without retrieval: {count_retrieval_false / count_total}\n")
        f.write(f"Proportion of samples with retrieval: {count_retrieval_true / count_total}\n")

        f.write(f"model_path: {model_path}\n")
        f.write(f"data_path: {data_path}\n")
        f.write("Overall Mean Metrics:\n")
        for key, value in mean_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write(f"output_len: {mean_all_output_len}\n\n")

        f.write("Mean Metrics for Samples with Retrieval:\n")
        for key, value in mean_retrieval_true_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write(f"output_len: {mean_retrieval_outputs_len}\n\n")

        f.write("Mean Metrics for Samples without Retrieval:\n")
        for key, value in mean_retrieval_false_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write(f"output_len: {mean_no_retrieval_outputs_len}\n")
