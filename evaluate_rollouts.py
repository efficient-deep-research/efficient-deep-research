import copy
import os
import asyncio
import json
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import argparse
from spacy.lang.en import English

from utils.response_format_validators import detect_variant_markers, is_valid_history

from utils.response_quarity_evaluators import evaluate_response_quality_async, extract_final_answer


def plot_distribution(
    data,
    title="Distribution of First Reason Length",
    bins=40,
    name="url",
    xlabel="First Reason Length",
    ylabel="Frequency",
    output_path="",
):
    # Set plot style
    sns.set(style="whitegrid")

    # Create histogram and kernel density estimation plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=bins, color="skyblue", edgecolor="black")

    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # Display the plot
    plt.show()
    # plt.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_path, f"{name}.png"), dpi=300, bbox_inches="tight")

    quantiles = np.arange(0.8, 1.0, 0.03)  # Quantiles from 0.8 to 0.9
    quantile_values = np.quantile(data, quantiles)  # Values corresponding to quantiles
    total_count = len(data)  # Total data count

    print(f"NAME: {name}")
    print("Quantile statistics:")
    for q, value in zip(quantiles, quantile_values):
        count_below = np.sum(np.array(data) <= value)  # Count of values less than or equal to current quantile
        percentage = count_below / total_count * 100  # Percentage
        print(f"Quantile {q:.2f}: Value = {value:.2f}, Count = {count_below}, Percentage = {percentage:.2f}%")


def merge_questions(root_path):
    """
    Merge Question data from all qualifying JSON files under the specified path.

    Args:
        root_path (str): Root directory path.

    Returns:
        dict: Merged dictionary with Question as key and list as value.
    """
    merged_dict = defaultdict(list)

    if os.path.isdir(root_path):
        for rollout_folder in tqdm(
            sorted(os.listdir(root_path)), total=len(os.listdir(root_path)), desc="rollout_folder"
        ):
            if not rollout_folder.startswith("rollout_"):
                continue
            rollout_path = os.path.join(root_path, rollout_folder)
            if not os.path.isdir(rollout_path) or not rollout_folder.startswith("rollout_"):
                continue
            pattern = re.compile(r"^turn_(\d+)\.json$")
            max_num = None

            for file_name in sorted(os.listdir(rollout_path)):
                match = pattern.match(file_name)
                if match:
                    num = int(match.group(1))
                    if max_num is None or num > max_num:
                        max_num = num

            if max_num is not None:
                file_name = f"turn_{max_num}.json"
                file_path = os.path.join(rollout_path, file_name)
                print(f"Detected file: {file_path} for merging.")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for sample in data:
                            question = sample["item"]["Question"].strip()
                            merged_dict[question].append(sample)

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return dict(merged_dict)


def main(args: argparse.Namespace):
    root_path = args.root_path
    model = args.model
    max_concurrent = args.max_concurrent
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    # collect all rollouts
    print("Merge question-answer pairs...")
    all_rollouts = merge_questions(root_path)
    print(f"len of all question-answer pairs: {len(all_rollouts)}")

    # evaluate rollouts with DeepResearchGym Critaria
    print("Evaluate response quality...")
    eval_criteria = ["Clarity", "Insightfulness"]
    all_rollouts = asyncio.run(
        evaluate_response_quality_async(
            input_data=all_rollouts,
            model=model,
            eval_criteria=eval_criteria,
            max_concurrent_requests=max_concurrent,
            output_path=output_path,
        )
    )

    # evaluate other metrics and calculate preference scores
    final_answer_word_count = []
    final_answer_sentence_count = []
    kpr = []
    kpc = []
    clarity = []
    insightfulness = []
    search_count = []

    nlp = English()
    nlp.add_pipe("sentencizer")
    print("Evaluate other metrics, calculate preference scores...")
    for question, entries in all_rollouts.items():
        for entry in entries:
            # add other metrics to entry
            final_answer = extract_final_answer(entry["output"])
            entry["final_answer_word_count"] = len((final_answer.split(" "))) if final_answer else 0
            entry["final_answer_sentence_count"] = len(list(nlp(final_answer).sents)) if final_answer else 0
            entry["error_special_token"] = detect_variant_markers(entry["output"])
            entry["is_valid_history"] = is_valid_history(entry["history"])

            # set preference score
            is_valid_rollout = entry["finished"] and (not entry["error_special_token"]) and entry["is_valid_history"]
            if is_valid_rollout:
                entry["preference_score"] = (
                    entry["kpr"]["supported_rate"] / 100
                    + entry["criteria_evaluation"]["Clarity"]["rating"] / 10
                    + entry["criteria_evaluation"]["Insightfulness"]["rating"] / 10
                )
            else:
                entry["preference_score"] = -1

            # collect distribution data
            final_answer_word_count.append(entry["final_answer_word_count"])
            final_answer_sentence_count.append(entry["final_answer_sentence_count"])
            kpr.append(entry["kpr"]["supported_rate"])
            kpc.append(entry["kpr"]["contradicted_rate"])
            clarity.append(entry["criteria_evaluation"]["Clarity"]["rating"])
            insightfulness.append(entry["criteria_evaluation"]["Insightfulness"]["rating"])
            search_count.append(entry["search_count"])

    # save preference evaluated rollouts
    evaluated_rollouts_path = os.path.join(output_path, "evaluated_rollouts.json")
    with open(evaluated_rollouts_path, "w", encoding="utf-8") as f:
        json.dump(all_rollouts, f, ensure_ascii=False, indent=4)
    print(f"Saving evaluated rollouts to {evaluated_rollouts_path}")

    # plot distribution
    metrics_distribution_path = os.path.join(output_path, "metrics_distribution")
    os.makedirs(metrics_distribution_path, exist_ok=True)
    plot_distribution(
        final_answer_word_count,
        title="final_answer_word_count",
        name="final_answer_word_count",
        xlabel="final_answer_word_count",
        ylabel="count",
        output_path=metrics_distribution_path,
    )
    plot_distribution(
        final_answer_sentence_count,
        title="final_answer_sentence_count",
        name="final_answer_sentence_count",
        xlabel="final_answer_sentence_count",
        ylabel="count",
        output_path=metrics_distribution_path,
    )
    plot_distribution(
        kpr, title="kpr", name="kpr", xlabel="kpr", ylabel="count", output_path=metrics_distribution_path
    )
    plot_distribution(
        kpc, title="kpc", name="kpc", xlabel="kpc", ylabel="count", output_path=metrics_distribution_path
    )
    plot_distribution(
        clarity,
        title="clarity",
        name="clarity",
        xlabel="clarity",
        ylabel="count",
        output_path=metrics_distribution_path,
    )
    plot_distribution(
        insightfulness,
        title="insightfulness",
        name="insightfulness",
        xlabel="insightfulness",
        ylabel="count",
        output_path=metrics_distribution_path,
    )
    plot_distribution(
        search_count,
        title="search_count",
        name="search_count",
        xlabel="search_count",
        ylabel="count",
        output_path=metrics_distribution_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and process questions from given root path.")
    parser.add_argument("--root_path", type=str, required=True, help="Path to the directory containing question data.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        help="OpenAI model to use for evaluation (e.g., 'o3-mini', 'gpt-4o-mini').",
    )
    parser.add_argument("--max_concurrent", type=int, default=100, help="Maximum concurrent API requests")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the directory to save the selected data."
    )

    args = parser.parse_args()

    main(args)
