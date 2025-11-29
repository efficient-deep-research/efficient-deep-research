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

from utils import extract_final_information, delete_invalid_spaces_from_citation, validate_citation_format
from utils.response_format_validators import detect_variant_markers, validate_history
from utils.response_quarity_evaluators import evaluate_response_quality_async


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
    plt.savefig(os.path.join(output_path, f"{name}.pdf"), dpi=300, bbox_inches="tight")
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


def merge_questions(root_path, filtering_criteria: list[str] = []):
    """
    Merge Question data from all qualifying JSON files under the specified path.

    Args:
        root_path (str): Root directory path.
        filtering_criteria (list[str], optional): List of filtering criteria to apply. Defaults to None. Vailue options: 'finished', 'valid_history', 'no_variant_markers', 'valid_citation_format'.

    Returns:
        dict: Merged dictionary with Question as key and list as value.
    """
    merged_dict = defaultdict(list)
    total_rollouts = 0
    valid_rollouts = 0
    finished_errors = 0
    history_errors = 0
    variant_marker_errors = 0
    citation_format_errors = 0

    print(f"Filtering criteria: {filtering_criteria}")
    
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
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for sample in data:
                        total_rollouts += 1
                        question = sample["item"]["Question"].strip()
                        output = sample["output"].strip()
                        final_answer = extract_final_information(output) # extract final answer from output
                        final_answer = delete_invalid_spaces_from_citation(final_answer) # delete invalid spaces from citation

                        # define conditions for a valid rollout
                        is_finished = sample.get("finished", False)
                        is_valid_history = validate_history(sample["history"])
                        no_valiant_markers = not detect_variant_markers(output)
                        is_valid_citation = validate_citation_format(final_answer, sample["executed_search_urls"].keys())
                        
                        conditions = {}
                        if "finished" in filtering_criteria:
                            conditions["finished"] = is_finished
                        elif "valid_history" in filtering_criteria:
                            conditions["valid_history"] = is_valid_history
                        elif "no_variant_markers" in filtering_criteria:
                            conditions["no_variant_markers"] = no_valiant_markers
                        elif "valid_citation_format" in filtering_criteria:
                            conditions["valid_citation_format"] = is_valid_citation["is_valid"]
                    
                        is_valid_rollout = all(conditions.values()) if filtering_criteria else True
                        # validate samples and only keep valid ones
                        if is_valid_rollout:
                            valid_rollouts += 1
                            # append the final answer to the sample and add to merged dict
                            sample_copy = copy.deepcopy(sample)
                            sample_copy["final_answer"] = final_answer
                            merged_dict[question].append(sample_copy)

                        # log invalid rollout info (file_path, question, judgement)
                        if not is_finished:
                            finished_errors += 1
                        if not is_valid_history:
                            history_errors += 1
                        if not no_valiant_markers:
                            variant_marker_errors += 1
                        if not is_valid_citation["is_valid"]:
                            citation_format_errors += 1
                        print(f'Invalid rollout [{file_path}]: Q="{question[:50]}..." | Finished={is_finished} | ValidHistory={is_valid_history} | NoVariants={no_valiant_markers} | ValidCite={is_valid_citation["is_valid"]} | CiteErrors={is_valid_citation["errors"]}')

    invalid_rollouts = total_rollouts - valid_rollouts
    print(f"Total rollouts: {total_rollouts}, Remaining rollouts: {valid_rollouts} ({(valid_rollouts / total_rollouts * 100):.2f}%), Filtered rollouts: {invalid_rollouts} ({(invalid_rollouts / total_rollouts * 100):.2f}%)")
    print(f"Finished errors: {finished_errors} ({(finished_errors / total_rollouts * 100):.2f}%), History errors: {history_errors} ({(history_errors / total_rollouts * 100):.2f}%), Variant marker errors: {variant_marker_errors} ({(variant_marker_errors / total_rollouts * 100):.2f}%), Citation format errors: {citation_format_errors} ({(citation_format_errors / total_rollouts * 100):.2f}%)")

    summary = {
        "total_rollouts": total_rollouts,
        "filtering_criteria": filtering_criteria,
        "filtered_rollouts": invalid_rollouts,
        "remaining_rollouts": valid_rollouts,
        "finished_errors": finished_errors,
        "history_errors": history_errors,
        "variant_marker_errors": variant_marker_errors,
        "citation_format_errors": citation_format_errors,
    }


    return dict(merged_dict), summary


def main(args: argparse.Namespace):
    root_path = args.root_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    # collect all rollouts
    print("Merge question-answer pairs...")
    all_rollouts, merged_summary = merge_questions(root_path, filtering_criteria=args.filtering_criteria)
    print(f"len of all question-answer pairs: {len(all_rollouts)}")
    merged_questions_path = os.path.join(output_path, "merged_questions.json")
    with open(merged_questions_path, "w", encoding="utf-8") as f:
        json.dump(all_rollouts, f, ensure_ascii=False, indent=4)
    print(f"Saving merged question-answer pairs to {merged_questions_path}")

    # evaluate rollouts with DeepResearchGym Criteria
    print("Evaluate response quality...")
    eval_criteria = ["Clarity", "Insightfulness"]
    all_rollouts = asyncio.run(
        evaluate_response_quality_async(
            input_data=all_rollouts,
            eval_criteria=eval_criteria,
            eval_kpr=args.eval_kpr,
            max_key_points=args.max_key_points,
            max_concurrent_requests=args.max_concurrent_requests,
            output_path=output_path,
        )
    )
    # rewrite merged_questions_path with evaluated rollouts
    with open(merged_questions_path, "w", encoding="utf-8") as f:
        json.dump(all_rollouts, f, ensure_ascii=False, indent=4)
    print(f"Saving evaluated question-answer pairs to {merged_questions_path}")

    # evaluate other metrics and calculate preference scores
    final_answer_word_count = []
    final_answer_sentence_count = []
    kpr = []
    kpc = []
    clarity = []
    insightfulness = []
    search_count = []
    preference_scores = []

    nlp = English()
    nlp.add_pipe("sentencizer")
    print("Evaluate other metrics, calculate preference scores...")
    for question, entries in all_rollouts.items():
        for entry in entries:
            # add other metrics to entry
            final_answer = entry["final_answer"]
            entry["final_answer_word_count"] = len((final_answer.split(" "))) if final_answer else 0
            entry["final_answer_sentence_count"] = len(list(nlp(final_answer).sents)) if final_answer else 0
            
            # calculate preference score
            if args.eval_kpr:
                entry["preference_score"] = (
                    entry["kpr_evaluation"]["supported_rate"] / 100
                    + entry["criteria_evaluation"]["Clarity"]["rating"] / 10
                    + entry["criteria_evaluation"]["Insightfulness"]["rating"] / 10
                )
                kpr.append(entry["kpr_evaluation"]["supported_rate"])
                kpc.append(entry["kpr_evaluation"]["contradicted_rate"])
            else:
                entry["preference_score"] = (
                    entry["criteria_evaluation"]["Clarity"]["rating"] / 10
                    + entry["criteria_evaluation"]["Insightfulness"]["rating"] / 10
                )
            
            # collect distribution data
            final_answer_word_count.append(entry["final_answer_word_count"])
            final_answer_sentence_count.append(entry["final_answer_sentence_count"])
            clarity.append(entry["criteria_evaluation"]["Clarity"]["rating"])
            insightfulness.append(entry["criteria_evaluation"]["Insightfulness"]["rating"])
            search_count.append(entry["search_count"])
            preference_scores.append(entry["preference_score"])

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
        search_count,
        title="search_count",
        name="search_count",
        xlabel="search_count",
        ylabel="count",
        output_path=metrics_distribution_path,
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
        preference_scores,
        title="preference_scores",
        name="preference_scores",
        xlabel="preference_scores",
        ylabel="count",
        output_path=metrics_distribution_path,
    )
    
    if args.eval_kpr:
        plot_distribution(
            kpr, title="kpr", name="kpr", xlabel="kpr", ylabel="count", output_path=metrics_distribution_path
        )
        plot_distribution(
            kpc, title="kpc", name="kpc", xlabel="kpc", ylabel="count", output_path=metrics_distribution_path
        )

    # print summary
    summary = {
        "evaluated_questions": len(all_rollouts),
        "average_final_answer_word_count": np.mean(final_answer_word_count),
        "average_final_answer_sentence_count": np.mean(final_answer_sentence_count),
        "average_search_count": np.mean(search_count),
        "average_clarity": np.mean(clarity),
        "average_insightfulness": np.mean(insightfulness),
        "average_preference_score": np.mean(preference_scores),
    }
    summary.update(merged_summary)
    
    if args.eval_kpr:
        summary.update(
            {
                "average_kpr": np.mean(kpr),
                "variance_kpr": np.var(kpr),
                "average_kpc": np.mean(kpc),
            }
        )
    
    print("----------------------------")
    print("Summary of evaluated rollouts:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # dump summaries into json file
    summary_path = os.path.join(output_path, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"Saving summary to {summary_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and process questions from given root path.")
    parser.add_argument("--root_path", type=str, required=True, help="Path to the directory containing question data.")
    parser.add_argument("--filtering_criteria", type=str, nargs="+", help="List of filtering criteria to apply.")
    parser.add_argument("--eval_kpr", action="store_true", help="Whether to evaluate KPR.")
    parser.add_argument("--max_key_points", type=int, default=20, help="Maximum number of key points to evaluate.")
    parser.add_argument("--max_concurrent_requests", type=int, default=100, help="Maximum concurrent API requests")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the directory to save the selected data."
    )

    args = parser.parse_args()

    main(args)
