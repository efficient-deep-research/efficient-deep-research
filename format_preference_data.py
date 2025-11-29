import json
import argparse
import os
from utils.prompts import get_qa_instruction


def get_prompt(question: str, max_search_limit: int, initial_search_result: str):
    instruction = get_qa_instruction(max_search_limit, question, initial_search_result)

    return [{"role": "user", "content": instruction}]


def main(args: argparse.Namespace):
    evaluated_rollouts_file = args.evaluated_rollouts_file
    output_path = args.output_path
    max_search_limit = args.instruction_search_limit
    part_number = evaluated_rollouts_file.split("/")[-3]

    with open(evaluated_rollouts_file, "r") as f:
        all_rollouts = json.load(f)
    print(f"Loaded {len(all_rollouts)} items from {evaluated_rollouts_file}")

    train_dataset = []
    test_dataset = []
    for question, entries in all_rollouts.items():
        rollouts = []
        for i, entry in enumerate(entries):
            rollouts.append({"preference_score": entry["preference_score"], "rollout": entry["output"], "search_count": entry["search_count"]})
            if i == 0:  # extract initial search result from the first entry
                source = entry["item"]["source"]
                for info in entry["all_info"]:
                    if "initial_search_webpage_analysis" in info.keys():
                        initial_search_result = info["initial_search_webpage_analysis"]

        # Sort rollouts by preference score
        sorted_rollouts = sorted(
            rollouts, key=lambda x: x["preference_score"] if x["preference_score"] is not None else -1, reverse=True
        )

        if len(sorted_rollouts) > 0:
            max_score = sorted_rollouts[0]["preference_score"]
            min_score = sorted_rollouts[-1]["preference_score"]

            score_gap = round(max_score - min_score, 1)
            if score_gap >= args.min_score_gap: # add questions to train dataset only if score gap is large enough

                # If multiple rollouts have the same max score, choose the one with the highest search count
                max_score_rollouts = [r for r in sorted_rollouts if r["preference_score"] == max_score]
                chosen = sorted(max_score_rollouts, key=lambda x: x["search_count"], reverse=True)[0]

                # If multiple rollouts have the same min score, choose the one with the lowest search count
                min_score_rollouts = [r for r in sorted_rollouts if r["preference_score"] == min_score]
                rejected = sorted(min_score_rollouts, key=lambda x: x["search_count"])[0]

                train_dataset.append(
                    {
                        "source": source,
                        "question": question,
                        "prompt": get_prompt(question, max_search_limit, initial_search_result),
                        "chosen": [{"role": "assistant", "content": chosen["rollout"]}],
                        "rejected": [{"role": "assistant", "content": rejected["rollout"]}],
                        "score_chosen": chosen["preference_score"],
                        "score_rejected": rejected["preference_score"],
                        "score_gap": score_gap,
                        "search_count_chosen": chosen["search_count"],
                        "search_count_rejected": rejected["search_count"],
                    }
                )
            else: # add questions with small score gap to test dataset
                test_dataset.append(
                    {
                        "source": source,
                        "question": question,
                        "prompt": get_prompt(question, max_search_limit, initial_search_result),
                        "chosen": [{"role": "assistant", "content": sorted_rollouts[0]["rollout"]}],
                        "rejected": [{"role": "assistant", "content": sorted_rollouts[-1]["rollout"]}],
                        "score_chosen": sorted_rollouts[0]["preference_score"],
                        "score_rejected": sorted_rollouts[-1]["preference_score"],
                        "score_gap": score_gap,
                        "search_count_chosen": sorted_rollouts[0]["search_count"],
                        "search_count_rejected": sorted_rollouts[-1]["search_count"],
                    }
                )

    print(f"Dataset: Items")
    print(f"  - Train: {len(train_dataset)} items")
    print(f"  - Test: {len(test_dataset)} items")

    # Optionally also save as JSONL for compatibility
    if args.save_jsonl:
        os.makedirs(output_path, exist_ok=True)

        train_file = os.path.join(output_path, "train.jsonl")
        with open(train_file, "w") as f:
            for item in train_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
        test_file = os.path.join(output_path, "test.jsonl")
        with open(test_file, "w") as f:
            for item in test_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
        print(f"Also saved JSONL files to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format JSON data with prompt, key, and answer formatting.")
    parser.add_argument("--evaluated_rollouts_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--min_score_gap", type=float, default=0, help="Minimum score gap between chosen and rejected.")
    parser.add_argument("--instruction_search_limit", type=int, default=10, help="Maximum number of searches per question.")
    parser.add_argument("--save_jsonl", action="store_true", help="Also save as JSONL files for compatibility")

    args = parser.parse_args()

    main(args)
