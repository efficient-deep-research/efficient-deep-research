import json
import argparse
import os
from utils.prompts import get_qa_instruction, get_task_instruction


def get_prompt(question: str, max_search_limit: int):
    instruction = get_qa_instruction(max_search_limit)
    user_prompt = get_task_instruction(question)

    return [{"role": "user", "content": instruction + user_prompt}]


def main(args: argparse.Namespace):
    evaluated_rollouts_file = args.evaluated_rollouts_file
    output_path = args.output_path
    max_search_limit = args.instruction_search_limit

    with open(evaluated_rollouts_file, "r") as f:
        all_rollouts = json.load(f)
    print(f"Loaded {len(all_rollouts)} items from {evaluated_rollouts_file}")

    dataset = []
    for question, entries in all_rollouts.items():
        rollouts = []
        for entry in entries:
            if entry["preference_score"] > 0:  # only keep valid rollouts
                rollouts.append({"preference_score": entry["preference_score"], "rollout": entry["output"]})

        sorted_rollouts = sorted(
            rollouts, key=lambda x: x["preference_score"] if x["preference_score"] is not None else -1, reverse=True
        )

        if len(sorted_rollouts) > 0:
            max_score = sorted_rollouts[0]["preference_score"]
            min_score = sorted_rollouts[-1]["preference_score"]
            if max_score - min_score > 0:
                chosen = sorted_rollouts[0]
                rejected = sorted_rollouts[-1]
                dataset.append(
                    {
                        "prompt": get_prompt(question, max_search_limit),
                        "chosen": [{"role": "assistant", "content": chosen["rollout"]}],
                        "rejected": [{"role": "assistant", "content": rejected["rollout"]}],
                        "score_chosen": chosen["preference_score"],
                        "score_rejected": rejected["preference_score"],
                    }
                )

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "preference_data.json")
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved preference dataset with {len(dataset)} items to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format JSON data with prompt, key, and answer formatting.")
    parser.add_argument("--evaluated_rollouts_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument(
        "--instruction_search_limit", type=int, default=10, help="Maximum number of searches per question."
    )
    args = parser.parse_args()

    main(args)
