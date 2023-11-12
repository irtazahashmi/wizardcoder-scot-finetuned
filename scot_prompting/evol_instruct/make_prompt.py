import json
import random


EVOL_INSTRUCT_DATASET_PATH = "../../data/fine_tuning/EvolInstruct-Code-80k.json"
PROMPT_DATASET_PATH = "../../data/fine_tuning/EvolInstruct-SCoT-Prompts.jsonl"


def generate_prompts(sample=1.5):
    """
    Enhanced a percentage of the Evol-Instruct dataset with the SCoT prompting technique.

    :param sample: The sampling percentage to take from the dataset.
    """
    training_dataset_full = json.load(open(EVOL_INSTRUCT_DATASET_PATH))
    sample_len = int(sample * len(training_dataset_full) / 100)

    print(
        f"Sampling {sample_len} entries from the {len(training_dataset_full)} entries dataset..."
    )

    # The first 3 instructions are used as examples already.
    sampled_idx = random.sample(range(3, len(training_dataset_full)), sample_len)

    output_f = open(PROMPT_DATASET_PATH, "w")
    example = open("prompt_example.txt", "r").read()
    for idx in sampled_idx:
        output_f.write(
            json.dumps(
                {
                    "instruction": training_dataset_full[idx]["instruction"],
                    "prompt": example
                    + "\n"
                    + training_dataset_full[idx]["instruction"]
                    + "\n# The pesudo code of the above instruction:",
                    "output": training_dataset_full[idx]["output"],
                }
            )
            + "\n"
        )
        output_f.flush()


if __name__ == "__main__":
    generate_prompts()
