import json
import os
import time

import openai


PROMPT_START_IDX = 0
PROMPT_END_IDX = None
PROMPT_DATASET_PATH = "../../data/evaluation/human_eval.jsonl"
EVALUATION_DATASET_PATH = "../../data/evaluation/human_eval-SCoT.jsonl"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPEN_API_TIMEOUT_SECONDS = 20


def generate_scot_dataset(prompt_start_idx=0, prompt_end_idx=None):
    """
    Generate the training dataset by inferring instructions to the ChatGPT API.

    :param prompt_start_idx: Starting index of the input dataset. Default is 0.
    :param prompt_end_idx: Ending index of the input dataset. Default is None.
    """
    if not OPENAI_API_KEY:
        print("Please provide an API KEY for the OPENAI API!")

    openai.api_key = OPENAI_API_KEY

    prompts_f = open(PROMPT_DATASET_PATH, "r").readlines()[prompt_start_idx:]
    output_f = open(EVALUATION_DATASET_PATH, "a")
    example = open("prompt_example.txt", "r").read()
    for idx, prompt in enumerate(prompts_f):
        # Build prompt
        prompt = json.loads(prompt)
        scot_prompt = (
            example
            + "\n\n"
            + prompt["prompt"]
            + "    pass\n\n"
            + "# The pseudo code of the above function:\n"
        )

        # Enhance with ChatGPT
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=[{"role": "user", "content": scot_prompt}],
                max_tokens=300,
                temperature=0.8,
                n=1,
                top_p=0.95,
            )
        except Exception as e:
            print(f"Instruction {prompt_start_idx}:", e)
            break

        for choice in response.choices:
            assert choice.message.role == "assistant"
            prompt["prompt"] += (
                "    pass\n\n"
                + "# The pseudo code of the above function:\n"
                + choice.message.content
            )

        # Update output file
        output_f.write(json.dumps(prompt) + "\n")
        output_f.flush()

        print(f"Enhanced instruction at index {prompt_start_idx}")
        prompt_start_idx += 1

        # Stop if end is reached
        if prompt_end_idx and prompt_start_idx > prompt_end_idx:
            break

        time.sleep(OPEN_API_TIMEOUT_SECONDS)


if __name__ == "__main__":
    generate_scot_dataset(
        prompt_start_idx=PROMPT_START_IDX, prompt_end_idx=PROMPT_END_IDX
    )
