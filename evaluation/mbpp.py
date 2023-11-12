import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


MODEL_PATH = os.getenv("MODEL_PATH", f"../models/WizardCoder-1B-V1.0/")
MODEL = os.path.basename(os.path.normpath(MODEL_PATH))
USE_SCOT = os.getenv("USE_SCOT", "True") == "True"
INPUT_DATASET = (
    "../data/evaluation/mbpp-SCoT.jsonl"
    if USE_SCOT
    else "../data/evaluation/mbpp.jsonl"
)
EVALUATION_START_IDX = 0
EVALUATION_END_IDX = None
TEMPERATURE = 0.2
MAX_TOKEN_LENGTH = 2048
PREDICTIONS = 1
SEQUENCES_PER_ITERATION = 1
OUTPUT_PATH = (
    f"../data/output/mbpp_scot/{MODEL}/"
    if USE_SCOT
    else f"../data/output/mbpp/{MODEL}/"
)
EVALUATIONS_PATH = (
    f"../data/output/mbpp_scot/{MODEL}/mbpp-SCoT-evaluation.jsonl"
    if USE_SCOT
    else f"../data/output/mbpp/{MODEL}/mbpp-evaluation.jsonl"
)
DECODING_STYLE = None

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except Exception:
    pass


def generate_prompt(input):
    """
    Generate prompt for the model.

    :param input: The input prompt.
    :return: The prompt in the correct format.
    """
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input}

### Response:"""
    return INSTRUCTION


def process_completions(completion):
    """
    Process the MBPP completions.

    :param completion: The completion to process.
    :return: The update completion.
    """
    processed_completion = completion
    if "```python" in processed_completion:
        def_line = processed_completion.index("```python")
        processed_completion = processed_completion[def_line:].strip()
        processed_completion = processed_completion.replace("```python", "")
        try:
            next_line = processed_completion.index("\n```")
            processed_completion = processed_completion[:next_line].strip()
        except:
            print(f"Error: {completion}")
            print("================\n")
            return completion

    if '__name__ == "__main__"' in processed_completion:
        next_line = processed_completion.index('if __name__ == "__main__":')
        processed_completion = processed_completion[:next_line].strip()

    if "# Example usage" in processed_completion:
        next_line = processed_completion.index("# Example usage")
        processed_completion = processed_completion[:next_line].strip()

    if "# Test examples" in processed_completion:
        next_line = processed_completion.index("# Test examples")
        processed_completion = processed_completion[:next_line].strip()

    return processed_completion


def get_model(
    load_8bit: bool = False,
    base_model: str = "bigcode/starcoder",
):
    """
    Load the Code LLM.

    :param load_8bit: Load using 8 bits encoding.
    :param base_model: Model path
    :return: The loaded model.
    """
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float32,
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    if not load_8bit and device != "cpu":
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return tokenizer, model


def generate_evaluations(evaluation_start_idx=0, evaluation_end_idx=None):
    """
    Evaluate the model using the MBPP dataset.

    :param evaluation_start_idx: The starting task to evaluate.
    :param evaluation_end_idx: The last task to evaluate.
    """
    # Open the dataset
    problems = open(INPUT_DATASET, "r").readlines()[evaluation_start_idx:]
    num_samples = len(problems)
    print("Number of samples to evaluate: {}".format(num_samples))

    # Load the model
    tokenizer, model = get_model(base_model=MODEL_PATH)
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        temperature=TEMPERATURE,
        max_length=MAX_TOKEN_LENGTH,
        num_return_sequences=SEQUENCES_PER_ITERATION,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.95,
    )

    print(f"Loaded {MODEL_PATH}")

    # Generate the code result for each instruction in the dataset
    for i, problem in enumerate(problems):
        problem = json.loads(problem)
        prompt = problem["prompt"].replace("    ", "\t")
        prompt_batch = [generate_prompt(prompt)]

        encoding = tokenizer(
            prompt_batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
        ).to(device)

        # Establish how many results the model should provide based on the decoding style
        if DECODING_STYLE == "sampling":
            loops = int(PREDICTIONS / SEQUENCES_PER_ITERATION)
        else:
            loops = 1

        output_f = open(EVALUATIONS_PATH, "a")
        for _ in range(loops):
            # Generate code based on the given instruction
            with torch.no_grad():
                gen_tokens = model.generate(
                    **encoding, generation_config=generation_config
                )

            if gen_tokens is not None:
                gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            else:
                gen_seqs = None

            # Save generated code to file
            if gen_seqs is not None:
                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[1]
                    completion_seq = completion_seq.replace("\t", "    ")
                    all_code = gen_seq.replace("\t", "    ")

                    output_seq = {
                        "task_id": problem["task_id"],
                        "prompt": prompt,
                        "completion": process_completions(completion_seq),
                        "all_code": all_code,
                    }

                    # Update output file
                    output_f.write(json.dumps(output_seq) + "\n")
                    output_f.flush()

                print("Processed MBPP task {}".format(problem["task_id"]))
            else:
                print("Error while processing MBPP task".format(problem["task_id"]))

        evaluation_start_idx += 1

        # Stop if end is reached
        if evaluation_end_idx and evaluation_start_idx > evaluation_end_idx:
            break


if __name__ == "__main__":
    generate_evaluations(
        evaluation_start_idx=EVALUATION_START_IDX, evaluation_end_idx=EVALUATION_END_IDX
    )
