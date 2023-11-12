# S-CoT prompting

S-CoT prompting is used to generate pseudo-code to attach to the instructions in the Evol-Instruct dataset and to the functions to complete in the HumanEval and MBPP evaluation datasets.
The pesudo-code is generated using the GPT3.5 model available through the OpenAI API.

Based on:
- Paper: https://arxiv.org/abs/2305.06599v3
- Code: https://figshare.com/articles/software/SCoT_Prompting/23821146/1

## Requirements
- Python >= 3.9
- OpenAI api key for ChatGPT

```shell
pip install -r requirements.txt
```

## How to run it

First, the OpenAI API KEY must be set as environment variable.

```shell
OPENAI_API_KEY=YOUR_OPEN_AI_API_KEY
```

### Evol-Instruct

Run the following script to generate the prompts to infer:

```shell
cd evol_instruct

# Generates the data/fine_tuning/EvolInstruct-SCoT-Prompts.jsonl dataset
python make_prompt.py
```

Run the following scripts to generate the S-CoT dataset using GPT-3.5

```shell
cd evol_instruct

# You must set the OPENAI_API_KEY environment variables using your API KEY
# Set the PROMPT_START_IDX variable in the script to start from the script last stopped
# The data/fine_tuning/EvolInstruct-SCoT-FineTune.jsonl dataset will be generated
python chatgpt_infer.py
```

### HumanEval

Run the following scripts to generate the S-CoT dataset using GPT-3.5

```shell
cd human_eval

# You must set the OPENAI_API_KEY environment variables using your API KEY
# Set the PROMPT_START_IDX variable in the script to start from the script last stopped
# The data/fine_tuning/human_eval-SCoT.jsonl dataset will be generated
python chatgpt_infer.py
```

### MBPP and MBCPP

Run the following scripts to generate the S-CoT dataset using GPT-3.5

```shell
cd mbpp

# You must set the OPENAI_API_KEY environment variables using your API KEY
# Set the PROMPT_START_IDX variable in the script to start from the script last stopped
# The data/fine_tuning/mbpp-SCoT.jsonl dataset will be generated
# To generate prompts for C code, set the constant LANGUAGE="C"
python chatgpt_infer.py
```
