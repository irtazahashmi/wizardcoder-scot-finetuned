# Evaluation

Evaluation is based on the pass@1 metrics using the HumanEval, HumenEval-SCoT, MBPP and MBPP-SCoT datasets.
The datasets are available at the following HuggingFace link:

https://huggingface.co/ML4SE2023-G1-WizardCoder

The hardware consisted on a GPU instance rented from [DataCrunch](https://datacrunch.io/) with the following specifications:

| NVidia RTX A6000 48GB 1A6000.10V |
|----------------------------------|
| 2 GPUs                           |
| 48GB VRAM per GPU                |
| 60 GB RAM                        |
| 10 CPUs                          |
| 100GB SSD Storage                |
| Ubuntu 20.04                     |
| CUDA 11.6                        |

## Requirements

The following requirements were installed to evaluate the models using Anaconda as environment:

```shell
# Create conda environment
conda create -n g1-wizard-coder python=3.10
conda activate g1-wizard-coder

# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

pip install -r requirements.txt
```

## Inference

To use WizardCoder for inference over an input dataset:

```shell
# Assuming that WizadrCoder-1B1-V1.0 has been downloaded in the models folder
python inference_wizardcoder.py \
--base_model "../models/WizardCoder-1B-V1.0/" \
--input_data_path "../data/evaluation/simple.jsonl" \
--output_data_path "../data/output/output_simple.jsonl"
```

The instructions are wrapped in the following prompt:

```shell
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
```

## Evaluate

### HumanEval

1. Add the dataset to the `../data/evaluation/` folder;
2. Run the following command:

```shell
# Assuming that it is evaluating the ../data/evaluation/human_eval-SCoT.jsonl dataset
# Adjust the script accordingly to evaluate the dataset of need
# To run multiple instances on different GPU devices, set the CUDA_VISIBLE_DEVICES environment variable accordingly
# The results of the following command are saved in:
# ../data/output/human_eval_scot/WizardCoder-1B-V1.0/human_eval-SCoT-evaluation.jsonl
MODEL_PATH="WizardLM/WizardCoder-1B-V1.0" USE_SCOT="True" CUDA_VISIBLE_DEVICES=0 python humaneval.py
```

3. Process the results and get the scores:

```shell
# The pass@1 score is displayed in the output and saved to the chosen txt file
evaluate_functional_correctness "../data/output/human_eval_scot/WizardCoder-1B-V1.0/human_eval-SCoT-evaluation.jsonl" --problem_file="../data/evaluation/human_eval-SCoT.jsonl" | tee "../data/output/human_eval_scot/WizardCoder-1B-V1.0/evaluation_output.txt"
```

### MBPP

1. Add the dataset to the `../data/evaluation/` folder;
2. Run the following command:

```shell
# Assuming that it is evaluating the ../data/evaluation/mbpp-SCoT.jsonl dataset
# Adjust the script accordingly to evaluate the dataset of need
# To run multiple instances on different GPU devices, set the CUDA_VISIBLE_DEVICES environment variable accordingly
# The results of the following command are saved in:
# ../data/output/mbpp_scot/WizardCoder-1B-V1.0/mbpp-SCoT-evaluation.jsonl
MODEL_PATH="WizardLM/WizardCoder-1B-V1.0" USE_SCOT="True" CUDA_VISIBLE_DEVICES=0 python mbpp.py
```

3. Process the results and get the scores:

```shell
# The pass@1 score is displayed in the output and saved to the chosen txt file
evaluate_functional_correctness "../data/output/mbpp_scot/WizardCoder-1B-V1.0/mbpp-SCoT-evaluation.jsonl" --problem_file="../data/evaluation/mbpp-SCoT.jsonl" | tee "../data/output/mbpp_scot/WizardCoder-1B-V1.0/evaluation_output.txt"
```
