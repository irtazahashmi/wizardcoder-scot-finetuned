# Fine-Tuning

The fine-tuning of the WizardCoder models is based on the [LlamaX](https://github.com/AetherCortex/Llama-X/tree/main) and [WizardCoder fine-tuning](https://github.com/nlpxucan/abcd/tree/main/WizardCoder#fine-tuning) instructions.

We fine-tuned WizardCoder-1B-V1.0 with the following hyperparameters:

| Hyperparameter | WizardCoder-1B-V1.0 |
|----------------|---------------------|
| Batch size     | 16                  |
| Learning rate  | 2e-5                |
| Epochs         | 3                   |
| Max length     | 2048                |
| Warmup step    | 30                  |
| LR scheduler   | cosine              |

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

The following requirements were installed to train the model using Anaconda as environment:

```shell
# Create conda environment
conda create -n g1-wizard-coder python=3.10
conda activate g1-wizard-coder

# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

pip install -r requirements.txt
```

## Training 

To train the model, then run:

```shell
deepspeed train_wizardcoder.py \
    --model_name_or_path "/your/path/to/model/" \
    --data_path "/your/path/to/code_instruction_data.json" \
    --output_dir "/your/path/to/ckpt" \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 30 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True
```

The model can be either stored locally (in the `models` folder) or on HuggingFace.
The fine-tuned models are available at the following HuggingFace link:

https://huggingface.co/ML4SE2023-G1-WizardCoder

## DelftBlue

We also provide a script to run the fine-tuning on the DelftBlue system.
Load the files in your DelftBlue instance and run the following command to submit a job:

```bash
sbatch dhpc.sh
```
