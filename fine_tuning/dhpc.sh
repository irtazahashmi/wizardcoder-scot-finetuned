#!/bin/bash
#
#SBATCH --job-name="wizard-coder-scot"
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --account=Education-EEMCS-Courses-IN4334

module load 2022r2
module load python/3.8.12
module load miniconda3
module load cuda/11.7
module load nvhpc

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n g1-wizard-coder python=3.10
conda activate g1-wizard-coder
srun pip install torch torchvision torchaudio
srun pip install -r requirements.txt
srun deepspeed train_wizardcoder.py \
    --model_name_or_path "WizardLM/WizardCoder-1B-V1.0" \
    --data_path "../data/fine_tuning/EvolInstruct-SCoT-FineTune.jsonl" \
    --output_dir "out" \
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
