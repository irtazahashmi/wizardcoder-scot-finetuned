# G1-WizardCoder: Enhancing WizardCoder with Structured Chain-of-Thought (S-CoT) Prompting

IN4334 ML4SE

Group1 WizardCoder

The aim of this repository is to investigate if S-CoT prompting can improve the performance of WizardCoder on the pass@1 metric. 
S-CoT prompting is applied in the evaluation by enhancing
the instructions of the HumanEval and MBPP datasets. The S-CoT
enhancement of the evaluation datasets allows to study its effect
when used just as a prompting technique, independently of the
S-CoT fine-tuning of the model. For the fine-tuning, S-CoT is used to enhance a sample of about 1200 entries from the Evol-
Instruct 80k dataset. The resulting dataset is then used for the
training task. The current WizardCoder model and the new S-CoT
fine-tuned one are compared on both versions of HumanEval
and MBPP (S-CoT and not) on the pass@1 metric.

The following table shows the results of pass@1(%) on HumanEval and MBPP compared to HumanEval-SCoT and MBPP-SCoT using WizardCoder-1B, WizardCoder-SCoT-1B and WizardCoder-15B. 

| **Dataset**    | **WizardCoder-1B-V1.0** | **WizardCoder-SCoT-1B-V1.0** | **WizardCoder-15B-V1.0** |
|----------------|-------------------------|------------------------------|--------------------------|
| HumanEval      | 23.78                   | **17.68**                    | 57.3                     |
| HumanEval-SCoT | **44.51**               | **27.44**                    | **57.3**                 |
| MBPP           | 23.4                    | **19.4**                     | 51.8                     |
| MBPP-SCoT      | **40**                  | **28**                       | **45.6**                 |

### Environment Setup

#### Pip
Create and activate a virtual environment with Python>=3.10.

```shell
# Linux/macOS
python3 -m venv g1-wizard-coder
source g1-wizard-coder/bin/activate

# Windows
.\g1-wizard-coder\Scripts\activate
py -m venv g1-wizard-coder
```

Install the requirements:

```shell
pip install torch torchvision torchaudio

pip install -r requirements.txt
```


#### Conda
Create a conda environment with Python>=3.10.

```shell
conda create -n g1-wizard-coder python=3.10

conda activate g1-wizard-coder
```

Install the requirements:

```shell
# latest GPU version with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# OR, For installing Pytorch CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch

pip install -r requirements.txt
```

### Datasets and Models

The datasets need to be placed in the `data` subfolders:

- `evaluate`: Add the datasets necessary for the evaluation (HumanEval, MBPP, etc.) in the `.jsonl` format, HumanEval, MBPP and MBCPP already present.
- `fine_tuning`: Add the dataset to use for fine-tuning in the `.jsonl` format, the Evol_Instruct dataset enhanced with S-CoT is already provided.
- `output`: Holds the results of the model inferences in `.jsonl` format.

The models stored locally can be placed in the `models` folder.

The produced datasets and models for this work can be found at the following link on HuggingFace:

https://huggingface.co/ML4SE2023-G1-WizardCoder 

### S-CoT Prompting

Scripts for S-CoT prompt generation, see [README](scot_prompting/README.md)

### Fine-Tuning

Scripts for fine-tuning WizardCoder, see [README](fine_tuning/README.md).

### Evaluation

Scripts for evaluating WizardCoder, see [README](evaluation/README.md).

### Code Quality

To ensure the readability and quality of the code, the Black formatter has been included in the requirements.
To check if the code follows the quality standard, run:

```shell
black --check evaluation/ fine_tuning/ scot_prompting/
```

To format a python file using the Black formatter:

```shell
black /path/to/file/file_name.py
```

### Contributors

Project developed for the course [IN4334 - Analytics and Machine Learning for Software Engineering](https://malihehizadi.github.io/ml4se/) in Q1 of the 2023/2024 academic year at TU Delft.

**Group1 Members:**
- Irtaza Hashmi - @irtazahashmi
- Radek Kargul - @rkargul
- Eleni Papadopoulou - @EleniPP
- Rohan Sobha - @Iodine98
- Alexander Schnapp - @aschnapp
- Mattia Bonfanti - @MattiaBonfanti-CS
