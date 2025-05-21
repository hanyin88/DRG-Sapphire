# DRG-Sapphire

This repository contains the codebase for the paper **[Reinforcement Learning for Out-of-Distribution Reasoning in LLMs: An Empirical Study on Diagnosis-Related Group Coding]**. It includes source code, data preprocessing tools, configuration recipes, and scripts for training and inference.

- The `src/` directory contains the core source code. The core code is refactored based on the great [Open-R1 project](https://github.com/huggingface/open-r1).
- The `data/` directory includes scripts for data preprocessing and construction.
- The `recipes/` directory contains configuration files.
- The `scripts/` directory provides example scripts for training and inference.

---

## Local Setup

To set up the environment locally using Conda:

1. Create the environment:
   ```
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```
   conda activate DRG-Sapphire
   ```

---

## Data Access

To prepare the dataset:

1. Obtain access to the MIMIC-IV database:  
   https://physionet.org/content/mimiciv/

2. Follow the data preprocessing and splitting steps described in the [DRG-LLaMA repository](https://github.com/hanyin88/DRG-LLaMA/).

3. Update the data paths in `paths.yaml` to reflect your local setup.

4. Run all Python files in the `data/` directory **in order**. These scripts will:
   - Preprocess the dataset
   - Construct the Chain-of-Thought (CoT) dataset
   - Split data into Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) subsets

> ⚠️ **Note:** For Steps 3 and 4, ensure you update the input files depending on whether you're using the DRG-Small dataset or the full dataset. The current scripts are configured for the full dataset.

---

## Running SFT

To run supervised fine-tuning (SFT), use the example script:

```
bash scripts/sft.sh
```

---

## Running GRPO

To run Generalized Reinforcement Pretraining with Optimization (GRPO), use:

```
bash scripts/grpo.sh
```

This script launches `src/open_r1/grpo.py`.

Details of files in the `src/open_r1` folder:
- `grpo.py`: Contains the baseline implementation of vanilla GRPO with dense rewards.
- `CustomGRPOTrainer.py` and `CustomGRPOPosTrainer.py`: Provide customized trainers with dynamic resampling strategies.
- `grpo_best_config.py`: Runs the best-performing GRPO configuration using DAPO loss, strict reward, and KL decay.

---

## Running Inference

To run inference after training:

```
bash scripts/inference.sh
```
