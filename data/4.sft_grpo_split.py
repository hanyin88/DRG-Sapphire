import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def main():
    # Load paths from YAML config
    with open("paths.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Read the inference results CSV
    df = pd.read_csv(config["all_train_cot_csv_path"])

    # Remove rare solutions
    solution_counts = df['solution'].value_counts()
    rare_solutions = solution_counts[solution_counts < 2].index
    df = df[~df['solution'].isin(rare_solutions)]

    # Load pretraining corpus
    df2 = pd.read_csv(config["ms_drg_corpora_path"])

    # Define split ratios
    splits = config["splits"]  # Should be a list of [sft_pct, grpo_pct] pairs
    RANDOM_SEED = 42

    # DRG introduction
    drg_intro = '''
    MS-DRG (Medicare Severity Diagnosis-Related Groups) is a system used by the U.S. Centers for Medicare & Medicaid Services to classify hospital inpatient stays for payment purposes. It groups patients with similar clinical conditions and resource needs into categories to determine reimbursement amounts.

    Each MS-DRG is assigned based on the patient's principal diagnosis, secondary diagnoses, procedures performed, age, discharge status, and other factors. The goal is to ensure fair and consistent hospital reimbursement based on the severity of the illness and the complexity of care required.

    CC and MCC in MS-DRG
    CC (Complication or Comorbidity): A secondary diagnosis that increases the complexity of care and resource utilization.
    MCC (Major Complication or Comorbidity): A more severe secondary condition that has a significant impact on resource use and hospital reimbursement.
    MCCs have a greater effect on the DRG weight than CCs due to increased patient care complexity.\n
    '''

    for sft_pct, grpo_pct in splits:
        # Create output directory
        folder_name = f"{sft_pct}_{grpo_pct}"
        out_dir = os.path.join(config["split_training_output_dir"], folder_name)
        os.makedirs(out_dir, exist_ok=True)

        # Train/test split
        df_grpo, df_sft = train_test_split(
            df,
            test_size=sft_pct / 100.0,
            random_state=RANDOM_SEED,
            stratify=df['solution']
        )

        print(f"\n=== Split {sft_pct}:{grpo_pct} ===")
        print(f"SFT rows:  {len(df_sft)}")
        print(f"GRPO rows: {len(df_grpo)}")

        # Process SFT portion
        df_sft['explanation'] = df_sft['explanation'].str.replace("<think>", "", regex=False).str.replace("</think>", "", regex=False)
        df_sft['completion'] = df_sft.apply(
            lambda row: f"<think>\n{row['explanation']}\n</think>\n<answer>\n{row['solution']}\n</answer>",
            axis=1
        )
        df_sft = df_sft[['problem', 'completion']].rename(columns={'problem': 'prompt'})
        df_sft['prompt'] = df_sft['prompt'].apply(lambda x: f"{drg_intro}{x}")

        # Process pretraining data
        df2_copy = df2.copy()
        df2_copy['solution'] = df2_copy.apply(
            lambda row: (
                "<think>\nTo answer this question, I need to check what the MS-DRG codes are.\n</think>\n"
                f"<answer>\n{row['completion']}\n</answer>"
            ),
            axis=1
        )
        df2_copy = df2_copy.drop(columns=['completion']).rename(columns={'solution': 'completion'})
        df2_copy['prompt'] = df2_copy['prompt'].apply(lambda x: f"{drg_intro}{x}")

        # Combine and save SFT
        combined_sft = pd.concat([df_sft, df2_copy], ignore_index=True)
        sft_csv_path = os.path.join(out_dir, "sft_split.csv")
        combined_sft.to_csv(sft_csv_path, index=False)
        print(f"SFT final CSV saved to: {sft_csv_path}")

        # Save GRPO data
        df_grpo = df_grpo[['problem', 'solution']]
        grpo_csv_path = os.path.join(out_dir, "grpo_split.csv")
        df_grpo.to_csv(grpo_csv_path, index=False)
        print(f"GRPO CSV saved to: {grpo_csv_path}")

        # Save GRPO HF dataset
        dataset = DatasetDict({"train": Dataset.from_pandas(df_grpo)})
        grpo_dataset_dir = os.path.join(out_dir, "grpo_train_merged")
        dataset.save_to_disk(grpo_dataset_dir)
        print(f"GRPO Hugging Face dataset saved to: {grpo_dataset_dir}")

    print("\nAll splits and transformations complete!")

if __name__ == "__main__":
    main()
