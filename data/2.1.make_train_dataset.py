import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load paths from YAML file
with open("paths.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load the main CSV file
df = pd.read_csv(config["train_merged_csv_path"])

# Ensure 'Description' is treated as a string
df['Description'] = df['Description'].astype(str)

# Keep only the required columns
df = df[['text', 'Description']]

# Modify the 'text' column with prompt formatting
df['text'] = df['text'].apply(lambda x: (
    "What is the most likely Medicare Severity Diagnosis Related Group (MS-DRG) "
    "based on the following discharge summary of a hospitalization? "
    "Provide the complete text description of the DRG code without including the numerical code. \n"
    f"***{x}***.")
)

# Rename columns to 'problem' and 'solution'
df.rename(columns={'text': 'problem', 'Description': 'solution'}, inplace=True)

# Save the cleaned dataset as Hugging Face Dataset
all_train_dataset = Dataset.from_pandas(df)
all_train_dataset.save_to_disk(config["all_train_dataset_path"])

# Filter out rare DRG codes (appearing fewer than 5 times) to allow for splitting
solution_counts = df['solution'].value_counts()
rare_solutions = solution_counts[solution_counts < 5].index
df_rare = df[df['solution'].isin(rare_solutions)]
df = df[~df['solution'].isin(rare_solutions)]

# First split: 1% for SFT cold start
df_remaining, df_sft_cold_start = train_test_split(
    df, test_size=0.01, random_state=42, stratify=df['solution']
)

# Second split: 20% of remaining data for DRG_small
_, df_DRG_small = train_test_split(
    df_remaining, test_size=0.2, random_state=42, stratify=df_remaining['solution']
)

# Display summary of dataset sizes
print(f"SFT cold start size: {len(df_sft_cold_start)}, DRG_small size: {len(df_DRG_small)}, df rare size: {len(df_rare)}")

# Save to Hugging Face Datasets
sft_cold_start_dataset = Dataset.from_pandas(df_sft_cold_start)
DRG_small_dataset = Dataset.from_pandas(df_DRG_small)

sft_cold_start_dataset.save_to_disk(config["sft_cold_start_dataset_path"])
DRG_small_dataset.save_to_disk(config["DRG_small_dataset_path"])