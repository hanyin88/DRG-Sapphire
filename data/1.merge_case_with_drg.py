import pandas as pd
import yaml

# ---------------------------
# Script Overview:
# This script reads paths from a YAML config file and processes both the training and testing datasets
# by merging clinical inputs with two reference files:
# (1) id2label.csv — from DRG-LLaMA
# (2) DRG_34.csv — from DRG-LLaMA
#
# The script outputs two merged CSVs — one for training and one for testing — based on the specified paths in the YAML.
# ---------------------------

# Load input/output paths from config.yaml
with open("paths.yaml", "r") as f:
    paths = yaml.safe_load(f)

# Function to process one data split (either 'train' or 'test')
def process_split(split_name: str):
    # Extract relevant paths from YAML config
    input_path = paths[f"{split_name}_set_path"]
    id2label_path = paths["id2label_path"]
    drg_34_path = paths["drg_34_path"]
    output_path = paths[f"{split_name}_merged_csv_path"]

    # Load input CSVs
    df1 = pd.read_csv(input_path)             # Input examples with 'label' and 'drg_34_code'
    df2 = pd.read_csv(id2label_path)          # Mapping from label to DRG code
    df3 = pd.read_csv(drg_34_path, sep="\t")  # DRG metadata table (tab-separated)

    # Merge df1 with df2 on the 'label' column to add DRG labels
    df_merged = df1.merge(df2, on='label', how='inner')
    print(f"[{split_name}] After first merge: {df_merged.shape[0]} rows")

    # Merge the above result with df3 to add DRG descriptions or additional fields
    df_final = df_merged.merge(df3, left_on='drg_34_code', right_on='DRG', how='inner')
    print(f"[{split_name}] After second merge: {df_final.shape[0]} rows")

    # Save the final merged dataframe to the output path
    df_final.to_csv(output_path, index=False)
    print(f"[{split_name}] Saved to: {output_path}")

# Execute the pipeline for both training and testing data
process_split("train")
process_split("test")
