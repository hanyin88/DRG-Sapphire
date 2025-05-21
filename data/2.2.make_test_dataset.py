import pandas as pd
import yaml
from datasets import Dataset, DatasetDict

# Load paths from YAML file
with open("paths.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load the main CSV file
df = pd.read_csv(config["test_merged_csv_path"])

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
all_test_dataset = DatasetDict({"train": Dataset.from_pandas(df)})
all_test_dataset.save_to_disk(config["all_test_dataset_path"])
