# This script runs experiments on the test set using OpenAI's GPT-4o model via Azure, 
# producing results similar to those shown in Figure 1.

import os
import re
import logging
import argparse
from tqdm import tqdm
import sys

import pandas as pd
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer

from openai import AzureOpenAI  # Using Azure OpenAI client

# get today's date
today = pd.Timestamp.now().strftime("%Y-%m-%d")

# set random seed
seed = 42

def init_openai():
    # Change to your own azure openai details or set env vars
    endpoint = os.getenv("ENDPOINT_URL", "****")  
    deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "****") 
    client = AzureOpenAI(  
        azure_endpoint=endpoint,  
        api_key=api_key,  
        api_version="2024-05-01-preview",  
    )
    return client, deployment

# 1) DRG intro 
drg_intro = """
    MS-DRG (Medicare Severity Diagnosis-Related Groups) is a system used by the U.S. Centers for Medicare & Medicaid Services to classify hospital inpatient stays for payment purposes. It groups patients with similar clinical conditions and resource needs into categories to determine reimbursement amounts. Each MS-DRG is assigned based on the patient's principal diagnosis, secondary diagnoses, procedures performed, age, discharge status, and other factors. The goal is to ensure fair and consistent hospital reimbursement based on the severity of the illness and the complexity of care required.

    CC and MCC in MS-DRG
    CC (Complication or Comorbidity): A secondary diagnosis that increases the complexity of care and resource utilization.
    MCC (Major Complication or Comorbidity): A more severe secondary condition that has a significant impact on resource use and hospital reimbursement.
    MCCs have a greater effect on the DRG weight than CCs due to increased patient care complexity.

"""

# 2) New instruction for final answer format - for GPT-4o and o3-mini
answer_instruction = """

After your reasoning, please provide the final DRG description in square brackets. 
For example:
[ALCOHOL DRUG ABUSE OR DEPENDENCE WITHOUT REHABILITATION THERAPY WITHOUT MCC]

Only include the textual description (no numbers), and wrap it exactly in [ ]. You should be using MS-DRG version 34.0.
"""

def load_and_prepare_dataset(dataset_path: str) -> DatasetDict:
    ds = load_from_disk(dataset_path)
    # prepend the drg_intro
    ds = ds.map(lambda x: {"problem": drg_intro + x["problem"]})
    return ds

def extract_answer_if_valid(text: str) -> str:
    """
    Find the first [...] block in the model output and return its contents.
    """
    m = re.search(r"\[([^\]]+)\]", text)
    return m.group(1).strip() if m else None

def normalize(s: str) -> str:
    """
    Lowercase, strip, collapse whitespace for fair comparison.
    """
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())

def main():
    parser = argparse.ArgumentParser(
        description="Run Azure OpenAI chat on a HF dataset, extract [DRG] answers, compute accuracy, save CSV."
    )
    parser.add_argument("--dataset_path", type=str,
                        default="data/test_merged.csv")
    parser.add_argument("--output_csv",   type=str,
                        default=f"baseline/gpt4o_{today}.csv")
    parser.add_argument("--max_rows",     type=int, default=None)
    parser.add_argument("--log_file",     type=str, default=f"baseline/gpt4o_{today}.log",
                        help="Where to write detailed logs.")
    args = parser.parse_args()

    # --- logging setup ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        )
    logging.info("Starting inference run")

    client, deployment = init_openai()

    # --- load & prepare ---
    ds = load_and_prepare_dataset(args.dataset_path)["train"]
    if "solution" not in ds.column_names:
        logging.error("Dataset missing 'solution' column. Aborting.")
        sys.exit(1)
    df = ds.to_pandas()
    
    if args.max_rows:
        df = df.head(args.max_rows)

    system_prompt = (
        "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
        "First think through your chain of thought, then give the final answer."
    )

    outputs, matches = [], []

    def save_partial():
        """Save whatever we have so far to a _partial.csv file."""
        n = len(outputs)
        df_part = df.iloc[:n].copy()
        df_part["model_output"] = outputs
        df_part["match_answer"] = matches
        df_part["is_correct"] = df_part.apply(
            lambda r: normalize(r["match_answer"]) == normalize(r["solution"]), axis=1
        )
        partial_path = args.output_csv.replace(".csv", "_partial.csv")
        df_part.drop(columns=["problem"]).to_csv(partial_path, index=False)
        logging.info(f"Partial results saved to '{partial_path}'")

    # --- inference loop w/ error handling ---
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
            user_content = row["problem"] + answer_instruction
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ]
            try:
                resp = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_tokens=4096,
                    temperature=1.0,
                    seed=seed,
                    top_p=0.8,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stream=False
                )
                text = resp.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"Error for Case_id {idx}: {e}")
                text = ""

            outputs.append(text)
            matches.append(extract_answer_if_valid(text))

    except KeyboardInterrupt:
        logging.warning("Interrupted by user at row %d", idx)
        save_partial()
        sys.exit(1)

    except Exception:
        logging.exception("Unexpected error during inference")
        save_partial()
        sys.exit(1)

    # --- assemble final results & accuracy ---
    df["model_output"] = outputs
    df["match_answer"] = matches
    df["is_correct"] = df.apply(
        lambda r: normalize(r["match_answer"]) == normalize(r["solution"]), axis=1
    )

    total   = len(df)
    correct = df["is_correct"].sum()
    logging.info("Final accuracy: %s (%d/%d)", f"{correct/total:.2%}", correct, total)
    print(f"Accuracy: {correct/total:.2%} ({correct}/{total})")

    # drop problem to slim CSV
    df = df.drop(columns=["problem"])
    df.to_csv(args.output_csv, index=False)
    logging.info("Saved complete results to '%s'", args.output_csv)

if __name__ == "__main__":
    main()