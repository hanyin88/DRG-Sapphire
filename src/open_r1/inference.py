import re
import argparse
import pandas as pd
from datasets import load_from_disk, DatasetDict
from vllm import LLM, SamplingParams
from trl.data_utils import maybe_apply_chat_template
from transformers import AutoTokenizer
import torch

def load_and_prepare_dataset(dataset_path: str) -> DatasetDict:
    """
    Loads a HuggingFace dataset from `dataset_path` and:
      - Appends the DRG introduction to the 'problem' field
      - Converts it to a DatasetDict with only a 'train' split
    """
    drg_intro = """
    MS-DRG (Medicare Severity Diagnosis-Related Groups) is a system used by the U.S. Centers for Medicare & Medicaid Services to classify hospital inpatient stays for payment purposes. It groups patients with similar clinical conditions and resource needs into categories to determine reimbursement amounts. Each MS-DRG is assigned based on the patient's principal diagnosis, secondary diagnoses, procedures performed, age, discharge status, and other factors. The goal is to ensure fair and consistent hospital reimbursement based on the severity of the illness and the complexity of care required.

    CC and MCC in MS-DRG
    CC (Complication or Comorbidity): A secondary diagnosis that increases the complexity of care and resource utilization.
    MCC (Major Complication or Comorbidity): A more severe secondary condition that has a significant impact on resource use and hospital reimbursement.
    MCCs have a greater effect on the DRG weight than CCs due to increased patient care complexity.

    """

    dataset = load_from_disk(dataset_path)
    # Append the drg_intro to the start of each problem
    dataset = dataset.map(lambda x: {"problem": f"{drg_intro}{x['problem']}"})
    # Convert it to a DatasetDict with only a 'train' split
    # dataset = DatasetDict({"train": dataset})
    return dataset

def extract_answer_if_valid(text: str) -> str:
    """
    Checks if the entire string matches this pattern:

        ^<think>.*?</think>\\s*<answer>.*?</answer>$

    If it does, extracts the text inside <answer>...</answer>.
    Otherwise returns None.
    """
    pattern_full = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    pattern_extract = r"<answer>\s*(.*?)\s*</answer>"

    if not re.match(pattern_full, text, flags=re.DOTALL | re.MULTILINE):
        return None
    
    match = re.search(pattern_extract, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    extracted = match.group(1).strip()
    return extracted if extracted else None

def main():
    parser = argparse.ArgumentParser(
        description="Load a HF dataset, run vLLM inference, extract <answer> from outputs, and save CSV."
    )
    parser.add_argument("--dataset_path", type=str, default="data/all_test_dataset",
                        help="Path to the HuggingFace dataset on disk.")
    parser.add_argument("--model_name", type=str, default="outputs/Qwen2.5-7B-Instruct_qwenCOT_512",
                        help="HuggingFace or local model name/path for vLLM inference.")
    parser.add_argument("--output_csv", type=str, default="results/inference_results.csv",
                        help="Where to save the CSV with model outputs and extracted answers.")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Optionally limit the number of rows to process (for testing).")
    args = parser.parse_args()

    # 1) Load HF dataset
    dataset = load_and_prepare_dataset(args.dataset_path)
    ds = dataset["train"]

    # If there's no 'solution' column, create it as empty
    if "solution" not in ds.column_names:
        ds = ds.map(lambda x: {"solution": ""})

    # Convert to Pandas for easier iteration
    df = ds.to_pandas()
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    # 2) Build prompts for vLLM
    #    Provide a system prompt if needed; here's an example:
    system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"


    all_prompts = [
        maybe_apply_chat_template(
            {"prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["problem"]}
            ]},
            tokenizer
        )["prompt"]  # Extracts the processed prompt as a string
        for _, row in df.iterrows()
    ]

    # 3) Run vLLM inference
    llm = LLM(model=args.model_name,
            #   dtype = torch.bfloat16,
              gpu_memory_utilization=0.8,
              enable_prefix_caching=True,
              )
    sampling_params = SamplingParams(temperature=0.6, 
                                     min_tokens=10,
                                     max_tokens=4096,
                                     seed=43,
                                     top_p=0.95,
                                    
                                     )

    print(f"Running vLLM inference on {len(all_prompts)} prompts using model='{args.model_name}'...")
        
    outputs = llm.generate(
        all_prompts,
        sampling_params,
        # use_tqdm=True,
        # chat_template=chat_template,
    )

    # 4) Extract model outputs and <answer> blocks
    all_model_outputs = []
    match_answers = []
    for out in outputs:
        if out.outputs:
            text_out = out.outputs[0].text
        else:
            text_out = ""
        all_model_outputs.append(text_out)

        # Attempt to extract <answer> text
        extracted = extract_answer_if_valid(text_out)
        match_answers.append(extracted)

    df["model_output"] = all_model_outputs
    df["match_answer"] = match_answers
    
    # remove the column of problem to save place
    df = df.drop(columns=['problem'])
    

    # 5) Save to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"Saved inference results to '{args.output_csv}'.")

if __name__ == "__main__":
    main()
