import json
import yaml
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_from_disk

def main():
    # 1. Load paths from YAML
    with open("paths.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Load HF dataset from disk
    dataset = load_from_disk(config["all_train_dataset_path"])

    # 3. Convert to DataFrame
    df = pd.DataFrame(dataset['train'])

    # only use first 5 rows for now (optional)
    # df = df.head(5)

    drg_intro = '''
    MS-DRG (Medicare Severity Diagnosis-Related Groups) is a system used by the U.S. Centers for Medicare & Medicaid Services to classify hospital inpatient stays for payment purposes. It groups patients with similar clinical conditions and resource needs into categories to determine reimbursement amounts.

    Each MS-DRG is assigned based on the patient's principal diagnosis, secondary diagnoses, procedures performed, age, discharge status, and other factors. The goal is to ensure fair and consistent hospital reimbursement based on the severity of the illness and the complexity of care required.

    CC and MCC in MS-DRG
    CC (Complication or Comorbidity): A secondary diagnosis that increases the complexity of care and resource utilization.
    MCC (Major Complication or Comorbidity): A more severe secondary condition that has a significant impact on resource use and hospital reimbursement.
    MCCs have a greater effect on the DRG weight than CCs due to increased patient care complexity.\n
    '''

    # 4. Create prompts
    prompts = [
        [
            {
                "role": "user",
                "content": f"{drg_intro}{row['problem']}. The answer is {row['solution']}. Explain the reason why the DRG code is assigned. In your reasoning step, assume you don't know the right DRG code yet."
            }
        ]
        for _, row in df.iterrows()
    ]

    # 5. Initialize the LLM and sampling parameters
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048
    )

    # 6. Run inference
    outputs = llm.chat(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    # 7. Collect and attach outputs
    explanations = [r.outputs[0].text for r in outputs]
    df["explanation"] = explanations

    # 8. Save output to CSV
    df.to_csv(config["all_train_cot_csv_path"], index=False)

if __name__ == "__main__":
    main()
