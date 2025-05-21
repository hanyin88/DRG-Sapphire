import re
import pandas as pd

# 1. Read the CSV file and extract the "Description" column.
input_file = "open-r1/data/from_old/DRG_34.csv"
df = pd.read_csv(input_file, sep="\t", header=0)
# Convert the Description column to a list of normalized strings.
descriptions = df["Description"].astype(str).tolist()
normalized_descriptions = [desc.upper().strip() for desc in descriptions]

def format_reward(completions, **kwargs):
    """
    Reward function that checks if the entire content is enclosed as:
       <think>...</think>
       <answer>...</answer>
    in that exact order (with optional whitespace in between).

    Returns:
      -1 if no match,
       0.0 if there's a match.
    """

    # Use the new pattern to prevent some reward hacking.
    pattern = r"^\s*<think>.*?</think>\s*<answer>(?:(?!<answer>|</answer>).)*?</answer>\s*$"
    format_scores = []

    for comp in completions:
        content = comp[0]["content"]
        # Check if the content matches the entire pattern.
        if re.fullmatch(pattern, content, re.DOTALL | re.MULTILINE):
            format_scores.append(0.0)   # match => 0.0
        else:
            format_scores.append(-2)    # no match => -2

    return format_scores

def accuracy_reward(completions, solution, **kwargs):
    """
    Evaluates the accuracy of predicted DRG codes against the gold solution.
    Returns a list of floats (the accuracy scores), one per completion.
    
    Steps:
      1. If the content does NOT match the entire pattern:
            ^<think>.*?</think>\s*<answer>.*?</answer>$
         then return 0.0 immediately.
    
      2. Otherwise, extract the text from <answer>...</answer>.
         First, check whether the extracted answer (normalized to uppercase) exactly matches one
         element of the normalized Description column. If it does not, return -1.5.
    
      3. If the predicted answer is empty, return -1.5.
    
      4. Then, if the entire gold solution is found in the predicted answer, return 2.0.
    
      5. Else, parse the code into a principal and status part:
            - If the principal is found in the predicted answer => 1.5
            - Else if the status is found => 0.5
            - Else => -0.5
    Possible final scores: 2.0, 1.5, 0.5, -0.5, 0.0, or -1.5.
    """
    # Pattern for an entire string with <think>...</think> followed by <answer>...</answer>.
    pattern = r"^\s*<think>.*?</think>\s*<answer>(?:(?!<answer>|</answer>).)*?</answer>\s*$"
    
    def parse_code(code):
        """
        Attempts to split the code into a principal diagnosis and a status part
        using the keywords "WITH" or "WITHOUT".
        Returns (principal, status) if successful; otherwise, returns None.
        """
        code = code.upper().strip()
        if code == "NAN":
            return None
        
        # Look for a status keyword: "WITH" or "WITHOUT"
        if re.search(r'\b(WITH|WITHOUT)\b', code, re.IGNORECASE):
            split_pattern = re.compile(r'\s+(WITH|WITHOUT)\s+', re.IGNORECASE)
            parts = split_pattern.split(code, maxsplit=1)
            if len(parts) >= 3:
                principal = parts[0].strip()
                status = (parts[1] + " " + parts[2]).strip()  # e.g., "WITH xyz"
                return principal, status
            else:
                return None
        else:
            return None

    accuracy_scores = []

    for comp, sol in zip(completions, solution):
        content = comp[0]["content"]
        sol_norm = sol.upper().strip()
        
        # 1. Check if content matches the entire <think>...</think><answer>...</answer> pattern.
        if not re.fullmatch(pattern, content, re.DOTALL | re.MULTILINE):
            accuracy_scores.append(0.0)
            continue
        
        # 2. Extract the text within <answer>...</answer>.
        match_answer = re.search(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL | re.IGNORECASE)
        if not match_answer:
            accuracy_scores.append(0.0)
            continue
        
        predicted = match_answer.group(1).strip()
        predicted_norm = predicted.upper()
        
        # 2a. Check if the predicted answer is empty.
        if not predicted_norm:
            accuracy_scores.append(-1.5)
            continue
        
        # 2b. First check: must exactly match one element of the normalized descriptions.
        if predicted_norm not in normalized_descriptions:
            accuracy_scores.append(-1.5)
            continue
        
        # 3a. Full match: if the entire gold solution is found in predicted, return 2.0.
        if sol_norm in predicted_norm:
            accuracy_scores.append(2.0)
            continue
        
        # # 3b. Partial matching: attempt to parse the code and then match principal or status.
        # parsed = parse_code(sol_norm)
        # if parsed is not None:
        #     sol_principal, sol_status = parsed
        #     if sol_principal and (sol_principal in predicted_norm):
        #         accuracy_scores.append(1.5)
        #     elif sol_status and (sol_status in predicted_norm):
        #         accuracy_scores.append(0.5)
        #     else:
        #         accuracy_scores.append(-0.5)
        # else:
        #     accuracy_scores.append(-0.5)
        
        else:
            # Fallback: if no specific condition is met, append a default reward (e.g., -0.5)
            accuracy_scores.append(0.0)

    return accuracy_scores

# # Example usage:
# completions = [ [{"content": "<think>Some thoughts</think><answer>EXAMPLE DESCRIPTION</answer>"}] ]
# solution = ["example description"]
# scores = accuracy_reward(completions, solution)
# print(scores)
