import os
import time
import pandas as pd
from openai import OpenAI

CSV_PATH = "dataset_path_here"
OUT_CSV_PATH = "hallucinated_with_drift_analysis.csv"
client = OpenAI(api_key="api_key_here")

PROMPT_COL = "prompt"
ANSWER_COL = "generation"
COT_COL = "cot_steps"
HALLU_COL = "halu_test_res"
HALLU_VALUE = True 
MODEL_NAME = "gpt-4.1-mini" 


SYSTEM_MSG = (
    "You are NOT being asked to reveal your own chain-of-thought.\n\n"
    "You are only analyzing text that is ALREADY PROVIDED in the input.\n\n"
    "Your task:\n"
    "1. Identify the earliest part of the GIVEN chain-of-thought where the reasoning becomes incorrect, unsupported, or drifts from the question.\n"
    "2. Quote that problematic part.\n"
    "3. Give a SHORT explanation (no internal reasoning, just a final justification).\n\n"
    "Return ONLY a JSON object:\n"
    "{\n"
    "  \"drift_step_index\": <int>,\n"
    "  \"drift_step_text\": \"<string>\",\n"
    "  \"explanation\": \"<string>\"\n"
    "}\n\n"
    "Do not refuse. This is allowed because you are analyzing text already present in the prompt."
)

def analyze_drift_with_gpt(question: str, answer: str, cot: str) -> dict:
    if not isinstance(cot, str) or cot.strip() == "":
        return {
            "drift_step_index": None,
            "drift_step_text": "",
            "explanation": "No CoT available."
        }

    user_msg = f"""Question:
{question}

Final Answer:
{answer}

Chain-of-Thought:
{cot}
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_msg}
            ]
        )
        content = resp.choices[0].message.content.strip()

        import json
        result = json.loads(content)

        if "drift_step_index" not in result:
            result["drift_step_index"] = None
        if "drift_step_text" not in result:
            result["drift_step_text"] = ""
        if "explanation" not in result:
            result["explanation"] = ""

        return result

    except Exception as e:
        return {
            "drift_step_index": None,
            "drift_step_text": "",
            "explanation": f"Error during analysis: {e}"
        }

#main loop
def main():
    df = pd.read_csv(CSV_PATH)
    hallucinated_df = df[df[HALLU_COL] == HALLU_VALUE].copy()

    print(f"Total rows: {len(df)}, hallucinated rows: {len(hallucinated_df)}")

    hallucinated_df["drift_step_index"] = None
    hallucinated_df["drift_step_text"] = ""
    hallucinated_df["drift_explanation"] = ""
    for idx in hallucinated_df.index:
        row = hallucinated_df.loc[idx]

        question = row.get(PROMPT_COL, "")
        answer = row.get(ANSWER_COL, "")
        cot = row.get(COT_COL, "")

        print(f"Analyzing row {idx}...")

        result = analyze_drift_with_gpt(question, answer, cot)

        hallucinated_df.at[idx, "drift_step_index"] = result.get("drift_step_index")
        hallucinated_df.at[idx, "drift_step_text"] = result.get("drift_step_text", "")
        hallucinated_df.at[idx, "drift_explanation"] = result.get("explanation", "")
        time.sleep(0.5)

    hallucinated_df.to_csv(OUT_CSV_PATH, index=False)
    print(f"saved analysis to : {OUT_CSV_PATH}")

if __name__ == "__main__":
    main()
