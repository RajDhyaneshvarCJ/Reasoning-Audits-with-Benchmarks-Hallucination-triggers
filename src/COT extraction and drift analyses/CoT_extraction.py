import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os

API_KEY = "api-key-here"



client = OpenAI(api_key=API_KEY)

INPUT_FILE = "full_dataset.csv"       
OUTPUT_FILE = "CoT_dataset.csv"      
N = 4000
df = pd.read_csv(INPUT_FILE)
df = df.head(N)
def get_cot_for_row(prompt, generation):
    """
    CoT extraction.
    """
    msg = f"""
    You are evaluating a Q/A pair for reasoning quality.

    PROMPT:
    {prompt}

    ANSWER:
    {generation}

    Task:
    - Break down the reasoning behind the answer in 2–4 short numbered steps.
    - Focus on *what the model probably did* logically.
    - you must reason it out as if you are the model and why you gave the answer
    - MUST provide complete chain of thought.
    

    Return ONLY the steps.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": msg}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"ERROR: {e}"


# Create new column
cot_results = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    if i > N:
      break   
    cot = get_cot_for_row(row["prompt"], row["generation"])
    cot_results.append(cot)

df["cot_steps"] = cot_results
df.to_csv(OUTPUT_FILE, index=False)

print("\n cot dataset:", OUTPUT_FILE)
