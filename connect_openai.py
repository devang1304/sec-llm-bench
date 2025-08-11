# connect_openai.py

import csv
import time
import os
from openai import OpenAI
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass
import csv
import time
import os

# --- OpenAI authentication via .env ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "Missing OpenAI API key. Set OPENAI_API_KEY in your environment or .env file.")


# --- Choose Model configuration ---
MODEL = os.getenv("OPENAI_MODEL")

# --- Model root ---
MODEL_ROOT = MODEL

# --- Initialize OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY)


def send_prompt(prompt_content: str) -> str | None:
    """Sends a prompt and returns the full response string, or None on error."""

    response = client.responses.create(
        model=MODEL,
        input=prompt_content,
        # temperature=float(os.getenv("TEMPERATURE", 0.1)),
        top_p=float(os.getenv("TOP_P", 1.0)),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", 1024)),
    )
    return response.output_text


# -----------------------
# Batch runner
# -----------------------


TASK = ["data/mcq.csv", "data/rcm.csv",
        "data/vsp.csv", "data/ate.csv", "data/taa.csv"]

# Create directory for results if it doesn't exist
os.makedirs(f"results_{MODEL_ROOT}", exist_ok=True)

output_headers = ["#", "response"]
LOG_FILE = "run.log"

for input_csv in TASK:
    print(f"Processing file: {input_csv}...")
    start_time = time.time()

    output_csv = f"results_{MODEL_ROOT}/results_{input_csv[5:8]}_{MODEL_ROOT}.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(input_csv, mode="r", newline="", encoding="utf-8") as infile, \
            open(output_csv, mode="w", newline="", encoding="utf-8") as outfile:

        print(
            f"Reading from '{infile.name}' and saving to '{outfile.name}'...")
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=output_headers)
        writer.writeheader()

        count = 0
        for row in reader:
            count += 1
            prompt_text = row["prompt_text"]
            print(f"Processing prompt: {infile.name} - {count}...")

            out = send_prompt(prompt_text)
            if out is not None:
                writer.writerow({"#": count, "response": out.strip()[0]})
            else:
                writer.writerow({"#": count, "response": "Error"})

            time.sleep(0.2)  # gentle rate limiting

    elapsed = time.time() - start_time
    print(
        f"Finished {count} prompts from '{infile.name}' -> '{outfile.name}' in {elapsed:.2f}s.")
    with open(LOG_FILE, "a") as logf:
        logf.write(
            f"Processed {count} prompts by {MODEL} from '{input_csv}' in {elapsed:.2f} seconds.\n")

print("Processing complete.")
