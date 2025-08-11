# sec-llm-bench/huggingface.py

import csv
import time
import os
from huggingface_hub import InferenceClient

# Load environment variables from a .env file if present
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass
import csv
import time
import os

# --- Hugging Face authentication via .env ---
# .env can contain HUGGINGFACE_API_KEY or HF_TOKEN
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise EnvironmentError(
        "Missing Hugging Face token. Set HUGGINGFACE_API_KEY in your environment or .env file.")


MODEL = os.getenv("HF_MODEL")
MODEL_ROOT = os.getenv("HF_MODEL_ROOT", MODEL.replace("/", "-").replace("@", "-").replace(":", "-").lower())

# MODEL = "deepseek-ai/DeepSeek-V3"
# MODEL_ROOT = "deepseek-v3"

client = InferenceClient(
    provider=os.getenv("HF_PROVIDERS", "auto"),
    token=HF_TOKEN,
)

# --- API Request Function (now returns the response string) ---


def send_prompt(prompt_content: str) -> str | None:
    """Sends a prompt and returns the full response string, or None on error."""

    # Calling Huggingface API
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        temperature=float(os.getenv("TEMPERATURE", 0.1)),
        top_p=float(os.getenv("TOP_P", 1.0)),
        max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", 1024)),
    )

    return (response.choices[0].message)

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
                writer.writerow({"#": count, "response": out.content})
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
