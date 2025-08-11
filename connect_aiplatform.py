# connect_aiplatform.py

import os
import csv
import json
import requests
from google.auth import default
from google.auth.transport.requests import Request
import time
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# GCP Model Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID")

# GCP Authentication
creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
if not creds.valid or creds.expired:
    creds.refresh(Request())
token = creds.token


# --- Llama 4 API Connectors ---
ENDPOINT = "us-east5-aiplatform.googleapis.com"
REGION = "us-east5"
MODEL = "meta/llama-4-maverick-17b-128e-instruct-maas"
MODEL_ROOT = "llama-4-maverick"

# --- Claude API Connectors ---
# MODEL="claude-sonnet-4@20250514"
# MODEL_ROOT="claude-sonnet-4"
# # Pick one region:
# REGION="global"
# REGION="us-east5"

# if REGION == "global":
#     ENDPOINT = f"aiplatform.googleapis.com"
# else:
#     ENDPOINT = f"{REGION}-aiplatform.googleapis.com"


# --- API Request Function (now returns the response string) ---

def send_prompt(prompt_content: str) -> str | None:
    """Sends a prompt and returns the full response string, or None on error."""

    # Calling GCP AI Platform Model
    root_url = (f"https://{ENDPOINT}/v1/projects/{PROJECT_ID}")

    if "claude" in MODEL:
        url = (f"{root_url}/publishers/anthropic/models/{MODEL}:streamRawPredict")
    elif "llama" in MODEL or "deepseek" in MODEL:
        url = (f"{root_url}/locations/{REGION}/endpoints/openapi/chat/completions")

    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json"}
    payload = {
        "model": (MODEL),
        "stream": True,
        "messages": [{"role": "user", "content": prompt_content}],
        "temperature": float(os.getenv("TEMPERATURE", 0.1)),
        "top_p": float(os.getenv("TOP_P", 1.0)),
        "max_output_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", 1024)),
    }

    try:
        response = requests.post(url, headers=headers,
                                 json=payload, stream=True)
        response.raise_for_status()

        response_parts = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line and raw_line.startswith("data:"):
                line = raw_line.removeprefix("data: ").strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    response_parts.append(content)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        # Join all the streamed parts into a single string and return it
        return "".join(response_parts)

    except requests.exceptions.RequestException as e:
        print(f"\n--- ERROR for prompt: {prompt_content[:50]}... --- \n{e}")
        return None  # Indicate failure

# -----------------------
# Batch runner
# -----------------------


TASK = ["data/mcq.csv", "data/rcm.csv",
        "data/vsp.csv", "data/ate.csv", "data/taa.csv"]

# Create directory for results if it doesn't exist
output_dir = f"responses/{MODEL_ROOT}"
os.makedirs(output_dir, exist_ok=True)

output_headers = ["#", "response"]
LOG_FILE = "run.log"

for input_csv in TASK:
    print(f"Processing file: {input_csv}...")
    start_time = time.time()

    output_csv = f"{output_dir}/{MODEL_ROOT}_{input_csv[5:8]}.csv"
    # os.makedirs(os.path.dirname(output_csv), exist_ok=True)

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
                writer.writerow({"#": count, "response": out.strip()})
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
