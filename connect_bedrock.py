# connect_bedrock.py
# Batch runner for AWS Bedrock models using native invoke_model for ALL providers

import os
import csv
import time
import json

import boto3
from botocore.exceptions import ClientError

try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# --- AWS / Bedrock configuration via environment ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL = os.getenv("BEDROCK_MODEL_ID")
if not MODEL:
    raise EnvironmentError(
        "Missing Bedrock model id. Set BEDROCK_MODEL_ID in your environment or .env file."
    )
INFERENCE_PROFILE_ARN = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN")  # optional

# Inference params
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
TOP_P = float(os.getenv("TOP_P", 1.0))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", 1024))

# --- Model root for output folder naming ---
MODEL_ROOT = MODEL

# --- Initialize Bedrock Runtime client ---
client = boto3.client("bedrock-runtime", region_name=AWS_REGION)


# -----------------------
# Provider-native request builders
# -----------------------

def _build_native_request(model_id: str, prompt: str) -> dict:
    mid = model_id.lower()

    # Anthropic (Claude)
    if mid.startswith("anthropic."):
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        }

    # Meta Llama (uses messages)
    if mid.startswith("meta.") or "llama" in mid:
        # Bedrock Llama supports messages; parameters commonly named temperature/top_p/max_gen_len
        return {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_gen_len": MAX_OUTPUT_TOKENS,
        }

    # Mistral (chat)
    if mid.startswith("mistral."):
        # Mistral chat supports messages; params often max_tokens/temperature/top_p
        return {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }

    # Cohere Command (text)
    if mid.startswith("cohere."):
        # Cohere on Bedrock uses `message` for chat; keep simple single-turn
        return {
            "message": prompt,
            "temperature": TEMPERATURE,
            "p": TOP_P,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }

    # Amazon Titan Text (G1, Lite/Express)
    if mid.startswith("amazon.titan") or mid.startswith("amazon.titan-text"):
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "temperature": TEMPERATURE,
                "topP": TOP_P,
                "maxTokenCount": MAX_OUTPUT_TOKENS,
            },
        }

    # AI21 (Jamba/ Jurassic)
    if mid.startswith("ai21."):
        return {
            "prompt": prompt,
            "temperature": TEMPERATURE,
            "maxTokens": MAX_OUTPUT_TOKENS,
            "topP": TOP_P,
        }

    # Fallback: generic messages payload (many providers accept this)
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }


def _extract_text(model_id: str, model_response: dict) -> str:
    """Try common fields across providers to get the text output."""
    # Anthropic-style
    try:
        content = model_response.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            t = content[0].get("text")
            if isinstance(t, str):
                return t
    except Exception:
        pass

    # Titan-style
    try:
        results = model_response.get("results")
        if isinstance(results, list) and results:
            out = results[0].get("outputText")
            if isinstance(out, str):
                return out
    except Exception:
        pass

    # Cohere
    try:
        gens = model_response.get("generations")
        if isinstance(gens, list) and gens:
            out = gens[0].get("text")
            if isinstance(out, str):
                return out
    except Exception:
        pass

    # Meta Llama (some variants)
    for key in ("generation", "output"):
        try:
            val = model_response.get(key)
            if isinstance(val, str) and val:
                return val
        except Exception:
            pass

    # Bedrock Converse-like output (if ever returned here)
    try:
        blocks = (
            model_response.get("output", {})
            .get("message", {})
            .get("content", [])
        )
        if isinstance(blocks, list) and blocks:
            parts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
            if any(parts):
                return "".join(parts)
    except Exception:
        pass

    # Last resort: common text fields
    for key in ("text", "answer"):
        val = model_response.get(key)
        if isinstance(val, str):
            return val
    return ""


def send_prompt(prompt_content: str) -> str | None:
    """Send a single prompt using **native invoke_model** for ALL providers."""
    body_dict = _build_native_request(MODEL, prompt_content)
    request_body = json.dumps(body_dict)

    try:
        kwargs = {"body": request_body}
        if INFERENCE_PROFILE_ARN:
            kwargs["inferenceProfileArn"] = INFERENCE_PROFILE_ARN
        else:
            kwargs["modelId"] = MODEL

        response = client.invoke_model(**kwargs)
        model_response = json.loads(response["body"].read())
        return _extract_text(MODEL, model_response)
    except ClientError as e:
        print(f"Bedrock ClientError: {e}")
        return None
    except Exception as e:
        print(
            f"ERROR invoking Bedrock. If you see ValidationException about on-demand throughput, set BEDROCK_INFERENCE_PROFILE_ARN to an inference profile that contains {MODEL}. Original: {e}"
        )
        return None


# -----------------------
# Batch runner (unchanged)
# -----------------------

TASK = [
    "data/mcq.csv",
    "data/rcm.csv",
    "data/vsp.csv",
    "data/ate.csv",
    "data/taa.csv",
]

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
                writer.writerow({"#": count, "response": out.strip()})
            else:
                writer.writerow({"#": count, "response": "Error"})

            time.sleep(0.5)  # gentle rate limiting

    elapsed = time.time() - start_time
    print(
        f"Finished {count} prompts from '{infile.name}' -> '{outfile.name}' in {elapsed:.2f}s.")
    with open(LOG_FILE, "a") as logf:
        logf.write(
            f"Processed {count} prompts by {MODEL} from '{input_csv}' in {elapsed:.2f} seconds.\n"
        )

print("Processing complete.")