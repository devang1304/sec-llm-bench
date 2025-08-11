# cti-code/gcp_vertexai.py

import os
import csv
import time
from typing import Optional

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# Model Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")

MODEL = os.getenv("GCP_MODEL")

# Derive a filesystem-friendly root for outputs
def _sanitize_model_root(name: str) -> str:
    root = name.lower().replace("/", "-").replace("@", "-").replace(":", "-").replace(".", "_")
    return "".join(ch for ch in root if ch.isalnum() or ch in ("-", "_"))

MODEL_ROOT = os.getenv("MODEL_ROOT", _sanitize_model_root(MODEL))


# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=REGION)

# Build a singleton model instance to avoid re-instantiation per prompt
_MODEL_INSTANCE: Optional[GenerativeModel] = None


def _get_model() -> GenerativeModel:
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = GenerativeModel(model_name=MODEL)
    return _MODEL_INSTANCE


def _generation_config() -> GenerationConfig:
    return GenerationConfig(
        temperature=float(os.getenv("TEMPERATURE", 0.1)),
        top_p=float(os.getenv("TOP_P", 1.0)),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", 1024)),
        response_mime_type="text/plain",
    )


def _safety_settings() -> list[SafetySetting]:
    # Match user's preference: BLOCK_NONE across categories
    thresholds = HarmBlockThreshold.BLOCK_NONE
    return [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=thresholds),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=thresholds),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=thresholds),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=thresholds),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_UNSPECIFIED, threshold=thresholds),
    ]

# --- API Request Function (returns the response string) ---


def send_prompt(prompt_content: str, *, stream: bool = True) -> str | None:
    """Sends a prompt with optional SYS_PROMPT and returns the full response text, or None on error."""
    try:
        model = _get_model()
        gen_cfg = _generation_config()
        safety_cfg = _safety_settings()

        response = model.generate_content(
            prompt_content,
            generation_config=gen_cfg,
            safety_settings=safety_cfg,
            stream=False,
        )
        return response.text
    except Exception as e:
        print(
            f"\n--- GEMINI SDK ERROR for prompt: {prompt_content[:80]}... ---\n{e}")
        return None

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
