# SEC-LLM-Bench

A minimal benchmarking framework for evaluating security-focused LLMs across five tasks:
- MCQ – Multiple Choice Questions
- RCM – Response Classification & Mapping
- VSP – Vulnerability Span Prediction
- ATE – ATT&CK Technique Extraction
- TAA – Technique Attribution Accuracy

## Repo layout
```
/data/                    # prompt files (per task)
/responses/<model>/<...>  # CSV outputs from runs
connect_hf.py             # Hugging Face batch runner
connect_openai.py         # OpenAI batch runner
connect_vertexai.py       # Vertex AI (Gemini) batch runner
connect_aiplatform.py     # GCP AI Platform batch runner
evaluate.py               # aggregates + scores responses from above
.env.example              # Example environment variables
README.md
requirements.txt
```
## Prerequisites
- Python 3.10–3.12
- Accounts/quotas on relevant providers
- Network access to model endpoints

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data & outputs
- Prompts live in: `/data`
- Outputs saved as: `/responses/<model-name>/<model-name>_<task>.csv`
- Example: `/responses/llama-3.1/llama-3.1_mcq.csv`

## Tasks
- MCQ – multiple-choice questions
- RCM – response classification/mapping
- VSP – vulnerability span prediction
- ATE – ATT&CK technique extraction
- TAA – technique attribution accuracy

## Credentials & configuration
- Copy .env.example → .env and fill in your keys/project info.
- OpenAI – API key required
- Hugging Face – API key required
- Vertex AI & GCP AI Platform –
    - GCP_PROJECT_ID, GCP_REGION
    - GCP Application Default Credentials (ADC) required: `gcloud auth application-default login` <br>
        or set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON.
## Evaluation
```bash
python evaluate.py
python evaluate.py --models gpt-5,deepseek-v3 --tasks mcq,vsp
python evaluate.py --output evaluation_results.csv
```

## References

This repository is based on ideas and methodology from the following paper:

> Md Tanvirul Alam, Dipkamal Bhusal, Le Nguyen, Nidhi Rastogi. "CTIBench: A Benchmark for Evaluating LLMs in Cyber Threat Intelligence" *NeurIPS 2024*.  
> [Paper Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/5acd3c628aa1819fbf07c39ef73e7285-Paper-Datasets_and_Benchmarks_Track.pdf)

We thank the authors for their work and open-source contributions.