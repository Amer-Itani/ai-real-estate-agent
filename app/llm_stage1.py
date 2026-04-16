import os
import json
from typing import Any, Dict

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearRemodAdd",
    "TotRmsAbvGrd",
    "Neighborhood",
    "KitchenQual",
    "ExterQual",
]


def call_llm(prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        print("\n=== RAW LLM OUTPUT ===")
        print(content)
        print("======================\n")

        return json.loads(content)

    except Exception as e:
        print("LLM ERROR:", e)
        return {}


def build_prompt_v1(query: str) -> str:
    return f"""
You are a data extraction system.

Extract features and return ONLY valid JSON.

NO explanation.
NO text.
NO markdown.

STRICT FORMAT:
{{
  "OverallQual": number or null,
  "GrLivArea": number or null,
  "GarageCars": number or null,
  "TotalBsmtSF": number or null,
  "FullBath": number or null,
  "YearRemodAdd": number or null,
  "TotRmsAbvGrd": number or null,
  "Neighborhood": string or null,
  "KitchenQual": one of [Po, Fa, TA, Gd, Ex] or null,
  "ExterQual": one of [Po, Fa, TA, Gd, Ex] or null
}}

Query:
{query}

Return JSON ONLY.
"""


def build_prompt_v2(query: str) -> str:
    return f"""
You are a structured data extractor.

Infer values when reasonable, but still return ONLY JSON.

Rules:
- good → Gd
- excellent → Ex
- average → TA

Same JSON format as below:

{{
  "OverallQual": number or null,
  "GrLivArea": number or null,
  "GarageCars": number or null,
  "TotalBsmtSF": number or null,
  "FullBath": number or null,
  "YearRemodAdd": number or null,
  "TotRmsAbvGrd": number or null,
  "Neighborhood": string or null,
  "KitchenQual": string or null,
  "ExterQual": string or null
}}

Query:
{query}

Return JSON ONLY.
"""


def evaluate_output(data: dict) -> float:
    if not data:
        return 0

    filled = sum(1 for k in FEATURES if data.get(k) is not None)
    return filled / len(FEATURES)


def extract_features_from_query(query: str) -> Dict[str, Any]:

    # Run both prompts
    data_v1 = call_llm(build_prompt_v1(query))
    data_v2 = call_llm(build_prompt_v2(query))

    score_v1 = evaluate_output(data_v1)
    score_v2 = evaluate_output(data_v2)

    # choose best prompt
    if score_v2 > score_v1:
        selected = data_v2
        version = "v2"
        score = score_v2
    else:
        selected = data_v1
        version = "v1"
        score = score_v1

    quality_map = {
        "Po": 2,
        "Fa": 4,
        "TA": 5,
        "Gd": 7,
        "Ex": 9
    }

    if isinstance(selected.get("OverallQual"), str):
        selected["OverallQual"] = quality_map.get(selected["OverallQual"], None)

    missing = [k for k in FEATURES if selected.get(k) is None]

    return {
        "extracted_features": selected,
        "missing_features": missing,
        "completeness_score": round(score, 2),
        "prompt_version": version,
    }