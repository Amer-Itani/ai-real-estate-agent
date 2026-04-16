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
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except:
        return {}


def build_prompt_v1(query: str) -> str:
    return f"""
You are an AI real estate assistant.

Extract structured features STRICTLY.

Return ONLY JSON.

Features:
{FEATURES}

Rules:
- Use null if missing
- Map quality: poor→Po, fair→Fa, average→TA, good→Gd, excellent→Ex
- Do NOT guess values

Query:
{query}
"""


def build_prompt_v2(query: str) -> str:
    return f"""
You are a smart real estate assistant.

Extract features EVEN if implicit.

Infer reasonable values if possible.

Return ONLY JSON.

Features:
{FEATURES}

Rules:
- If user says "good house" → OverallQual ~ 6–7
- If "large house" → higher sqft
- Still use null if truly unknown

Query:
{query}
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

    if score_v2 > score_v1:
        selected = data_v2
        version = "v2"
        score = score_v2
    else:
        selected = data_v1
        version = "v1"
        score = score_v1

    missing = [k for k in FEATURES if selected.get(k) is None]

    return {
        "extracted_features": selected,
        "missing_features": missing,
        "completeness_score": round(score, 2),
        "prompt_version": version,
    }