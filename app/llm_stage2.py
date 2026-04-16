import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def interpret_prediction(features: dict, prediction: float) -> str:
    prompt = f"""
You are a real estate expert.

House features:
{features}

Predicted price:
{prediction}

Dataset reference:
- Median price: 180000
- Typical range: 40000 to 750000

Tasks:
1. Say if price is below, around, or above median
2. Explain key drivers (size, quality, location)
3. Keep explanation simple

Answer clearly:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content.strip()