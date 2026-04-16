import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def interpret_prediction(features: dict, prediction: float) -> str:
    prompt = f"""
You are a real estate expert.

Explain the predicted house price based on features.

Features:
{features}

Predicted price:
{prediction}

Guidelines:
- Explain if price is high or low
- Mention key drivers (size, quality, location)
- Keep it simple and clear
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content.strip()