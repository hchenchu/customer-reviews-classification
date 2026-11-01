"""LLM sentiment and topic classification for Toys, Toys, Toys Inc."""

import json
import os
import re

import requests
import streamlit as st

# Keep the same Groq integration as the pizza app.
BUSINESS = "Toys, Toys, Toys Inc."
CATEGORIES = ("price", "quality", "variety", "delivery", "service")
SENTIMENTS = ("positive", "negative")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
URL = "https://api.groq.com/openai/v1/chat/completions"

GUIDE = """Classify a customer comment about toys sold by Toys, Toys, Toys Inc. using the following rules:

1. sentiment: "positive" or "negative". If the intent cannot be determined,
   use "unclear". Sarcasm is negative. If praise and criticism both appear,
   classify sentiment based on what the customer is unhappy about.
2. category: exactly ONE primary topic:
   - "price": value for money, affordability, cost, or perceived value.
   - "quality": toy durability, safety, materials, reliability, workmanship.
   - "variety": range of toys, product selection, themes, types, or options.
   - "delivery": how fast or late the order is delivered.
   - "service": refunds, returns, support, order problems, or tracking shipments.
   If no primary category can be inferred, use "unclear".
   When multiple topics are present, prioritize the customer's complaint;
   otherwise choose the most emphasized topic.
   Distinguish slow delivery (delivery) from poor shipment tracking (service).
Return ONLY a JSON object with these exact keys:
{"sentiment": "positive", "category": "quality"}
Do not add explanations or extra keys."""


def _parse_classification(answer):
    """Validate LLM output and force unsupported values to 'unclear'."""
    try:
        raw = answer.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
    except (ValueError, TypeError):
        return {
            "sentiment": "unclear",
            "category": "unclear",
        }

    sentiment = str(result.get("sentiment", "unclear")).strip().lower()
    category = str(result.get("category", "unclear")).strip().lower()
    return {
        "sentiment": sentiment if sentiment in SENTIMENTS else "unclear",
        "category": category if category in CATEGORIES else "unclear",
    }


def label_comment(text):
    """Return validated sentiment and primary topic for a customer comment."""
    if not text or not text.strip():
        raise ValueError("A written comment is required")
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError) as exc:
        raise RuntimeError("Set GROQ_API_KEY in Streamlit secrets") from exc

    prompt = f"You label customer comments for {BUSINESS}.\n{GUIDE}\n\nComment: {text.strip()}"
    response = requests.post(
        URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    return _parse_classification(answer)
