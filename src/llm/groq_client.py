"""
Groq LLM Client
Wraps the Groq API with JSON-safe generation and error handling.
"""

import os
import json
import re
from typing import Optional
from groq import Groq


class GroqClient:
    """Unified interface to the Groq API using open-source LLMs."""

    MODEL_HEAVY = "llama-3.3-70b-versatile"   # deep reasoning
    MODEL_FAST  = "llama-3.1-8b-instant"       # quick inference

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API key is required.")
        self.client = Groq(api_key=api_key)

    # ------------------------------------------------------------------ #
    # Public helpers                                                        #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        system: str = "",
        model: str = "heavy",
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> str:
        """Generate a plain-text response."""
        model_id = self.MODEL_HEAVY if model == "heavy" else self.MODEL_FAST
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(f"Groq API error: {exc}") from exc

    def generate_json(
        self,
        prompt: str,
        system: str = "",
        model: str = "heavy",
        max_tokens: int = 2000,
    ) -> dict:
        """Generate and parse a JSON response — robust against markdown fences."""
        json_system = (
            system
            + "\n\nCRITICAL: Respond ONLY with valid JSON. "
            "No markdown, no backticks, no prose before or after."
        )
        raw = self.generate(prompt, json_system, model, max_tokens)
        return self._parse_json(raw)

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()

        # Strip ```json … ``` fences
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last-resort: extract first JSON object
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(
                f"Could not parse JSON from LLM response. First 300 chars:\n{text[:300]}"
            )
