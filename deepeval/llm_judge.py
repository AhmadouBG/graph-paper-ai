import os
import re
import json
import asyncio
from typing import Optional, Tuple, Type
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()


class NvidiaJudgeModel(DeepEvalBaseLLM):
    """
    Custom DeepEval LLM judge that routes through the NVIDIA-hosted
    OpenAI-compatible endpoint (openai/gpt-oss-120b).
    """

    def __init__(self, model: str = "openai/gpt-oss-120b"):
        self.model_name = model
        api_key = os.getenv("OPENAI_API_KEY_NVDIA")
        base_url = "https://integrate.api.nvidia.com/v1"
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=300.0)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=180.0)

    def load_model(self):
        return self.client

    def _call_api(self, prompt: str, schema=None) -> str:
        """
        Raw API call.
        When `schema` is provided, prepends a strict system instruction so the
        model returns a bare JSON object — no response_format param used because
        the NVIDIA endpoint ignores / breaks on it for reasoning models.
        """
        messages = []

        if schema is not None:
            # Build field list so the model knows the exact keys to emit
            try:
                fields = list(schema.model_fields.keys())
                field_hint = ", ".join(f'"{f}"' for f in fields)
            except AttributeError:
                field_hint = "(see context)"

            system_instruction = (
                "IMPORTANT: Your entire reply must be a single valid JSON object. "
                "Do NOT include any explanation, chain-of-thought, markdown fences, "
                "or text outside the JSON object. "
                f"The JSON object must have exactly these keys: {field_hint}."
            )
            # Prepend the JSON instruction to the user prompt directly
            # (some endpoints ignore the system role)
            prompt = system_instruction + "\n\n" + prompt

        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=1,
            top_p=1,
            max_tokens=4096,
            stream=False,
        )

        # Surface chain-of-thought reasoning when available
        reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
        if reasoning:
            print("[Reasoning]", reasoning)

        return completion.choices[0].message.content or ""

    def generate(self, prompt: str, schema=None):
        """
        Synchronous generation.

        - If `schema` is a Pydantic model class (used by DeepEval synthesizer /
          metrics internally), the model is prompted for JSON and the response
          is validated against the schema, returning a model instance.
        - Otherwise returns (str, float) as expected by plain metric calls.
        """
        content = self._call_api(prompt, schema=schema)

        if schema is not None:
            stripped = content.strip()

            # 1. Strip markdown fences ```json ... ``` or ``` ... ```
            if stripped.startswith("```"):
                stripped = stripped.split("```", 2)[-1]
                stripped = stripped.rsplit("```", 1)[0].strip()
                if "\n" in stripped:
                    first, rest = stripped.split("\n", 1)
                    if not first.strip().startswith("{"):
                        stripped = rest.strip()

            # 2. Regex fallback: grab the first {...} block from mixed-text responses
            if not stripped.startswith("{"):
                match = re.search(r"\{.*\}", stripped, re.DOTALL)
                if match:
                    stripped = match.group(0)

            try:
                data = json.loads(stripped)
                return schema(**data)
            except Exception as e:
                raise ValueError(
                    f"NvidiaJudgeModel: failed to parse schema '{schema.__name__}'.\n"
                    f"--- Raw response ---\n{content}\n--- Error ---\n{e}"
                )

        # Plain call — return (text, cost) tuple
        return content, 0.0

    async def _async_call_api(self, prompt: str, schema=None) -> str:
        """Async API call for concurrent DeepEval metric evaluation."""
        messages = []

        if schema is not None:
            try:
                fields = list(schema.model_fields.keys())
                field_hint = ", ".join(f'"{f}"' for f in fields)
            except AttributeError:
                field_hint = "(see context)"

            system_instruction = (
                "IMPORTANT: Your entire reply must be a single valid JSON object. "
                "Do NOT include any explanation, chain-of-thought, markdown fences, "
                "or text outside the JSON object. "
                f"The JSON object must have exactly these keys: {field_hint}."
            )
            prompt = system_instruction + "\n\n" + prompt

        messages.append({"role": "user", "content": prompt})

        completion = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=1,
            top_p=1,
            max_tokens=4096,
            stream=False,
        )

        reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
        if reasoning:
            print("[Async Reasoning]", reasoning)

        return completion.choices[0].message.content or ""

    async def a_generate(self, prompt: str, schema=None):
        """Asynchronous generation for DeepEval async evaluation."""
        content = await self._async_call_api(prompt, schema=schema)

        if schema is not None:
            stripped = content.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("```", 2)[-1]
                stripped = stripped.rsplit("```", 1)[0].strip()
                if "\n" in stripped:
                    first, rest = stripped.split("\n", 1)
                    if not first.strip().startswith("{"):
                        stripped = rest.strip()

            if not stripped.startswith("{"):
                match = re.search(r"\{.*\}", stripped, re.DOTALL)
                if match:
                    stripped = match.group(0)

            try:
                data = json.loads(stripped)
                return schema(**data)
            except Exception as e:
                raise ValueError(
                    f"NvidiaJudgeModel Async: failed to parse schema '{schema.__name__}'.\n"
                    f"--- Raw response ---\n{content}\n--- Error ---\n{e}"
                )

        return content, 0.0

    def get_model_name(self) -> str:
        return self.model_name


# Singleton judge instance — imported by other deepeval files
bedrock_judge = NvidiaJudgeModel()