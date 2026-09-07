# llm_judge.py
import asyncio
import json
import os
import re
import threading
import time
from collections import deque
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()


class RollingRateLimiter:
    """
    Enforces both a per-minute and a per-hour request cap using rolling windows.
    Safer than fixed time.sleep() pacing because it reacts to the ACTUAL
    request history instead of assuming a constant interval.
    """

    def __init__(self, max_per_minute: int, max_per_hour: int):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self._minute_window = deque()
        self._hour_window = deque()
        self._lock = threading.Lock()

    def _sleep_needed(self) -> float:
        now = time.time()
        while self._minute_window and now - self._minute_window[0] > 60:
            self._minute_window.popleft()
        while self._hour_window and now - self._hour_window[0] > 3600:
            self._hour_window.popleft()

        wait = 0.0
        if len(self._minute_window) >= self.max_per_minute:
            wait = max(wait, 60 - (now - self._minute_window[0]) + 0.25)
        if len(self._hour_window) >= self.max_per_hour:
            wait = max(wait, 3600 - (now - self._hour_window[0]) + 0.25)
        return wait

    def acquire_sync(self):
        with self._lock:
            wait = self._sleep_needed()
        if wait > 0:
            print(f"⏳ Rate limit guard: sleeping {wait:.1f}s")
            time.sleep(wait)
        with self._lock:
            now = time.time()
            self._minute_window.append(now)
            self._hour_window.append(now)

    async def acquire_async(self):
        wait = self._sleep_needed()
        if wait > 0:
            print(f"⏳ Rate limit guard: sleeping {wait:.1f}s")
            await asyncio.sleep(wait)
        now = time.time()
        self._minute_window.append(now)
        self._hour_window.append(now)


class NvidiaJudgeModel(DeepEvalBaseLLM):
    """
    Custom DeepEval LLM judge routing through Baseten's OpenAI-compatible endpoint.
    Tuned for gpt-oss-120b on Baseten (15 req/min, 203 req/hour).
    """

    # Safety margins under Baseten's hard caps (15/min, 203/hour)
    MAX_PER_MINUTE = 12
    MAX_PER_HOUR = 190

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        reasoning_effort: str = "low", 
        max_tokens: int = 2048,          
    ):
        self.model_name = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://inference.baseten.co/v1"
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=600.0)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=600.0)

        self._semaphore = None
        self._limiter = RollingRateLimiter(self.MAX_PER_MINUTE, self.MAX_PER_HOUR)

    def get_semaphore(self):
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(1)
        return self._semaphore

    def load_model(self):
        return self.client

    def _build_messages(self, prompt: str, schema=None):
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
        return [{"role": "user", "content": prompt}]

    def _call_api(self, prompt: str, schema=None) -> str:
        messages = self._build_messages(prompt, schema)

        for attempt in range(8):
            self._limiter.acquire_sync()
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    top_p=1,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                    stream=False,
                )
                reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
                if reasoning:
                    print("[Reasoning]", reasoning[:300], "...")
                return completion.choices[0].message.content or ""
            except Exception as e:
                if "429" in str(e) or "RateLimit" in type(e).__name__:
                    wait_time = (attempt + 1) * 6
                    print(f"⚠️ Rate limit hit (429). Retrying in {wait_time}s... "
                          f"(attempt {attempt+1}/8)")
                    time.sleep(wait_time)
                else:
                    raise e

        raise RuntimeError("NvidiaJudgeModel: Exceeded maximum rate limit retries.")

    def generate(self, prompt: str, schema=None):
        content = self._call_api(prompt, schema=schema)
        return self._parse(content, schema)

    async def _async_call_api(self, prompt: str, schema=None) -> str:
        messages = self._build_messages(prompt, schema)
        sem = self.get_semaphore()

        async with sem:
            for attempt in range(8):
                await self._limiter.acquire_async()
                try:
                    completion = await self.async_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,
                        top_p=1,
                        max_tokens=self.max_tokens,
                        reasoning_effort=self.reasoning_effort,
                        stream=False,
                    )
                    reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
                    if reasoning:
                        print("[Async Reasoning]", reasoning[:300], "...")
                    return completion.choices[0].message.content or ""
                except Exception as e:
                    if "429" in str(e) or "RateLimit" in type(e).__name__:
                        wait_time = (attempt + 1) * 6
                        print(f"⚠️ Async Rate limit hit (429). Retrying in {wait_time}s... "
                              f"(attempt {attempt+1}/8)")
                        await asyncio.sleep(wait_time)
                    else:
                        raise e

        raise RuntimeError("NvidiaJudgeModel Async: Exceeded maximum rate limit retries.")

    async def a_generate(self, prompt: str, schema=None):
        content = await self._async_call_api(prompt, schema=schema)
        return self._parse(content, schema)

    def _parse(self, content: str, schema):
        if schema is None:
            return content, 0.0

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
                f"NvidiaJudgeModel: failed to parse schema '{schema.__name__}'.\n"
                f"--- Raw response ---\n{content}\n--- Error ---\n{e}"
            )

    def get_model_name(self) -> str:
        return self.model_name

nvidia_judge = NvidiaJudgeModel(reasoning_effort="low", max_tokens=2048)