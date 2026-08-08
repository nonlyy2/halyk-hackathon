import json
import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
ALIBABA_CLOUD_API_KEY = os.environ.get("ALIBABA_CLOUD_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
# Gemini speaks the OpenAI protocol on this path, so it needs no transport of its own.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
OLLAMA_BASE_URL = "http://localhost:11434"

# Starting points only -- COVENANT_MODEL_SMALL / COVENANT_MODEL_COMPLEX in .env override any of
# them, so choosing a model never means editing this file. The pairs below are what each provider
# should be run with absent a reason to differ; see .env.example for the alternatives.
_MODEL_BY_BACKEND_AND_SIZE = {
    # Flash both ways on purpose: every Gemini Pro model reports `limit: 0` on the free tier,
    # so the choice is between Flash generations, not between Flash and Pro.
    ("gemini", "small"): "gemini-2.5-flash",
    ("gemini", "complex"): "gemini-3.5-flash",
    ("anthropic", "small"): "claude-haiku-4-5-20251001",
    ("anthropic", "complex"): "claude-sonnet-5",
    ("huggingface", "small"): "Qwen/Qwen3-8B:nscale",
    ("huggingface", "complex"): "openai/gpt-oss-120b:cerebras",  # qwen3.5-27b
    ("alibaba", "small"): "qwen3-8b",
    ("alibaba", "complex"): "qwen3-32b",
    # non-thinking instruct models on purpose: ollama serves qwen3 with a generic chatml template
    # (--no-jinja), so neither `think: false` nor `/no_think` reaches the model and it spends the
    # whole num_predict budget on invisible reasoning instead of answering.
    # 3b gets doc_type right but collapses `is_authoritative_for_covenants` to always-False (it
    # can't hold the 3-part conjunction), which starves every downstream stage -- so 7b for both.
    ("ollama", "small"): "qwen2.5:14b-instruct",
    ("ollama", "complex"): "qwen2.5:14b-instruct",
}

MAX_RETRIES = 4
DEFAULT_MAX_TOKENS = 4096


def _hf_extra_body(model: str) -> dict:
    # Qwen3 hybrid-thinking models on the HF router burn their whole output budget on invisible
    # chain-of-thought unless thinking is disabled via chat_template_kwargs. Other model families
    # (e.g. gpt-oss, which reports reasoning in a separate field) reject that param with a 400 --
    # so only send it for models that actually need it.
    if "qwen" in model.lower():
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def _select_backend() -> str:
    # COVENANT_BACKEND pins the provider regardless of which keys happen to be present: adding a
    # key for one provider otherwise silently re-tags every cache and orphans the existing runs.
    forced = os.environ.get("COVENANT_BACKEND")
    if forced:
        return forced
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if GEMINI_API_KEY:
        return "gemini"
    if HF_TOKEN:
        return "huggingface"
    if ALIBABA_CLOUD_API_KEY:
        return "alibaba"
    return "ollama"


@dataclass
class Client:
    backend: str
    model: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                if self.backend == "anthropic":
                    return self._complete_anthropic(system, user, max_tokens, temperature)
                if self.backend == "huggingface":
                    return self._complete_openai_compatible(
                        HF_ROUTER_BASE_URL,
                        HF_TOKEN,
                        system,
                        user,
                        max_tokens,
                        temperature,
                        extra_body=_hf_extra_body(self.model),
                    )
                if self.backend == "gemini":
                    return self._complete_openai_compatible(
                        GEMINI_BASE_URL,
                        GEMINI_API_KEY,
                        system,
                        user,
                        max_tokens,
                        temperature,
                    )
                if self.backend == "alibaba":
                    return self._complete_openai_compatible(
                        DASHSCOPE_BASE_URL,
                        ALIBABA_CLOUD_API_KEY,
                        system,
                        user,
                        max_tokens,
                        temperature,
                        extra_body={"enable_thinking": False},
                    )
                return self._complete_ollama(system, user, max_tokens, temperature)
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt >= MAX_RETRIES - 1:
                    break
                # A 429 is a quota window, not congestion: free tiers meter per minute, so backing
                # off for a couple of seconds just spends another attempt on the same refusal.
                # Honour Retry-After when the server sends one, otherwise wait out the window.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 429:
                    retry_after = exc.response.headers.get("retry-after")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else 20.0 * (attempt + 1)
                    )
                else:
                    delay = 2**attempt
                time.sleep(delay)
        raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts") from last_error

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> dict:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            # Retrying a malformed generation only helps if the retry can come out different, and at
            # temperature 0 decoding is greedy -- the same prompt returns the same broken bytes every
            # time, burning all MAX_RETRIES to no effect. Nudge the temperature up after the first
            # failure so a retry is actually a fresh sample; the first attempt stays reproducible.
            attempt_temperature = temperature if attempt == 0 else max(temperature, 0.1 * attempt)
            raw = self.complete(
                system, user, max_tokens=max_tokens, temperature=attempt_temperature
            ).strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw
                if raw.endswith("```"):
                    raw = raw.rsplit("```", 1)[0]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            # small models sometimes wrap a valid JSON object in leading/trailing commentary
            # despite being told not to -- parse just the object, not the surrounding text.
            start = raw.find("{")
            if start < 0:
                last_error = ValueError(f"no JSON object found in response: {raw!r}")
                continue
            try:
                return json.JSONDecoder().raw_decode(raw, start)[0]
            except json.JSONDecodeError as exc:
                # occasional malformed generation (not a network issue) -- a fresh completion
                # usually comes back well-formed, so retry the call itself, not just re-parse.
                last_error = exc
        raise RuntimeError(
            f"LLM returned unparseable JSON after {MAX_RETRIES} attempts"
        ) from last_error

    def _complete_anthropic(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _complete_openai_compatible(
        self,
        base_url: str,
        api_key: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        extra_body: dict | None = None,
    ) -> str:
        # temperature 0 (the default) gives greedy, reproducible decoding for the extraction and
        # classification stages; callers that want sampled diversity (spec self-consistency voting)
        # pass a nonzero temperature explicitly.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **(extra_body or {}),
        }
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        # content can be null when a model emits only reasoning / an empty turn -- coerce to ""
        # so callers get a retryable empty string instead of a None that crashes on .strip().
        return response.json()["choices"][0]["message"]["content"] or ""

    def _complete_ollama(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        # ollama defaults to a 4096-token context and silently truncates the prompt -- a 46KB
        # credit agreement would lose most of its text. Size the window to the actual prompt
        # (~3 chars/token) instead of allocating a fixed 32k KV cache on every small call.
        num_ctx = min(32768, max(8192, (len(system) + len(user)) // 3 + max_tokens + 512))
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {"num_predict": max_tokens, "temperature": temperature, "num_ctx": num_ctx},
        }
        response = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=600)
        response.raise_for_status()
        return response.json()["message"]["content"]


def get_client(size: str = "complex") -> Client:
    """The client for a tier, with the model overridable without touching code.

    COVENANT_MODEL_SMALL / COVENANT_MODEL_COMPLEX exist because the spec cache is keyed by model
    name: swapping models by editing this table meant the new name no longer matched the cached
    files, and every scenario silently fell through to its fallback answer instead of erroring.
    """
    backend = _select_backend()
    override = os.environ.get(f"COVENANT_MODEL_{size.upper()}")
    if override:
        return Client(backend=backend, model=override)

    model = _MODEL_BY_BACKEND_AND_SIZE.get((backend, size))
    if model is None:
        known = sorted({b for b, _ in _MODEL_BY_BACKEND_AND_SIZE})
        raise SystemExit(
            f"no default model for backend {backend!r} (tier {size!r}).\n"
            f"set COVENANT_MODEL_{size.upper()} in .env, "
            f"or use one of the backends with defaults: {', '.join(known)}"
        )
    return Client(backend=backend, model=model)
