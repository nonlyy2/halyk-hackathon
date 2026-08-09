import json
import os
import sys
import threading
import time
from dataclasses import dataclass

import httpx
from dotenv import dotenv_values, find_dotenv, load_dotenv

load_dotenv()

_ENV_PATH = find_dotenv(usecwd=True)
_key_lock = threading.Lock()
_last_seen_key: dict[str, str] = {}


def api_key(*names: str) -> str | None:
    """The credential as it stands RIGHT NOW, re-read from .env on every call.

    A free-tier key is spent in a few hundred requests and then replaced, and a run outlives
    several of them. Reading the value once at import meant a key swapped into .env changed
    nothing until the process restarted -- and restarting mid-stage throws away whatever was in
    flight and re-spends the calls that produced it.

    The file wins over the process environment, because editing .env is how a key gets replaced.
    Re-reading costs one small file read per request, against a hundred-odd requests a run.
    """
    from_file = dotenv_values(_ENV_PATH) if _ENV_PATH else {}
    for name in names:
        value = from_file.get(name) or os.environ.get(name)
        if not value:
            continue
        with _key_lock:
            changed = _last_seen_key.get(name) not in (None, value)
            _last_seen_key[name] = value
        if changed:
            # A new key carries its own untouched allowance, so everything learned about which
            # models were spent applied to the old one and must not be held against this one.
            with _exhausted_lock:
                _exhausted.clear()
            print(f"  {name} changed -- exhausted-quota state cleared", flush=True)
        return value
    return None


def gemini_key() -> str | None:
    return api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")


HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
# A dedicated Alibaba workspace gets its own host, so the endpoint has to be configurable rather
# than fixed to the shared one. COVENANT_BASE_URL overrides whichever backend is selected, which
# also covers any other OpenAI-compatible gateway.
DASHSCOPE_BASE_URL = os.environ.get(
    "ALIBABA_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
).rstrip("/")
# Gemini speaks the OpenAI protocol on this path, so it needs no transport of its own.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
OLLAMA_BASE_URL = "http://localhost:11434"

# Starting points only -- COVENANT_MODEL_SMALL / COVENANT_MODEL_COMPLEX in .env override any of
# them, so choosing a model never means editing this file. The pairs below are what each provider
# should be run with absent a reason to differ; see .env.example for the alternatives.
_MODEL_BY_BACKEND_AND_SIZE = {
    # Flash-lite both ways on purpose. Every Gemini Pro model reports `limit: 0` on the free tier,
    # so the choice is only between Flash generations -- and of those, measured rather than
    # documented, gemini-3.1-flash-lite allows 500 requests a day while the rest allow 20. A run
    # makes on the order of a hundred calls, so anything metered at 20 cannot carry a single stage.
    ("gemini", "small"): "gemini-3.1-flash-lite",
    ("gemini", "complex"): "gemini-3.1-flash-lite",
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
# Seconds to leave between requests. Free tiers meter per minute, and reacting to a 429 after the
# fact is not enough: the stages fire several calls back to back, blow the window, and then spend
# the whole retry budget waiting for it to reopen. Pacing the requests avoids the refusal instead.
MIN_REQUEST_INTERVAL = float(os.environ.get("COVENANT_MIN_INTERVAL", "0"))
_last_request_at = 0.0
# Stages run several scenarios at once, so the pacing clock is shared state: without the lock two
# threads read the same `_last_request_at`, both decide they may go now, and the interval that
# exists to stay inside a per-minute quota stops holding.
_pace_lock = threading.Lock()

# Models whose quota was spent, and when. Asking a model whose daily allowance is gone wastes a
# request and a retry backoff on every call for the rest of the run, so the refusal is remembered --
# but only for a while. A permanent mark is worse than no mark at all: a day boundary crossing
# mid-run, a quota raised, or a refusal misread as daily all lock the pipeline out of a model that
# is actually answering, and with the alternates behind it also spent the run simply stops. Costing
# one probe per model per interval buys the guarantee that it can always recover.
EXHAUSTED_RETRY_AFTER = float(os.environ.get("COVENANT_EXHAUSTED_RETRY", "600"))
_exhausted: dict[str, float] = {}
_exhausted_lock = threading.Lock()


def _mark_exhausted_if_daily(model: str, exc: Exception) -> None:
    """A 429 is two different things: a per-minute window, which reopens in seconds, and a daily
    allowance, which does not reopen for hours. Only the second is worth remembering -- the
    providers say which in the error body."""
    body = getattr(getattr(exc, "response", None), "text", "") or ""
    if "PerDay" in body or "per day" in body.lower():
        with _exhausted_lock:
            _exhausted[model] = time.monotonic()


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
    if api_key("ANTHROPIC_API_KEY"):
        return "anthropic"
    if gemini_key():
        return "gemini"
    if api_key("HF_TOKEN"):
        return "huggingface"
    if api_key("ALIBABA_CLOUD_API_KEY"):
        return "alibaba"
    return "ollama"


@dataclass
class Client:
    """A tier's model, plus the models to fall through to when its quota runs out.

    Free tiers meter each model separately -- Gemini allows on the order of twenty requests per
    model per day -- so no single model can carry a run of a hundred-odd calls, while several
    together can. `model` stays fixed no matter which alternate actually served a call, because the
    caches are keyed by it: rotating the recorded name would make finished work look unfinished.
    """

    backend: str
    model: str
    alternates: tuple[str, ...] = ()

    def _models(self) -> list[str]:
        chain = [self.model, *self.alternates]
        now = time.monotonic()
        with _exhausted_lock:
            live = [
                m
                for m in chain
                if now - _exhausted.get(m, -EXHAUSTED_RETRY_AFTER) >= EXHAUSTED_RETRY_AFTER
            ]
        # never return nothing: with every model spent the call still has to fail with the real
        # error from the provider rather than an IndexError from here.
        return live or chain

    def _pace(self) -> None:
        # Only metered providers need this. A local Ollama has no quota, so spacing its calls out
        # would be pure waiting -- and this pipeline makes over a hundred of them.
        global _last_request_at
        if self.backend == "ollama" or MIN_REQUEST_INTERVAL <= 0:
            return
        with _pace_lock:
            wait = _last_request_at + MIN_REQUEST_INTERVAL - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last_request_at = time.monotonic()

    def _base_url(self, default: str) -> str:
        return (os.environ.get("COVENANT_BASE_URL") or default).rstrip("/")

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        last_error: Exception | None = None
        models = self._models()
        active = 0
        for attempt in range(MAX_RETRIES + len(self.alternates)):
            serving = models[active]
            self._pace()
            try:
                if self.backend == "anthropic":
                    return self._complete_anthropic(serving, system, user, max_tokens, temperature)
                if self.backend == "huggingface":
                    return self._complete_openai_compatible(
                        self._base_url(HF_ROUTER_BASE_URL),
                        api_key("HF_TOKEN"),
                        serving,
                        system,
                        user,
                        max_tokens,
                        temperature,
                        extra_body=_hf_extra_body(serving),
                    )
                if self.backend == "gemini":
                    return self._complete_openai_compatible(
                        self._base_url(GEMINI_BASE_URL),
                        gemini_key(),
                        serving,
                        system,
                        user,
                        max_tokens,
                        temperature,
                    )
                if self.backend == "alibaba":
                    return self._complete_openai_compatible(
                        self._base_url(DASHSCOPE_BASE_URL),
                        api_key("ALIBABA_CLOUD_API_KEY"),
                        serving,
                        system,
                        user,
                        max_tokens,
                        temperature,
                        extra_body={"enable_thinking": False},
                    )
                return self._complete_ollama(serving, system, user, max_tokens, temperature)
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt >= MAX_RETRIES - 1:
                    break
                # A 429 is a quota window, not congestion: free tiers meter per minute, so backing
                # off for a couple of seconds just spends another attempt on the same refusal.
                # Honour Retry-After when the server sends one, otherwise wait out the window.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 429 and active < len(models) - 1:
                    # this model's quota is spent -- another one's is not, so switch instead of
                    # waiting out a window that will not reopen until tomorrow.
                    _mark_exhausted_if_daily(serving, exc)
                    active += 1
                    print(
                        f"  quota exhausted on {serving}, switching to {models[active]}", flush=True
                    )
                    continue
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
        self, model: str, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=model,
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
        model: str,
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
            "model": model,
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

    def _complete_ollama(
        self, model: str, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        # ollama defaults to a 4096-token context and silently truncates the prompt -- a 46KB
        # credit agreement would lose most of its text. Size the window to the actual prompt
        # (~3 chars/token) instead of allocating a fixed 32k KV cache on every small call.
        num_ctx = min(32768, max(8192, (len(system) + len(user)) // 3 + max_tokens + 512))
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {"num_predict": max_tokens, "temperature": temperature, "num_ctx": num_ctx},
        }
        response = httpx.post(
            f"{self._base_url(OLLAMA_BASE_URL)}/api/chat", json=payload, timeout=600
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


_MODEL_NAME_HINTS = {
    "gemini": ("gemini-", "gemma-"),
    "anthropic": ("claude-",),
    "alibaba": ("qwen", "glm-", "deepseek-", "kimi-"),
}


def _warn_if_foreign_model(backend: str, model: str) -> None:
    """Say something when the chosen model plainly belongs to a different provider.

    Backend and model are configured separately, so changing one and forgetting the other sends,
    say, a Gemini model name to Alibaba's endpoint -- which answers 404 and surfaces as a bare
    "LLM call failed after 4 attempts", with nothing pointing at the real cause.
    """
    hints = _MODEL_NAME_HINTS.get(backend)
    owners = [b for b, prefixes in _MODEL_NAME_HINTS.items() if model.lower().startswith(prefixes)]
    if hints and owners and backend not in owners:
        print(
            f"warning: COVENANT_BACKEND={backend} but the model {model!r} looks like "
            f"{owners[0]}'s -- check .env, this usually means one of the two was changed alone",
            file=sys.stderr,
        )


def get_client(size: str = "complex") -> Client:
    """The client for a tier, with the model overridable without touching code.

    COVENANT_MODEL_SMALL / COVENANT_MODEL_COMPLEX exist because the spec cache is keyed by model
    name: swapping models by editing this table meant the new name no longer matched the cached
    files, and every scenario silently fell through to its fallback answer instead of erroring.
    """
    backend = _select_backend()
    override = os.environ.get(f"COVENANT_MODEL_{size.upper()}")
    if override:
        # comma-separated: the first is the model of record, the rest are fallen through to as each
        # one's quota runs out (see Client).
        names = [n.strip() for n in override.split(",") if n.strip()]
        _warn_if_foreign_model(backend, names[0])
        return Client(backend=backend, model=names[0], alternates=tuple(names[1:]))

    model = _MODEL_BY_BACKEND_AND_SIZE.get((backend, size))
    if model is None:
        known = sorted({b for b, _ in _MODEL_BY_BACKEND_AND_SIZE})
        raise SystemExit(
            f"no default model for backend {backend!r} (tier {size!r}).\n"
            f"set COVENANT_MODEL_{size.upper()} in .env, "
            f"or use one of the backends with defaults: {', '.join(known)}"
        )
    return Client(backend=backend, model=model)
