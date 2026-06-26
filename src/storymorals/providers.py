"""Provider routing: turn a (provider, model, prompt) into a generated moral.

All providers except Anthropic expose an OpenAI-compatible chat-completions API,
so they share one code path with a different base_url + key. Anthropic uses its
own SDK (anthropic.Anthropic / messages.create).
"""
from __future__ import annotations

import os
import re
import time

# Light system instruction. The paper's full socio-demographic prompt is sent as
# the user message (prompt_template.txt); this just nudges chat models to emit
# only the moral, matching the paper's downstream cleaning step.
SYSTEM_PROMPT = (
    "You output only the moral of the story: a single, complete sentence in the "
    "requested language. No preamble, no quotation marks, no explanation."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_moral(text: str) -> str:
    """Strip stray <think> blocks / surrounding quotes / whitespace."""
    text = _THINK_RE.sub("", text or "").strip()
    if len(text) >= 2 and text[0] in "\"'“「" and text[-1] in "\"'”」":
        text = text[1:-1].strip()
    return text


class ProviderError(RuntimeError):
    pass


class FatalProviderError(ProviderError):
    """Won't be fixed by retrying (bad key, unknown model, malformed request)."""


# Exception class-name fragments that mean "stop retrying this model now".
_FATAL = ("AuthenticationError", "PermissionDenied", "NotFoundError",
          "BadRequestError", "InvalidRequest")


def _is_fatal(exc: Exception) -> bool:
    return any(frag in type(exc).__name__ for frag in _FATAL)


# --- client construction (cached per provider+timeout) ----------------------
_clients: dict[tuple, object] = {}


def _get_client(provider: str, pcfg: dict, timeout: float):
    cache_key = (provider, timeout)
    if cache_key in _clients:
        return _clients[cache_key]
    key = os.environ.get(pcfg["api_key_env"])
    if not key:
        raise FatalProviderError(
            f"Missing API key: set {pcfg['api_key_env']} (provider {provider!r})"
        )
    # max_retries=0: we run our own retry/backoff loop below, so the SDK
    # shouldn't silently retry on top (which would multiply wall-clock time).
    if pcfg["sdk"] == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=0)
    elif pcfg["sdk"] == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=pcfg.get("base_url") or None,
                        timeout=timeout, max_retries=0)
    else:
        raise FatalProviderError(f"Unknown sdk {pcfg['sdk']!r} for provider {provider!r}")
    _clients[cache_key] = client
    return client


# Some OpenAI models (GPT-5 family, o-series) reject `max_tokens` and require
# `max_completion_tokens`. We auto-detect per model and cache the right param so
# it only costs one extra attempt the first time.
_TOKEN_PARAM: dict[str, str] = {}


def _call_openai(client, model, system, user, max_tokens, temperature):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    def _build(token_param):
        kw = {"model": model, "messages": messages, token_param: max_tokens}
        if temperature is not None:
            kw["temperature"] = temperature
        return kw

    param = _TOKEN_PARAM.get(model, "max_tokens")
    try:
        resp = client.chat.completions.create(**_build(param))
    except Exception as exc:  # noqa: BLE001
        if param == "max_tokens" and "max_completion_tokens" in str(exc):
            _TOKEN_PARAM[model] = "max_completion_tokens"
            resp = client.chat.completions.create(**_build("max_completion_tokens"))
        else:
            raise
    return resp.choices[0].message.content or ""


def _call_anthropic(client, model, system, user, max_tokens, temperature):
    # temperature is intentionally NOT sent (removed on Opus 4.8/4.7; default
    # behavior matches the paper). thinking left off — single-sentence task.
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def generate(provider: str, pcfg: dict, model: str, user_prompt: str,
             *, max_tokens: int = 1024, temperature=None, max_retries: int = 5,
             timeout: float = 120.0) -> str:
    """Generate one moral, with retry/backoff + per-call timeout.

    Retries transient errors (rate limits, 5xx, timeouts) with exponential
    backoff. Fails immediately on fatal errors (bad key, unknown model) so a
    misconfigured model doesn't burn minutes retrying. Raises ProviderError on
    final failure; the caller records it as a MODEL_ERROR row.
    """
    client = _get_client(provider, pcfg, timeout)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            if pcfg["sdk"] == "anthropic":
                raw = _call_anthropic(client, model, SYSTEM_PROMPT, user_prompt,
                                      max_tokens, temperature)
            else:
                raw = _call_openai(client, model, SYSTEM_PROMPT, user_prompt,
                                   max_tokens, temperature)
            moral = clean_moral(raw)
            if not moral:
                # Empty body usually means a refusal / safety stop with no text.
                raise ProviderError("empty response (possible refusal)")
            return moral
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_fatal(exc):
                raise FatalProviderError(f"{model}: {type(exc).__name__}: {exc}") from exc
            if attempt < max_retries:
                time.sleep(min(30, 2 ** attempt))
    raise ProviderError(f"{model}: failed after {max_retries} attempts: {last_exc}")
