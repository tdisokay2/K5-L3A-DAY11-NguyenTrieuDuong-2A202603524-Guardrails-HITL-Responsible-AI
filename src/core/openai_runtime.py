"""
OpenAI SDK runtime — dùng cho:

  Blue Team → OpenRouter liquid/lfm-3-1.7b:free (create_blue_pair)
  Red Team  → OpenAI gpt-4o-mini (create_openai_pair) khi RED_TEAM_PROVIDER=openai

Gemini Red Team dùng Google ADK trong agents/*.py — không đi qua file này.
"""
from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dataclasses import dataclass, field
from typing import Any, Callable

from core.config import (
    get_red_model,
    get_red_provider,
    get_blue_model,
    get_blue_provider,
    blue_client_kwargs,
    red_openai_client_kwargs,
)


@dataclass
class OpenAIAgent:
    name: str
    instruction: str
    provider: str = "openai"


@dataclass
class _MockInvocationContext:
    user_id: str = "student"


@dataclass
class OpenAIRunner:
    """Optional ADK-style plugins + Chat Completions."""

    app_name: str
    model: str
    plugins: list = field(default_factory=list)
    provider: str = "openai"
    temperature: float = 0.4
    client_kwargs: dict = field(default_factory=dict)
    input_hooks: list[Callable[[str], str | None]] = field(default_factory=list)
    output_hooks: list[Callable[[str], str]] = field(default_factory=list)

    def _client(self):
        from openai import OpenAI

        # Own the retry budget here so SDK retries do not multiply our attempts.
        return OpenAI(**{"max_retries": 0, **(self.client_kwargs or {})})

    async def _completion_with_backoff(self, client, **kwargs):
        """Retry temporary provider throttling, preserving the assigned model."""
        from openai import RateLimitError

        for attempt in range(4):
            try:
                return client.chat.completions.create(**kwargs)
            except RateLimitError as error:
                # A 429 can describe exhausted credit/spend quota, which waiting
                # cannot fix. Inspect structured codes without dumping API bodies.
                details = error.body if isinstance(error.body, dict) else {}
                details = details.get("error", details)
                if not isinstance(details, dict):
                    details = {}
                codes = (getattr(error, "code", None), getattr(error, "type", None),
                         details.get("code"), details.get("type"))
                quota_codes = {
                    "insufficient_quota", "credit_balance_exhausted",
                    "organization_spend_limit_exceeded", "organization_usage_limit_exceeded",
                    "billing_hard_limit_reached",
                }
                quota_code = next((code for code in codes if isinstance(code, str)
                                   and code in quota_codes), None)
                account = "OpenAI" if self.provider == "openai" else self.provider
                if quota_code:
                    raise RuntimeError(
                        f"{account} API quota/billing limit reached ({quota_code}). "
                        f"Check the {account} account's API billing, credits, and usage limits "
                        "for this key/project before rerunning. Retrying will not restore quota."
                    ) from None
                delay = 5 * 2 ** attempt + random.uniform(0, 1)
                hint = error.response.headers.get("retry-after")
                if hint:
                    try:
                        delay = float(hint)
                    except ValueError:
                        try:
                            deadline = parsedate_to_datetime(hint)
                            if deadline.tzinfo is None:
                                deadline = deadline.replace(tzinfo=timezone.utc)
                            delay = (deadline - datetime.now(timezone.utc)).total_seconds()
                        except (TypeError, ValueError, OverflowError):
                            pass
                if not math.isfinite(delay) or delay < 0:
                    delay = 5 * 2 ** attempt
                if attempt == 3 or delay > 60:
                    raise RuntimeError(
                        f"{self.provider} returned HTTP 429 for {self.model}. "
                        "The provider is still rate limited. "
                        f"Check your {account} API rate limits and billing/usage quota, "
                        "then retry after any temporary limit has cleared. "
                        "This is separate from the lab's local rate limiter."
                    ) from None
                print(f"Provider HTTP 429; retry {attempt + 1}/3 in {delay:.1f}s...", flush=True)
                await asyncio.sleep(delay)

    async def chat(self, agent: OpenAIAgent, user_message: str) -> str:
        for hook in self.input_hooks:
            blocked = hook(user_message)
            if blocked:
                return blocked

        block_msg = await self._run_input_plugins(user_message)
        if block_msg is not None:
            return block_msg

        client = self._client()
        completion = await self._completion_with_backoff(client,
            model=self.model,
            messages=[
                {"role": "system", "content": agent.instruction},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
        )
        text = (completion.choices[0].message.content or "").strip()

        for hook in self.output_hooks:
            text = hook(text)

        text = await self._run_output_plugins(text)
        return text

    async def _run_input_plugins(self, user_message: str) -> str | None:
        if not self.plugins:
            return None
        try:
            from google.genai import types
        except ImportError:
            return None

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        ctx = _MockInvocationContext()
        for plugin in self.plugins:
            cb = getattr(plugin, "on_user_message_callback", None)
            if cb is None:
                continue
            try:
                result = await cb(
                    invocation_context=ctx, user_message=user_content
                )
            except TypeError:
                result = cb(invocation_context=ctx, user_message=user_content)
            if result is None:
                continue
            return _content_to_text(result)
        return None

    async def _run_output_plugins(self, text: str) -> str:
        if not self.plugins or not text:
            return text
        try:
            from google.genai import types
        except ImportError:
            return text

        content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )

        class _Resp:
            pass

        llm_response = _Resp()
        llm_response.content = content

        class _Ctx:
            pass

        for plugin in self.plugins:
            cb = getattr(plugin, "after_model_callback", None)
            if cb is None:
                continue
            try:
                out = await cb(callback_context=_Ctx(), llm_response=llm_response)
            except TypeError:
                out = cb(callback_context=_Ctx(), llm_response=llm_response)
            if out is not None and getattr(out, "content", None) is not None:
                llm_response = out
        return _content_to_text(llm_response.content) or text


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    chunks = []
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks)


def _make_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    model: str,
    provider: str,
    client_kwargs: dict,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    agent = OpenAIAgent(name=name, instruction=instruction, provider=provider)
    runner = OpenAIRunner(
        app_name=app_name,
        model=model,
        provider=provider,
        client_kwargs=client_kwargs,
        plugins=list(plugins or []),
        input_hooks=list(input_hooks or []),
        output_hooks=list(output_hooks or []),
        temperature=temperature,
    )
    return agent, runner


def create_blue_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Blue Team — always OpenRouter liquid/lfm-3-1.7b:free."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=get_blue_model(),
        provider=get_blue_provider(),
        client_kwargs=blue_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )


def create_openai_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    model: str | None = None,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Red Team OpenAI path (default = soft model; advance may pass harder)."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=model or get_red_model(),
        provider=get_red_provider(),
        client_kwargs=red_openai_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )
