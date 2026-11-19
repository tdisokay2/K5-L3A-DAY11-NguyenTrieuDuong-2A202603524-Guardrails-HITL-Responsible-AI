"""
Lab 11 — Helper Utilities
"""
import asyncio
import random

from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Retry temporary Gemini overload for the lab's independent chat requests."""
    if isinstance(runner, OpenAIRunner) or getattr(runner, "provider", None) in ("openrouter", "openai"):
        # These runtimes already manage their own provider retry budget.
        return await _chat_with_agent_once(agent, runner, user_message, session_id)

    from google.genai.errors import ServerError

    for attempt in range(4):
        try:
            return await _chat_with_agent_once(agent, runner, user_message, session_id)
        except ServerError as error:
            if error.code != 503 or session_id is not None:
                # Do not replay a caller-owned conversation that may have
                # partially changed its state before the server error.
                raise
            if attempt == 3:
                raise RuntimeError(
                    "Gemini is still unavailable (HTTP 503: high demand) after retries. "
                    "Wait a few minutes, then rerun this checkpoint."
                ) from None
            delay = 5 * 2 ** attempt + random.uniform(0, 1)
            print(f"Gemini HTTP 503; retry {attempt + 1}/3 in {delay:.1f}s...", flush=True)
            await asyncio.sleep(delay)


async def _chat_with_agent_once(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    final_response = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=content
    ):
        if hasattr(event, "content") and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_response += part.text

    return final_response, session
