"""Generierungsschritt der RAG-Pipeline mit Claude (Streaming)."""

from collections.abc import AsyncIterator

import anthropic

from .config import settings

SYSTEM_PROMPT = (
    "Du bist ein Assistent in einem Lern-Dashboard über RAG. Beantworte die Frage ausschließlich auf Basis "
    "der mitgelieferten Quellen. Belege jede Aussage mit der Quellennummer in eckigen Klammern, z. B. [2]. "
    "Wenn die Quellen die Frage nicht beantworten, sage klar: „Dazu enthalten die Quellen keine Information.“ "
    "Nutze kein Wissen außerhalb der Quellen. Antworte knapp auf Deutsch."
)

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None


def llm_available() -> bool:
    return _client is not None


def build_user_prompt(question: str, sources: list[dict]) -> str:
    blocks = "\n".join(
        f'<quelle id="{i}" titel="{s["title"]}">\n{s["text"]}\n</quelle>' for i, s in enumerate(sources, start=1)
    )
    return f"<quellen>\n{blocks}\n</quellen>\n\nFrage: {question}"


async def stream_answer(question: str, sources: list[dict]) -> AsyncIterator[tuple[str, str]]:
    """Liefert ("delta", text) während der Generierung und am Ende ("done", Hinweis)."""
    async with _client.beta.messages.stream(
        model=settings.claude_model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        output_config={"effort": settings.claude_effort},
        # Bei einer Ablehnung durch Sicherheitsklassifikatoren springt serverseitig ein Ersatzmodell ein
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": build_user_prompt(question, sources)}],
    ) as stream:
        async for text in stream.text_stream:
            yield "delta", text
        final = await stream.get_final_message()

    if final.stop_reason == "refusal":
        yield "done", "Das Modell hat die Anfrage abgelehnt."
    elif final.stop_reason == "max_tokens":
        yield "done", "Antwort wurde wegen des Token-Limits abgeschnitten."
    else:
        yield "done", f"Antwort von {final.model} · {final.usage.input_tokens} Input- / {final.usage.output_tokens} Output-Token"
