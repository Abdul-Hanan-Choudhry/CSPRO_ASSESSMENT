"""LLM clients behind one small interface: a real Ollama client and a deterministic fake."""

import json
import re
from typing import Protocol

import httpx

from app.config import Settings
from app.errors import LLMUnavailableError
from app.retry import call_with_retries
from app.schemas import LLMAnswer

Message = dict[str, str]


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[Message]) -> str:
        """Return the raw assistant text. Raises LLMUnavailableError if the runtime fails."""
        ...

    def health(self) -> str:
        """Return a short status string. Raises LLMUnavailableError if not ready."""
        ...


class OllamaClient:
    """Talks to a local Ollama server (https://github.com/ollama/ollama/blob/main/docs/api.md)."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.model = settings.ollama_model
        self._retries = settings.dependency_retries
        self._http = httpx.Client(
            base_url=settings.ollama_url,
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=5.0),
            transport=transport,
        )

    def complete(self, messages: list[Message]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Constrained decoding to our schema; we still validate strictly afterwards because
            # the schema cannot express our semantic rules (verbatim excerpts, known sources).
            "format": LLMAnswer.model_json_schema(),
            "options": {"temperature": 0, "seed": 7, "num_ctx": 4096},
        }
        response = self._request("POST", "/api/chat", json=body)
        try:
            return response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMUnavailableError("Ollama returned an unexpected response shape") from exc

    def health(self) -> str:
        tags = self._request("GET", "/api/tags").json()
        names = {m.get("name") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise LLMUnavailableError(
                f"model '{self.model}' not pulled: run `ollama pull {self.model}`"
            )
        return f"ollama ready, model {self.model}"

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            # Only connection failures are retried: a timeout already spent the full budget,
            # and retrying a slow CPU generation would multiply the user's wait.
            response = call_with_retries(
                lambda: self._http.request(method, path, **kwargs),
                retries=self._retries,
                retry_on=(httpx.ConnectError,),
            )
        except httpx.ConnectError as exc:
            # DEBUG: start the runtime with `ollama serve`, then check OLLAMA_URL.
            raise LLMUnavailableError("Ollama not reachable: start it with `ollama serve`") from exc
        except httpx.TimeoutException as exc:
            # DEBUG: CPU inference of a 3B model can exceed the budget on long contexts. Raise
            # LLM_TIMEOUT_S or switch OLLAMA_MODEL to llama3.2:1b.
            raise LLMUnavailableError("LLM request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"LLM transport error: {type(exc).__name__}") from exc
        if response.status_code == 404:
            raise LLMUnavailableError(
                f"model '{self.model}' not found: run `ollama pull {self.model}`"
            )
        if response.status_code >= 400:
            raise LLMUnavailableError(f"Ollama returned HTTP {response.status_code}")
        return response


_DOC_RE = re.compile(r'<document source_id="([^"]+)"[^>]*>\n(.*?)\n</document>', re.S)


class FakeLLM:
    """Deterministic stand-in for tests and offline demos.

    With `responses`, returns them in order (the last one repeats). Without, cites the first
    sentence of the first document in the prompt, or abstains if there are none.
    Every received message list is kept in `calls` so tests can inspect the exact prompt.
    """

    model = "fake-llm"

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if self._responses:
            return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return self._default_answer(messages[-1]["content"])

    def health(self) -> str:
        return "fake llm ready"

    @staticmethod
    def _default_answer(prompt: str) -> str:
        match = _DOC_RE.search(prompt)
        if not match:
            return json.dumps(
                {"status": "insufficient_evidence", "answer": "No records found.", "citations": []}
            )
        source_id, text = match.groups()
        sentence = text.split(". ")[0].strip()
        return json.dumps(
            {
                "status": "answered",
                "answer": sentence,
                "citations": [{"source_id": source_id, "excerpt": sentence}],
            }
        )


def build_llm(settings: Settings) -> LLMClient:
    return FakeLLM() if settings.llm_backend == "fake" else OllamaClient(settings)
