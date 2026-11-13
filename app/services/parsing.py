"""Strict parsing of the model's JSON answer. Shared by the app and finetune/eval.py."""

from pydantic import ValidationError

from app.errors import LLMOutputError
from app.schemas import LLMAnswer


def parse_answer(raw: str) -> LLMAnswer:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return LLMAnswer.model_validate_json(text)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "body"
        raise LLMOutputError(f"invalid JSON answer at {where}: {first['msg']}") from exc
