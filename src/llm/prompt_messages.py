from typing import Literal, NotRequired, TypedDict

from jinja2 import Template


class MessageTemplate(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: Template
    cache_control: NotRequired[dict[str, str]]


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    cache_control: NotRequired[dict[str, str]]
    tool_calls: NotRequired[list[dict[str, str]]]
    tool_call_id: NotRequired[str]
