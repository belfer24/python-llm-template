from typing import Literal, NotRequired, TypedDict

from jinja2 import Template


class MessageTemplate(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: Template
    cache_control: NotRequired[dict[str, str]]


class Message(MessageTemplate):
    role: Literal["system", "user", "assistant"]
    content: str
    cache_control: NotRequired[dict[str, str]]
