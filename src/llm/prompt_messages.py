from typing import Literal, NotRequired, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    cache_control: NotRequired[dict[str, str]]
