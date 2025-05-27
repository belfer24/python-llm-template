import inspect
from dataclasses import dataclass
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints


@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str


@dataclass
class ToolResult:
    call_id: str
    result: Any
    error: str | None = None


def parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    if not docstring:
        return "", {}

    lines = [line.strip() for line in docstring.split("\n") if line.strip()]
    description = lines[0]

    param_descriptions = {}
    for line in lines[1:]:
        if ":" in line:
            param_name, desc = line.split(":", 1)
            param_descriptions[param_name.strip()] = desc.strip()

    return description, param_descriptions


def python_type_to_json_schema(python_type: type) -> str:
    if get_origin(python_type) is Union:
        args = get_args(python_type)
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return python_type_to_json_schema(non_none_type)

    type_mapping = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}

    return type_mapping.get(python_type, "string")


def generate_tool_definition(func: Callable) -> dict:
    sig = inspect.signature(func)
    docstring = inspect.getdoc(func)
    type_hints = get_type_hints(func)

    description, param_descriptions = parse_docstring(docstring)

    tool_def = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    for param_name, param in sig.parameters.items():
        param_type = type_hints[param_name]
        json_type = python_type_to_json_schema(param_type)

        tool_def["function"]["parameters"]["properties"][param_name] = {
            "type": json_type,
            "description": param_descriptions.get(param_name, param_name),
        }

        if param.default == inspect.Parameter.empty:
            tool_def["function"]["parameters"]["required"].append(param_name)

    return tool_def
