from src.llm.tool_helpers import generate_tool_definition


def get_weather(city: str) -> str:
    """
    Get the weather for a given city
    city: The city to get the weather for
    """
    return f"The weather in {city} is sunny."


def test_generate_tool_definition() -> None:
    tool_def = generate_tool_definition(get_weather)
    assert tool_def is not None
    assert tool_def["type"] == "function"
    assert tool_def["function"]["name"] == "get_weather"
    assert tool_def["function"]["description"] == "Get the weather for a given city"
    assert tool_def["function"]["parameters"]["type"] == "object"
    assert tool_def["function"]["parameters"]["properties"]["city"]["type"] == "string"
    assert tool_def["function"]["parameters"]["properties"]["city"]["description"] == "The city to get the weather for"
    assert tool_def["function"]["parameters"]["required"] == ["city"]
