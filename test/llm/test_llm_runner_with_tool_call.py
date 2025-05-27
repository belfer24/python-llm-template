import time

from jinja2 import Template

from src.llm import models
from src.llm.llm_runner import LLMRunner
from src.llm.output_parsers import parse_text
from src.llm.prompt_censor import do_not_censor_prompt
from src.llm.prompt_messages import MessageTemplate


def get_weather(city: str) -> str:
    """
    Get the weather for a given city
    city: The city to get the weather for
    """
    time.sleep(
        0.01
    )  # langfuse orders nodes by time created, but truncates at milliseconds. Sleep is necessary for nodes to appear in order.
    return f"The weather in {city} is sunny."


def test_llm_runner():
    template = "What is the weather in {{city}}?"

    message_template: MessageTemplate = {
        "role": "user",
        "content": Template(template),
    }

    llm_runner = LLMRunner(
        parse_output=parse_text,
        prompt_template=[message_template],
        model=models.ANTHROPIC.CLAUDE_3_5_SONNET_06_20,
        tools=[get_weather],
    )

    prompt_input = dict(city="Montreal")

    response = llm_runner.query_llm(prompt_input=prompt_input, query_source="test", censor_func=do_not_censor_prompt)
    assert "sunny" in response
