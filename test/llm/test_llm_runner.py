from jinja2 import Template

from src.llm import models
from src.llm.llm_runner import LLMRunner
from src.llm.output_parsers import parse_text
from src.llm.prompt_censor import do_not_censor_prompt
from src.llm.prompt_messages import MessageTemplate


def test_llm_runner():
    template = "{{input}}"

    message_template: MessageTemplate = {
        "role": "user",
        "content": Template(template),
    }

    llm_runner = LLMRunner(
        parse_output=parse_text,
        prompt_template=[message_template],
        model=models.ANTHROPIC.CLAUDE_3_5_HAIKU_10_22,
    )

    prompt_input = dict(input="Hello!")

    response = llm_runner.run(prompt_input=prompt_input, query_source="test", censor_func=do_not_censor_prompt)
    assert response
