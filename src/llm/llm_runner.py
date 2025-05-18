import logging
from typing import Callable, Optional, TypeVar, cast

import litellm
import openai
from litellm.types import utils as litellm_types

from src.llm import exception as llm_exception
from src.llm import models as llm_models
from src.llm.llm_tracer import LLMTracer
from src.llm.prompt_messages import Message, MessageTemplate

T = TypeVar("T")


class LLMRunner[T]:
    def __init__(
        self,
        output_parser: Callable[[str, str, str], T],
        prompt_template: list[MessageTemplate],
        prompt_private_input_variables: Optional[list[str]] = None,
        model: str = llm_models.OPENAI.GPT_4O_2024_08_06,
        max_tokens: int = 4096,
        # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        cache_control_index: int | None = None,
    ):
        self._logger = logging.getLogger(__name__)
        self.output_parser = output_parser
        self.prompt_private_input_variables = prompt_private_input_variables if prompt_private_input_variables else []
        self.prompt_template = prompt_template
        self.model = model
        self.max_tokens = max_tokens
        self.cache_control_index = cache_control_index

    def get_concrete_prompt(self, prompt_input: dict[str, str]) -> list[Message]:
        concrete_prompt: list[Message] = []
        for template_message in self.prompt_template:
            concrete_message: Message = template_message.copy()  # type: ignore
            concrete_message["content"] = template_message["content"].render(**prompt_input)
            concrete_prompt.append(concrete_message)
        return concrete_prompt

    def _call_llm(
        self,
        prompt_input: dict[str, str],
        use_prompt_caching: bool = False,
    ) -> litellm_types.ModelResponse:
        concrete_prompt = self.get_concrete_prompt(prompt_input)
        if self.cache_control_index is not None and use_prompt_caching:
            concrete_prompt[self.cache_control_index]["cache_control"] = {"type": "ephemeral"}

        response = cast(
            litellm_types.ModelResponse,
            litellm.completion(
                model=self.model,
                messages=concrete_prompt,
                max_tokens=self.max_tokens,
            ),
        )
        return response

    def query_llm(
        self,
        prompt_input: dict[str, str],
        query_source: str,
        censor_func: Callable[[dict[str, str], list[str]], dict[str, str]],
        parent_tracer: LLMTracer | None = None,
        use_prompt_caching: bool = False,
    ) -> T:
        try:
            censored_input = censor_func(prompt_input, self.prompt_private_input_variables)
            tracer = LLMTracer(
                run_name=self.__class__.__name__,
                tracer_input=censored_input,
                metadata={"query_source": query_source},
                parent=parent_tracer,
            )
            tracer.init_llm_call(censored_input, self.prompt_template, self.model)
            response = self._call_llm(prompt_input, use_prompt_caching)
            raw_llm_output = cast(str, response.choices[0].message.content)  # type: ignore
            tracer.end_llm_call(raw_llm_output, response.usage)  # type: ignore
            parsed_output = self.output_parser.parse(raw_llm_output)
            tracer.end_run(raw_llm_output, error=None)
            return parsed_output

        except openai.APIError as e:
            raise llm_exception.LLMResponseException(
                tracer=tracer, query_source=query_source, model=self.model, prompt_input=prompt_input
            ) from e

        except AttributeError as e:
            raise llm_exception.LLMResponseParsingException(query_source=query_source, model=self.model) from e

        except llm_exception.LLMOutputParsingException:
            tracer.end_run(raw_llm_output, error="Failed to parse output.")
            raise

        except Exception as e:
            raise llm_exception.LLMUnknownException(tracer, query_source, self.model) from e
