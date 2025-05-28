import json
import logging
from typing import Callable, Optional, TypeVar

import litellm
import openai
from litellm.types import utils as litellm_types

from src.llm import exception as llm_exception
from src.llm import models as llm_models
from src.llm.llm_tracer import LLMTracer
from src.llm.prompt_messages import Message, MessageTemplate
from src.llm.tool_helpers import ToolCall, ToolResult, generate_tool_definition

T = TypeVar("T")


class LLMRunner[T]:
    def __init__(
        self,
        parse_output: Callable[[str, str, str], T],
        prompt_template: list[MessageTemplate],
        prompt_private_input_variables: Optional[list[str]] = None,
        model: str = llm_models.OPENAI.GPT_4O_2024_08_06,
        max_tokens: int = 4096,
        # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        cache_control_index: int | None = None,
        tools: list[Callable] | None = None,
        max_tool_iterations: int = 5,
    ):
        self._logger = logging.getLogger(__name__)
        self.parse_output = parse_output
        self.prompt_private_input_variables = prompt_private_input_variables if prompt_private_input_variables else []
        self.prompt_template = prompt_template
        self.model = model
        self.max_tokens = max_tokens
        self.cache_control_index = cache_control_index

        self.tools = tools or []
        self.max_tool_iterations = max_tool_iterations
        self._tool_registry = {tool.__name__: tool for tool in self.tools}
        self._tool_definitions = [generate_tool_definition(tool) for tool in self.tools] or None

    def get_concrete_prompt(self, prompt_input: dict[str, str]) -> list[Message]:
        concrete_prompt: list[Message] = []
        for template_message in self.prompt_template:
            concrete_message: Message = template_message.copy()  # type: ignore
            concrete_message["content"] = template_message["content"].render(**prompt_input)
            concrete_prompt.append(concrete_message)
        return concrete_prompt

    def _make_llm_request(
        self,
        concrete_prompt: list[Message],
        censored_concrete_prompt: list[Message],
        tracer: LLMTracer,
        use_prompt_caching: bool = False,
    ) -> litellm_types.Message:
        if self.cache_control_index is not None and use_prompt_caching:
            concrete_prompt[self.cache_control_index]["cache_control"] = {"type": "ephemeral"}

        tracer.init_llm_call(censored_concrete_prompt, self.model)
        response = litellm.completion(
            model=self.model,
            messages=concrete_prompt,
            max_tokens=self.max_tokens,
            tools=self._tool_definitions,
        )
        raw_llm_output = str(response.choices[0].message.content)  # type: ignore
        tracer.end_llm_call(raw_llm_output, response.usage)  # type: ignore
        return response.choices[0].message  # type: ignore

    def _get_tool_function(self, tool_name: str) -> Callable:
        try:
            return self._tool_registry[tool_name]
        except KeyError as e:
            raise llm_exception.ToolNotFoundException(tool_name) from e

    def _execute_tool_call(self, tool_call: ToolCall, tracer: LLMTracer) -> ToolResult:
        try:
            tool_func = self._get_tool_function(tool_call.name)
            tracer.init_tool_use(tool_call)
            result = tool_func(**tool_call.arguments)
            tool_result = ToolResult(call_id=tool_call.call_id, result=result)
            tracer.end_tool_use(tool_result)
            return tool_result

        except llm_exception.ToolNotFoundException as e:
            return ToolResult(call_id=tool_call.call_id, result=None, error=str(e))

        except Exception as e:
            self._logger.exception(f"Error executing tool {tool_call.name}")
            return ToolResult(call_id=tool_call.call_id, result=None, error=str(e))

    def _extract_tool_calls(self, message: litellm_types.Message) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)  # type: ignore
                except json.JSONDecodeError:
                    arguments = {}

                tool_calls.append(
                    ToolCall(
                        name=tool_call.function.name,  # type: ignore
                        arguments=arguments,
                        call_id=tool_call.id,
                    )
                )

        return tool_calls

    def _handle_tool_execution(
        self,
        messages: list[Message],
        censored_messages: list[Message],
        tracer: LLMTracer,
        use_prompt_caching: bool = False,
    ) -> str:
        conversation_messages = messages.copy()

        for _ in range(self.max_tool_iterations):
            response_message = self._make_llm_request(
                conversation_messages, censored_messages, tracer, use_prompt_caching
            )
            tool_calls = self._extract_tool_calls(response_message)

            if not tool_calls:
                return str(response_message.content)

            assistant_message: Message = {
                "role": "assistant",
                "content": str(response_message.content),
                "tool_calls": [  # type: ignore
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in tool_calls
                ],
            }
            conversation_messages.append(assistant_message)
            censored_messages.append(assistant_message)

            for tool_call in tool_calls:
                tool_result = self._execute_tool_call(tool_call, tracer)

                result_content = str(tool_result.result) if tool_result.error is None else f"Error: {tool_result.error}"

                tool_message: Message = {
                    "role": "tool",
                    "tool_call_id": tool_result.call_id,
                    "content": result_content,
                }
                conversation_messages.append(tool_message)

        final_response_message = self._make_llm_request(
            conversation_messages, censored_messages, tracer, use_prompt_caching
        )
        final_content = str(final_response_message.content)

        return final_content

    def _handle_llm_interaction(
        self,
        concrete_prompt: list[Message],
        censored_concrete_prompt: list[Message],
        tracer: LLMTracer,
        use_prompt_caching: bool,
    ) -> str:
        if self.tools:
            return self._handle_tool_execution(concrete_prompt, censored_concrete_prompt, tracer, use_prompt_caching)
        response_message = self._make_llm_request(concrete_prompt, censored_concrete_prompt, tracer, use_prompt_caching)
        return str(response_message.content)

    def run(
        self,
        prompt_input: dict[str, str],
        query_source: str,
        censor_func: Callable[[dict[str, str], list[str]], dict[str, str]],
        parent_tracer: LLMTracer | None = None,
        use_prompt_caching: bool = False,
    ) -> T:
        censored_input = censor_func(prompt_input, self.prompt_private_input_variables)
        censored_concrete_prompt = self.get_concrete_prompt(censored_input)
        tracer = LLMTracer(
            run_name=self.__class__.__name__,
            tracer_input=censored_input,
            metadata={"query_source": query_source},
            parent=parent_tracer,
        )
        concrete_prompt = self.get_concrete_prompt(prompt_input)
        raw_llm_output = ""
        try:
            raw_llm_output = self._handle_llm_interaction(
                concrete_prompt, censored_concrete_prompt, tracer, use_prompt_caching
            )
            parsed_output = self.parse_output(raw_llm_output, query_source, self.model)
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
