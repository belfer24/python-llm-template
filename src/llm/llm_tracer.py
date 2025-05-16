from __future__ import annotations

import time
import uuid
from typing import Any

from langfuse import Langfuse
from litellm.types.utils import Usage as LiteLLmUsage

from src.llm.prompt_messages import Message


class LLMTracer:
    def __init__(
        self,
        run_name: str,
        tracer_input: dict[str, str],
        metadata: dict[str, Any] | None = None,
        parent: LLMTracer | None = None,
    ) -> None:
        self.langfuse = Langfuse(host="https://us.cloud.langfuse.com", public_key="", secret_key="")
        self.run_id = str(uuid.uuid4())
        self.run_name = run_name
        self.tracer_input = tracer_input
        self.metadata = metadata or {}
        self.parent = parent

        if parent is None:
            self.trace = self.langfuse.trace(
                name=run_name,
                id=self.run_id,
                metadata=self.metadata,
                input=self.tracer_input,
            )
        else:
            self.trace = parent.trace

        self.span = self.trace.span(
            name=run_name,
            input=self.tracer_input,
            metadata=self.metadata,
        )

    def init_llm_call(self, prompt_input: dict[str, str], prompt_template: list[Message], model: str) -> None:
        self.llm_generation = self.span.generation(
            name=f"{self.run_name}_generation",
            model=model,
            prompt_template=str(prompt_template),  # Convert to string for storage
            prompt_input=prompt_input,
        )

    def end_llm_call(self, output: str, llm_usage_information: LiteLLmUsage) -> None:
        if self.llm_generation:
            self.llm_generation.end(
                output=output,
                usage_details={
                    "input": llm_usage_information.prompt_tokens,
                    "output": llm_usage_information.completion_tokens,
                },
                finish_time=time.time(),
            )

    def end_run(self, output: str | dict, error: str | None = None) -> None:
        if self.span:
            if error:
                self.span.end(
                    output=output,
                    status="error",
                    status_message=error,
                    finish_time=time.time(),
                )
            else:
                self.span.end(
                    output=output,
                    status="success",
                    finish_time=time.time(),
                )
