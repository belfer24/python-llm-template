from __future__ import annotations

import os
import time
import uuid
from typing import Any

from langfuse import Langfuse
from litellm.types.utils import Usage as LiteLLmUsage

from src.llm.prompt_messages import Message, MessageTemplate


class LLMTracer:
    def __init__(
        self,
        run_name: str,
        tracer_input: dict[str, str],
        metadata: dict[str, Any] | None = None,
        parent: LLMTracer | None = None,
    ) -> None:
        self.langfuse = Langfuse(
            host="https://us.cloud.langfuse.com",
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        )
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
            self.span = self.trace.span(
                name=run_name,
                input=self.tracer_input,
                metadata=self.metadata,
            )
        else:
            self.trace = parent.trace
            self.span = parent.span.span(
                name=run_name,
                input=self.tracer_input,
                metadata=self.metadata,
            )

    def init_llm_call(self, llm_input: list[Message], model: str) -> None:
        self.llm_generation = self.span.generation(
            name=f"{self.run_name}_generation",
            model=model,
            input=llm_input,
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
