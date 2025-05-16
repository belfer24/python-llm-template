from src.llm.llm_tracer import LLMTracer
from src.logging import utils as logging_utils


class LLMException(Exception):
    pass


class LLMResponseException(LLMException):
    def __init__(self, tracer: LLMTracer, query_source: str, model: str, prompt_input: dict):
        tracer.end_run("", error=self.__class__.__name__)
        char_counts = {key: len(str(value)) for key, value in prompt_input.items()}
        super().__init__(
            logging_utils.format_log_msg(
                msg=f"Failed to get response for query made by {query_source} with model {model}. Prompt input character counts: {char_counts}",
                component="LLM",
            )
        )


class LLMResponseParsingException(LLMException):
    def __init__(self, query_source: str, model: str):
        super().__init__(
            logging_utils.format_log_msg(
                msg=f"Failed to process response for query made by {query_source} with model {model}",
                component="LLM",
            )
        )


class LLMOutputParsingException(LLMException):
    def __init__(self, query_source: str, model: str, output_parser: str):
        super().__init__(
            logging_utils.format_log_msg(
                msg=f"{output_parser} failed to parse output for query made by {query_source} with model {model}",
                component="LLM",
            )
        )


class LLMUnknownException(LLMException):
    def __init__(self, tracer: LLMTracer, query_source: str, model: str):
        tracer.end_run("", error=self.__class__.__name__)
        super().__init__(
            logging_utils.format_log_msg(
                msg=f"Unknown error occurred for query made by {query_source} with model {model}",
                component="LLM",
            )
        )
