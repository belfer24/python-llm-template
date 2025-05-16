from enum import StrEnum


class OPENAI(StrEnum):
    """https://platform.openai.com/docs/models/"""

    GPT_4_ORIGINAL = "gpt-4"  # Use only for token count, not for LLM inference
    GPT_4_06_13 = "gpt-4-0613"
    GPT_4_TURBO_04_09 = "gpt-4-turbo-2024-04-09"
    GPT_4_TURBO_PREVIEW_01_25 = "gpt-4-0125-preview"
    GPT_4O_MINI_07_18 = "gpt-4o-mini-2024-07-18"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4O_2024_08_06 = "gpt-4o-2024-08-06"


class ANTHROPIC(StrEnum):
    """https://www.anthropic.com/api"""

    CLAUDE_3_7_SONNET_02_19 = "claude-3-7-sonnet-20250219"
    CLAUDE_3_5_SONNET_06_20 = "claude-3-5-sonnet-20240620"
    CLAUDE_3_5_SONNET_10_22 = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU_10_22 = "claude-3-5-haiku-20241022"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
