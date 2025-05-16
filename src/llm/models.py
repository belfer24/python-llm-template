from enum import StrEnum


class OPENAI(StrEnum):
    """https://platform.openai.com/docs/models/"""

    GPT_4O_MINI_07_18 = "gpt-4o-mini-2024-07-18"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4O_2024_08_06 = "gpt-4o-2024-08-06"
    GPT_4_1_2025_04_14 = "gpt-4.1-2025-04-14"


class ANTHROPIC(StrEnum):
    """https://www.anthropic.com/api"""

    CLAUDE_3_7_SONNET_02_19 = "claude-3-7-sonnet-20250219"
    CLAUDE_3_5_SONNET_06_20 = "claude-3-5-sonnet-20240620"
    CLAUDE_3_5_SONNET_10_22 = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU_10_22 = "claude-3-5-haiku-20241022"
