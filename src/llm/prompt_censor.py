from typing import Callable


def censor_prompt(prompt_input_values: dict[str, str], private_input_variables: list[str]) -> dict[str, str]:
    censored_dict = prompt_input_values.copy()
    for key in private_input_variables:
        censored_dict[key] = f"{{{key}}}"
    return censored_dict


def do_not_censor_prompt(prompt_input_values: dict[str, str], private_input_variables: list[str]) -> dict[str, str]:
    return prompt_input_values.copy()


def select_censor_function(code_storage_allowed: bool) -> Callable[[dict[str, str], list[str]], dict[str, str]]:
    return do_not_censor_prompt if code_storage_allowed else censor_prompt
