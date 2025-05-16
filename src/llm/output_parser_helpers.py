import json
import re
from typing import Any

import yaml


def clean_yaml_output_content(output: str) -> str:
    pattern = r"(?:.*?\n)?```yaml\n((?:[^`]|`(?!``)|``(?!`)|\n\s+```[a-zA-Z]*[\s\S]*?\n\s+```)*)\n```"
    match = re.search(pattern, output, re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    return output


def clean_json_output_content(output: str) -> str:
    patterns = [
        (r"```json\n((?:[^`]|`(?!``)|``(?!`))*)\n```", 1),
        (r"(\[|\{)[\s\S]*\{([\s\S]*)\}[\s\S]*(\]|\})", 0),
        (r"\{([\s\S]*)\}", 0),
        (r"```python\n((?:[^`]|`(?!``)|``(?!`))*)\n```", 1),
        (r"```\n((?:[^`]|`(?!``)|``(?!`))*)\n```", 1),
    ]

    for pattern, group in patterns:
        match = re.search(pattern, output, re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(group)

    return output


def parse_json(text: str) -> Any:
    output = clean_json_output_content(text)
    json_object = json.loads(output, strict=False)
    return json_object


def parse_yaml(text: str) -> Any:
    output = clean_yaml_output_content(text)
    yaml_object = yaml.safe_load(output)
    return yaml_object
