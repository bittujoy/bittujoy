from typing import Dict

from core.constants import DEFAULT_PROMPT_TEMPLATE


def build_troubleshooting_prompt(masked_output: str) -> str:
    return DEFAULT_PROMPT_TEMPLATE.format(masked_output=masked_output)


def build_connection_test_prompt(provider: str, model_name: str, target: str) -> str:
    return (
        f"Connection test for provider {provider} using model {model_name} is successful. "
        f"Target: {target}."
    )
