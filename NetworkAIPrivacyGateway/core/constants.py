from enum import Enum


class CommandType(str, Enum):
    PING = "ping"
    TRACEROUTE = "traceroute"
    TELNET = "telnet"


DEFAULT_PROMPT_TEMPLATE = (
    "You are an expert Network Troubleshooting Engineer.\n"
    "Analyze the following network diagnostic output.\n\n"
    "Provide:\n"
    "1. Summary\n"
    "2. Root Cause\n"
    "3. Observations\n"
    "4. Security Findings\n"
    "5. Recommendations\n\n"
    "Network Output:\n"
    "{masked_output}\n"
)


RAW_OUTPUT_LABEL = "Raw diagnostic output"
MASKED_OUTPUT_LABEL = "Masked network diagnostic output"
PROMPT_LABEL = "LLM prompt"  
LLM_RESPONSE_LABEL = "LLM response (masked)"
FINAL_REPORT_LABEL = "Final readable report"
