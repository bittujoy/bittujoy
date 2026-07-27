import time
from typing import Dict, Optional

import streamlit as st

from config import AppSettings, get_settings
from core.llm_client import LLMClient, LLMClientError, LLMResponse, OllamaClient, OpenAIClient
from core.logger import initialize_logger
from core.masker import MaskingEngine
from core.network_tools import NetworkCommandError, run_ping, run_telnet, run_traceroute
from core.prompt_builder import build_troubleshooting_prompt
from core.utils import validate_ipv4_address


def create_llm_client(
    provider: str,
    server_url: str,
    model_name: str,
    api_key: str,
    auth_header: str = "Authorization",
    auth_prefix: str = "Bearer",
) -> LLMClient:
    if provider.lower() == "openai":
        return OpenAIClient(
            server_url=server_url,
            model_name=model_name,
            api_key=api_key,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
        )
    if provider.lower() == "ollama":
        return OllamaClient(
            server_url=server_url,
            model_name=model_name,
            api_key=api_key,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
        )
    raise ValueError("Unsupported provider selected.")


def execute_diagnostics(
    source_ip: str,
    destination_ip: str,
    port: int,
    timeout_settings: Dict[str, int],
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    if source_ip:
        outputs["source_note"] = f"Source endpoint: {source_ip}"
    try:
        outputs["ping"] = run_ping(destination_ip, timeout_settings["ping"])
    except NetworkCommandError as exc:
        outputs["ping"] = f"Ping failed: {str(exc)}"
    try:
        outputs["traceroute"] = run_traceroute(destination_ip, timeout_settings["traceroute"])
    except NetworkCommandError as exc:
        outputs["traceroute"] = f"Traceroute failed: {str(exc)}"
    try:
        outputs["telnet"] = run_telnet(destination_ip, port, timeout_settings["telnet"])
    except NetworkCommandError as exc:
        outputs["telnet"] = f"Telnet failed: {str(exc)}"
    return outputs


def build_raw_diagnostic_text(outputs: Dict[str, str], source_ip: str, destination_ip: str, port: int) -> str:
    lines = ["Network Diagnostic Report:"]
    if source_ip:
        lines.append(f"Source IP: {source_ip}")
    lines.extend([f"Destination IP: {destination_ip}", f"Port: {port}", ""])
    for command, output in outputs.items():
        if command == "source_note":
            continue
        lines.append(f"=== {command.upper()} ===")
        lines.append(output or "No output available.")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    st.set_page_config(
        page_title="Privacy Masking Demo",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("Privacy Masking Demo")
    st.caption("Simple demo for testing masking before sending diagnostics to the LLM and restoring them before display.")

    settings: AppSettings = get_settings()
    initialize_logger(settings.log_directory)

    with st.expander("Demo Inputs", expanded=True):
        col1, col2, col3 = st.columns([1, 1, 0.7])
        source_ip = col1.text_input("Source IP", value="")
        destination_ip = col2.text_input("Destination IP", value="")
        port = col3.number_input("Port", min_value=1, max_value=65535, value=22)
        run_demo = st.button("Run Demo")

    if run_demo:
        validation_errors: list[str] = []
        if not destination_ip.strip():
            validation_errors.append("Destination IP is required to run the demo.")
        elif not validate_ipv4_address(destination_ip):
            validation_errors.append("Destination IP is invalid.")
        if source_ip and not validate_ipv4_address(source_ip):
            validation_errors.append("Source IP is invalid.")

        if validation_errors:
            for error in validation_errors:
                st.error(error)
            st.stop()

        timeout_settings = {
            "ping": 10,
            "traceroute": 30,
            "telnet": 10,
        }

        with st.spinner("Running demo..."):
            command_outputs = execute_diagnostics(source_ip, destination_ip, port, timeout_settings)
            raw_output = build_raw_diagnostic_text(command_outputs, source_ip, destination_ip, port)
            masker = MaskingEngine()
            masked_output = masker.mask(raw_output)
            prompt_text = build_troubleshooting_prompt(masked_output)

            if not settings.llm_api_key:
                st.warning("The LLM API key is not configured in the environment file, so the demo cannot complete.")
                st.stop()

            try:
                client = create_llm_client(
                    provider=settings.llm_provider,
                    server_url=settings.llm_server_url,
                    model_name=settings.llm_model_name,
                    api_key=settings.llm_api_key,
                    auth_header=settings.llm_auth_header,
                    auth_prefix=settings.llm_auth_prefix,
                )
                llm_response = client.analyze(prompt_text)
            except (LLMClientError, Exception) as exc:
                st.error(f"LLM request failed: {exc}")
                st.stop()

            final_report = masker.unmask(llm_response.raw_response)

        st.success("Demo completed.")

        st.subheader("1. Raw diagnostic output")
        st.code(raw_output, language="text")

        st.subheader("2. Masked payload sent to the LLM")
        st.code(prompt_text, language="text")

        st.subheader("3. Raw response from the LLM")
        st.code(llm_response.raw_response, language="text")

        st.subheader("4. Final restored output")
        st.code(final_report, language="text")
    else:
        st.info("Enter a destination IP and click Run Demo to see the masking flow in action.")


if __name__ == "__main__":
    main()
