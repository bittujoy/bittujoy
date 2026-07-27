import time
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from core.constants import (
    FINAL_REPORT_LABEL,
    LLM_RESPONSE_LABEL,
    MASKED_OUTPUT_LABEL,
    PROMPT_LABEL,
    RAW_OUTPUT_LABEL,
)
from core.llm_client import (
    LLMClient,
    LLMClientError,
    LLMResponse,
    OllamaClient,
    OpenAIClient,
)
from core.logger import initialize_logger
from core.masker import MaskingEngine
from core.network_tools import NetworkCommandError, run_ping, run_telnet, run_traceroute
from core.prompt_builder import build_troubleshooting_prompt
from core.utils import validate_ipv4_address


def create_llm_client(provider: str, server_url: str, model_name: str, api_key: str) -> LLMClient:
    if provider.lower() == "openai":
        return OpenAIClient(server_url=server_url, model_name=model_name, api_key=api_key)
    if provider.lower() == "ollama":
        return OllamaClient(server_url=server_url, model_name=model_name, api_key=api_key)
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
        page_title="Network AI Privacy Gateway",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Network AI Privacy Gateway")
    st.caption("Protect sensitive network data by tokenizing before sending diagnostics to an LLM.")

    settings: Optional[AppSettings] = None
    logger = initialize_logger(Path("logs"))

    with st.sidebar.expander("LLM Configuration", expanded=True):
        provider = st.selectbox("Provider", ["OpenAI", "Ollama"])
        server_url = st.text_input("Server URL", value="http://localhost:11434")
        model_name = st.text_input("Model Name", value="gpt-4o-mini")
        api_key = st.text_input("API Key", type="password")
        if st.button("Test Connection"):
            try:
                client = create_llm_client(provider, server_url, model_name, api_key)
                connected = client.connect()
                if connected:
                    st.success("Connection successful.")
                else:
                    st.error("Connection failed: invalid provider or model.")
            except Exception as exc:
                st.error(f"Connection test failed: {exc}")

    st.markdown("---")

    with st.expander("Network Diagnostic Panel", expanded=True):
        cols = st.columns([1, 1, 1, 0.7, 0.7, 0.7])
        source_ip = cols[0].text_input("Source IP")
        destination_ip = cols[1].text_input("Destination IP")
        port = cols[2].number_input("Port", min_value=1, max_value=65535, value=22)
        ping_button = cols[3].button("Execute Ping")
        traceroute_button = cols[4].button("Execute Traceroute")
        telnet_button = cols[5].button("Execute Telnet")
        run_all_button = st.button("Run All Diagnostics")

    diagnostics_requested = run_all_button or ping_button or traceroute_button or telnet_button
    validation_errors: list[str] = []

    if diagnostics_requested:
        if not destination_ip.strip():
            validation_errors.append("Destination IP is required to run diagnostics.")
        elif not validate_ipv4_address(destination_ip):
            validation_errors.append("Destination IP is invalid.")
        if source_ip and not validate_ipv4_address(source_ip):
            validation_errors.append("Source IP is invalid.")

        if validation_errors:
            for error in validation_errors:
                st.error(error)
            st.stop()
    else:
        if not destination_ip.strip():
            st.info("Enter a destination IP and click a diagnostic button to begin.")
        elif source_ip and not validate_ipv4_address(source_ip):
            st.warning("Source IP is invalid and will not be used until corrected.")

    timeout_settings = {
        "ping": 10,
        "traceroute": 30,
        "telnet": 10,
    }

    command_outputs: Dict[str, str] = {}
    executed = False
    start_time = time.monotonic()

    if run_all_button or ping_button or traceroute_button or telnet_button:
        with st.spinner("Running network diagnostics..."):
            if ping_button or run_all_button:
                try:
                    command_outputs["ping"] = run_ping(destination_ip, timeout_settings["ping"])
                except NetworkCommandError as exc:
                    command_outputs["ping"] = str(exc)
            if traceroute_button or run_all_button:
                try:
                    command_outputs["traceroute"] = run_traceroute(destination_ip, timeout_settings["traceroute"])
                except NetworkCommandError as exc:
                    command_outputs["traceroute"] = str(exc)
            if telnet_button or run_all_button:
                try:
                    command_outputs["telnet"] = run_telnet(destination_ip, port, timeout_settings["telnet"])
                except NetworkCommandError as exc:
                    command_outputs["telnet"] = str(exc)
            executed = True

    if executed:
        raw_output = build_raw_diagnostic_text(command_outputs, source_ip, destination_ip, port)
        st.success(f"Diagnostics executed in {time.monotonic() - start_time:.2f} seconds.")
        st.session_state["raw_output"] = raw_output
        st.session_state["command_outputs"] = command_outputs
    else:
        raw_output = st.session_state.get("raw_output", "")

    if raw_output:
        with st.expander(RAW_OUTPUT_LABEL, expanded=True):
            st.code(raw_output, language="text")

        masker = MaskingEngine()
        masked_output = masker.mask(raw_output)
        prompt_text = build_troubleshooting_prompt(masked_output)

        with st.expander("Privacy Gateway", expanded=True):
            st.write("Original Text")
            st.code(raw_output, language="text")
            st.write("Masked Text")
            st.code(masked_output, language="text")
            st.write("Mask Mapping Table")
            mapping_table = masker.get_mapping_table()
            st.table([
                {"Original Value": row.original_value, "Token": row.token}
                for row in mapping_table
            ])

        with st.expander(PROMPT_LABEL, expanded=True):
            st.code(prompt_text, language="text")

        model_response: Optional[LLMResponse] = None
        if st.button("Send to LLM"):
            if not api_key.strip():
                st.error("API key is required to send data to the LLM.")
            else:
                try:
                    client = create_llm_client(provider, server_url, model_name, api_key)
                    llm_response = client.analyze(prompt_text)
                    st.session_state["llm_response"] = llm_response.raw_response
                    st.session_state["api_latency"] = llm_response.latency_seconds
                    st.session_state["provider"] = llm_response.provider
                    model_response = llm_response
                except (LLMClientError, Exception) as exc:
                    st.error(f"LLM request failed: {exc}")

        if "llm_response" in st.session_state:
            model_response = LLMResponse(
                raw_response=st.session_state["llm_response"],
                latency_seconds=st.session_state.get("api_latency", 0.0),
                provider=st.session_state.get("provider", provider),
            )

        if model_response:
            with st.expander(LLM_RESPONSE_LABEL, expanded=True):
                st.code(model_response.raw_response, language="text")
                st.info(f"API latency: {model_response.latency_seconds:.2f} seconds")

            final_report = masker.unmask(model_response.raw_response)
            with st.expander(FINAL_REPORT_LABEL, expanded=True):
                st.code(final_report, language="text")

            if st.button("Download Final Report"):
                st.download_button(
                    label="Download final report",
                    data=final_report,
                    file_name="final_report.txt",
                    mime="text/plain",
                )
    else:
        st.info("Execute diagnostics to begin the Privacy Gateway demonstration.")


if __name__ == "__main__":
    main()
