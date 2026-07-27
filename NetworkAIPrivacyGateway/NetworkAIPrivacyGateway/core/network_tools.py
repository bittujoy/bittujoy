import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from core.utils import get_os_type


class NetworkCommandError(Exception):
    pass


def _build_ping_command(destination: str, timeout: int) -> List[str]:
    if get_os_type() == "windows":
        return ["ping", "-n", "4", "-w", str(timeout * 1000), destination]
    return ["ping", "-c", "4", "-W", str(timeout), destination]


def _build_traceroute_command(destination: str, timeout: int) -> List[str]:
    if get_os_type() == "windows":
        return ["tracert", "-d", destination]
    return ["traceroute", "-q", "1", destination]


def _build_telnet_probe(destination: str, port: int, timeout: int) -> List[str]:
    if get_os_type() == "windows":
        return ["powershell", "-NoProfile", "-Command", f"Test-NetConnection -ComputerName {destination} -Port {port} -WarningAction SilentlyContinue"]
    return ["bash", "-lc", f"timeout {timeout} bash -c 'cat < /dev/tcp/{destination}/{port}'"]


def _run_subprocess(command: List[str], timeout: int) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        output = result.stdout.strip() or result.stderr.strip()
        return output
    except subprocess.TimeoutExpired as exc:
        raise NetworkCommandError(f"Command timed out after {timeout} seconds") from exc
    except subprocess.SubprocessError as exc:
        raise NetworkCommandError("Failed to execute network command") from exc


def run_ping(destination: str, timeout: int) -> str:
    command = _build_ping_command(destination, timeout)
    return _run_subprocess(command, timeout)


def run_traceroute(destination: str, timeout: int) -> str:
    command = _build_traceroute_command(destination, timeout)
    return _run_subprocess(command, timeout)


def run_telnet(destination: str, port: int, timeout: int) -> str:
    command = _build_telnet_probe(destination, port, timeout)
    return _run_subprocess(command, timeout)
