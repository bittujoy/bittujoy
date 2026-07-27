import ipaddress
import platform
from pathlib import Path
from typing import Tuple


def validate_ipv4_address(candidate: str) -> bool:
    try:
        ipaddress.IPv4Address(candidate.strip())
        return True
    except (ipaddress.AddressValueError, ValueError):
        return False


def get_os_type() -> str:
    name = platform.system().lower()
    if "windows" in name:
        return "windows"
    return "linux"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_lines(text: str) -> Tuple[str, ...]:
    return tuple(line for line in text.splitlines() if line.strip())
