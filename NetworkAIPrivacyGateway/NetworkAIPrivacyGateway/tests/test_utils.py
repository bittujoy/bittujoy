from core.utils import validate_ipv4_address, get_os_type


def test_validate_ipv4_address_accepts_valid_ip():
    assert validate_ipv4_address("10.0.0.1")


def test_validate_ipv4_address_rejects_invalid_ip():
    assert not validate_ipv4_address("300.300.300.300")


def test_get_os_type_returns_supported_string():
    assert get_os_type() in {"windows", "linux"}
