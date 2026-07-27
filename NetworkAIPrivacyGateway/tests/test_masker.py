from core.masker import MaskingEngine


def test_ipv4_masking_is_deterministic():
    engine = MaskingEngine()
    original = "Ping 10.10.10.1 and 10.10.10.1"
    masked = engine.mask(original)

    assert "IP1" in masked
    assert masked.count("IP1") == 2
    assert engine.unmask(masked) == original


def test_mapping_table_contains_original_values():
    engine = MaskingEngine()
    engine.mask("Connect to 192.168.0.5 and 10.0.0.10")
    table = engine.get_mapping_table()
    values = {entry.original_value for entry in table}
    assert "192.168.0.5" in values
    assert "10.0.0.10" in values
