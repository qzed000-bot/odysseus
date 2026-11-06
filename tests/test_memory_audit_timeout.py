from pathlib import Path


def test_memory_audit_uses_its_own_llm_timeout():
    source = Path("src/request_timeout.py").read_text()
    start = source.index("_TIMEOUT_EXEMPT_PREFIXES =")
    end = source.index("\n)\n", start)
    timeout_exemptions = source[start:end]

    assert '"/api/memory/audit"' in timeout_exemptions
