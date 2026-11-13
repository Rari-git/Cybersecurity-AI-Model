import pytest
from utils.zenoh_conn import validate_endpoint_policy


def test_production_requires_tls():
    with pytest.raises(ValueError, match="forbidden"):
        validate_endpoint_policy("tcp/127.0.0.1:7447", "production", True)


def test_development_tcp_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="LUMEN_ALLOW_INSECURE_LOCAL"):
        validate_endpoint_policy("tcp/127.0.0.1:7447", "development", False)


def test_development_tcp_is_loopback_only():
    with pytest.raises(ValueError, match="loopback"):
        validate_endpoint_policy("tcp/192.168.1.50:7447", "development", True)


def test_explicit_local_development_tcp_is_accepted():
    assert validate_endpoint_policy("tcp/localhost:7447", "development", True) == "tcp"
    assert validate_endpoint_policy("tcp/127.0.0.1:7447", "development", True) == "tcp"


def test_tls_is_accepted_in_production():
    assert validate_endpoint_policy("tls/127.0.0.1:7447", "production", False) == "tls"


@pytest.mark.parametrize("endpoint", [
    "127.0.0.1:7447", "tcp/127.0.0.1", "tls/:7447", "tcp/host:99999", "udp/host:7447",
])
def test_malformed_endpoint_is_rejected(endpoint):
    with pytest.raises(ValueError):
        validate_endpoint_policy(endpoint, "development", True)


def test_unknown_environment_is_rejected():
    with pytest.raises(ValueError, match="LUMEN_ENV"):
        validate_endpoint_policy("tls/127.0.0.1:7447", "test", False)


def test_tls_identity_path_component_is_restricted_in_build_config(monkeypatch, tmp_path):
    import sys
    import types
    from utils import zenoh_conn
    from utils.cert_lifecycle import CertAuthority

    class FakeConfig:
        def __init__(self):
            self.values = {}

        def insert_json5(self, key, value):
            self.values[key] = value

    monkeypatch.setitem(sys.modules, "zenoh", types.SimpleNamespace(Config=FakeConfig))

    authority = CertAuthority.create_session(
        tmp_path,
        identities=("router", "gateway", "ids", "honeypot"),
    )
    try:
        monkeypatch.setenv("LUMEN_ENV", "production")
        monkeypatch.setenv("LUMEN_ZENOH_ENDPOINT", "tls/127.0.0.1:7447")
        monkeypatch.setenv("LUMEN_CERT_POINTER", str(authority.pointer_path))
        monkeypatch.setenv("LUMEN_CERT_REVOCATIONS", str(authority.revocation_path))

        with pytest.raises(ValueError, match="TLS identity"):
            zenoh_conn.build_client_config("../gateway")

        cfg = zenoh_conn.build_client_config("gateway")
        assert "transport/link/tls/enable_mtls" in cfg.values
    finally:
        authority.cleanup()