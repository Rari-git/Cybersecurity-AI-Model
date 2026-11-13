"""Process-local network denial for the V6 offline authority pipeline.

This guard blocks Python socket connection, DNS lookup and datagram-send APIs
while a protected operation runs. It is a defense-in-depth control, not a
replacement for an OS firewall, network namespace, or physically air-gapped
host: native extensions and child processes require OS-level isolation.
"""
from __future__ import annotations

import socket
from contextlib import contextmanager
from unittest.mock import patch
from typing import Iterator


class OfflineNetworkAccessRejected(RuntimeError):
    """Raised if offline authority code attempts network access."""


def _deny_network(*_args, **_kwargs):
    raise OfflineNetworkAccessRejected(
        "V6 adaptive authority is offline; network operations are prohibited"
    )


@contextmanager
def offline_network_guard() -> Iterator[None]:
    """Temporarily deny common Python socket/DNS APIs in the current process."""
    with (
        patch.object(socket.socket, "connect", _deny_network),
        patch.object(socket.socket, "connect_ex", _deny_network),
        patch.object(socket.socket, "sendto", _deny_network),
        patch.object(socket.socket, "sendmsg", _deny_network, create=True),
        patch.object(socket, "create_connection", _deny_network),
        patch.object(socket, "getaddrinfo", _deny_network),
        patch.object(socket, "gethostbyname", _deny_network),
        patch.object(socket, "gethostbyname_ex", _deny_network),
    ):
        yield
