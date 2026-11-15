"""Fail-closed isolation policy and capability reporting.

C2.3 provides application-level staging controls only. It does not verify or
enforce Windows OS containment and therefore never authorizes scenario execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import platform


@dataclass(frozen=True)
class IsolationPolicy:
    network_allowed: bool = False
    subprocess_allowed: bool = False
    production_access_allowed: bool = False
    require_os_containment: bool = True

    def __post_init__(self) -> None:
        if self.network_allowed or self.subprocess_allowed or self.production_access_allowed:
            raise ValueError(
                "Staging policy must deny network, subprocess and production access"
            )


@dataclass(frozen=True)
class IsolationReport:
    platform_name: str
    # Configuration rejects dangerous options, but Python does not enforce
    # process-wide network/subprocess denial. Keep these claims separate.
    application_restrictions_configured: bool
    application_restrictions_enforced: bool
    application_restrictions_verified: bool
    os_containment_verified: bool
    os_backend: str | None
    checks: tuple[str, ...]
    staging_ready: bool
    execution_ready: bool
    limitations: tuple[str, ...]


def assess_isolation(policy: IsolationPolicy = IsolationPolicy()) -> IsolationReport:
    """Report C2.3 staging readiness without claiming OS isolation.

    `require_os_containment=False` may describe a relaxed future policy, but it
    does not turn application-level checks into OS containment and never
    authorizes scenario execution.
    """
    checks = (
        "network use is disallowed by staging configuration only",
        "subprocess use is disallowed by staging configuration only",
        "production data and PKI access are disallowed by staging configuration only",
        "workspace and resource guards are application-level controls, not OS enforcement",
    )
    limitations = (
        "Python application restrictions are not an OS security boundary",
        "C2.3 does not configure Windows Firewall, Job Objects or a VM boundary",
        "no arbitrary or untrusted code may be executed by this staging package",
        "OS-level process and network containment is not verified",
    )
    return IsolationReport(
        platform_name=f"{platform.system()} {platform.release()}",
        application_restrictions_configured=True,
        application_restrictions_enforced=False,
        application_restrictions_verified=False,
        os_containment_verified=False,
        os_backend=None,
        checks=checks,
        staging_ready=True,
        execution_ready=False,
        limitations=limitations,
    )
