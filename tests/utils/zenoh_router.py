"""Generate a generation-bound Zenoh router config from the active mTLS bundle."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import json5

from utils.cert_lifecycle import CertAuthority, CertRevocationLedger, CertStoreError
from utils.key_lifecycle import atomic_write_bytes


def _replace_one(text: str, pattern: str, replacement: str, label: str) -> str:
    text2, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise CertStoreError(f"router config template must contain exactly one {label} field")
    return text2





def _key_exprs_overlap(left: str, right: str) -> bool:
    """Return whether two segment-based Zenoh expressions can match one key.

    Supports literal segments, ``*`` (one segment), and ``**`` (zero or more
    segments). The matcher is deliberately conservative for malformed
    wildcard syntax: malformed expressions are rejected by the caller.
    """
    from functools import lru_cache

    def parse(expr: str) -> tuple[str, ...]:
        if not isinstance(expr, str) or not expr or expr.startswith("/") or expr.endswith("/"):
            raise CertStoreError(f"invalid Zenoh key expression: {expr!r}")
        parts = tuple(expr.split("/"))
        for part in parts:
            if not part or ("*" in part and part not in {"*", "**"}):
                raise CertStoreError(f"unsupported wildcard syntax in Zenoh key expression: {expr!r}")
        return parts

    a, b = parse(left), parse(right)

    @lru_cache(maxsize=None)
    def visit(i: int, j: int) -> bool:
        if i == len(a) and j == len(b):
            return True
        if i < len(a) and a[i] == "**":
            # ** may match zero segments or consume a segment matched by b.
            if visit(i + 1, j):
                return True
            if j < len(b):
                if b[j] == "**":
                    return visit(i, j + 1)
                return visit(i, j + 1)
            return False
        if j < len(b) and b[j] == "**":
            if visit(i, j + 1):
                return True
            if i < len(a):
                return visit(i + 1, j)
            return False
        if i >= len(a) or j >= len(b):
            return False
        if a[i] == "*" or b[j] == "*" or a[i] == b[j]:
            return visit(i + 1, j + 1)
        return False

    return visit(0, 0)


def validate_router_acl_template(template_path) -> dict:
    """Fail closed unless the JSON5 router template enforces the E1/E2 trust boundary.

    The shipped launcher accepts ``LUMEN_ZENOH_ROUTER_CONFIG`` as an operator
    override. Validate the rendered template structurally so an older or
    weakened override cannot silently remove the internal Gateway/IDS ACLs.
    """
    path = Path(template_path)
    try:
        document = json5.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CertStoreError(f"invalid Zenoh router JSON5 template {path}: {exc}") from exc

    if not isinstance(document, dict):
        raise CertStoreError("Zenoh router template root must be an object")
    access = document.get("access_control")
    if not isinstance(access, dict) or access.get("enabled") is not True:
        raise CertStoreError("Zenoh router ACL must be enabled")
    if access.get("default_permission") != "deny":
        raise CertStoreError("Zenoh router ACL must retain default_permission: deny")

    rules = access.get("rules")
    policies = access.get("policies")
    if not isinstance(rules, list) or not isinstance(policies, list):
        raise CertStoreError("Zenoh router ACL rules and policies must be arrays")
    rule_map: dict[str, dict] = {}
    for rule in rules:
        if not isinstance(rule, dict) or not isinstance(rule.get("id"), str):
            raise CertStoreError("every Zenoh ACL rule must be an object with a string id")
        rule_id = rule["id"]
        if rule_id in rule_map:
            raise CertStoreError(f"duplicate Zenoh ACL rule id: {rule_id}")
        rule_map[rule_id] = rule
    required = {
        "in_put_internal_ids_telemetry": ("internal/ids/telemetry/**", "put", "ingress"),
        "out_put_internal_ids_telemetry": ("internal/ids/telemetry/**", "put", "egress"),
        "sub_internal_ids_telemetry": ("internal/ids/telemetry/**", "declare_subscriber", "ingress"),
        "in_put_pep_ack": ("control/pep_ack/**", "put", "ingress"),
        "out_put_pep_ack": ("control/pep_ack/**", "put", "egress"),
        "sub_pep_ack": ("control/pep_ack/**", "declare_subscriber", "ingress"),
        "in_put_policy_ctrl": ("control/policy/**", "put", "ingress"),
        "out_put_policy_ctrl": ("control/policy/**", "put", "egress"),
        "sub_policy_ctrl": ("control/policy/**", "declare_subscriber", "ingress"),
    }
    for rule_id, (key_expr, message, flow) in required.items():
        rule = rule_map.get(rule_id)
        if not isinstance(rule, dict):
            raise CertStoreError(f"Zenoh router ACL missing required rule: {rule_id}")
        if (rule.get("permission") != "allow" or rule.get("key_exprs") != [key_expr]
                or rule.get("messages") != [message] or rule.get("flows") != [flow]):
            raise CertStoreError(f"Zenoh router ACL rule is weakened or malformed: {rule_id}")

    grants: dict[str, set[str]] = {rule_id: set() for rule_id in required}
    for policy in policies:
        if not isinstance(policy, dict):
            continue
        subjects = policy.get("subjects", [])
        referenced_rules = policy.get("rules", [])
        if not isinstance(subjects, list) or not isinstance(referenced_rules, list):
            raise CertStoreError("Zenoh ACL policy subjects/rules must be arrays")
        for rule_id in set(referenced_rules) & grants.keys():
            grants[rule_id].update(str(subject) for subject in subjects)

    subject_rows = access.get("subjects")
    if not isinstance(subject_rows, list):
        raise CertStoreError("Zenoh ACL subjects must be an array")
    subject_ids = {
        str(subject.get("id")): subject
        for subject in subject_rows
        if isinstance(subject, dict) and subject.get("id") is not None
    }
    for role in ("gateway", "ids"):
        subject = subject_ids.get(role)
        if not isinstance(subject, dict) or not subject.get("cert_common_names"):
            raise CertStoreError(f"Zenoh ACL subject {role!r} must have a certificate CN binding")

    # Identity envelopes and PEP ACKs have a strict two-principal boundary.
    strict_subjects = {
        "in_put_internal_ids_telemetry": {"gateway"},
        "in_put_pep_ack": {"gateway"},
        "out_put_internal_ids_telemetry": {"ids"},
        "out_put_pep_ack": {"ids"},
        "sub_internal_ids_telemetry": {"ids"},
        "sub_pep_ack": {"ids"},
    }
    for rule_id, expected in strict_subjects.items():
        if grants[rule_id] != expected:
            raise CertStoreError(
                f"Zenoh ACL rule {rule_id} must be assigned only to {sorted(expected)} "
                f"(found {sorted(grants[rule_id])})"
            )

    # Closed-world validation: inspect every allow rule, not just the required
    # rule IDs. A new expression such as internal/** or */pep_ack/** must not
    # grant another role access through an overlap with a protected namespace.
    protected_exprs = ("internal/**", "control/pep_ack/**")
    for rule_id, rule in rule_map.items():
        if rule.get("permission") != "allow":
            continue
        expressions = rule.get("key_exprs", [])
        messages = rule.get("messages", [])
        flows = rule.get("flows", [])
        if not isinstance(expressions, list) or not isinstance(messages, list) or not isinstance(flows, list):
            raise CertStoreError(f"malformed ACL expression/messages/flows in rule {rule_id}")
        overlaps = any(
            _key_exprs_overlap(expression, protected)
            for expression in expressions
            for protected in protected_exprs
        )
        if not overlaps:
            continue

        allowed_roles: set[str] | None = None
        for message in messages:
            for flow in flows:
                if message == "put" and flow == "ingress":
                    allowed = {"gateway"}
                elif message == "put" and flow == "egress":
                    allowed = {"ids"}
                elif message == "declare_subscriber" and flow == "ingress":
                    allowed = {"ids"}
                else:
                    raise CertStoreError(
                        f"unsupported allow operation overlaps protected Zenoh namespace in rule {rule_id}: "
                        f"{message}/{flow}"
                    )
                allowed_roles = allowed if allowed_roles is None else (allowed_roles & allowed)
        if allowed_roles is None:
            raise CertStoreError(f"protected overlapping ACL rule {rule_id} has no explicit operation")
        granted: set[str] = set()
        for policy in policies:
            if not isinstance(policy, dict):
                raise CertStoreError("ACL policies must be objects")
            if rule_id in policy.get("rules", []):
                subjects = policy.get("subjects", [])
                if not isinstance(subjects, list):
                    raise CertStoreError(f"ACL policy subjects must be an array for {rule_id}")
                granted.update(str(subject) for subject in subjects)
        if granted and not granted.issubset(allowed_roles):
            raise CertStoreError(
                f"ACL rule {rule_id} overlaps a protected namespace and grants unauthorized subjects "
                f"{sorted(granted - allowed_roles)} (allowed: {sorted(allowed_roles)})"
            )

    # The policy-command route has legacy V5 operator/monitoring principals in
    # the production template. Require the actual IDS -> Gateway path while
    # restricting any optional subscribers/publishers to those named roles.
    policy_allowlists = {
        "in_put_policy_ctrl": ({"ids"}, {"ids", "controller", "lab_tester"}),
        "out_put_policy_ctrl": ({"gateway"}, {"gateway", "dashboard", "lab_tester"}),
        "sub_policy_ctrl": ({"gateway"}, {"gateway", "dashboard", "lab_tester"}),
    }
    for rule_id, (required, allowed) in policy_allowlists.items():
        if not required.issubset(grants[rule_id]) or not grants[rule_id].issubset(allowed):
            raise CertStoreError(
                f"Zenoh ACL rule {rule_id} must include {sorted(required)} and may only "
                f"include {sorted(allowed)} (found {sorted(grants[rule_id])})"
            )
    return document

def build_router_config(template_path, pointer_path, ledger_path, output_path) -> tuple[str, Path]:
    """Render TLS paths and per-role cert CNs; return config digest and bundle dir.

    The ACL subject IDs and policy references stay stable (``ids``, ``gateway``),
    but the cert_common_names set becomes generation-specific. A revoked current
    serial is mapped to a non-issued sentinel CN, never to an empty/wildcard
    subject, because Zenoh interprets omitted subject match lists as wildcards.
    """
    pointer, bundle_dir, manifest = CertAuthority.inspect_active(pointer_path)
    ledger = CertRevocationLedger(ledger_path)
    revoked = ledger.revoked_serials()  # validates/fails closed before rendering
    identities = manifest.get("identities", {})
    quarantine = set(pointer.get("quarantined", []))

    router_info = identities.get("router")
    if not isinstance(router_info, dict) or ("router", router_info.get("serial")) in revoked:
        raise CertStoreError("router certificate is revoked or missing")
    ca_path = (bundle_dir / "ca.pem").resolve()
    router_cert = (bundle_dir / router_info["cert_file"]).resolve()
    router_key = (bundle_dir / router_info["key_file"]).resolve()
    for path in (ca_path, router_cert, router_key):
        if not path.is_file():
            raise CertStoreError(f"router TLS file missing: {path}")

    text = Path(template_path).read_text(encoding="utf-8")
    # JSON5 path strings are valid JSON string literals after json.dumps.
    replacements = {
        "root_ca_certificate": ca_path.as_posix(),
        "listen_private_key": router_key.as_posix(),
        "listen_certificate": router_cert.as_posix(),
    }
    for field, value in replacements.items():
        pattern = rf"(\b{field}\s*:\s*)('[^']*'|\"[^\"]*\")"
        text = _replace_one(text, pattern, lambda m, v=json.dumps(value): m.group(1) + v,
                            field)

    # Force expiration-driven link closure on TLS mTLS listeners (keep valid JSON5 comma).
    if re.search(r"\bclose_link_on_expiration\s*:", text):
        text = re.sub(r"(\bclose_link_on_expiration\s*:\s*)(true|false)",
                      r"\1true", text, count=1)
    else:
        text = _replace_one(
            text,
            r"(\benable_mtls\s*:\s*true)\s*,?",
            r"\1,\n        close_link_on_expiration: true,",
            "enable_mtls",
        )

    for identity, info in identities.items():
        if identity == "router":
            continue
        if identity in quarantine or (identity, info.get("serial")) in revoked:
            allowed = [f"__lumen_denied__{identity}"]
        else:
            allowed = [str(info["common_name"])]
        pattern = (r"(\{\s*id\s*:\s*[\"']?" + re.escape(identity) +
                   r"[\"']?\s*,\s*cert_common_names\s*:\s*)\[[^\]]*\](\s*\})")
        text = _replace_one(text, pattern,
                            lambda m, values=json.dumps(allowed): m.group(1) + values + m.group(2),
                            f"ACL subject {identity}")

    # Make publication atomic; router reads the fully rendered version at exec.
    data = text.encode("utf-8")
    atomic_write_bytes(Path(output_path), data)
    digest = hashlib.sha256(data).hexdigest()
    state_digest = hashlib.sha256((
        pointer["bundle_id"] + "|" + str(pointer.get("revision", 0)) + "|" +
        json.dumps(sorted(revoked)) + "|" + digest
    ).encode("utf-8")).hexdigest()
    return state_digest, bundle_dir