"""Patch 03.1 tests: ephemeral session keys, incident-triggered REKEY, fail-closed behaviour.

Self-contained: builds its own public template and session in a temp dir, so it does not
depend on config/ or on the existing fixtures (the 82 regression tests are untouched).
"""
import json
import os
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from cryptography.hazmat.primitives import serialization

from utils import crypto_utils
from utils.crypto_utils import CommandSigner, KeyRotationPending, _canonical_bytes
from utils.key_lifecycle import (KeyAuthority, KeyHealthMonitor, RevocationLedger,
                                 RevocationStoreError, load_revocations, sweep_stale_sessions)

TOPIC = "home/actuators/door/lock"


def _template(directory: Path, signers=("controller",)) -> Path:
    doc = {"version": 1, "signers": {
        s: {"active_key_id": "static-old", "keys": {"static-old": {
            "public_key": "nonexistent.pem", "allowed_topics": [TOPIC]}}} for s in signers}}
    path = directory / "template.json"
    path.write_text(json.dumps(doc))
    return path


class SessionTestCase(unittest.TestCase):
    """Creates a fresh ephemeral session and points the process environment at it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.clock = [1000.0]
        self.authority = KeyAuthority.create_session(
            _template(self.tmp), base_dir=self.tmp, signers=["controller"],
            audit_path=self.tmp / "audit.jsonl", clock=lambda: self.clock[0])
        env = {**self.authority.env_public(), **self.authority.env_signer("controller"),
               "LUMEN_SEQUENCE_DB": str(self.tmp / "seq.db")}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        CommandSigner._reset_process_revocations()
        self.addCleanup(CommandSigner._reset_process_revocations)

    def sign(self):
        return CommandSigner.sign_command(TOPIC, "LOCK")

    def private_key(self):
        return serialization.load_pem_private_key(
            self.authority.private_key_path("controller").read_bytes(), password=None)

    def active(self):
        return self.authority.status()["controller"]["active_key_id"]


class StartupProvisioningTests(SessionTestCase):
    def test_fresh_session_signs_and_verifies_without_any_preconfigured_secret(self):
        frame = self.sign()
        ok, reason = CommandSigner.verify_command(TOPIC, frame, last_seen_nonce=0)
        self.assertEqual((ok, reason), (True, "AUTHENTICATED"))

    def test_every_session_gets_different_keys(self):
        other = KeyAuthority.create_session(_template(self.tmp), base_dir=self.tmp,
                                            audit_path=self.tmp / "a2.jsonl")
        self.assertNotEqual(self.authority.private_key_path("controller").read_bytes(),
                            other.private_key_path("controller").read_bytes())
        self.assertNotEqual(self.active(), other.status()["controller"]["active_key_id"])

    def test_trust_store_and_ledger_contain_no_private_material(self):
        for path in (self.authority.trust_path, self.authority.revocation_path):
            self.assertNotIn("PRIVATE", path.read_text())
        for pub in self.authority.keys_dir.iterdir():
            self.assertIn("BEGIN PUBLIC KEY", pub.read_text())

    @unittest.skipUnless(os.name == "posix", "POSIX mode bits")
    def test_private_key_and_session_dir_are_owner_only(self):
        self.assertEqual(stat.S_IMODE(self.authority.private_key_path("controller").stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.authority.session_dir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.authority.signing_dir.stat().st_mode), 0o700)

    def test_verifier_environment_never_contains_private_key_path(self):
        self.assertNotIn("LUMEN_COMMAND_PRIVATE_KEY", self.authority.env_public())
        self.assertIn("LUMEN_COMMAND_PRIVATE_KEY", self.authority.env_signer("controller"))

    def test_audit_log_has_key_ids_but_no_key_material(self):
        self.authority.rotate("controller", "test")
        text = (self.tmp / "audit.jsonl").read_text()
        self.assertIn("key_rotated", text)
        self.assertNotIn("PRIVATE", text)
        self.assertNotIn("BEGIN", text)

    def test_cleanup_overwrites_and_removes_private_keys(self):
        key_path = self.authority.private_key_path("controller")
        self.assertTrue(key_path.exists())
        self.authority.cleanup()
        self.assertFalse(self.authority.session_dir.exists())

    def test_sweep_removes_only_old_session_dirs(self):
        old = Path(tempfile.mkdtemp(prefix="lumen-session-", dir=self.tmp))
        os.utime(old, (1, 1))
        removed = sweep_stale_sessions(self.tmp, max_age_sec=3600)
        self.assertEqual(removed, 1)
        self.assertFalse(old.exists())
        self.assertTrue(self.authority.session_dir.exists())


class RotationTests(SessionTestCase):
    def test_rotation_invalidates_old_key_and_new_key_works(self):
        old_frame = self.sign()
        old_key = self.active()
        new_key = self.authority.rotate("controller", "test")
        self.assertNotEqual(old_key, new_key)
        ok, reason = CommandSigner.verify_command(TOPIC, old_frame, last_seen_nonce=0)
        self.assertFalse(ok)
        self.assertIn(reason, ("UNTRUSTED_SIGNER", "KEY_REVOKED"))
        ok, reason = CommandSigner.verify_command(TOPIC, self.sign(), last_seen_nonce=0)
        self.assertEqual((ok, reason), (True, "AUTHENTICATED"))
        self.assertIn(("controller", old_key), load_revocations(self.authority.revocation_path))

    def test_nonce_stays_monotonic_across_rotation(self):
        n1 = json.loads(self.sign())["nonce"]
        self.authority.rotate("controller", "test")
        n2 = json.loads(self.sign())["nonce"]
        self.assertGreater(n2, n1)

    def test_revoked_key_rejected_before_trust_store_commit(self):
        frame = self.sign()
        self.authority.ledger.revoke("controller", self.active(), "incident")
        ok, reason = CommandSigner.verify_command(TOPIC, frame, last_seen_nonce=0)
        self.assertEqual((ok, reason), (False, "KEY_REVOKED"))

    def test_signing_with_revoked_active_key_is_refused(self):
        self.authority.ledger.revoke("controller", self.active(), "incident")
        with self.assertRaises(KeyRotationPending):
            CommandSigner._sign_once(TOPIC, "LOCK")
        self.assertIsInstance(KeyRotationPending("x", 1.0), ValueError)   # legacy callers unaffected

    def test_poll_rotates_only_when_active_key_revoked(self):
        self.assertEqual(self.authority.poll(), [])
        key = self.active()
        self.authority.ledger.revoke("controller", key, "incident")
        events = self.authority.poll()
        self.assertEqual([e["event"] for e in events], ["rotated"])
        self.assertNotEqual(self.active(), key)
        self.assertEqual(self.authority.poll(), [])

    def test_signer_rides_out_in_flight_rotation(self):
        self.authority.ledger.revoke("controller", self.active(), "incident")
        timer = threading.Timer(0.3, self.authority.poll)
        timer.start()
        self.addCleanup(timer.join)              # rotation tail (audit, old pub cleanup) must finish before tmp teardown
        frame = self.sign()                      # blocks briefly, then signs with the NEW key
        ok, reason = CommandSigner.verify_command(TOPIC, frame, last_seen_nonce=0)
        self.assertEqual((ok, reason), (True, "AUTHENTICATED"))

    def test_scheduled_rotation_by_age(self):
        self.authority.max_key_age_sec = 60.0
        self.assertEqual(self.authority.poll(), [])
        self.clock[0] += 61
        events = self.authority.poll()
        self.assertEqual(events[0]["reason"], "scheduled")

    def test_rotation_cap_suspends_and_stays_locked_down(self):
        self.authority.max_rotations_per_hour = 1
        self.authority.ledger.revoke("controller", self.active(), "incident-1")
        self.assertEqual(self.authority.poll()[0]["event"], "rotated")
        self.authority.ledger.revoke("controller", self.active(), "incident-2")
        events = self.authority.poll()
        self.assertEqual(events[0]["event"], "suspended")
        self.assertEqual(self.authority.poll(), [])          # no retry storm
        self.assertTrue(self.authority.status()["controller"]["suspended"])
        with self.assertRaises(KeyRotationPending):          # signer locked out: fail closed
            CommandSigner._sign_once(TOPIC, "LOCK")

    def test_trust_store_cache_picks_up_rotation(self):
        # Atomic replace changes mtime, so a long-lived verifier process re-reads the store.
        before = crypto_utils._load_trust_store()["controller"]["active_key_id"]
        self.authority.rotate("controller", "test")
        after = crypto_utils._load_trust_store()["controller"]["active_key_id"]
        self.assertNotEqual(before, after)


class FailClosedTests(SessionTestCase):
    def test_corrupt_ledger_makes_verification_fail_closed(self):
        frame = self.sign()
        self.authority.revocation_path.write_text("{not json")
        ok, reason = CommandSigner.verify_command(TOPIC, frame, last_seen_nonce=0)
        self.assertEqual((ok, reason), (False, "REVOCATION_STORE_ERROR (fail-closed)"))
        with self.assertRaises(RevocationStoreError):
            CommandSigner.validate_trust_store()

    def test_ledger_writer_never_overwrites_a_corrupt_ledger(self):
        self.authority.revocation_path.write_text("garbage")
        with self.assertRaises(RevocationStoreError):
            RevocationLedger(self.authority.revocation_path).revoke("controller", "k1", "x")
        self.assertEqual(self.authority.revocation_path.read_text(), "garbage")

    def test_missing_ledger_means_nothing_revoked(self):
        self.authority.revocation_path.unlink()
        ok, _ = CommandSigner.verify_command(TOPIC, self.sign(), last_seen_nonce=0)
        self.assertTrue(ok)

    def test_in_process_revocation_holds_even_if_ledger_cannot_be_written(self):
        frame = self.sign()
        CommandSigner.revoke_in_process("controller", self.active())
        ok, reason = CommandSigner.verify_command(TOPIC, frame, last_seen_nonce=0)
        self.assertEqual((ok, reason), (False, "KEY_REVOKED"))

    def test_revocations_are_monotone_and_idempotent(self):
        ledger = RevocationLedger(self.authority.revocation_path)
        self.assertTrue(ledger.revoke("controller", "k-a", "r"))
        self.assertFalse(ledger.revoke("controller", "k-a", "again"))
        self.assertIn(("controller", "k-a"), load_revocations(self.authority.revocation_path))

    def test_invalid_ids_rejected_by_ledger(self):
        with self.assertRaises(ValueError):
            RevocationLedger(self.authority.revocation_path).revoke("../evil", "k", "x")


def _frame(nonce=1, sender="controller", key="k1", sig="a" * 128, action="LOCK") -> bytes:
    return json.dumps({"action": action, "timestamp": 1.0, "nonce": nonce, "sender_id": sender,
                       "key_id": key, "signature": sig}).encode()


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.t = [0.0]
        self.m = KeyHealthMonitor(window_sec=30, replay_threshold=3, tamper_threshold=4,
                                  cooldown_sec=20, clock=lambda: self.t[0])

    def test_equivocation_is_immediate_and_needs_a_differing_frame(self):
        self.m.on_accepted(TOPIC, _frame(nonce=7))
        same = self.m.on_rejected(TOPIC, "REPLAY_DUPLICATE_NONCE (Received: 7, Expected > 7)", _frame(nonce=7))
        self.assertFalse(same.revoke)                       # byte-identical = plain replay
        diff = self.m.on_rejected(TOPIC, "REPLAY_DUPLICATE_NONCE (Received: 7, Expected > 7)",
                                  _frame(nonce=7, action="UNLOCK"))
        self.assertTrue(diff.revoke)
        self.assertEqual((diff.sender_id, diff.key_id), ("controller", "k1"))

    def test_replay_threshold_inside_window(self):
        self.m.on_accepted(TOPIC, _frame(nonce=1))
        res = [self.m.on_rejected(TOPIC, "REPLAY_DUPLICATE_NONCE x", _frame(nonce=1)).revoke for _ in range(3)]
        self.assertEqual(res, [False, False, True])

    def test_replay_events_outside_window_do_not_accumulate(self):
        self.m.on_accepted(TOPIC, _frame(nonce=1))
        for _ in range(2):
            self.m.on_rejected(TOPIC, "REPLAY_DUPLICATE_NONCE x", _frame(nonce=1))
        self.t[0] += 31
        self.assertFalse(self.m.on_rejected(TOPIC, "REPLAY_DUPLICATE_NONCE x", _frame(nonce=1)).revoke)

    def test_tampering_threshold(self):
        res = [self.m.on_rejected(TOPIC, "INVALID_SIGNATURE", _frame()).revoke for _ in range(4)]
        self.assertEqual(res, [False, False, False, True])

    def test_cooldown_blocks_trigger_storm_but_not_equivocation(self):
        for _ in range(4):
            self.m.on_rejected(TOPIC, "INVALID_SIGNATURE", _frame())
        for _ in range(10):                                  # attacker keeps flooding the NEW key
            self.assertFalse(self.m.on_rejected(TOPIC, "INVALID_SIGNATURE", _frame(key="k2")).revoke)
        self.m.on_accepted(TOPIC, _frame(nonce=9, key="k2"))
        eq = self.m.on_rejected(TOPIC, "REPLAY_DUPLICATE_NONCE x", _frame(nonce=9, key="k2", action="UNLOCK"))
        self.assertTrue(eq.revoke)
        self.t[0] += 21
        res = [self.m.on_rejected(TOPIC, "INVALID_SIGNATURE", _frame(key="k3")).revoke for _ in range(4)]
        self.assertTrue(res[-1])

    def test_reasons_without_key_bound_evidence_never_trigger(self):
        for reason in ("KEY_REVOKED", "UNTRUSTED_SIGNER", "TOPIC_NOT_AUTHORIZED", "MALFORMED_PACKET (fields)",
                       "REPLAY_STORE_ERROR (fail-closed)", "INVALID_ACTION"):
            for _ in range(50):
                self.assertFalse(self.m.on_rejected(TOPIC, reason, _frame()).revoke, reason)

    def test_unparseable_frames_never_trigger(self):
        for raw in (b"\xff\xfe", b"[]", b'{"nonce": 1}', _frame(sender="../x")):
            for _ in range(20):
                self.assertFalse(self.m.on_rejected(TOPIC, "INVALID_SIGNATURE", raw).revoke)

    def test_accepted_ledger_is_bounded(self):
        m = KeyHealthMonitor(max_accepted=10)
        for n in range(1, 100):
            m.on_accepted(TOPIC, _frame(nonce=n))
        self.assertLessEqual(len(m._accepted), 10)

    def test_real_signature_equivocation_detected_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            authority = KeyAuthority.create_session(_template(tmp), base_dir=tmp)
            env = {**authority.env_public(), **authority.env_signer("controller"),
                   "LUMEN_SEQUENCE_DB": str(tmp / "seq.db")}
            with mock.patch.dict(os.environ, env):
                good = CommandSigner.sign_command(TOPIC, "LOCK")
                packet = json.loads(good)
                self.m.on_accepted(TOPIC, good.encode())
                # Two holders of the same key sign different content under one nonce.
                priv = serialization.load_pem_private_key(
                    authority.private_key_path("controller").read_bytes(), password=None)
                packet["action"] = "UNLOCK"
                msg = _canonical_bytes(TOPIC, action="UNLOCK", timestamp=packet["timestamp"],
                                       nonce=packet["nonce"], sender_id="controller",
                                       key_id=packet["key_id"])
                packet["signature"] = priv.sign(msg).hex()
                forged = json.dumps(packet, separators=(",", ":"))
                ok, reason = CommandSigner.verify_command(TOPIC, forged, last_seen_nonce=packet["nonce"])
                self.assertFalse(ok)
                self.assertTrue(reason.startswith("REPLAY_DUPLICATE_NONCE"), reason)  # signature DID verify
                decision = self.m.on_rejected(TOPIC, reason, forged.encode())
                self.assertTrue(decision.revoke)
            authority.cleanup()


class _FakeSession:
    def __init__(self): self.puts = []
    def put(self, key, payload, **kw): self.puts.append((str(key), bytes(payload)))


class _FakeEnforcement:
    def __init__(self): self.calls = []
    def execute_mitigation(self, attack, topic, loss=0.0, payload="", critical=False, action=None, ttl=None):
        self.calls.append({"attack": attack, "topic": topic, "action": action, "critical": critical})
        return action or "DROP"


class GatewayRekeyIntegrationTests(SessionTestCase):
    """Drives the real ZenohInlineGateway._handle_actuator with fakes for the network/enforcement."""

    def _gateway(self, rekey=True):
        try:
            from agent.gateway_ids import ZenohInlineGateway
        except ImportError as exc:          # pragma: no cover - environment without zenoh
            self.skipTest(f"gateway import unavailable: {exc}")
        gw = object.__new__(ZenohInlineGateway)
        gw._stats_lock, gw._nonce_lock = threading.Lock(), threading.Lock()
        gw.stats, gw.last_seen_nonces = {}, {}
        gw.session, gw.enforcement = _FakeSession(), _FakeEnforcement()
        gw.key_monitor = KeyHealthMonitor(tamper_threshold=3, replay_threshold=3, cooldown_sec=0.0) if rekey else None
        gw.revocations = RevocationLedger(CommandSigner.revocation_path()) if rekey else None
        return gw

    @staticmethod
    def _tamper(frame: str) -> bytes:
        packet = json.loads(frame)
        packet["signature"] = ("0" if packet["signature"][0] != "0" else "1") + packet["signature"][1:]
        return json.dumps(packet, separators=(",", ":")).encode()

    def test_tampering_burst_revokes_alerts_rotates_and_traffic_resumes(self):
        gw = self._gateway()
        legit = self.sign()
        old_key = self.active()
        for _ in range(3):
            gw._handle_actuator(TOPIC, self._tamper(legit))
        self.assertIn(("controller", old_key), load_revocations(self.authority.revocation_path))
        rekeys = [c for c in gw.enforcement.calls if c["action"] == "REKEY"]
        self.assertEqual(len(rekeys), 1)
        self.assertTrue(rekeys[0]["critical"])
        # In-flight frame from the revoked key is refused and is NOT counted as an attack trigger.
        gw._handle_actuator(TOPIC, legit.encode())
        self.assertEqual(gw.enforcement.calls[-1]["attack"], "Revoked Key Use")
        self.assertEqual(gw.session.puts, [])
        # Authority rotates; the very next legitimate command flows end to end.
        self.assertEqual(self.authority.poll()[0]["event"], "rotated")
        gw._handle_actuator(TOPIC, self.sign().encode())
        self.assertEqual(len(gw.session.puts), 1)
        self.assertEqual(gw.stats["actuator_forwarded"], 1)

    def test_equivocation_revokes_on_first_event(self):
        gw = self._gateway()
        good = self.sign()
        gw._handle_actuator(TOPIC, good.encode())
        self.assertEqual(len(gw.session.puts), 1)
        packet = json.loads(good)
        packet["action"] = "UNLOCK"
        msg = _canonical_bytes(TOPIC, action="UNLOCK", timestamp=packet["timestamp"], nonce=packet["nonce"],
                               sender_id="controller", key_id=packet["key_id"])
        packet["signature"] = self.private_key().sign(msg).hex()
        gw._handle_actuator(TOPIC, json.dumps(packet, separators=(",", ":")).encode())
        self.assertEqual(len(gw.session.puts), 1)            # forged copy NOT forwarded
        self.assertTrue(any(c["action"] == "REKEY" for c in gw.enforcement.calls))
        self.assertIn(("controller", packet["key_id"]), load_revocations(self.authority.revocation_path))

    def test_plain_replays_below_threshold_do_not_rotate(self):
        gw = self._gateway()
        good = self.sign()
        gw._handle_actuator(TOPIC, good.encode())
        for _ in range(2):
            gw._handle_actuator(TOPIC, good.encode())
        self.assertFalse(any(c["action"] == "REKEY" for c in gw.enforcement.calls))
        self.assertEqual(load_revocations(self.authority.revocation_path), frozenset())

    def test_ledger_failure_still_revokes_in_process(self):
        gw = self._gateway()
        gw.revocations = mock.Mock(revoke=mock.Mock(side_effect=OSError("disk full")))
        legit = self.sign()
        for _ in range(3):
            gw._handle_actuator(TOPIC, self._tamper(legit))
        self.assertEqual(gw.stats["rekey_errors"], 1)
        gw._handle_actuator(TOPIC, legit.encode())
        self.assertEqual(gw.session.puts, [])                # still fail-closed

    def test_rekey_disabled_preserves_legacy_behaviour(self):
        gw = self._gateway(rekey=False)
        legit = self.sign()
        for _ in range(50):
            gw._handle_actuator(TOPIC, self._tamper(legit))
        self.assertFalse(any(c["action"] == "REKEY" for c in gw.enforcement.calls))
        self.assertEqual(load_revocations(self.authority.revocation_path), frozenset())
        gw._handle_actuator(TOPIC, legit.encode())
        self.assertEqual(len(gw.session.puts), 1)            # legit key unaffected


class EnforcementRekeyActionTests(unittest.TestCase):
    def test_rekey_action_is_critical_and_never_blocks_the_topic(self):
        try:
            from agent.enforcement import EnforcementEngine
        except ImportError as exc:          # pragma: no cover
            self.skipTest(f"enforcement import unavailable: {exc}")
        session = _FakeSession()
        engine = EnforcementEngine(session, is_inline_gateway=True)
        action = engine.execute_mitigation("Key Compromise Suspected", TOPIC, 99.0, "detail",
                                           critical=True, action="REKEY")
        self.assertEqual(action, "REKEY")
        alert = json.loads(session.puts[-1][1])
        self.assertEqual((alert["action"], alert["severity"], alert["block_ttl"]), ("REKEY", "critical", 0.0))
        self.assertFalse(engine.is_blocked(TOPIC))           # a rekey must not DoS legitimate commands


if __name__ == "__main__":
    unittest.main()
