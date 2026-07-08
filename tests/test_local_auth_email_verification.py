"""Unit tests for local-auth email verification (Resend-backed, transport injected).

These tests are docker-free: they exercise the email-verification code path in
``hypha.local_auth`` directly, injecting a REAL in-process fake email transport that
captures sent messages (NO unittest.mock / monkeypatch of internals). They cover:

- code generation + successful verification -> account creation -> token issuance
- wrong code (attempt counting)
- expired code (TTL)
- max attempts lockout
- resend throttling + resend delivering a fresh code
- cleanup / TTL eviction of abandoned pending-verification sessions
- Resend transport payload shape (using the injected fake to capture the HTTP call)
- dev fallback behaviour when RESEND_API_KEY is unset
"""

import asyncio
import time

import pytest

import hypha.local_auth as la
from hypha.local_auth import (
    CapturingEmailTransport,
    PendingVerificationStore,
    generate_verification_code,
    set_email_transport,
    reset_email_transport,
    build_email_transport,
    ResendEmailTransport,
    LoggingEmailTransport,
    VERIFICATION_CODE_TTL,
    MAX_VERIFICATION_ATTEMPTS,
    RESEND_THROTTLE_SECONDS,
)


@pytest.fixture
def capturing_transport():
    """Inject a real in-process capturing transport; reset afterwards."""
    transport = CapturingEmailTransport()
    set_email_transport(transport)
    yield transport
    reset_email_transport()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_generate_verification_code_is_numeric_and_fixed_length():
    for _ in range(50):
        code = generate_verification_code()
        assert code.isdigit()
        assert len(code) == 6


def test_generate_verification_code_is_random():
    codes = {generate_verification_code() for _ in range(200)}
    # With 10^6 space and 200 draws, collisions are extremely unlikely to make this <150.
    assert len(codes) > 150


# ---------------------------------------------------------------------------
# PendingVerificationStore: TTL, attempts, throttling, bounded eviction
# ---------------------------------------------------------------------------


def test_pending_store_add_and_get():
    store = PendingVerificationStore(ttl=100, max_entries=10)
    store.put("a@example.com", {"code": "123456", "name": "A", "password_hash": "h", "salt": "s"})
    entry = store.get("a@example.com")
    assert entry is not None
    assert entry["code"] == "123456"


def test_pending_store_ttl_expiry():
    store = PendingVerificationStore(ttl=1, max_entries=10)
    store.put("a@example.com", {"code": "123456"})
    assert store.get("a@example.com") is not None
    # Force time forward past the TTL without sleeping by rewriting created_at.
    store._entries["a@example.com"]["created_at"] = time.time() - 2
    assert store.get("a@example.com") is None  # expired entries are evicted on read
    assert "a@example.com" not in store._entries


def test_pending_store_bounded_eviction():
    store = PendingVerificationStore(ttl=1000, max_entries=3)
    for i in range(5):
        store.put(f"user{i}@example.com", {"code": "111111"})
    # Store must never grow beyond max_entries.
    assert len(store._entries) <= 3
    # The most-recently-added entries survive; the oldest were evicted.
    assert store.get("user4@example.com") is not None
    assert store.get("user0@example.com") is None


def test_pending_store_sweep_removes_expired():
    store = PendingVerificationStore(ttl=1, max_entries=10)
    store.put("a@example.com", {"code": "1"})
    store.put("b@example.com", {"code": "2"})
    store._entries["a@example.com"]["created_at"] = time.time() - 5
    removed = store.sweep_expired()
    assert removed == 1
    assert "a@example.com" not in store._entries
    assert "b@example.com" in store._entries


# ---------------------------------------------------------------------------
# Transport construction / dev fallback
# ---------------------------------------------------------------------------


def test_build_transport_uses_resend_when_key_present():
    transport = build_email_transport(api_key="re_test_key", from_email="noreply@x.com")
    assert isinstance(transport, ResendEmailTransport)


def test_build_transport_falls_back_to_logging_without_key():
    transport = build_email_transport(api_key=None, from_email="noreply@x.com")
    assert isinstance(transport, LoggingEmailTransport)
    # The logging (dev) transport reports it is NOT a real sender, so signup can
    # expose the code in dev only.
    assert transport.is_dev_fallback is True


def test_resend_transport_is_not_dev_fallback():
    transport = ResendEmailTransport(api_key="re_test", from_email="noreply@x.com")
    assert transport.is_dev_fallback is False


# ---------------------------------------------------------------------------
# End-to-end signup -> verify -> token  (with a real fake in-process server-less path)
# ---------------------------------------------------------------------------


class _FakeArtifactManager:
    """A minimal in-process artifact-manager stand-in (REAL object, not a mock).

    It emulates just enough of the artifact-manager surface used by the local-auth
    signup/verify path: collection create/read, list, and item create.
    """

    def __init__(self):
        self._collections = {}  # alias/id -> collection dict
        self._items = {}  # collection_id -> list of item dicts
        self._counter = 0

    async def create(self, workspace=None, alias=None, type=None, manifest=None,
                     config=None, parent_id=None, stage=False):
        self._counter += 1
        if parent_id is None:
            # Creating a collection
            cid = f"{workspace}/{alias}"
            col = {"id": cid, "alias": alias, "type": type, "manifest": manifest}
            self._collections[cid] = col
            self._collections[f"{workspace}/{alias}"] = col
            self._items.setdefault(cid, [])
            return col
        else:
            item = {"id": f"item-{self._counter}", "manifest": manifest, "type": type}
            self._items.setdefault(parent_id, []).append(item)
            return item

    async def read(self, artifact_id):
        if artifact_id in self._collections:
            return self._collections[artifact_id]
        # search items
        for items in self._items.values():
            for it in items:
                if it["id"] == artifact_id:
                    return it
        raise Exception(f"Artifact {artifact_id} does not exist")

    async def list(self, collection_id):
        return list(self._items.get(collection_id, []))


class _FakeServer:
    def __init__(self, artifact_manager):
        self._am = artifact_manager

    async def get_service(self, name):
        assert name == "public/artifact-manager"
        return self._am


@pytest.fixture
def fake_server():
    return _FakeServer(_FakeArtifactManager())


@pytest.mark.asyncio
async def test_signup_sends_code_and_does_not_create_account_yet(fake_server, capturing_transport):
    result = await la.signup_handler(
        fake_server, None,
        name="Alice", email="alice@example.com", password="Password123!",
    )
    assert result["success"] is True
    assert result.get("verification_required") is True
    # The account is NOT created until verification succeeds.
    am = fake_server._am
    users = []
    for items in am._items.values():
        users.extend(items)
    assert not any(u["manifest"].get("email") == "alice@example.com" for u in users)
    # An email was captured by the injected transport.
    assert len(capturing_transport.sent) == 1
    sent = capturing_transport.sent[0]
    assert sent["to"] == "alice@example.com"
    assert sent["code"].isdigit()


@pytest.mark.asyncio
async def test_verify_success_creates_account(fake_server, capturing_transport):
    await la.signup_handler(
        fake_server, None,
        name="Bob", email="bob@example.com", password="Password123!",
    )
    code = capturing_transport.sent[-1]["code"]
    verify = await la.verify_email_handler(
        fake_server, None, email="bob@example.com", code=code,
    )
    assert verify["success"] is True
    assert "user_id" in verify
    # Now the account exists and login works, issuing a token.
    login = await la.login_handler(
        fake_server, None, email="bob@example.com", password="Password123!",
    )
    assert login["success"] is True
    assert login["token"]


@pytest.mark.asyncio
async def test_verify_wrong_code_fails_and_counts_attempt(fake_server, capturing_transport):
    await la.signup_handler(
        fake_server, None,
        name="Carol", email="carol@example.com", password="Password123!",
    )
    res = await la.verify_email_handler(
        fake_server, None, email="carol@example.com", code="000000",
    )
    assert res["success"] is False
    entry = la._PENDING_VERIFICATIONS.get("carol@example.com")
    assert entry is not None
    assert entry["attempts"] == 1


@pytest.mark.asyncio
async def test_verify_expired_code(fake_server, capturing_transport):
    await la.signup_handler(
        fake_server, None,
        name="Dan", email="dan@example.com", password="Password123!",
    )
    code = capturing_transport.sent[-1]["code"]
    # Age the pending entry beyond TTL.
    la._PENDING_VERIFICATIONS._entries["dan@example.com"]["created_at"] = (
        time.time() - VERIFICATION_CODE_TTL - 5
    )
    res = await la.verify_email_handler(
        fake_server, None, email="dan@example.com", code=code,
    )
    assert res["success"] is False
    assert "expired" in res["error"].lower()


@pytest.mark.asyncio
async def test_verify_max_attempts_lockout(fake_server, capturing_transport):
    await la.signup_handler(
        fake_server, None,
        name="Eve", email="eve@example.com", password="Password123!",
    )
    correct = capturing_transport.sent[-1]["code"]
    for _ in range(MAX_VERIFICATION_ATTEMPTS):
        r = await la.verify_email_handler(
            fake_server, None, email="eve@example.com", code="999999",
        )
        assert r["success"] is False
    # After max attempts the pending session is invalidated; the correct code no longer works.
    r = await la.verify_email_handler(
        fake_server, None, email="eve@example.com", code=correct,
    )
    assert r["success"] is False
    assert "too many" in r["error"].lower() or "no pending" in r["error"].lower()


@pytest.mark.asyncio
async def test_resend_throttling(fake_server, capturing_transport):
    await la.signup_handler(
        fake_server, None,
        name="Frank", email="frank@example.com", password="Password123!",
    )
    assert len(capturing_transport.sent) == 1
    # Immediate resend should be throttled.
    r = await la.resend_code_handler(fake_server, None, email="frank@example.com")
    assert r["success"] is False
    assert "wait" in r["error"].lower() or "throttl" in r["error"].lower()
    assert len(capturing_transport.sent) == 1  # no new email

    # After the throttle window passes, resend delivers a fresh code.
    la._PENDING_VERIFICATIONS._entries["frank@example.com"]["last_sent_at"] = (
        time.time() - RESEND_THROTTLE_SECONDS - 1
    )
    r = await la.resend_code_handler(fake_server, None, email="frank@example.com")
    assert r["success"] is True
    assert len(capturing_transport.sent) == 2
    # The resent code is the one that now verifies.
    new_code = capturing_transport.sent[-1]["code"]
    verify = await la.verify_email_handler(
        fake_server, None, email="frank@example.com", code=new_code,
    )
    assert verify["success"] is True


@pytest.mark.asyncio
async def test_abandoned_session_cleanup(fake_server, capturing_transport):
    await la.signup_handler(
        fake_server, None,
        name="Grace", email="grace@example.com", password="Password123!",
    )
    assert "grace@example.com" in la._PENDING_VERIFICATIONS._entries
    # Age it out and run the sweeper.
    la._PENDING_VERIFICATIONS._entries["grace@example.com"]["created_at"] = (
        time.time() - VERIFICATION_CODE_TTL - 10
    )
    removed = la._PENDING_VERIFICATIONS.sweep_expired()
    assert removed >= 1
    assert "grace@example.com" not in la._PENDING_VERIFICATIONS._entries
    # Verifying an abandoned/cleaned session fails cleanly (no leak).
    res = await la.verify_email_handler(
        fake_server, None, email="grace@example.com", code="123456",
    )
    assert res["success"] is False


@pytest.mark.asyncio
async def test_duplicate_email_rejected_before_sending_code(fake_server, capturing_transport):
    # Create + verify a first account.
    await la.signup_handler(
        fake_server, None, name="H", email="dup@example.com", password="Password123!",
    )
    code = capturing_transport.sent[-1]["code"]
    await la.verify_email_handler(fake_server, None, email="dup@example.com", code=code)
    sent_before = len(capturing_transport.sent)
    # Second signup with same email must be rejected and NOT send a new code.
    res = await la.signup_handler(
        fake_server, None, name="H2", email="dup@example.com", password="Password123!",
    )
    assert res["success"] is False
    assert "already registered" in res["error"].lower()
    assert len(capturing_transport.sent) == sent_before


@pytest.mark.asyncio
async def test_dev_fallback_returns_code_in_response(fake_server):
    """When no RESEND_API_KEY is configured, the dev fallback exposes the code so
    developers can complete the flow without an email provider. Reset transport to
    the default (logging) transport to exercise this branch."""
    reset_email_transport()  # ensures we use the process default
    la.set_email_transport(LoggingEmailTransport(from_email="noreply@localhost"))
    try:
        res = await la.signup_handler(
            fake_server, None, name="I", email="dev@example.com", password="Password123!",
        )
        assert res["success"] is True
        assert res.get("dev_code") is not None
        assert res["dev_code"].isdigit()
        verify = await la.verify_email_handler(
            fake_server, None, email="dev@example.com", code=res["dev_code"],
        )
        assert verify["success"] is True
    finally:
        reset_email_transport()


@pytest.mark.asyncio
async def test_resend_transport_builds_expected_payload():
    """Verify the Resend transport shapes the HTTP call correctly by injecting a
    fake HTTP-post callable (REAL function, captures args) into the transport."""
    captured = {}

    async def fake_post(url, json=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers

        class _Resp:
            status_code = 200

            def json(self):
                return {"id": "email_123"}

            @property
            def text(self):
                return "ok"

        return _Resp()

    transport = ResendEmailTransport(
        api_key="re_secret", from_email="noreply@hypha.test", http_post=fake_post
    )
    await transport.send_verification_email("target@example.com", "654321")
    assert captured["url"] == "https://api.resend.com/emails"
    assert captured["headers"]["Authorization"] == "Bearer re_secret"
    assert captured["json"]["from"] == "noreply@hypha.test"
    assert captured["json"]["to"] == ["target@example.com"]
    assert "654321" in captured["json"]["html"]
    assert "654321" in captured["json"].get("text", "")
