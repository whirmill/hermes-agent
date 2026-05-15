import asyncio
from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.run import (
    GatewayRunner,
    _kanban_balanced_thread_name,
    _kanban_notifier_event_kinds,
)
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


class RecordingDiscordAdapter(RecordingAdapter):
    def __init__(self, thread_id="thread-1"):
        super().__init__()
        self.thread_id = thread_id
        self.created_threads = []

    async def create_handoff_thread(self, parent_chat_id, name, *, user_ids=None):
        self.created_threads.append({
            "parent_chat_id": parent_chat_id,
            "name": name,
            "user_ids": user_ids,
        })
        return self.thread_id


class FailingThreadDiscordAdapter(RecordingAdapter):
    async def create_handoff_thread(self, parent_chat_id, name, *, user_ids=None):
        return None


class DisconnectedAdapters(dict):
    """Expose a platform during collection, then simulate disconnect on get()."""

    def get(self, key, default=None):
        return None


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter, platform=Platform.TELEGRAM):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {platform: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


def _create_completed_subscription(summary="done once"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify once", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
        return tid
    finally:
        conn.close()


def _unseen_terminal_events(tid):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_dedupes_board_slugs_pointing_to_same_db(tmp_path, monkeypatch):
    db_path = tmp_path / "shared-kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    kb.write_board_metadata("alias-a", name="Alias A")
    kb.write_board_metadata("alias-b", name="Alias B")

    tid = _create_completed_subscription()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert "✅ Done" in adapter.sent[0]["text"]
    assert tid[:8] in adapter.sent[0]["text"]


def test_kanban_notifier_claim_prevents_second_watcher_send(tmp_path, monkeypatch):
    db_path = tmp_path / "single-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    tid = _create_completed_subscription()

    adapter1 = RecordingAdapter()
    adapter2 = RecordingAdapter()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter1)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter2)))

    assert len(adapter1.sent) == 1
    assert adapter2.sent == []


def test_kanban_notifier_rewinds_claim_if_adapter_disconnects(tmp_path, monkeypatch):
    db_path = tmp_path / "adapter-disconnect.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = DisconnectedAdapters({Platform.TELEGRAM: RecordingAdapter()})
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_kanban_db_path_is_test_isolated_from_real_home():
    hermes_home = Path(kb.kanban_home())
    production_db = Path.home() / ".hermes" / "kanban.db"
    assert kb.kanban_db_path().resolve() != production_db.resolve()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
    finally:
        conn.close()

    assert kb.kanban_db_path().resolve().is_relative_to(hermes_home.resolve())
    assert kb.kanban_db_path().resolve() != production_db.resolve()


class FailingAdapter:
    """Adapter whose send() always raises, simulating a transient send error."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


def test_kanban_notifier_rewinds_claim_on_send_exception(tmp_path, monkeypatch):
    """A raising adapter rewinds the claim so the next tick can retry.

    This is the second rewind path (distinct from the adapter-disconnect path
    in test_kanban_notifier_rewinds_claim_if_adapter_disconnects). Here the
    adapter is connected and the send call actually fires; the claim must
    still rewind so the event isn't lost when send() raises mid-tick.
    """
    db_path = tmp_path / "send-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FailingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Send was attempted (so we exercised the failure path, not just the
    # disconnect path) and the claim was rewound — the unseen-events query
    # still returns the event for retry on the next tick.
    assert adapter.attempts >= 1, "send should have been attempted at least once"
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_notifier_redelivers_same_kind_on_dispatch_cycle(tmp_path, monkeypatch):
    """A retry cycle (crashed → reclaimed → crashed) notifies the user twice.

    Before #21398 the notifier auto-unsubscribed on any terminal event kind
    (gave_up / crashed / timed_out), so the second crash in a respawn cycle
    silently dropped — the subscription was already gone. This test pins the
    new contract: subscription survives non-final terminal events; the
    cursor handles dedup.

    Two crashes ten seconds apart on the same task — both should land on
    the adapter.
    """
    db_path = tmp_path / "redeliver-cycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_NOTIFIER_MODE", "verbose")
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="cycle test", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        # First crash — fired by the dispatcher when the worker PID dies.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # First crash delivered.
    assert len(adapter.sent) == 1
    assert "crashed" in adapter.sent[0]["text"].lower()

    # Subscription survives — the cursor advanced past event #1, but the
    # row is still there.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, (
            "Subscription must survive a crashed event so a respawn-cycle "
            "second crash also notifies the user (issue #21398)."
        )

        # Second crash — same task, same dispatcher (or a respawn). Append
        # another event to simulate the dispatcher firing crashed a second
        # time during retry.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    # New tick: the second event has a fresh id past the cursor advance,
    # so it gets claimed and delivered.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 2, (
        f"Second crashed event should also notify; got {len(adapter.sent)} "
        f"deliveries (texts: {[d['text'] for d in adapter.sent]})"
    )
    assert "crashed" in adapter.sent[1]["text"].lower()



def test_notifier_renders_spawned_intro_with_operational_audit(tmp_path, monkeypatch):
    db_path = tmp_path / "spawned-intro.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_NOTIFIER_MODE", "audit")
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="Implement portfolio review automation",
            body=(
                "Wire automatic review workers for portfolio cards.\n\n"
                "Acceptance criteria:\n"
                "- Planner creates explicit review cards\n"
                "- Review runs before completion\n"
            ),
            assignee="dev-worker",
            workspace_kind="dir",
            workspace_path="/tmp/portfolio",
        )
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(conn, tid, kind="created")
        kb._append_event(conn, tid, kind="spawned", payload={"pid": 1234})
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "avviato" in text
    assert "Obiettivo" in text
    assert "Assignee" in text and "dev-worker" in text
    assert "Repo/workspace" in text and "/tmp/portfolio" in text
    assert "Planner creates explicit review cards" in text
    assert "chain-of-thought" in text
    assert "hermes kanban --board default show" in text
    assert "hermes kanban --board default tail" in text
    assert "hermes kanban --board default log" in text
    assert "created" not in text.lower()


def test_notifier_renders_stage_updates_and_comment_preview(tmp_path, monkeypatch):
    db_path = tmp_path / "stage-updates.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_NOTIFIER_MODE", "audit")
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="stage test", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        assert kb.claim_task(conn, tid, claimer="worker", ttl_seconds=600)
        assert kb.heartbeat_worker(conn, tid, note="testing: pytest tests/gateway/test_kanban_notifier.py")
        kb.add_comment(conn, tid, author="worker", body="review-ready: changed gateway notifier")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    texts = [m["text"] for m in adapter.sent]
    assert len(texts) == 2
    assert "stage" in texts[0].lower()
    assert "pytest tests/gateway/test_kanban_notifier.py" in texts[0]
    assert "tail" in texts[0]
    assert "review-ready: changed gateway notifier" in texts[1]


def test_notifier_completion_includes_run_metadata_and_detail_commands(tmp_path, monkeypatch):
    db_path = tmp_path / "completion-audit.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_NOTIFIER_MODE", "audit")
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="completion audit", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        assert kb.claim_task(conn, tid, claimer="worker", ttl_seconds=600)
        kb.complete_task(
            conn,
            tid,
            summary="Implemented notifier audit trail",
            metadata={
                "changed_files": ["gateway/run.py", "hermes_cli/kanban_db.py"],
                "tests_run": ["pytest tests/gateway/test_kanban_notifier.py"],
                "diff_path": "/tmp/notifier.diff",
            },
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "done" in text
    assert "Implemented notifier audit trail" in text
    assert "Audit operativo" in text
    assert "gateway/run.py" in text
    assert "pytest tests/gateway/test_kanban_notifier.py" in text
    assert "/tmp/notifier.diff" in text
    assert "hermes kanban --board default runs" in text


def test_discord_channel_subscription_moves_to_activity_thread(tmp_path, monkeypatch):
    db_path = tmp_path / "discord-thread-routing.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_NOTIFIER_MODE", "audit")
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="thread routing", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="discord",
            chat_id="parent-channel",
            user_id="811564178748997643",
        )
        kb._append_event(conn, tid, kind="spawned", payload={"pid": 1234})
    finally:
        conn.close()

    adapter = RecordingDiscordAdapter(thread_id="activity-thread")
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter, Platform.DISCORD)))

    assert len(adapter.created_threads) == 1
    assert adapter.created_threads[0]["parent_chat_id"] == "parent-channel"
    assert adapter.created_threads[0]["user_ids"] == ["811564178748997643"]
    assert len(adapter.sent) == 1
    assert adapter.sent[0]["chat_id"] == "parent-channel"
    assert adapter.sent[0]["metadata"] == {"thread_id": "activity-thread"}
    assert "avviato" in adapter.sent[0]["text"]

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()
    assert len(subs) == 1
    assert subs[0]["thread_id"] == "activity-thread"
    assert subs[0]["last_event_id"] > 0


def test_discord_channel_subscription_falls_back_to_parent_if_thread_creation_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "discord-thread-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_NOTIFIER_MODE", "audit")
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="thread failure", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="discord", chat_id="parent-channel")
        kb._append_event(conn, tid, kind="spawned", payload={"pid": 1234})
    finally:
        conn.close()

    adapter = FailingThreadDiscordAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter, Platform.DISCORD)))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["chat_id"] == "parent-channel"
    assert adapter.sent[0]["metadata"] == {}
    assert "avviato" in adapter.sent[0]["text"]
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="discord",
            chat_id="parent-channel",
            kinds=["spawned"],
        )
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()
    assert events == []
    assert len(subs) == 1
    assert subs[0]["thread_id"] == ""



def test_kanban_notifier_event_kinds_default_balanced(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_NOTIFIER_MODE", raising=False)

    assert _kanban_notifier_event_kinds() == ("completed", "blocked", "gave_up")


def test_kanban_notifier_event_kinds_verbose_and_audit(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_NOTIFIER_MODE", "verbose")
    assert _kanban_notifier_event_kinds() == (
        "completed", "blocked", "gave_up", "crashed", "timed_out"
    )

    monkeypatch.setenv("HERMES_KANBAN_NOTIFIER_MODE", "audit")
    assert _kanban_notifier_event_kinds() == (
        "spawned", "heartbeat", "commented",
        "completed", "blocked", "gave_up", "crashed", "timed_out",
    )


def test_kanban_verbose_timeout_handles_malformed_limit(tmp_path, monkeypatch):
    db_path = tmp_path / "malformed-timeout.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_NOTIFIER_MODE", "verbose")
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="timeout test", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(conn, tid, kind="timed_out", payload={"limit_seconds": "not-a-number"})
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert "Timed out" in adapter.sent[0]["text"]
    assert "not-a-number" not in adapter.sent[0]["text"]


def test_kanban_balanced_thread_name_is_outcome_first_and_scannable():
    name = _kanban_balanced_thread_name(
        "Hermes Kanban t_1234567890abcdef — review PR #42 for notifier with detailed implementation chatter via gateway internals",
        task_id="t_1234567890abcdef",
    )

    assert len(name) <= 80
    assert name.startswith("Review PR #42 —")
    assert name.endswith("· t_123456")
    assert "Hermes" not in name
    assert "Kanban" not in name
