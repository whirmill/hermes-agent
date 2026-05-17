import asyncio
from types import SimpleNamespace

import pytest

from gateway.platforms.discord import ActionProposalView
from gateway.platforms.base import MessageEvent, MessageType, Platform, SessionSource
from gateway.run import GatewayRunner


def _interaction(user_id, role_ids=None, *, drop_roles=False):
    user_kwargs = {"id": user_id, "display_name": f"user-{user_id}"}
    if not drop_roles:
        user_kwargs["roles"] = [SimpleNamespace(id=r) for r in (role_ids or [])]
    return SimpleNamespace(user=SimpleNamespace(**user_kwargs))


def test_action_proposal_view_has_italian_cta_labels():
    view = ActionProposalView(
        proposal_id="ap_123",
        allowed_user_ids=set(),
        allowed_role_ids=set(),
    )

    labels = [child.label for child in view.children]
    assert labels == ["✅ Approva", "✏️ Modifica / commenta", "❌ Rifiuta"]


def test_action_proposal_view_uses_shared_user_or_role_auth():
    view = ActionProposalView(
        proposal_id="ap_123",
        allowed_user_ids=set(),
        allowed_role_ids={42},
    )

    assert view._check_auth(_interaction(999, role_ids=[42])) is True
    assert view._check_auth(_interaction(999, role_ids=[7])) is False


def test_action_proposal_view_preserves_empty_allowlist_backcompat():
    view = ActionProposalView(proposal_id="ap_123", allowed_user_ids=set())
    assert view.allowed_role_ids == set()
    assert view._check_auth(_interaction(999)) is True


class _ActionProposalDispatchAdapter:
    def __init__(self):
        self._pending_messages = {}
        self._active_sessions = {}
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)


def _decision_event():
    return MessageEvent(
        text="Approved action proposal",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="thread-1",
            chat_type="group",
            user_id="42",
            user_name="Lorenzo",
            thread_id="thread-1",
        ),
        internal=True,
    )


@pytest.mark.asyncio
async def test_action_proposal_decision_dispatches_idle_session_immediately():
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._draining = False
    adapter = _ActionProposalDispatchAdapter()
    event = _decision_event()

    ok = runner._dispatch_action_proposal_decision_event(
        "discord:group:thread-1",
        event,
        adapter,
        asyncio.get_running_loop(),
    )

    assert ok is True
    await asyncio.sleep(0.01)
    assert adapter.handled == [event]
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_action_proposal_decision_queues_when_session_active():
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._draining = False
    runner._queued_events = {}
    adapter = _ActionProposalDispatchAdapter()
    adapter._active_sessions["discord:group:thread-1"] = asyncio.Event()
    event = _decision_event()

    ok = runner._dispatch_action_proposal_decision_event(
        "discord:group:thread-1",
        event,
        adapter,
        asyncio.get_running_loop(),
    )

    assert ok is True
    assert adapter.handled == []
    assert adapter._pending_messages["discord:group:thread-1"] is event
