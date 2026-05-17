import time

from tools import action_proposal as ap


def setup_function(_):
    ap.clear_all_action_proposals()
    ap.set_action_proposal_decision_callback(None)


def test_register_resolve_once_and_snapshot_signature():
    proposal = ap.register_action_proposal(
        session_key="discord:chan:thread",
        title="Merge PR #12",
        body="Approve merge after checks pass",
        proposal_type="kanban_merge_gate",
        intent_key="merge:repo:12:sha1",
        payload={"pr": 12},
        source_snapshot={"head_sha": "sha1"},
        expires_in_seconds=60,
    )

    assert proposal.status == "pending"
    assert proposal.intent_key == "merge:repo:12:sha1"
    assert proposal.source_snapshot_hash

    result = ap.resolve_action_proposal(
        proposal.proposal_id,
        decision="approved",
        actor_id="111",
        actor_name="Lorenzo",
    )
    assert result.ok is True
    assert result.proposal.status == "approved"
    assert result.decision["actor_name"] == "Lorenzo"

    second = ap.resolve_action_proposal(
        proposal.proposal_id,
        decision="rejected",
        actor_id="111",
        actor_name="Lorenzo",
    )
    assert second.ok is False
    assert second.reason == "already_resolved"
    assert second.proposal.status == "approved"


def test_same_session_intent_key_supersedes_previous_pending_proposal():
    first = ap.register_action_proposal(
        session_key="s1",
        title="Old",
        body="old body",
        proposal_type="kanban_unblock",
        intent_key="task:t1:blocked",
        expires_in_seconds=60,
    )
    second = ap.register_action_proposal(
        session_key="s1",
        title="New",
        body="new body",
        proposal_type="kanban_unblock",
        intent_key="task:t1:blocked",
        expires_in_seconds=60,
    )

    assert first.proposal_id != second.proposal_id
    assert ap.get_action_proposal(first.proposal_id).status == "superseded"
    assert ap.get_action_proposal(second.proposal_id).status == "pending"

    stale = ap.resolve_action_proposal(
        first.proposal_id,
        decision="approved",
        actor_id="111",
        actor_name="Lorenzo",
    )
    assert stale.ok is False
    assert stale.reason == "already_resolved"


def test_expired_proposal_cannot_be_approved(monkeypatch):
    now = time.time()
    monkeypatch.setattr(ap.time, "time", lambda: now)
    proposal = ap.register_action_proposal(
        session_key="s1",
        title="Expired",
        body="body",
        proposal_type="chat_followup",
        expires_in_seconds=1,
    )

    monkeypatch.setattr(ap.time, "time", lambda: now + 2)
    result = ap.resolve_action_proposal(
        proposal.proposal_id,
        decision="approved",
        actor_id="111",
        actor_name="Lorenzo",
    )

    assert result.ok is False
    assert result.reason == "expired"
    assert result.proposal.status == "expired"


def test_build_decision_prompt_contains_policy_guardrails():
    proposal = ap.register_action_proposal(
        session_key="s1",
        title="Review blocked card",
        body="Worker is blocked waiting for review",
        proposal_type="kanban_review_gate",
        intent_key="review:t1",
        payload={"task_id": "t_123", "board": "zapbot"},
        source_snapshot={"task_status": "blocked"},
        expires_in_seconds=60,
    )
    result = ap.resolve_action_proposal(
        proposal.proposal_id,
        decision="approved",
        actor_id="111",
        actor_name="Lorenzo",
    )

    prompt = ap.build_decision_prompt(result.proposal, result.decision)
    assert "Action proposal decision" in prompt
    assert "kanban_review_gate" in prompt
    assert "Re-read" in prompt
    assert "do not bypass" in prompt
    assert "t_123" in prompt


def test_kanban_merge_policy_requires_auditable_preflight_fields():
    errors = ap.validate_action_proposal_policy(
        proposal_type="kanban_merge_gate",
        payload={"board": "zapbot", "task_id": "t_123"},
        source_snapshot={"task_status": "blocked"},
    )

    assert "payload.repo is required for kanban_merge_gate" in errors
    assert "payload.pr is required for kanban_merge_gate" in errors
    assert "source_snapshot.head_sha is required for kanban_merge_gate" in errors


def test_action_proposal_tool_rejects_ambiguous_kanban_proposal_before_callback():
    called = []

    result = ap.action_proposal_tool(
        title="Unblock task",
        body="Worker is waiting for review",
        proposal_type="kanban_unblock",
        callback=lambda args: called.append(args),
    )

    assert "payload.board is required" in result
    assert called == []


def test_action_proposal_tool_allows_well_formed_kanban_merge_payload():
    def callback(args):
        return ap.register_action_proposal(
            session_key="s1",
            title=args["title"],
            body=args["body"],
            proposal_type=args["proposal_type"],
            intent_key=args.get("intent_key"),
            payload=args.get("payload"),
            source_snapshot=args.get("source_snapshot"),
            expires_in_seconds=args.get("expires_in_seconds"),
        )

    result = ap.action_proposal_tool(
        title="Approve merge",
        body="Checks are green; approve merge if still current",
        proposal_type="kanban_merge_gate",
        intent_key="merge:zapbot:t_123:99:abc",
        payload={"board": "zapbot", "task_id": "t_123", "repo": "org/repo", "pr": 99},
        source_snapshot={"task_status": "blocked", "head_sha": "abc", "checks": "success"},
        callback=callback,
    )

    assert "proposal_id" in result
    assert "kanban_merge_gate" not in result  # tool result stays compact; details are server-side
