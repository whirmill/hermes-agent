"""Gateway action proposals with interactive approval CTAs.

This module provides two related pieces:

* an in-memory, resolve-once proposal primitive used by gateway adapters
  (Discord buttons today, text fallback elsewhere), and
* the ``action_proposal`` tool exposed to agents so follow-up messages can
  include auditable human-decision CTAs without bypassing policy/preflight.

The primitive is intentionally conservative: a button click records a human
intent and queues a follow-up turn. Any real mutation must still re-read source
state and pass the normal tool/policy/safety path in that later turn.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_ACTION_DECISIONS = {"approved", "needs_changes", "rejected"}
_DEFAULT_EXPIRES_SECONDS = 24 * 60 * 60
_MAX_EXPIRES_SECONDS = 7 * 24 * 60 * 60
KANBAN_ACTION_PROPOSAL_TYPES = {
    "kanban_unblock",
    "kanban_review_gate",
    "kanban_merge_gate",
    "reconciler_unblock_proposal",
}


@dataclass
class ActionProposal:
    proposal_id: str
    session_key: str
    title: str
    body: str
    proposal_type: str = "chat_followup"
    intent_key: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    source_snapshot: Dict[str, Any] = field(default_factory=dict)
    source_snapshot_hash: str = ""
    source: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    status: str = "pending"
    superseded_by: Optional[str] = None
    message_id: Optional[str] = None
    decision: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionProposalResolveResult:
    ok: bool
    reason: str
    proposal: ActionProposal
    decision: Optional[Dict[str, Any]] = None


_lock = threading.RLock()
_entries: Dict[str, ActionProposal] = {}
_intent_index: Dict[tuple[str, str], str] = {}
_decision_callback: Optional[Callable[[ActionProposal, Dict[str, Any]], Any]] = None


def _json_stable(value: Any) -> str:
    try:
        return json.dumps(value or {}, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def _snapshot_hash(snapshot: Dict[str, Any] | None) -> str:
    raw = _json_stable(snapshot or {})
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _clean_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _clamp_expires(seconds: Any) -> int:
    try:
        value = int(seconds)
    except Exception:
        value = _DEFAULT_EXPIRES_SECONDS
    if value <= 0:
        value = _DEFAULT_EXPIRES_SECONDS
    return min(value, _MAX_EXPIRES_SECONDS)


def clear_all_action_proposals() -> None:
    """Test/helper hook: remove every in-memory proposal."""
    with _lock:
        _entries.clear()
        _intent_index.clear()


def set_action_proposal_decision_callback(
    callback: Optional[Callable[[ActionProposal, Dict[str, Any]], Any]]
) -> None:
    """Install the gateway callback invoked after a proposal is resolved.

    The callback is intentionally process-local. Gateway adapters call
    :func:`notify_action_proposal_decision` after a successful button click;
    the GatewayRunner callback typically queues a synthetic follow-up turn for
    the same session.
    """
    global _decision_callback
    with _lock:
        _decision_callback = callback


def register_action_proposal(
    *,
    session_key: str,
    title: str,
    body: str,
    proposal_type: str = "chat_followup",
    intent_key: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    source_snapshot: Optional[Dict[str, Any]] = None,
    source: Optional[Dict[str, Any]] = None,
    expires_in_seconds: int = _DEFAULT_EXPIRES_SECONDS,
) -> ActionProposal:
    """Register a pending proposal, superseding any pending same intent.

    ``intent_key`` is scoped by ``session_key``. Registering a second pending
    proposal for the same pair marks the earlier one ``superseded`` so stale
    Discord buttons cannot approve outdated state.
    """
    session_key = str(session_key or "").strip()
    title = str(title or "").strip()
    body = str(body or "").strip()
    proposal_type = str(proposal_type or "chat_followup").strip() or "chat_followup"
    clean_intent = str(intent_key).strip() if intent_key is not None and str(intent_key).strip() else None
    now = time.time()
    proposal = ActionProposal(
        proposal_id=f"ap_{uuid.uuid4().hex[:12]}",
        session_key=session_key,
        title=title,
        body=body,
        proposal_type=proposal_type,
        intent_key=clean_intent,
        payload=_clean_mapping(payload),
        source_snapshot=_clean_mapping(source_snapshot),
        source_snapshot_hash=_snapshot_hash(source_snapshot),
        source=_clean_mapping(source),
        created_at=now,
        expires_at=now + _clamp_expires(expires_in_seconds),
    )

    with _lock:
        if clean_intent:
            key = (session_key, clean_intent)
            previous_id = _intent_index.get(key)
            previous = _entries.get(previous_id or "")
            if previous and previous.status == "pending":
                previous.status = "superseded"
                previous.superseded_by = proposal.proposal_id
            _intent_index[key] = proposal.proposal_id
        _entries[proposal.proposal_id] = proposal
    return proposal


def get_action_proposal(proposal_id: str) -> Optional[ActionProposal]:
    with _lock:
        return _entries.get(str(proposal_id or ""))


def mark_action_proposal_delivered(proposal_id: str, message_id: Optional[str]) -> bool:
    with _lock:
        proposal = _entries.get(str(proposal_id or ""))
        if not proposal:
            return False
        proposal.message_id = str(message_id) if message_id else None
        return True


def cancel_action_proposal(proposal_id: str, status: str = "cancelled") -> bool:
    with _lock:
        proposal = _entries.get(str(proposal_id or ""))
        if not proposal:
            return False
        if proposal.status == "pending":
            proposal.status = status
        return True


def resolve_action_proposal(
    proposal_id: str,
    *,
    decision: str,
    actor_id: Optional[str] = None,
    actor_name: Optional[str] = None,
    comment: Optional[str] = None,
) -> ActionProposalResolveResult:
    """Resolve a proposal once, enforcing expiry and stale states."""
    decision = str(decision or "").strip()
    if decision not in _ACTION_DECISIONS:
        raise ValueError(f"Unsupported action proposal decision: {decision}")

    with _lock:
        proposal = _entries.get(str(proposal_id or ""))
        if proposal is None:
            raise KeyError(f"Unknown action proposal: {proposal_id}")

        if proposal.status != "pending":
            return ActionProposalResolveResult(
                ok=False,
                reason="already_resolved",
                proposal=proposal,
                decision=proposal.decision,
            )

        if time.time() >= proposal.expires_at:
            proposal.status = "expired"
            return ActionProposalResolveResult(
                ok=False,
                reason="expired",
                proposal=proposal,
                decision=None,
            )

        decision_info = {
            "decision": decision,
            "actor_id": str(actor_id or ""),
            "actor_name": str(actor_name or ""),
            "comment": str(comment or ""),
            "resolved_at": time.time(),
        }
        proposal.status = decision
        proposal.decision = decision_info
        return ActionProposalResolveResult(
            ok=True,
            reason="resolved",
            proposal=proposal,
            decision=decision_info,
        )


async def notify_action_proposal_decision(
    proposal: ActionProposal,
    decision: Dict[str, Any],
) -> Any:
    """Run the installed gateway decision callback, if any."""
    with _lock:
        callback = _decision_callback
    if callback is None:
        return None
    result = callback(proposal, decision)
    if inspect.isawaitable(result):
        return await result
    return result


def build_decision_prompt(proposal: ActionProposal, decision: Dict[str, Any]) -> str:
    """Build the synthetic follow-up text queued after a CTA click."""
    payload = _json_stable(proposal.payload)
    snapshot = _json_stable(proposal.source_snapshot)
    decision_name = decision.get("decision", "")
    actor = decision.get("actor_name") or decision.get("actor_id") or "authorized user"
    comment = decision.get("comment") or ""
    return (
        "[Action proposal decision]\n"
        f"Proposal ID: {proposal.proposal_id}\n"
        f"Type: {proposal.proposal_type}\n"
        f"Intent key: {proposal.intent_key or ''}\n"
        f"Decision: {decision_name}\n"
        f"Actor: {actor}\n"
        f"Title: {proposal.title}\n"
        f"Body: {proposal.body}\n"
        f"Comment: {comment}\n"
        f"Payload JSON: {payload}\n"
        f"Source snapshot hash: {proposal.source_snapshot_hash}\n"
        f"Source snapshot JSON: {snapshot}\n\n"
        "Policy: treat this as an auditable human decision only. Re-read the "
        "relevant source state (Kanban task/board, PR head SHA/checks, run/PID, "
        "or other referenced object) before any mutation. If the source changed, "
        "expired, or no longer matches the snapshot, report it as stale/no-op. "
        "do not bypass dangerous-command prompts, board manual-only policy, or "
        "other preflight/safety gates. Do not create another proposal for this "
        "same intent unless fresh state requires a materially different decision."
    )


def _tool_error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def validate_action_proposal_policy(
    *,
    proposal_type: str,
    payload: Optional[Dict[str, Any]] = None,
    source_snapshot: Optional[Dict[str, Any]] = None,
) -> list[str]:
    """Return conservative policy/preflight validation errors.

    Kanban CTA proposals are only useful if the follow-up turn can re-read and
    compare canonical state. Require stable identifiers and snapshot fields for
    review/unblock/merge gates so a later approval cannot operate on a stale or
    ambiguous card/PR.
    """
    ptype = str(proposal_type or "chat_followup").strip() or "chat_followup"
    data = _clean_mapping(payload)
    snapshot = _clean_mapping(source_snapshot)
    errors: list[str] = []
    if ptype in KANBAN_ACTION_PROPOSAL_TYPES:
        if not data.get("board"):
            errors.append("payload.board is required for Kanban action proposals")
        if not data.get("task_id"):
            errors.append("payload.task_id is required for Kanban action proposals")
        if not snapshot:
            errors.append("source_snapshot is required for Kanban action proposals")
        if ptype == "kanban_merge_gate":
            if not data.get("repo"):
                errors.append("payload.repo is required for kanban_merge_gate")
            if not (data.get("pr") or data.get("pull_request")):
                errors.append("payload.pr is required for kanban_merge_gate")
            if not (snapshot.get("head_sha") or snapshot.get("merge_sha")):
                errors.append("source_snapshot.head_sha is required for kanban_merge_gate")
    return errors


def action_proposal_tool(
    *,
    title: str,
    body: str,
    proposal_type: str = "chat_followup",
    intent_key: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    source_snapshot: Optional[Dict[str, Any]] = None,
    expires_in_seconds: int = _DEFAULT_EXPIRES_SECONDS,
    callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> str:
    """Create and deliver an interactive action proposal."""
    if not str(title or "").strip():
        return _tool_error("title is required")
    if not str(body or "").strip():
        return _tool_error("body is required")
    if callback is None:
        return _tool_error("action_proposal is only available in gateway sessions")

    payload = _clean_mapping(payload)
    source_snapshot = _clean_mapping(source_snapshot)
    policy_errors = validate_action_proposal_policy(
        proposal_type=proposal_type,
        payload=payload,
        source_snapshot=source_snapshot,
    )
    if policy_errors:
        return _tool_error("; ".join(policy_errors))

    try:
        proposal = callback(
            {
                "title": title,
                "body": body,
                "proposal_type": proposal_type,
                "intent_key": intent_key,
                "payload": payload,
                "source_snapshot": source_snapshot,
                "expires_in_seconds": _clamp_expires(expires_in_seconds),
            }
        )
    except Exception as exc:
        logger.exception("action_proposal callback failed: %s", exc)
        return _tool_error(f"failed to create action proposal: {exc}")

    if isinstance(proposal, ActionProposal):
        return json.dumps(
            {
                "proposal_id": proposal.proposal_id,
                "status": proposal.status,
                "intent_key": proposal.intent_key,
                "expires_at": proposal.expires_at,
                "message_id": proposal.message_id,
            },
            ensure_ascii=False,
        )
    if isinstance(proposal, dict):
        return json.dumps(proposal, ensure_ascii=False, default=str)
    return _tool_error("action_proposal callback returned an invalid result")


ACTION_PROPOSAL_SCHEMA = {
    "name": "action_proposal",
    "description": (
        "Create an interactive, auditable call-to-action proposal in the current "
        "gateway chat. Use it for follow-up operations that need a human decision, "
        "especially Kanban unblock/review/merge gates. The buttons record intent; "
        "future execution must still re-read source state and pass policy/preflight."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short proposal title."},
            "body": {"type": "string", "description": "Human-readable operation summary."},
            "proposal_type": {
                "type": "string",
                "description": (
                    "Structured proposal type, e.g. chat_followup, kanban_unblock, "
                    "kanban_review_gate, kanban_merge_gate, kanban_create, "
                    "reconciler_unblock_proposal."
                ),
                "default": "chat_followup",
            },
            "intent_key": {
                "type": "string",
                "description": "Stable dedupe key. Same session+intent supersedes older pending proposals.",
            },
            "payload": {
                "type": "object",
                "description": (
                    "Structured references needed later (task_id, board, repo, pr, sha, etc.). "
                    "Kanban review/unblock/merge proposals must include at least board+task_id; "
                    "merge gates also require repo+pr."
                ),
                "additionalProperties": True,
            },
            "source_snapshot": {
                "type": "object",
                "description": (
                    "Snapshot fields that must be re-read before executing (status, sha, etc.). "
                    "Required for Kanban proposal types; merge gates must include head_sha/merge_sha."
                ),
                "additionalProperties": True,
            },
            "expires_in_seconds": {
                "type": "integer",
                "description": "Expiry in seconds. Defaults to 24h; dangerous-command approvals remain separate.",
                "default": _DEFAULT_EXPIRES_SECONDS,
            },
        },
        "required": ["title", "body"],
    },
}


def check_action_proposal_requirements() -> bool:
    return True


from tools.registry import registry

registry.register(
    name="action_proposal",
    toolset="clarify",
    schema=ACTION_PROPOSAL_SCHEMA,
    handler=lambda args, **kw: action_proposal_tool(
        title=args.get("title", ""),
        body=args.get("body", ""),
        proposal_type=args.get("proposal_type", "chat_followup"),
        intent_key=args.get("intent_key"),
        payload=args.get("payload"),
        source_snapshot=args.get("source_snapshot"),
        expires_in_seconds=args.get("expires_in_seconds", _DEFAULT_EXPIRES_SECONDS),
        callback=kw.get("callback"),
    ),
    check_fn=check_action_proposal_requirements,
    emoji="✅",
)
