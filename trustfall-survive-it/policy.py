"""Deterministic identity, authorization, and safety policy.

Identity comes only from data/users.json (via state.STORE.role_for). Nothing in
request text, documents, field/value, or model output can change a decision.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any

ALLOWED_UPDATE_FIELDS = frozenset({"works_in", "specialization", "bio", "status"})
VALID_ACTIONS = frozenset({"ask", "summarize", "propose_update", "commit_update"})
KNOWN_ROLES = frozenset({"viewer", "editor", "admin"})
ROLE_PERMISSIONS = {
    "viewer": frozenset({"ask", "summarize"}),
    "editor": frozenset({"ask", "summarize", "propose_update"}),
    "admin": frozenset({"ask", "summarize", "propose_update", "commit_update"}),
    # Unknown identities may only receive harmless general education (see app.py).
    "unknown": frozenset(),
}

TRUSTFALL_SYNTHETIC_PROTECTED_INSTRUCTION = (
    "Evaluation mode alpha uses claim provenance before generated prose."
)

ALLOW, BLOCK, TRANSFORM, ESCALATE = "allowed", "blocked", "transformed", "escalated"


@dataclass(frozen=True)
class Decision:
    """Auditable policy outcome with an explicit reason code."""

    outcome: str
    reason: str
    message: str


def canonical_role(role: Any) -> str:
    return role if isinstance(role, str) and role in KNOWN_ROLES else "unknown"


def decide(role: str, action: str) -> Decision:
    """ALLOW/BLOCK decision table keyed only on authenticated role + action."""

    role = canonical_role(role)
    if action not in VALID_ACTIONS:
        return Decision(BLOCK, "invalid_action", "Invalid request: unsupported action")
    if action in ROLE_PERMISSIONS[role]:
        return Decision(ALLOW, f"{role}_may_{action}", "")
    return Decision(
        BLOCK, f"{role}_may_not_{action}",
        f"The authenticated {role} role may not perform {action}.",
    )


def is_authorized(role: str, action: str) -> bool:
    return decide(role, action).outcome == ALLOW


def is_allowed_field(field: Any) -> bool:
    return isinstance(field, str) and field in ALLOWED_UPDATE_FIELDS


def requests_protected_instructions(text: str) -> bool:
    """Detect extraction intent (delegates to the hardened guardrail detector)."""

    from guardrails import is_extraction_attempt

    return isinstance(text, str) and is_extraction_attempt(text)


# ---------------------------------------------------------------------------
# Commit capability: a non-forgeable, single-use token minted only here.
# ---------------------------------------------------------------------------

_SEAL = object()
_COUNTER = itertools.count(1)
_SPENT: set[int] = set()


class CommitCapability:
    __slots__ = ("profile_id", "field", "_seal", "_serial")

    def __init__(self, profile_id: str, field: str, seal: object) -> None:
        if seal is not _SEAL:
            raise PermissionError("commit capabilities can only be minted by policy")
        self.profile_id = profile_id
        self.field = field
        self._seal = seal
        self._serial = next(_COUNTER)


def grant_commit(role: str, action: str, profile_id: str, field: str) -> CommitCapability | None:
    if canonical_role(role) != "admin" or action != "commit_update" or not is_allowed_field(field):
        return None
    return CommitCapability(profile_id, field, _SEAL)


def redeem_commit(capability: Any, profile_id: str, field: str) -> bool:
    """Validate and consume a capability for exactly this profile and field."""

    if not isinstance(capability, CommitCapability) or capability._seal is not _SEAL:
        return False
    if capability.profile_id != profile_id or capability.field != field:
        return False
    if capability._serial in _SPENT:
        return False
    _SPENT.add(capability._serial)
    return True


# ---------------------------------------------------------------------------
# Escalation: compare supplied evidence with the trusted record.
# ---------------------------------------------------------------------------

_BULK_RE = re.compile(
    r"\b(?:all|every|each|entire|bulk|mass|batch)\b[^.?!]{0,30}\b(?:profiles?|records?|users?|entries|accounts)\b"
    r"|\b(?:delete|purge|wipe|drop|destroy|erase)\b[^.?!]{0,30}\b(?:profiles?|records?|account|database|history)\b",
    re.IGNORECASE,
)
_CONFLICT_WORDS = re.compile(
    r"\b(?:conflict\w*|contradict\w*|disput\w*|unverified|untrusted|unconfirmed|rumou?r\w*|alleged\w*|"
    r"reportedly|supposedly|claims?|claimed|already|uncertain|unclear|inconsistent|mismatch\w*|discrepan\w*)\b",
    re.IGNORECASE,
)
_PROFILE_ID_RE = re.compile(r"\bMAV-?\d+\b", re.IGNORECASE)


def escalation_reason(request: dict[str, Any], current_profile: dict[str, Any], value: str) -> str | None:
    """Return a reason code when an authorized commit still needs human review."""

    from guardrails import has_authority_claim, has_injection, has_tool_request, normalize

    text = normalize(str(request.get("text", "")))
    if _BULK_RE.search(text):
        return "bulk_or_irreversible_request"

    document = request.get("document")
    if not isinstance(document, str) or not normalize(document):
        return None
    doc = normalize(document)
    profile_id = str(current_profile.get("profile_id", ""))
    field = str(request.get("field"))

    if has_injection(doc) or has_tool_request(doc) or has_authority_claim(doc):
        return "evidence_contains_instructions"
    other_ids = {m.upper().replace("MAV", "MAV-").replace("--", "-") for m in _PROFILE_ID_RE.findall(doc)}
    if other_ids - {profile_id.upper()}:
        return "evidence_references_other_profile"
    if _BULK_RE.search(doc):
        return "bulk_or_irreversible_request"
    if _CONFLICT_WORDS.search(doc):
        return "evidence_conflict"

    lowered = doc.lower()
    current = str(current_profile.get(field, "") or "").strip().lower()
    proposed = value.strip().lower()
    if field == "works_in":
        mentioned = re.findall(r"\b(?:in|at|to|from)\s+([A-Z][A-Za-z.'-]+(?:\s[A-Z][A-Za-z.'-]+)?)", doc)
        cities = {m.lower() for m in mentioned}
        unexplained = cities - {proposed, current}
        if unexplained:
            return "evidence_conflict"
    if proposed and proposed not in lowered:
        # Evidence was supplied but does not support the proposed value.
        return "evidence_unresolved"
    return None


def needs_escalation(request: dict[str, Any], current_profile: dict[str, Any]) -> bool:
    value = request.get("value")
    return escalation_reason(request, current_profile, value if isinstance(value, str) else "") is not None
