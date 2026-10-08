"""TRUSTFALL application: deterministic trust and decision layer.

Pipeline for every request (fail closed at each step):
  1. Contract validation          -> BLOCK malformed input, never crash
  2. Identity                     -> role from data/users.json only
  3. Extraction guard             -> BLOCK protected-instruction requests
  4. Authorization decision table -> BLOCK by role/action (policy.decide)
  5. Action handler               -> claim verification / document sanitizing /
                                     gated proposal or commit (+ ESCALATE)
  6. Output guard                 -> contract shape, legal enums, redaction
"""

from __future__ import annotations

from typing import Any

from actions import propose_update, update_profile
from claim_verifier import verify_profile_text
from guardrails import (
    educational_answer,
    has_tool_request,
    is_extraction_attempt,
    normalize,
    output_is_safe,
    sanitize,
    validate_value,
)
from policy import (
    ALLOW,
    BLOCK,
    ESCALATE,
    TRANSFORM,
    canonical_role,
    decide,
    escalation_reason,
    grant_commit,
    is_allowed_field,
)
from state import STORE

REQUIRED_RESPONSE_KEYS = {"status", "answer", "claim_status", "citations", "tool_calls"}
VALID_STATUSES = {ALLOW, BLOCK, TRANSFORM, ESCALATE}
VALID_CLAIM_STATUSES = {"verified", "unverified", "unknown", "document_claim", "not_applicable"}
DOCUMENT_CITATION = {"source": "document", "label": "user_supplied_document"}
MAX_TEXT = 4000
MAX_DOCUMENT = 20000
MAX_SUMMARY = 700


def _response(
    status: str,
    answer: str,
    *,
    claim_status: str = "not_applicable",
    citations: list[dict[str, Any]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "answer": answer,
        "claim_status": claim_status,
        "citations": citations or [],
        "tool_calls": tool_calls or [],
    }


def _invalid(message: str) -> dict[str, Any]:
    return _response(BLOCK, f"Invalid request: {message}")


def _blocked_extraction() -> dict[str, Any]:
    return _response(BLOCK, "Protected instructions cannot be disclosed.")


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def _summarize_document(document: str) -> str:
    """Attribute every retained sentence to the document; drop instruction spans."""

    cleaned = sanitize(document, document=True)
    kept = cleaned.text
    if len(kept) > MAX_SUMMARY:
        kept = kept[: MAX_SUMMARY - 3].rsplit(" ", 1)[0] + "..."
    if kept:
        summary = (
            f"Document summary (untrusted, user-supplied): the document states: {kept} "
            "These are document claims and are not established by trusted profile data."
        )
    else:
        summary = "The supplied document contained no summarizable claims."
    if cleaned.removed:
        summary += f" {len(cleaned.removed)} instruction-like span(s) were removed and not executed."
    if cleaned.authority_claims:
        summary += " Authority or approval claims in the document carry no authority."
    return summary


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_ask(role: str, text: str, profile_id: str, profile: dict[str, Any] | None) -> dict[str, Any]:
    if has_tool_request(text):
        return _response(
            BLOCK,
            "Questions cannot invoke tools or change profile state; updates require an authorized propose_update or commit_update request.",
        )
    education = educational_answer(text)
    if education:
        return _response(ALLOW, education)
    if role == "unknown":
        return _response(BLOCK, "Unknown identities may only receive harmless general education.")

    cleaned = sanitize(text, document=False)
    if cleaned.removed and not cleaned.text:
        return _response(BLOCK, "Instructions embedded in the request were not executed, and no question remained.")
    if profile is None:
        return _response(ALLOW, "No trusted profile was found.", claim_status="unknown")

    result = verify_profile_text(cleaned.text or text, profile)
    status = TRANSFORM if (result.needs_transform or cleaned.removed or cleaned.authority_claims) else ALLOW
    answer = result.answer
    if cleaned.removed:
        answer += f" {len(cleaned.removed)} embedded instruction(s) in the request were ignored."
    if cleaned.authority_claims:
        answer += " Authority claims in the request text do not change trusted data or permissions."
    return _response(status, answer, claim_status=result.claim_status, citations=result.citations)


def _handle_summarize(text: str, document: Any) -> dict[str, Any]:
    if not isinstance(document, str) or not normalize(document):
        return _invalid("summarize requires a non-empty document")
    if has_tool_request(text):
        return _response(BLOCK, "Summaries cannot invoke tools or change profile state.")
    return _response(
        TRANSFORM,
        _summarize_document(document),
        claim_status="document_claim",
        citations=[dict(DOCUMENT_CITATION)],
    )


def _handle_update(role: str, action: str, request: dict[str, Any], profile_id: str,
                   profile: dict[str, Any] | None) -> dict[str, Any]:
    if profile is None:
        return _invalid("unknown profile_id")
    field = request.get("field")
    if not is_allowed_field(field):
        return _response(BLOCK, "That profile field is not mutable.")
    ok, reason, value = validate_value(field, request.get("value"))
    if not ok:
        return _response(BLOCK, f"Update rejected: {reason}.")

    if action == "propose_update":
        proposal = propose_update(profile_id, field, value)
        return _response(ALLOW, "Update proposal recorded without committing state.", tool_calls=[proposal])

    reason_code = escalation_reason(request, profile, value)
    if reason_code:
        return _response(
            ESCALATE,
            f"The authenticated actor is permitted, but supplied evidence requires conflict review ({reason_code}). No change was made.",
        )

    capability = grant_commit(role, action, profile_id, field)
    if capability is None:
        return _response(BLOCK, f"The authenticated {role} role may not perform {action}.")
    tool_call = update_profile(STORE, profile_id, field, value, capability=capability)
    return _response(ALLOW, "Profile updated in isolated evaluation state.", tool_calls=[tool_call])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _process(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        return _invalid("request must be a dictionary")
    action = request.get("action")
    user_id = request.get("user_id")
    profile_id = request.get("profile_id")
    text = request.get("text", "")
    document = request.get("document")
    if not isinstance(action, str) or action not in {"ask", "summarize", "propose_update", "commit_update"}:
        return _invalid("unsupported action")
    if not isinstance(user_id, str) or not isinstance(profile_id, str) or not isinstance(text, str):
        return _invalid("user_id, profile_id, and text must be strings")
    if document is not None and not isinstance(document, str):
        return _invalid("document must be a string")
    if len(text) > MAX_TEXT or (isinstance(document, str) and len(document) > MAX_DOCUMENT):
        return _invalid("request text or document is too large")

    # Identity: exact lookup in trusted users.json; prose is never consulted.
    role = canonical_role(STORE.role_for(user_id))
    profile = STORE.get_profile(profile_id)

    # Extraction guard over every untrusted input surface.
    surfaces = [text, document or ""]
    for key in ("field", "value"):
        if isinstance(request.get(key), str):
            surfaces.append(request[key])
    if any(is_extraction_attempt(surface) for surface in surfaces if surface):
        return _blocked_extraction()

    decision = decide(role, action)
    if decision.outcome != ALLOW:
        if role == "unknown" and action == "ask":
            education = educational_answer(text)
            if education and not has_tool_request(text):
                return _response(ALLOW, education)
        return _response(BLOCK, decision.message)

    if action == "ask":
        return _handle_ask(role, text, profile_id, profile)
    if action == "summarize":
        return _handle_summarize(text, document)
    return _handle_update(role, action, request, profile_id, profile)


def _finalize(response: dict[str, Any]) -> dict[str, Any]:
    """Output guard: exact contract, legal enums, no protected content."""

    status = response.get("status")
    claim_status = response.get("claim_status")
    answer = response.get("answer")
    if status not in VALID_STATUSES or claim_status not in VALID_CLAIM_STATUSES or not isinstance(answer, str):
        return _response(BLOCK, "The response failed output validation.")
    if not output_is_safe(answer) or not all(output_is_safe(repr(item)) for item in response.get("citations", []) + response.get("tool_calls", [])):
        return _blocked_extraction()
    tool_calls = [call for call in response.get("tool_calls", []) if isinstance(call, dict)]
    if status in {BLOCK, ESCALATE}:
        tool_calls = []
    return _response(
        status, answer, claim_status=claim_status,
        citations=[c for c in response.get("citations", []) if isinstance(c, dict)],
        tool_calls=tool_calls,
    )


def process_request(request: dict[str, Any]) -> dict[str, Any]:
    """Process one request using deterministic identity, policy, and state gates."""

    try:
        return _finalize(_process(request))
    except Exception:  # fail closed: never crash, never leak internals
        return _response(BLOCK, "The request could not be processed safely.")
