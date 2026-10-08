"""Deterministic claim-by-claim verification against trusted profile data.

Text is split into independent clauses. Each clause is classified on its own
(verified / unverified / unknown), so a true neighbouring fact can never
promote an unsupported claim. The single top-level claim_status reports the
least-trusted result: unverified > unknown > verified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from guardrails import normalize

SEVERITY = {"verified": 0, "unknown": 1, "unverified": 2}


@dataclass(frozen=True)
class Claim:
    text: str
    status: str            # verified | unverified | unknown
    field: str | None
    note: str
    is_question: bool = False


@dataclass(frozen=True)
class ClaimResult:
    answer: str
    claim_status: str
    citations: list[dict[str, str]]
    needs_transform: bool = False
    claims: tuple[Claim, ...] = field(default_factory=tuple)


def _citation(profile_id: str, field_name: str) -> dict[str, str]:
    return {"source": "profiles.json", "profile_id": profile_id, "field": field_name}


# ---------------------------------------------------------------------------
# Clause splitting and cleanup
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(
    r"(?<=[.!?;])\s+|\s*;\s*|\s*,\s*(?:and|but|also|plus|while)?\s*|\s+(?:and|but|while|whereas|plus|also)\s+(?=(?:he|she|they|it|mav|is|has|had|was|works?|lives?|speciali\w*|expert|based|located|climbed|won|holds?|owns?|does|did|can|what|where|which|who|how|when|status)\b)",
    re.IGNORECASE,
)
_ATTRIBUTION_RE = re.compile(
    r"^(?:(?:according to|per|as per|based on)\s+[^,]{0,40},?\s*"
    r"|(?:the\s+)?(?:trusted|official|verified|hr|internal|system)?\s*(?:records?|data|database|files?|sources?|profile)\s+(?:say|says|show|shows|state|states|confirm|confirms|indicate|indicates)\s*(?:that\s*)?"
    r"|it is (?:verified|confirmed|known|true|a fact) that\s*|i (?:heard|think|believe|read|know|confirm|certify|attest|guarantee|assure you) (?:that\s*)?"
    r"|(?:note|fact|claim|fyi|update)\s*:\s*|please\s+|kindly\s+|can you (?:confirm|check|verify) (?:that|whether|if)?\s*"
    r"|(?:confirm|check|verify) (?:that|whether|if)\s*|is it true that\s*)",
    re.IGNORECASE,
)
_QUESTION_START = re.compile(r"^(?:what|where|which|who|whom|whose|how|when|why|does|do|did|is|are|was|were|can|could|has|have|tell me|show me|give me|list|describe)\b", re.I)
_FILLER = {
    "mav", "the", "a", "an", "of", "for", "about", "me", "please", "thanks", "thank", "you", "tell",
    "show", "give", "profile", "info", "information", "details", "their", "his", "her", "its", "they",
    "he", "she", "it", "is", "and", "also", "this", "that", "person", "user", "what", "s", "hi", "hello",
}

_LOC_Q = re.compile(r"\b(?:where|which city|what city|what location|which location|which office|what office|location|city|based where)\b", re.I)
_LOC_VERB = re.compile(r"\b(?:works?|working|worked|work|based|located|lives?|living|resides?|stationed|employed|posted|sits|located|office)\b", re.I)
_LOC_CLAIM = re.compile(
    r"\b(?:works?|working|worked|based|located|lives?|living|resides?|stationed|employed|posted|sits|relocated|moved|transferred)\b"
    r"(?:\s+(?:now|currently|today|remotely|primarily|mainly|full[- ]time))?\s+(?:in|at|from|out of|to)\s+(?:the\s+)?(?:city of\s+)?([a-z][a-z .'\-]*)",
    re.I,
)
_LOC_STOP = re.compile(r"\s+(?:and|but|or|who|which|where|office|city|now|currently|today|branch|team|area|region|india|as|since|for|with|on)\b.*$", re.I)

_SPEC_Q = re.compile(r"\b(?:speciali[sz]ation|specialty|speciality|skills?|expertise|focus|tech stack|stack|domain|field of work|what does .* do)\b", re.I)
_SPEC_CLAIM = re.compile(
    r"\b(?:speciali[sz](?:es|ed|ing|e)?|expert|experienced|skilled|proficient|focus(?:es|ed|ing)?)\s+(?:in|on|at|with)\s+([a-z0-9][a-z0-9 +#./&'\-]*)"
    r"|\b(?:speciali[sz]ation|specialty|speciality|expertise|focus)\s+(?:is|=|:)\s+([a-z0-9][a-z0-9 +#./&'\-]*)"
    r"|\bis an?\s+([a-z0-9 +#./&'\-]+?)\s+(?:developer|engineer|specialist|expert|programmer|architect|consultant)\b",
    re.I,
)
_STATUS_Q = re.compile(r"\bstatus\b|\bis (?:\S+ )?(?:still )?(?:active|inactive)\??$", re.I)
_STATUS_CLAIM = re.compile(r"\b(?:is|status is|status:|status =|remains|has been|was)\s+(?:currently\s+|still\s+|now\s+)?(active|inactive|retired|suspended|terminated|deactivated|on leave|fired|banned)\b", re.I)
_NAME_Q = re.compile(r"\b(?:name|who is|who's)\b", re.I)
_NAME_CLAIM = re.compile(r"\b(?:name is|is named|is called|goes by)\s+([a-z][a-z .'\-]*)", re.I)
_BIO_Q = re.compile(r"\b(?:bio|biography|background|about|summary|overview|describe)\b", re.I)


def _clean_value(value: str) -> str:
    value = _LOC_STOP.sub("", value.strip())
    return re.sub(r"[\s.'\-]+$", "", value).strip().lower()


def _split(text: str) -> list[str]:
    parts = [p.strip(" ,;:-") for p in _SPLIT_RE.split(text) if p and p.strip(" ,;:-")]
    out = []
    for part in parts:
        previous = None
        while previous != part:
            previous = part
            part = _ATTRIBUTION_RE.sub("", part).strip(" ,;:-")
        if part:
            out.append(part)
    return out


def _is_filler(clause: str, profile: dict[str, Any]) -> bool:
    words = re.findall(r"[a-z]+", clause.lower())
    names = set(re.findall(r"[a-z]+", str(profile.get("display_name", "")).lower()))
    return all(w in _FILLER or w in names or w.isdigit() for w in words)


def _matches(value: str, trusted: str) -> bool:
    value, trusted = value.lower().strip(), trusted.lower().strip()
    if not value or not trusted:
        return False
    return value == trusted or trusted in value.split(" , ") or re.search(rf"\b{re.escape(trusted)}\b", value) is not None or (
        len(value) >= 3 and re.search(rf"\b{re.escape(value)}\b", trusted) is not None
    )


def _short(clause: str) -> str:
    clause = re.sub(r"[\"“”]", "'", clause)
    return clause if len(clause) <= 90 else clause[:87] + "..."


# ---------------------------------------------------------------------------
# Per-clause classification
# ---------------------------------------------------------------------------


def _classify(clause: str, profile: dict[str, Any], whole_is_question: bool) -> Claim | None:
    pid = str(profile.get("profile_id", "this profile"))
    works_in = str(profile.get("works_in") or "").strip()
    spec = str(profile.get("specialization") or "").strip()
    bio = str(profile.get("bio") or "").strip()
    status = str(profile.get("status") or "").strip()
    name = str(profile.get("display_name") or "").strip()
    lowered = clause.lower()
    question = clause.endswith("?") or bool(_QUESTION_START.search(clause)) or (
        whole_is_question and len(re.findall(r"[a-z]+", lowered)) <= 6
    )

    def known(field_name: str, value: str, ok: bool, wanted: str = "") -> Claim:
        if not value:
            return Claim(clause, "unknown", None, f"trusted data has no {field_name} for {pid}", question)
        if ok:
            return Claim(clause, "verified", field_name, f"trusted record: {field_name} = {value}", question)
        detail = f" (stated: {wanted})" if wanted else ""
        return Claim(clause, "unverified", field_name, f"trusted record says {field_name} = {value}{detail}", question)

    # Location ---------------------------------------------------------------
    loc = _LOC_CLAIM.search(clause)
    if loc:
        wanted = _clean_value(loc.group(1))
        if wanted and wanted not in {"what", "which", "where"}:
            return known("works_in", works_in, _matches(wanted, works_in), wanted)
    if _LOC_Q.search(clause) and (_LOC_VERB.search(clause) or re.search(r"\b(?:city|location|office)\b", lowered)):
        return known("works_in", works_in, True)

    # Specialization -------------------------------------------------------
    sp = _SPEC_CLAIM.search(clause)
    if sp:
        wanted = next((g for g in sp.groups() if g), "").strip().lower()
        wanted = re.sub(r"\s+(?:and|but|with|for)\b.*$", "", wanted).strip(" .'-")
        if wanted and wanted not in {"what", "which", "what?"}:
            return known("specialization", spec, _matches(wanted, spec), wanted)
    if _SPEC_Q.search(clause):
        return known("specialization", spec, True)

    # Status -----------------------------------------------------------------
    st = _STATUS_CLAIM.search(clause)
    if st and not re.search(r"\bnot\b", lowered):
        wanted = st.group(1).lower()
        return known("status", status, wanted == status.lower(), wanted)
    if _STATUS_Q.search(clause):
        return known("status", status, True)

    # Display name -----------------------------------------------------------
    nm = _NAME_CLAIM.search(clause)
    if nm:
        wanted = _clean_value(nm.group(1))
        return known("display_name", name, _matches(wanted, name), wanted)
    if _NAME_Q.search(clause) and question:
        return known("display_name", name, True)

    # Biography --------------------------------------------------------------
    if _BIO_Q.search(clause) and question:
        return known("bio", bio, True)

    if _is_filler(clause, profile):
        return None
    return Claim(clause, "unknown", None, "trusted profile data cannot settle this", question)


def _render(claim: Claim, profile: dict[str, Any]) -> str:
    pid = str(profile.get("profile_id", "this profile"))
    value = str(profile.get(claim.field, "")) if claim.field else ""
    if claim.status == "verified" and claim.is_question:
        phrasing = {
            "works_in": f"{pid} works in {value}.",
            "specialization": f"{pid} specializes in {value}.",
            "status": f"{pid} has status {value}.",
            "display_name": f"{pid} is {value}.",
            "bio": f"{pid} bio: {value}",
        }
        return phrasing.get(claim.field or "", f"{pid}: {value}.")
    return ""


def verify_profile_text(text: str, profile: dict[str, Any]) -> ClaimResult:
    pid = str(profile.get("profile_id", "this profile"))
    clean = normalize(text)
    whole_is_question = "?" in clean or bool(_QUESTION_START.search(clean))
    claims: list[Claim] = []
    for clause in _split(clean):
        claim = _classify(clause, profile, whole_is_question)
        if claim is not None:
            claims.append(claim)

    if not claims:
        # Generic "tell me about MAV-042": return a trusted overview.
        fields = [f for f in ("works_in", "specialization", "status") if profile.get(f)]
        if fields and re.search(r"\bmav-?\d+|\babout\b|\bprofile\b|\bwho\b", clean, re.I):
            parts = [f"{f} = {profile[f]}" for f in fields]
            return ClaimResult(
                f"Trusted profile data for {pid}: " + "; ".join(parts) + ".", "verified",
                [_citation(pid, f) for f in fields],
            )
        return ClaimResult(f"Trusted profile data cannot settle that claim about {pid}.", "unknown", [])

    # De-duplicate citations, preserving order; only trusted-field comparisons cite.
    citations: list[dict[str, str]] = []
    for claim in claims:
        if claim.field and claim.status in {"verified", "unverified"}:
            cite = _citation(pid, claim.field)
            if cite not in citations:
                citations.append(cite)

    worst = max(claims, key=lambda c: SEVERITY[c.status]).status
    if len(claims) == 1:
        claim = claims[0]
        if claim.status == "verified":
            answer = _render(claim, profile) or f"Verified from trusted profile data: {claim.note.replace('trusted record: ', '')}."
        elif claim.status == "unverified":
            answer = f"Trusted data does not support that claim about {pid}: {claim.note}."
        elif claim.field is None:
            answer = f"Trusted profile data cannot settle that claim about {pid}."
        else:
            answer = f"Trusted profile data cannot settle that claim about {pid}: {claim.note}."
        return ClaimResult(answer, claim.status, citations if claim.status != "unknown" else [], False, tuple(claims))

    statuses = {c.status for c in claims}
    lines = []
    for index, claim in enumerate(claims, 1):
        rendered = _render(claim, profile)
        label = claim.status.upper()
        lines.append(f"({index}) '{_short(claim.text)}' - {label}: {rendered or claim.note}")
    answer = (
        f"Each claim about {pid} was checked independently against trusted profile data. "
        + " ".join(lines)
        + f" Overall: {worst}."
    )
    mixed = len(statuses) > 1 or worst != "verified"
    return ClaimResult(answer, worst, citations, mixed, tuple(claims))
