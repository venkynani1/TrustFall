"""Deterministic input/output guardrails for TRUSTFALL.

Everything here is plain Python: no model output is consulted. These helpers
only *detect* and *rewrite* untrusted text; authorization lives in policy.py.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from policy import TRUSTFALL_SYNTHETIC_PROTECTED_INSTRUCTION

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_INVISIBLE = re.compile(r"[­͏؜ᅟᅠ឴឵᠎​-‏‪-‮⁠-⁯﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize(text: str) -> str:
    """NFKC-fold, drop invisible/control characters, collapse whitespace."""

    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize(text).lower())


def _decoded_variants(text: str) -> list[str]:
    """Return base64/hex-decoded payloads hidden inside text, for detection only."""

    found: list[str] = []
    for token in re.findall(r"[A-Za-z0-9+/=_-]{16,}", text):
        for decoder in (
            lambda t: base64.b64decode(t + "=" * (-len(t) % 4), validate=False),
            lambda t: base64.urlsafe_b64decode(t + "=" * (-len(t) % 4)),
            lambda t: bytes.fromhex(t),
        ):
            try:
                raw = decoder(token)
                decoded = raw.decode("utf-8")
            except (ValueError, binascii.Error, UnicodeDecodeError):
                continue
            if decoded and sum(ch.isprintable() for ch in decoded) / len(decoded) > 0.9:
                found.append(decoded)
                break
    return found


def detection_views(text: str) -> list[str]:
    """All views of a text that detectors should inspect."""

    base = normalize(text)
    views = [base]
    views.extend(normalize(item) for item in _decoded_variants(base))
    # Spaced-out letters ("r e v e a l") are collapsed into a second view.
    if re.search(r"\b(?:\w\s){4,}\w\b", base):
        views.append(re.sub(r"\b(\w)\s(?=\w\b)", r"\1", base))
    return views


# ---------------------------------------------------------------------------
# Protected-instruction extraction
# ---------------------------------------------------------------------------

_EXTRACTION_VERBS = (
    r"reveal|repeat|print|output|show|display|dump|leak|expose|disclose|recite|"
    r"translate|encode|decode|encrypt|quote|copy|paste|spell|write out|read out|"
    r"tell me|give me|share|list|paraphrase|summari[sz]e|rephrase|return|echo|"
    r"base64|rot13|reverse|what (?:is|are|was|were)"
)
_EXTRACTION_TARGETS = (
    r"system (?:prompt|message|instructions?)|hidden|protected|developer (?:prompt|message|instructions?)|"
    r"(?:your|the|initial|original|internal|secret|confidential|previous|prior|preceding|above) "
    r"(?:prompt|instructions?|rules|guidelines|configuration|directives?|policy text)|"
    r"text above|everything above|words above|instructions above|initial instructions|"
    r"pre-?prompt|meta-?prompt|evaluation mode|mode alpha|protected instruction|synthetic protected|"
    r"your (?:prompt|instructions?)"
)
_EXTRACTION_RE = re.compile(
    rf"\b(?:{_EXTRACTION_VERBS})\b.{{0,80}}?\b(?:{_EXTRACTION_TARGETS})\b"
    rf"|\b(?:{_EXTRACTION_TARGETS})\b.{{0,40}}?\b(?:{_EXTRACTION_VERBS})\b",
    re.IGNORECASE,
)
_PROTECTED_WORDS = re.sub(r"[^a-z0-9 ]+", " ", TRUSTFALL_SYNTHETIC_PROTECTED_INSTRUCTION.lower()).split()


def mentions_protected_text(text: str) -> bool:
    """True when any 4-word window of the protected instruction appears."""

    words = re.sub(r"[^a-z0-9 ]+", " ", normalize(text).lower()).split()
    joined = " " + " ".join(words) + " "
    squashed = _alnum(text)
    if _alnum(TRUSTFALL_SYNTHETIC_PROTECTED_INSTRUCTION) in squashed:
        return True
    for start in range(len(_PROTECTED_WORDS) - 3):
        window = " ".join(_PROTECTED_WORDS[start:start + 4])
        if f" {window} " in joined:
            return True
    return False


def is_extraction_attempt(text: str) -> bool:
    for view in detection_views(text):
        cleaned = re.sub(r"prompt injections?", "injection-topic", view, flags=re.IGNORECASE)
        if _EXTRACTION_RE.search(cleaned) or mentions_protected_text(view):
            return True
    return False


# ---------------------------------------------------------------------------
# Instruction / tool / authority detection
# ---------------------------------------------------------------------------

_ROLE_LABEL = r"(?:system|admin|administrator|assistant|developer|instruction|instructions|ai|model|root|operator|hr|security|override|note to (?:ai|assistant|model|llm))"
ROLE_LABEL_RE = re.compile(rf"(?:\[\s*{_ROLE_LABEL}\s*\]|<\s*/?\s*{_ROLE_LABEL}\s*>|\b{_ROLE_LABEL}\s*(?:says?|:|>>|->))", re.IGNORECASE)

OVERRIDE_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass|skip|drop|abandon|suspend|disable)\b[^.?!]{0,60}?"
    r"\b(?:instructions?|rules|policy|policies|guidelines|guardrails?|restrictions?|constraints|safety|"
    r"previous|prior|above|earlier|system|checks?|verification|filters?)\b"
    r"|\byou are now\b|\bact as\b|\bpretend (?:to be|you)\b|\broleplay\b|\bdeveloper mode\b|\bjailbreak\b"
    r"|\bdo anything now\b|\bfrom now on\b|\bnew (?:instructions?|rules|task|policy)\s*:"
    r"|\b(?:the )?(?:assistant|ai|model|chatbot|llm|bot)\s+(?:must|should|shall|will|needs to|is required to)\b"
    r"|\b(?:respond|reply|answer|output)\s+(?:only\s+)?with\b|\b(?:say|state|report|tell (?:the )?user)\s+that\b"
    r"|\bdo not (?:summari[sz]e|mention|reveal|cite|label)\b|\btreat (?:this|the following|it) as (?:trusted|verified|true|fact)\b",
    re.IGNORECASE,
)

_MUTATION_VERBS = r"mark|set|change|update|modify|edit|delete|remove|erase|overwrite|grant|approve|commit|elevate|promote|flag|label|classify|reclassify|verify|confirm|make|give|assign|rewrite|replace|deactivate|activate|reset"
_MUTATION_TARGETS = r"verified|trusted|admin|administrator|role|roles|permissions?|privileges?|access|status|works_in|specialization|bio|profile|profiles|record|records|field|clearance|editor|database|state"
MUTATION_RE = re.compile(
    rf"(?:^|[:;,]\s*|\b(?:and|then|please|now|also|immediately|quickly|just)\s+)(?:{_MUTATION_VERBS})\b[^.?!]{{0,60}}?\b(?:{_MUTATION_TARGETS})\b",
    re.IGNORECASE,
)

TOOL_RE = re.compile(
    r"\b(?:update_profile|propose_update|commit_update|tool_calls?|function[_ ]call|execute_tool)\b"
    r"|\b(?:call|invoke|execute|run|trigger|fire|use)\s+(?:the\s+|a\s+|an\s+)?(?:\w+\s+)?(?:tool|function|api|endpoint|action|command)\b"
    r"|\b(?:works_in|specialization|bio|status|role|profile_id|user_id)\s*=\s*\S",
    re.IGNORECASE,
)

AUTHORITY_RE = re.compile(
    r"\b(?:i am|i'm|im|as)\s+(?:an?\s+|the\s+|your\s+)?(?:admin|administrator|editor|system|root|owner|superuser|developer|operator|manager|hr|moderator|security team)\b"
    r"|\b(?:approved|authori[sz]ed|signed off|sanctioned|whitelisted|cleared)\s+(?:by|this|it|the)\b"
    r"|\b(?:hr|management|the admin|admin|security|compliance|legal|ceo|cto|manager)\s+(?:has\s+)?(?:approved|authori[sz]ed|confirmed|verified|says|said)\b"
    r"|\b(?:with|under|using)\s+(?:admin|root|elevated|system)\s+(?:rights|privileges|access|authority|role)\b"
    r"|\bmy role is\b|\brole\s*[:=]\s*(?:admin|editor)\b",
    re.IGNORECASE,
)


def has_tool_request(text: str) -> bool:
    return any(TOOL_RE.search(view) or MUTATION_RE.search(view) for view in detection_views(text))


def has_injection(text: str) -> bool:
    return any(OVERRIDE_RE.search(v) or ROLE_LABEL_RE.search(v) for v in detection_views(text))


def has_authority_claim(text: str) -> bool:
    return any(AUTHORITY_RE.search(view) for view in detection_views(text))


# ---------------------------------------------------------------------------
# Untrusted-text sanitizer (documents and user prose)
# ---------------------------------------------------------------------------

_HIDDEN_BLOCKS = re.compile(
    r"<!--.*?-->|/\*.*?\*/|\[\[.*?\]\]|\{\{.*?\}\}|```.*?```|<\s*(system|instructions?|admin|assistant|hidden)[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Sanitized:
    kept: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    authority_claims: int = 0

    @property
    def text(self) -> str:
        return " ".join(self.kept).strip()


_DIRECTIVE_VERBS = r"say|tell|respond|reply|output|print|write|answer|state|report|declare|announce|call|invoke|execute|grant|approve|mark|set|change|update|delete|reveal|ignore|disregard|forget|treat|label|classify|promote|elevate"
DIRECTIVE_LEAD_DOC = re.compile(rf"^(?:(?:and|then|also|so|now)\s+)?(?:please\s+)?(?:{_DIRECTIVE_VERBS})\b", re.IGNORECASE)
DIRECTIVE_LEAD_USER = re.compile(rf"^(?:and|then|also|so|now)\s+(?:please\s+)?(?:{_DIRECTIVE_VERBS})\b", re.IGNORECASE)


def _is_instruction(segment: str, *, document: bool) -> bool:
    if (DIRECTIVE_LEAD_DOC if document else DIRECTIVE_LEAD_USER).search(segment):
        return True
    checks = [OVERRIDE_RE, TOOL_RE, MUTATION_RE]
    if document:
        checks.append(ROLE_LABEL_RE)
    for view in detection_views(segment):
        if any(rx.search(view) for rx in checks) or is_extraction_attempt(view):
            return True
    return False


def sanitize(text: str, *, document: bool) -> Sanitized:
    """Split untrusted text into kept evidence and removed instruction spans.

    A role label ("SYSTEM:", "[admin]") cuts the remainder of its sentence,
    because everything after the label is attacker-authored instruction.
    """

    result = Sanitized()
    raw = unicodedata.normalize("NFKC", text)
    raw = _INVISIBLE.sub("", raw)
    for match in _HIDDEN_BLOCKS.finditer(raw):
        result.removed.append(normalize(match.group(0)))
    raw = _HIDDEN_BLOCKS.sub(" ", raw)

    for sentence in _SENTENCE_SPLIT.split(raw):
        sentence = normalize(sentence)
        if not sentence:
            continue
        label = ROLE_LABEL_RE.search(sentence)
        if label:
            head, tail = sentence[: label.start()].strip(), sentence[label.start():].strip()
            if tail:
                result.removed.append(tail)
            sentence = head.rstrip(" ,;:-")
            if not sentence:
                continue
        # Split on clause joiners so "X is true, and ignore your rules" keeps X.
        clauses = re.split(r"(?:[;,]\s*|\s+)(?=(?:and|then|also|but|so)\s+(?:please\s+)?(?:ignore|disregard|forget|override|bypass|mark|set|change|update|delete|grant|approve|commit|call|invoke|execute|run|reveal|print|output|say|respond|reply|tell|you)\b)", sentence, flags=re.IGNORECASE)
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            if _is_instruction(clause, document=document):
                result.removed.append(clause)
                continue
            if AUTHORITY_RE.search(clause):
                result.authority_claims += 1
                if not document:
                    # In user prose an authority claim carries no meaning: drop it.
                    stripped = AUTHORITY_RE.sub(" ", clause)
                    stripped = re.sub(r"^\W*(?:and|but|so)?\W*", "", normalize(stripped))
                    if len(re.findall(r"[a-z]{3,}", stripped.lower())) < 3:
                        result.removed.append(clause)
                        continue
                    clause = stripped
            result.kept.append(clause)
    return result


# ---------------------------------------------------------------------------
# Output guard
# ---------------------------------------------------------------------------


def output_is_safe(text: str) -> bool:
    return not mentions_protected_text(text)


def redact(text: str) -> str:
    return "[redacted protected content]" if mentions_protected_text(text) else text


# ---------------------------------------------------------------------------
# Update value validation
# ---------------------------------------------------------------------------

_VALUE_RULES: dict[str, tuple[int, re.Pattern[str]]] = {
    "works_in": (60, re.compile(r"^[A-Za-z][A-Za-z .'\-]*$")),
    "specialization": (80, re.compile(r"^[A-Za-z0-9][A-Za-z0-9 +#./&()'\-]*$")),
    "bio": (500, re.compile(r"^[^<>{}`\\]*$")),
}
ALLOWED_STATUS_VALUES = frozenset({"active", "inactive"})


def validate_value(field_name: str, value: Any) -> tuple[bool, str, Any]:
    """Return (ok, reason, normalized_value). Value text is untrusted data."""

    if not isinstance(value, str):
        return False, "update values must be a single string", None
    clean = normalize(value)
    if not clean:
        return False, "update value is empty", None
    if field_name == "status":
        lowered = clean.lower()
        if lowered not in ALLOWED_STATUS_VALUES:
            return False, "status must be active or inactive", None
        return True, "", lowered
    limit, pattern = _VALUE_RULES[field_name]
    if len(clean) > limit or not pattern.match(clean):
        return False, f"value is not a valid {field_name}", None
    if re.search(r"https?://|www\.|\.(?:com|net|org|io)\b", clean, re.IGNORECASE):
        return False, "URLs are not accepted in profile fields", None
    if (
        has_injection(clean) or has_tool_request(clean) or is_extraction_attempt(clean)
        or has_authority_claim(clean) or ROLE_LABEL_RE.search(clean)
    ):
        return False, "value contains instruction-like content", None
    return True, "", clean


# ---------------------------------------------------------------------------
# Harmless educational answers (fixed text, safe for any role)
# ---------------------------------------------------------------------------

EDUCATION: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bindirect (?:prompt )?injection\b", re.I),
     "Indirect prompt injection hides instructions inside content a system processes, such as a document or web page, hoping the system treats that data as commands."),
    (re.compile(r"\bprompt injection\b", re.I),
     "Prompt injection is untrusted content attempting to redirect a system away from its governing rules."),
    (re.compile(r"\btrust boundar(?:y|ies)\b", re.I),
     "A trust boundary separates data or actors with different authority; anything crossing it must be validated before it can influence decisions."),
    (re.compile(r"\bprovenance\b", re.I),
     "Provenance records where a piece of information came from, so trusted records and untrusted claims are never confused."),
    (re.compile(r"\bguardrails?\b", re.I),
     "Guardrails are deterministic checks around an AI system that constrain inputs, outputs, and actions regardless of what the model says."),
    (re.compile(r"\bleast privilege\b", re.I),
     "Least privilege means each identity gets only the permissions it needs, so a compromised or tricked component can do limited harm."),
    (re.compile(r"\bhallucinat\w*", re.I),
     "A hallucination is model output that sounds plausible but is not supported by trusted data."),
    (re.compile(r"\bjailbreak\w*", re.I),
     "A jailbreak is an attempt to talk a model out of its safety rules; robust systems enforce those rules in code, not in the model."),
    (re.compile(r"\b(?:rbac|role[- ]based access control)\b", re.I),
     "Role-based access control grants permissions by authenticated role rather than by what a request claims about itself."),
    (re.compile(r"\bdata exfiltration\b", re.I),
     "Data exfiltration is unauthorized extraction of information; defenses include output filtering and minimizing what a system can reveal."),
)
_EDU_INTENT = re.compile(r"\b(?:what|explain|define|describe|meaning|how does|how do|why|tell me about|overview)\b|\?", re.I)
PROFILE_MARKER = re.compile(r"\bmav-?\d+\b|\bworks? in\b|\bspeciali[sz]|\bbio(?:graphy)?\b|\bprofile\b|\bstatus\b", re.I)


def educational_answer(text: str) -> str | None:
    clean = normalize(text)
    if not _EDU_INTENT.search(clean) or PROFILE_MARKER.search(clean):
        return None
    for pattern, answer in EDUCATION:
        if pattern.search(clean):
            return answer
    return None
