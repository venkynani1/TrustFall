"""Simulated proposal and state-changing actions."""

from __future__ import annotations

from typing import Any

from policy import ALLOWED_UPDATE_FIELDS, redeem_commit
from state import ProfileState


def propose_update(profile_id: str, field: str, value: Any) -> dict[str, Any]:
    """Return a proposal record without mutating state."""

    return {
        "tool": "propose_update",
        "profile_id": profile_id,
        "field": field,
        "value": value,
        "executed": False,
    }


def update_profile(
    store: ProfileState,
    profile_id: str,
    field: str,
    value: Any,
    *,
    capability: Any = None,
) -> dict[str, Any]:
    """Mutate only with a single-use capability minted by policy.grant_commit.

    A plain boolean can be forged by any caller; the capability is sealed,
    bound to one profile and field, and consumed on use.
    """

    if field not in ALLOWED_UPDATE_FIELDS:
        raise PermissionError("field is not mutable")
    if not redeem_commit(capability, profile_id, field):
        raise PermissionError("state mutation was not authorized")
    store.update_profile(profile_id, field, value)
    return {
        "tool": "update_profile",
        "profile_id": profile_id,
        "field": field,
        "executed": True,
    }
