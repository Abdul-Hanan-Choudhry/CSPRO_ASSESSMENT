"""Read-access rules. The single source of truth for who may see which record.

The same rule is enforced twice at query time: as a Qdrant payload filter (so other customers'
data is never fetched) and as `can_read` on every returned hit (so a filter bug cannot leak it).
"""

from collections.abc import Mapping
from typing import Any

from qdrant_client import models

from app.config import Principal
from app.schemas import Visibility


def can_read(principal: Principal, payload: Mapping[str, Any]) -> bool:
    if principal.role == "admin":
        return True
    visibility = payload.get("visibility")
    if visibility == Visibility.PUBLIC_KB:
        return True
    # Fail closed: unknown visibility values, internal_kb, and other owners are all denied.
    return visibility == Visibility.CUSTOMER and payload.get("customer_id") == principal.customer_id


def qdrant_filter(principal: Principal) -> models.Filter | None:
    """Payload filter equivalent to `can_read`. None means no restriction (admin)."""
    if principal.role == "admin":
        return None
    return models.Filter(
        should=[
            _match("visibility", Visibility.PUBLIC_KB),
            models.Filter(
                must=[
                    _match("visibility", Visibility.CUSTOMER),
                    _match("customer_id", principal.customer_id),
                ]
            ),
        ]
    )


def _match(key: str, value: str | None) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))
