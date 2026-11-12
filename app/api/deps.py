"""Request dependencies: the service container and the authenticated principal."""

import hmac
from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader

from app.config import Principal
from app.container import Services
from app.errors import AuthenticationError, PermissionDeniedError

_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_principal(
    key: Annotated[str | None, Security(_api_key)],
    services: Annotated[Services, Depends(get_services)],
) -> Principal:
    if not key:
        raise AuthenticationError("missing X-API-Key header")
    match = None
    # Compare against every key in constant time so response timing does not reveal prefixes.
    for known, principal in services.settings.api_keys.items():
        if hmac.compare_digest(known.encode(), key.encode()):
            match = principal
    if match is None:
        raise AuthenticationError("invalid API key")
    return match


def require_admin(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
    if principal.role != "admin":
        raise PermissionDeniedError("admin role required")
    return principal
