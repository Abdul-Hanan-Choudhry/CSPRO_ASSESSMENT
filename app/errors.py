"""Domain exception hierarchy. Each subclass carries the HTTP status and a stable error code."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthenticated"


class PermissionDeniedError(AppError):
    status_code = 403
    code = "forbidden"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class DependencyUnavailableError(AppError):
    status_code = 503
    code = "dependency_unavailable"


class VectorStoreUnavailableError(DependencyUnavailableError):
    code = "vector_store_unavailable"


class LLMUnavailableError(DependencyUnavailableError):
    code = "llm_unavailable"


class LLMOutputError(AppError):
    """The model answered, but not in the required JSON contract even after one repair retry."""

    status_code = 502
    code = "bad_llm_output"
