"""Stable application errors and process exit codes.

Error objects contain structured, non-secret context suitable for JSONL
events. Provider response bodies and credentials must never be put in
``details``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from typing import Any, ClassVar


class ExitCode(IntEnum):
    """Version-1 process exit-code registry."""

    SUCCESS = 0
    CONFIGURATION = 2
    DATA_ACQUISITION = 10
    DATA_VALIDATION = 11
    MODEL = 20
    GENERATION = 30
    VALIDATION = 40
    INTEGRITY = 50
    RELEASE = 60
    NOT_IMPLEMENTED = 69
    INTERNAL = 70
    INTERRUPTED = 130


class PRPGError(Exception):
    """Base class for expected, user-facing PRPG failures."""

    default_exit_code: ClassVar[ExitCode] = ExitCode.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        exit_code: ExitCode | int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = ExitCode(exit_code or self.default_exit_code)
        self.details = dict(details or {})

    def as_event(self) -> dict[str, Any]:
        """Return a stable structured representation for logs or CLI output."""

        return {
            "event": "error",
            "error_type": type(self).__name__,
            "exit_code": int(self.exit_code),
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(PRPGError):
    """The configuration is absent, malformed, or scientifically invalid."""

    default_exit_code = ExitCode.CONFIGURATION


ConfigError = ConfigurationError


class DataAcquisitionError(PRPGError):
    """A bounded provider acquisition failed."""

    default_exit_code = ExitCode.DATA_ACQUISITION


class DataValidationError(PRPGError):
    """Acquired or transformed data failed a frozen quality rule."""

    default_exit_code = ExitCode.DATA_VALIDATION


class ModelError(PRPGError):
    """Model fitting, selection, or calibration failed."""

    default_exit_code = ExitCode.MODEL


class GenerationError(PRPGError):
    """Simulation or output generation failed."""

    default_exit_code = ExitCode.GENERATION


class ValidationError(PRPGError):
    """A structural or scientific validation gate failed."""

    default_exit_code = ExitCode.VALIDATION


class IntegrityError(PRPGError):
    """A checksum, fingerprint, state, or replay invariant failed."""

    default_exit_code = ExitCode.INTEGRITY


class ReleaseError(PRPGError):
    """Release authorization, finalization, or attestation failed."""

    default_exit_code = ExitCode.RELEASE


class NotImplementedStageError(PRPGError):
    """A documented command belongs to a later, not-yet-qualified phase."""

    default_exit_code = ExitCode.NOT_IMPLEMENTED
