from __future__ import annotations

import pytest

from prpg.errors import (
    ConfigurationError,
    DataAcquisitionError,
    DataValidationError,
    ExitCode,
    GenerationError,
    IntegrityError,
    ModelError,
    NotImplementedStageError,
    PRPGError,
    ReleaseError,
    ValidationError,
)


@pytest.mark.parametrize(
    ("error_type", "exit_code"),
    [
        (ConfigurationError, ExitCode.CONFIGURATION),
        (DataAcquisitionError, ExitCode.DATA_ACQUISITION),
        (DataValidationError, ExitCode.DATA_VALIDATION),
        (ModelError, ExitCode.MODEL),
        (GenerationError, ExitCode.GENERATION),
        (ValidationError, ExitCode.VALIDATION),
        (IntegrityError, ExitCode.INTEGRITY),
        (ReleaseError, ExitCode.RELEASE),
        (NotImplementedStageError, ExitCode.NOT_IMPLEMENTED),
    ],
)
def test_error_taxonomy_has_stable_exit_codes(
    error_type: type[PRPGError], exit_code: ExitCode
) -> None:
    error = error_type("example", details={"safe": True})

    assert error.exit_code == exit_code
    assert error.as_event() == {
        "event": "error",
        "error_type": error_type.__name__,
        "exit_code": int(exit_code),
        "message": "example",
        "details": {"safe": True},
    }
