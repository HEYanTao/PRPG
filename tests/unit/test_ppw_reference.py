from __future__ import annotations

import pytest

from prpg.errors import IntegrityError
from prpg.model.ppw_reference import (
    REFERENCE_BYTES,
    REFERENCE_SHA256,
    UPSTREAM_SHA256,
    verify_ppw_reference,
    verify_ppw_reference_content,
)


def test_vendored_corrected_ppw_reference_is_hash_pinned() -> None:
    result = verify_ppw_reference()
    assert result.vendored_bytes == REFERENCE_BYTES == 4_673
    assert result.vendored_sha256 == REFERENCE_SHA256
    assert result.upstream_sha256 == UPSTREAM_SHA256
    assert result.line_ending_transform == "CRLF-to-LF-only"


def test_reference_corruption_fails_closed() -> None:
    result = verify_ppw_reference()
    del result
    with pytest.raises(IntegrityError, match="hash or size"):
        verify_ppw_reference_content(b"changed", b"{}\n")
