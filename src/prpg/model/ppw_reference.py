"""Integrity verification for the vendored corrected PPW reference source."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from prpg.errors import IntegrityError

REFERENCE_PACKAGE = "prpg.vendor.patton_politis_white"
REFERENCE_FILENAME = "opt_block_length_REV_dec07.txt"
REFERENCE_SHA256 = "34fe6a7b169716dbdfa45672084272ec2438d0b7e24eaa3f897dbff62ced80e8"
REFERENCE_BYTES = 4_673
UPSTREAM_SHA256 = "55dea54a42f8095c7da4cd618fba651c08677b7b4fc5dfabe9b08b7730157ed0"
UPSTREAM_BYTES = 4_796
SOURCE_URL = "https://public.econ.duke.edu/~ap172/opt_block_length_REV_dec07.txt"


@dataclass(frozen=True, slots=True)
class PPWReferenceVerification:
    """Verified upstream and normalized-vendor identities."""

    source_url: str
    upstream_sha256: str
    upstream_bytes: int
    vendored_sha256: str
    vendored_bytes: int
    line_ending_transform: str


def verify_ppw_reference() -> PPWReferenceVerification:
    """Fail closed if packaged reference or source metadata changes."""

    root = files(REFERENCE_PACKAGE)
    source = root.joinpath(REFERENCE_FILENAME).read_bytes()
    metadata_bytes = root.joinpath("SOURCE.json").read_bytes()
    return verify_ppw_reference_content(source, metadata_bytes)


def verify_ppw_reference_content(
    source: bytes,
    metadata_bytes: bytes,
) -> PPWReferenceVerification:
    """Verify injected bytes; exposed for installation and corruption tests."""

    if not isinstance(source, bytes) or not isinstance(metadata_bytes, bytes):
        raise IntegrityError("PPW reference inputs must be bytes")
    if len(source) != REFERENCE_BYTES or _sha256(source) != REFERENCE_SHA256:
        raise IntegrityError("vendored PPW reference hash or size is invalid")
    try:
        metadata: Any = json.loads(metadata_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("PPW source metadata is not valid JSON") from error
    if not isinstance(metadata, dict):
        raise IntegrityError("PPW source metadata must be an object")
    canonical = (
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        + b"\n"
    )
    if metadata_bytes != canonical:
        raise IntegrityError("PPW source metadata is not canonical JSON")
    expected = {
        "license_url": "https://public.econ.duke.edu/~ap172/license.html",
        "normalized_lf_bytes": REFERENCE_BYTES,
        "normalized_lf_sha256": REFERENCE_SHA256,
        "retrieved_source_bytes": UPSTREAM_BYTES,
        "retrieved_source_sha256": UPSTREAM_SHA256,
        "source_url": SOURCE_URL,
        "vendored_transform": "CRLF-to-LF-only",
    }
    if metadata != expected:
        raise IntegrityError("PPW source metadata identity is invalid")
    required_fragments = (
        b"Bmax = ceil(min(3*sqrt(n),n/3));",
        b"DSBhat = 2*(sum(lam(kk/M).*acv)^2);",
        b"Bstar = [1;1];",
    )
    if any(fragment not in source for fragment in required_fragments):
        raise IntegrityError("PPW reference omits a binding corrected branch")
    return PPWReferenceVerification(
        SOURCE_URL,
        UPSTREAM_SHA256,
        UPSTREAM_BYTES,
        REFERENCE_SHA256,
        REFERENCE_BYTES,
        "CRLF-to-LF-only",
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
