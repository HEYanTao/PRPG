"""Lean G8 finalization and independent release-attestation boundaries.

This module consumes an already canonical ``DATA_VALIDATED`` run.  It does
not generate returns, repeat G7's all-row scientific validation, sign data,
manage Git, or implement crash recovery.  Its responsibilities are narrow:

* re-hash the immutable G7 consumer bundle and its bound structural evidence;
* reject every file or directory outside an explicit finalization allowlist;
* publish deterministic consumer and root ``SHA256SUMS`` plus minimal frozen
  manifests, then publish checksum-excluded ``COMPLETE`` exactly once; and
* validate a separately stored, independent-verifier attestation before
  publishing checksum-excluded ``RELEASED`` exactly once.

SHA-256 here proves byte integrity only.  It is not an authorship signature.
Unexpected or partial states fail closed and require a new run; this module
intentionally contains no recovery state machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, TypeAlias, cast
from urllib.parse import unquote, urlsplit

from prpg.data.calendar import CALENDAR_ID
from prpg.errors import GenerationError, IntegrityError, ReleaseError
from prpg.storage.csv_v1 import CSV_SCHEMA_VERSION, CSV_SERIALIZER_VERSION
from prpg.validation.structural import (
    CANONICAL_STRUCTURAL_PLAN,
    CONSUMER_MANIFEST_SCHEMA_ID,
    CONSUMER_MANIFEST_SCHEMA_VERSION,
    DATA_VALIDATED_SCHEMA_ID,
    DATA_VALIDATED_SCHEMA_VERSION,
    LOG_AGGREGATION_TOLERANCE,
    SIMPLE_RETURN_TOLERANCE,
    STRUCTURAL_REPORT_SCHEMA_ID,
    STRUCTURAL_REPORT_SCHEMA_VERSION,
    StructuralValidationPlan,
)

EVIDENCE_INDEX_SCHEMA_ID: Final = "prpg-g8-evidence-index-v1"
RUN_MANIFEST_SCHEMA_ID: Final = "prpg-g8-final-run-manifest-v1"
COMPLETE_SCHEMA_ID: Final = "prpg-complete-v1"
EXTERNAL_ATTESTATION_SCHEMA_ID: Final = "prpg-g8-external-attestation-v1"
RELEASED_SCHEMA_ID: Final = "prpg-released-v1"
RELEASE_SCHEMA_VERSION: Final = 1
MAX_JSON_BYTES: Final = 16 * 1024 * 1024
READ_CHUNK_BYTES: Final = 1024 * 1024

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]{0,127}\Z")
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_EVIDENCE_ID = re.compile(r"[A-Za-z][A-Za-z0-9._-]{0,127}\Z")

_LIFECYCLE_MARKERS: Final = frozenset(
    {"RUNNING", "INTERRUPTED", "FAILED", "DATA_VALIDATED", "COMPLETE", "RELEASED"}
)
_GENERATED_FILES: Final = frozenset(
    {
        "consumer/SHA256SUMS",
        "evidence-index.json",
        "run-manifest.json",
        "SHA256SUMS",
    }
)
_ALLOWED_AUDIT_TOP_LEVEL: Final = frozenset(
    {
        "audit_hidden",
        "config",
        "data",
        "logs",
        "model",
        "receipts",
        "validation",
        "run-state.json",
    }
)
_CONSUMER_SHARD_KEYS: Final = frozenset(
    {
        "work_unit_id",
        "work_unit_receipt_fingerprint",
        "family",
        "frequency",
        "shard_index",
        "first_path_entity",
        "last_path_entity",
        "first_path_id",
        "last_path_id",
        "path_count",
        "years",
        "periods_per_path",
        "rows",
        "relative_path",
        "bytes",
        "sha256",
        "csv_receipt_fingerprint",
    }
)

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
EvidencePurpose: TypeAlias = Literal[
    "clean_replay", "release_authorization", "additional"
]


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class ReleaseEvidence:
    """One already-written evidence file bound into the final evidence index."""

    evidence_id: str
    purpose: EvidencePurpose
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.evidence_id, str)
            or _EVIDENCE_ID.fullmatch(self.evidence_id) is None
        ):
            raise GenerationError("release evidence ID is invalid")
        if self.purpose not in {
            "clean_replay",
            "release_authorization",
            "additional",
        }:
            raise GenerationError("release evidence purpose is invalid")
        _checked_relative_path(self.relative_path, "release evidence path")
        _require_fingerprint(self.sha256, "release evidence SHA-256")

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "evidence_id": self.evidence_id,
            "purpose": self.purpose,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ReleaseFinalizationPlan:
    """Exact, prospective allowlist and provenance for one G8 finalization."""

    run_id: str
    generation_commit: str
    release_commit: str
    release_tag: str
    finalizer_identity: str
    immutable_audit_files: tuple[str, ...]
    evidence: tuple[ReleaseEvidence, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for value, label in (
            (self.run_id, "run ID"),
            (self.release_tag, "release tag"),
            (self.finalizer_identity, "finalizer identity"),
        ):
            _require_token(value, label)
        for value, label in (
            (self.generation_commit, "generation commit"),
            (self.release_commit, "release commit"),
        ):
            if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
                raise GenerationError(f"{label} is invalid")
        if self.generation_commit == self.release_commit:
            raise GenerationError("generation and release commits must be distinct")
        if (
            not isinstance(self.immutable_audit_files, tuple)
            or not self.immutable_audit_files
        ):
            raise GenerationError("immutable audit-file allowlist is empty")
        if tuple(sorted(set(self.immutable_audit_files))) != self.immutable_audit_files:
            raise GenerationError("immutable audit-file allowlist is not sorted/unique")
        for relative in self.immutable_audit_files:
            checked = _checked_relative_path(relative, "immutable audit file")
            first = checked.parts[0]
            if first not in _ALLOWED_AUDIT_TOP_LEVEL:
                raise GenerationError(
                    "immutable audit file has an unapproved top level"
                )
            if first == "consumer" or relative in _GENERATED_FILES:
                raise GenerationError("immutable audit file is release-generated")
            if relative in _LIFECYCLE_MARKERS:
                raise GenerationError("lifecycle marker cannot be an audit file")
        required = {"run-state.json", "validation/structural-report.json"}
        if not required.issubset(self.immutable_audit_files):
            raise GenerationError("finalization allowlist omits required frozen files")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise GenerationError("release evidence is empty")
        ids = [item.evidence_id for item in self.evidence]
        paths = [item.relative_path for item in self.evidence]
        if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
            raise GenerationError("release evidence IDs and paths must be unique")
        if tuple(sorted(ids)) != tuple(ids):
            raise GenerationError("release evidence must be sorted by evidence ID")
        purposes = [item.purpose for item in self.evidence]
        if purposes.count("clean_replay") != 1:
            raise GenerationError("release evidence requires one clean replay")
        if purposes.count("release_authorization") != 1:
            raise GenerationError("release evidence requires one release authorization")
        for item in self.evidence:
            if item.relative_path not in self.immutable_audit_files:
                raise GenerationError("release evidence is outside the audit allowlist")
        object.__setattr__(
            self,
            "fingerprint",
            _sha256_bytes(_canonical_bytes(self.identity())),
        )

    def identity(self) -> dict[str, JSONValue]:
        return {
            "schema_id": "prpg-g8-finalization-plan-v1",
            "schema_version": RELEASE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "generation_commit": self.generation_commit,
            "release_commit": self.release_commit,
            "release_tag": self.release_tag,
            "finalizer_identity": self.finalizer_identity,
            "immutable_audit_files": list(self.immutable_audit_files),
            "evidence": [item.as_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class CompleteVerification:
    """Read-only verified identity of a content-finalized run."""

    run_id: str
    root_sha256sums_sha256: str
    complete_sha256: str
    run_manifest_sha256: str
    data_validated_sha256: str
    generation_commit: str
    release_commit: str
    release_tag: str
    finalizer_identity: str
    covered_file_count: int


@dataclass(frozen=True, slots=True)
class ReleasedVerification:
    """Read-only verified identity of an independently released run."""

    complete: CompleteVerification
    attestation_uri: str
    attestation_sha256: str
    verifier_identity: str
    released_sha256: str


def finalize_complete(
    run_root: str | Path, *, plan: ReleaseFinalizationPlan
) -> CompleteVerification:
    """Finalize one canonical G7 run and atomically publish ``COMPLETE``.

    The operation never overwrites.  It requires the exact marker set
    ``{DATA_VALIDATED}``, an exact caller-supplied audit allowlist, and the
    manifest-derived consumer allowlist before publishing any new file.
    """

    root = _existing_safe_directory(run_root, "run root")
    if not isinstance(plan, ReleaseFinalizationPlan):
        raise GenerationError("release finalization plan is invalid")
    if plan.fingerprint != _sha256_bytes(_canonical_bytes(plan.identity())):
        raise IntegrityError("release finalization plan fingerprint is invalid")
    _require_marker_set(root, {"DATA_VALIDATED"})
    for relative in _GENERATED_FILES:
        _require_absent(root / relative, "release-generated file")

    consumer = _load_and_verify_consumer(root)
    data_validated = _verify_data_validated(
        root,
        cast(str, consumer["manifest_sha256"]),
        cast(dict[str, Any], consumer["manifest"]),
    )
    expected_staged = {
        "DATA_VALIDATED",
        *plan.immutable_audit_files,
        *cast(tuple[str, ...], consumer["files"]),
    }
    _verify_exact_tree(root, expected_staged)
    _verify_evidence(root, plan.evidence)

    consumer_entries = cast(dict[str, str], consumer["hashes"])
    consumer_checksum_content = _checksum_bytes(
        {
            PurePosixPath(relative).relative_to("consumer").as_posix(): digest
            for relative, digest in consumer_entries.items()
        }
    )
    _publish_no_replace(root / "consumer" / "SHA256SUMS", consumer_checksum_content)
    consumer_checksum_sha256 = _sha256_bytes(consumer_checksum_content)

    evidence_document = _evidence_index_document(plan)
    evidence_content = _canonical_bytes(evidence_document)
    _publish_no_replace(root / "evidence-index.json", evidence_content)
    evidence_index_sha256 = _sha256_bytes(evidence_content)

    root_covered_files = tuple(
        sorted(
            {
                *plan.immutable_audit_files,
                *cast(tuple[str, ...], consumer["files"]),
                "consumer/SHA256SUMS",
                "evidence-index.json",
                "run-manifest.json",
            }
        )
    )
    run_manifest = _run_manifest_document(
        plan=plan,
        data_validated=data_validated,
        consumer_manifest_sha256=cast(str, consumer["manifest_sha256"]),
        consumer_checksum_sha256=consumer_checksum_sha256,
        evidence_index_sha256=evidence_index_sha256,
        covered_files=root_covered_files,
    )
    run_manifest_content = _canonical_bytes(run_manifest)
    _publish_no_replace(root / "run-manifest.json", run_manifest_content)
    run_manifest_sha256 = _sha256_bytes(run_manifest_content)

    expected_before_root = {"DATA_VALIDATED", *root_covered_files}
    _verify_exact_tree(root, expected_before_root)
    root_entries = {
        relative: _hash_regular_file(root / relative, relative)[0]
        for relative in root_covered_files
    }
    root_checksum_content = _checksum_bytes(root_entries)
    _publish_no_replace(root / "SHA256SUMS", root_checksum_content)
    root_checksum_sha256 = _sha256_bytes(root_checksum_content)

    durable_root_checksum = _read_regular_file(
        root / "SHA256SUMS", "root SHA256SUMS", MAX_JSON_BYTES
    )
    if durable_root_checksum != root_checksum_content:
        raise IntegrityError("root SHA256SUMS changed before COMPLETE")
    verified_root_entries = _parse_checksum_bytes(
        durable_root_checksum, "root SHA256SUMS"
    )
    if verified_root_entries != root_entries:
        raise IntegrityError("root SHA256SUMS entries changed before COMPLETE")
    _verify_exact_tree(root, {"DATA_VALIDATED", "SHA256SUMS", *root_covered_files})
    final_manifest_content, final_manifest = _read_canonical_json(
        root / "run-manifest.json", "final run manifest"
    )
    if _sha256_bytes(final_manifest_content) != run_manifest_sha256:
        raise IntegrityError("final run manifest changed before COMPLETE")
    _verify_run_manifest(final_manifest, verified_root_entries)
    final_consumer = _load_and_verify_consumer(
        root,
        checksum_expected=True,
        trusted_root_hashes=verified_root_entries,
    )
    if (
        final_consumer["manifest_sha256"] != consumer["manifest_sha256"]
        or _sha256_bytes(
            _read_regular_file(
                root / "consumer" / "SHA256SUMS",
                "consumer SHA256SUMS",
                MAX_JSON_BYTES,
            )
        )
        != consumer_checksum_sha256
    ):
        raise IntegrityError("consumer bundle changed before COMPLETE")
    final_data_validated = _verify_data_validated(
        root,
        cast(str, final_consumer["manifest_sha256"]),
        cast(dict[str, Any], final_consumer["manifest"]),
    )
    if final_data_validated["sha256"] != data_validated["sha256"]:
        raise IntegrityError("DATA_VALIDATED changed before COMPLETE")
    _verify_evidence_index(root, final_manifest, verified_root_entries)
    complete_document = {
        "schema_id": COMPLETE_SCHEMA_ID,
        "schema_version": RELEASE_SCHEMA_VERSION,
        "run_id": plan.run_id,
        "root_sha256sums_sha256": root_checksum_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "data_validated_sha256": cast(str, data_validated["sha256"]),
        "finalizer_identity": plan.finalizer_identity,
    }
    _publish_no_replace(root / "COMPLETE", _canonical_bytes(complete_document))
    return verify_complete(root)


def verify_complete(run_root: str | Path) -> CompleteVerification:
    """Read-only, fail-closed verification of ``DATA_VALIDATED`` + ``COMPLETE``."""

    root = _existing_safe_directory(run_root, "run root")
    marker_set = _current_marker_set(root)
    if marker_set not in (
        {"DATA_VALIDATED", "COMPLETE"},
        {"DATA_VALIDATED", "COMPLETE", "RELEASED"},
    ):
        raise IntegrityError("run does not have a valid COMPLETE lifecycle state")

    complete_content, complete = _read_canonical_json(
        root / "COMPLETE", "COMPLETE marker"
    )
    _require_exact_keys(
        complete,
        {
            "schema_id",
            "schema_version",
            "run_id",
            "root_sha256sums_sha256",
            "run_manifest_sha256",
            "data_validated_sha256",
            "finalizer_identity",
        },
        "COMPLETE marker",
    )
    if (
        complete["schema_id"] != COMPLETE_SCHEMA_ID
        or complete["schema_version"] != RELEASE_SCHEMA_VERSION
    ):
        raise IntegrityError("COMPLETE marker schema is invalid")
    run_id = _require_token(complete["run_id"], "COMPLETE run ID", integrity=True)
    finalizer = _require_token(
        complete["finalizer_identity"], "COMPLETE finalizer identity", integrity=True
    )
    root_checksum_sha256 = _require_fingerprint(
        complete["root_sha256sums_sha256"], "COMPLETE root checksum", integrity=True
    )
    run_manifest_sha256 = _require_fingerprint(
        complete["run_manifest_sha256"], "COMPLETE run manifest", integrity=True
    )
    data_validated_sha256 = _require_fingerprint(
        complete["data_validated_sha256"], "COMPLETE DATA_VALIDATED", integrity=True
    )

    root_checksum_content = _read_regular_file(
        root / "SHA256SUMS", "root SHA256SUMS", MAX_JSON_BYTES
    )
    if _sha256_bytes(root_checksum_content) != root_checksum_sha256:
        raise IntegrityError("root SHA256SUMS differs from COMPLETE")
    root_entries = _parse_checksum_bytes(root_checksum_content, "root SHA256SUMS")
    if set(root_entries) & (_LIFECYCLE_MARKERS | {"SHA256SUMS"}):
        raise IntegrityError("root SHA256SUMS contains an excluded lifecycle file")
    expected_tree = {
        *root_entries,
        "SHA256SUMS",
        "DATA_VALIDATED",
        "COMPLETE",
    }
    if "RELEASED" in marker_set:
        expected_tree.add("RELEASED")
    _verify_exact_tree(root, expected_tree)
    _verify_checksum_entries(root, root_entries, base=root)

    manifest_content, manifest = _read_canonical_json(
        root / "run-manifest.json", "final run manifest"
    )
    if _sha256_bytes(manifest_content) != run_manifest_sha256:
        raise IntegrityError("final run manifest differs from COMPLETE")
    _verify_run_manifest(manifest, root_entries)
    if manifest["run_id"] != run_id or manifest["finalizer_identity"] != finalizer:
        raise IntegrityError("COMPLETE identity differs from final run manifest")
    if manifest["data_validated_sha256"] != data_validated_sha256:
        raise IntegrityError("COMPLETE DATA_VALIDATED differs from final manifest")

    consumer = _load_and_verify_consumer(
        root, checksum_expected=True, trusted_root_hashes=root_entries
    )
    if manifest["consumer_manifest_sha256"] != consumer["manifest_sha256"]:
        raise IntegrityError("consumer manifest differs from final run manifest")
    consumer_checksum_content = _read_regular_file(
        root / "consumer" / "SHA256SUMS", "consumer SHA256SUMS", MAX_JSON_BYTES
    )
    if (
        _sha256_bytes(consumer_checksum_content)
        != manifest["consumer_sha256sums_sha256"]
    ):
        raise IntegrityError("consumer SHA256SUMS differs from final run manifest")
    _verify_data_validated(
        root,
        cast(str, consumer["manifest_sha256"]),
        cast(dict[str, Any], consumer["manifest"]),
    )
    data_content = _read_regular_file(
        root / "DATA_VALIDATED", "DATA_VALIDATED marker", MAX_JSON_BYTES
    )
    if _sha256_bytes(data_content) != data_validated_sha256:
        raise IntegrityError("DATA_VALIDATED differs from COMPLETE")
    _verify_evidence_index(root, manifest, root_entries)

    return CompleteVerification(
        run_id=run_id,
        root_sha256sums_sha256=root_checksum_sha256,
        complete_sha256=_sha256_bytes(complete_content),
        run_manifest_sha256=run_manifest_sha256,
        data_validated_sha256=data_validated_sha256,
        generation_commit=cast(str, manifest["generation_commit"]),
        release_commit=cast(str, manifest["release_commit"]),
        release_tag=cast(str, manifest["release_tag"]),
        finalizer_identity=finalizer,
        covered_file_count=len(root_entries),
    )


def build_external_release_attestation(
    run_root: str | Path, *, verifier_identity: str
) -> bytes:
    """Build canonical EV-015 bytes after independently verifying ``COMPLETE``.

    The caller stores these bytes outside the run root.  This function does
    not write the attestation or publish ``RELEASED``.
    """

    verifier = _require_token(verifier_identity, "verifier identity")
    complete = verify_complete(run_root)
    if verifier == complete.finalizer_identity:
        raise ReleaseError("G8 verifier must be independent of the finalizer")
    return _canonical_bytes(
        {
            "schema_id": EXTERNAL_ATTESTATION_SCHEMA_ID,
            "schema_version": RELEASE_SCHEMA_VERSION,
            "run_id": complete.run_id,
            "root_sha256sums_sha256": complete.root_sha256sums_sha256,
            "complete_sha256": complete.complete_sha256,
            "run_manifest_sha256": complete.run_manifest_sha256,
            "generation_commit": complete.generation_commit,
            "release_commit": complete.release_commit,
            "release_tag": complete.release_tag,
            "finalizer_identity": complete.finalizer_identity,
            "verifier_identity": verifier,
            "decision": "release",
        }
    )


def publish_released(
    run_root: str | Path, *, attestation_path: str | Path
) -> ReleasedVerification:
    """Verify external EV-015 and atomically publish checksum-excluded RELEASED."""

    root = _existing_safe_directory(run_root, "run root")
    _require_marker_set(root, {"DATA_VALIDATED", "COMPLETE"})
    complete = verify_complete(root)
    path = _external_attestation_path(root, attestation_path)
    content, attestation = _read_canonical_json(path, "external G8 attestation")
    verifier = _verify_attestation(attestation, complete)
    attestation_sha256 = _sha256_bytes(content)
    uri = path.as_uri()
    released = {
        "schema_id": RELEASED_SCHEMA_ID,
        "schema_version": RELEASE_SCHEMA_VERSION,
        "run_id": complete.run_id,
        "root_sha256sums_sha256": complete.root_sha256sums_sha256,
        "complete_sha256": complete.complete_sha256,
        "attestation_uri": uri,
        "attestation_sha256": attestation_sha256,
        "verifier_identity": verifier,
        "decision": "release",
    }
    _publish_no_replace(root / "RELEASED", _canonical_bytes(released))
    return verify_released(root, attestation_path=path)


def verify_released(
    run_root: str | Path, *, attestation_path: str | Path | None = None
) -> ReleasedVerification:
    """Read-only verification of a released root and its external attestation."""

    root = _existing_safe_directory(run_root, "run root")
    _require_marker_set(root, {"DATA_VALIDATED", "COMPLETE", "RELEASED"})
    complete = verify_complete(root)
    released_content, released = _read_canonical_json(
        root / "RELEASED", "RELEASED marker"
    )
    _require_exact_keys(
        released,
        {
            "schema_id",
            "schema_version",
            "run_id",
            "root_sha256sums_sha256",
            "complete_sha256",
            "attestation_uri",
            "attestation_sha256",
            "verifier_identity",
            "decision",
        },
        "RELEASED marker",
    )
    if (
        released["schema_id"] != RELEASED_SCHEMA_ID
        or released["schema_version"] != RELEASE_SCHEMA_VERSION
        or released["decision"] != "release"
        or released["run_id"] != complete.run_id
        or released["root_sha256sums_sha256"] != complete.root_sha256sums_sha256
        or released["complete_sha256"] != complete.complete_sha256
    ):
        raise IntegrityError("RELEASED marker differs from COMPLETE")
    attestation_sha256 = _require_fingerprint(
        released["attestation_sha256"], "RELEASED attestation", integrity=True
    )
    verifier = _require_token(
        released["verifier_identity"], "RELEASED verifier identity", integrity=True
    )
    uri = released["attestation_uri"]
    if not isinstance(uri, str):
        raise IntegrityError("RELEASED attestation URI is invalid")
    path = (
        _external_attestation_path(root, attestation_path)
        if attestation_path is not None
        else _path_from_file_uri(root, uri)
    )
    if path.as_uri() != uri:
        raise IntegrityError("RELEASED URI differs from the attestation path")
    content, attestation = _read_canonical_json(path, "external G8 attestation")
    if _sha256_bytes(content) != attestation_sha256:
        raise IntegrityError("external G8 attestation differs from RELEASED")
    if _verify_attestation(attestation, complete) != verifier:
        raise IntegrityError("RELEASED verifier differs from external attestation")
    return ReleasedVerification(
        complete=complete,
        attestation_uri=uri,
        attestation_sha256=attestation_sha256,
        verifier_identity=verifier,
        released_sha256=_sha256_bytes(released_content),
    )


def _load_and_verify_consumer(
    root: Path,
    *,
    checksum_expected: bool = False,
    trusted_root_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    content, manifest = _read_canonical_json(
        root / "consumer" / "consumer-manifest.json", "consumer manifest"
    )
    _verify_consumer_manifest_shape(manifest, CANONICAL_STRUCTURAL_PLAN)
    shards = cast(list[dict[str, Any]], manifest["shards"])
    manifest_sha256 = _sha256_bytes(content)
    if (
        trusted_root_hashes is not None
        and trusted_root_hashes.get("consumer/consumer-manifest.json")
        != manifest_sha256
    ):
        raise IntegrityError("consumer manifest differs from root SHA256SUMS")
    expected_files = {"consumer/consumer-manifest.json"}
    hashes = {"consumer/consumer-manifest.json": manifest_sha256}
    for record in shards:
        relative = cast(str, record["relative_path"])
        if relative in expected_files:
            raise IntegrityError("consumer manifest contains a duplicate shard path")
        expected_files.add(relative)
        if trusted_root_hashes is None:
            digest, byte_count = _hash_regular_file(root / relative, "consumer shard")
        else:
            trusted_digest = trusted_root_hashes.get(relative)
            if trusted_digest is None:
                raise IntegrityError("consumer shard is absent from root SHA256SUMS")
            digest = trusted_digest
            byte_count = _regular_file_size(root / relative, "consumer shard")
        if digest != record["sha256"] or byte_count != record["bytes"]:
            raise IntegrityError("consumer shard differs from its G7 manifest")
        hashes[relative] = digest
    expected_tree = set(expected_files)
    if checksum_expected:
        expected_tree.add("consumer/SHA256SUMS")
    _verify_exact_subtree(root, "consumer", expected_tree)
    if checksum_expected:
        checksum_content = _read_regular_file(
            root / "consumer" / "SHA256SUMS",
            "consumer SHA256SUMS",
            MAX_JSON_BYTES,
        )
        checksum_sha256 = _sha256_bytes(checksum_content)
        if (
            trusted_root_hashes is not None
            and trusted_root_hashes.get("consumer/SHA256SUMS") != checksum_sha256
        ):
            raise IntegrityError("consumer SHA256SUMS differs from root SHA256SUMS")
        checksum_entries = _parse_checksum_bytes(
            checksum_content, "consumer SHA256SUMS"
        )
        expected_entries = {
            PurePosixPath(relative).relative_to("consumer").as_posix(): digest
            for relative, digest in hashes.items()
        }
        if checksum_entries != expected_entries:
            raise IntegrityError("consumer SHA256SUMS entries differ from G7 manifest")
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "files": tuple(sorted(expected_files)),
        "hashes": hashes,
    }


def _verify_consumer_manifest_shape(
    manifest: Mapping[str, Any], plan: StructuralValidationPlan
) -> None:
    _require_exact_keys(
        manifest,
        {
            "schema_id",
            "schema_version",
            "calendar_id",
            "csv_schema_version",
            "csv_serializer_version",
            "plan_fingerprint",
            "storage_binding",
            "datasets",
            "work_unit_count",
            "csv_shard_count",
            "total_rows",
            "total_values",
            "shards",
        },
        "consumer manifest",
    )
    if (
        manifest["schema_id"] != CONSUMER_MANIFEST_SCHEMA_ID
        or manifest["schema_version"] != CONSUMER_MANIFEST_SCHEMA_VERSION
        or manifest["calendar_id"] != CALENDAR_ID
        or manifest["csv_schema_version"] != CSV_SCHEMA_VERSION
        or manifest["csv_serializer_version"] != CSV_SERIALIZER_VERSION
        or manifest["plan_fingerprint"] != plan.fingerprint
        or manifest["work_unit_count"] != plan.work_unit_count
        or manifest["csv_shard_count"] != plan.csv_shard_count
        or manifest["total_rows"] != plan.total_rows
        or manifest["total_values"] != plan.total_values
        or manifest["datasets"] != _expected_datasets(plan)
    ):
        raise IntegrityError("consumer manifest is not the canonical G7 geometry")
    binding = manifest["storage_binding"]
    if not isinstance(binding, dict) or set(binding) != {
        "simulation_fingerprint",
        "materialization_fingerprint",
        "toolchain_fingerprint",
        "run_fingerprint",
    }:
        raise IntegrityError("consumer manifest storage binding is invalid")
    for value in binding.values():
        _require_fingerprint(value, "consumer storage binding", integrity=True)
    shards = manifest["shards"]
    if (
        not isinstance(shards, list)
        or len(shards) != plan.csv_shard_count
        or any(not isinstance(item, dict) for item in shards)
    ):
        raise IntegrityError("consumer manifest shard list is invalid")
    row_total = 0
    for item in cast(list[dict[str, Any]], shards):
        if set(item) != _CONSUMER_SHARD_KEYS:
            raise IntegrityError("consumer shard manifest keys are invalid")
        relative = _checked_relative_path(
            item["relative_path"], "consumer shard path", integrity=True
        )
        if (
            len(relative.parts) < 5
            or relative.parts[:2] != ("consumer", "returns")
            or relative.suffix != ".csv"
        ):
            raise IntegrityError("consumer shard path is outside returns")
        _require_fingerprint(item["sha256"], "consumer shard SHA-256", integrity=True)
        _positive_int(item["bytes"], "consumer shard bytes")
        row_total += _positive_int(item["rows"], "consumer shard rows")
    if row_total != plan.total_rows:
        raise IntegrityError("consumer shard rows differ from canonical total")


def _expected_datasets(plan: StructuralValidationPlan) -> list[dict[str, JSONValue]]:
    result: list[dict[str, JSONValue]] = []
    for family, frequency, periods_per_year, paths, paths_per_shard in (
        ("LF", "monthly", 12, plan.lf_paths, plan.lf_paths_per_shard),
        ("LF", "quarterly", 4, plan.lf_paths, plan.lf_paths_per_shard),
        ("LF", "annual", 1, plan.lf_paths, plan.lf_paths_per_shard),
        ("HF", "daily", 252, plan.hf_paths, plan.hf_paths_per_shard),
        ("HF", "weekly", 52, plan.hf_paths, plan.hf_paths_per_shard),
    ):
        periods = plan.years * periods_per_year
        rows = paths * periods
        result.append(
            {
                "family": family,
                "frequency": frequency,
                "paths": paths,
                "periods_per_path": periods,
                "rows": rows,
                "values": rows * 3,
                "shards": (paths + paths_per_shard - 1) // paths_per_shard,
            }
        )
    return result


def _verify_data_validated(
    root: Path,
    consumer_manifest_sha256: str,
    consumer_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    content, marker = _read_canonical_json(
        root / "DATA_VALIDATED", "DATA_VALIDATED marker"
    )
    _require_exact_keys(
        marker,
        {
            "schema_id",
            "schema_version",
            "scientific_pass_fingerprint",
            "structural_report_fingerprint",
            "structural_report_sha256",
        },
        "DATA_VALIDATED marker",
    )
    if (
        marker["schema_id"] != DATA_VALIDATED_SCHEMA_ID
        or marker["schema_version"] != DATA_VALIDATED_SCHEMA_VERSION
    ):
        raise IntegrityError("DATA_VALIDATED marker schema is invalid")
    science = _require_fingerprint(
        marker["scientific_pass_fingerprint"],
        "DATA_VALIDATED scientific pass",
        integrity=True,
    )
    report_fingerprint = _require_fingerprint(
        marker["structural_report_fingerprint"],
        "DATA_VALIDATED structural report",
        integrity=True,
    )
    report_sha256 = _require_fingerprint(
        marker["structural_report_sha256"],
        "DATA_VALIDATED structural report SHA-256",
        integrity=True,
    )
    report_content, report = _read_canonical_json(
        root / "validation" / "structural-report.json", "structural report"
    )
    if _sha256_bytes(report_content) != report_sha256:
        raise IntegrityError("structural report differs from DATA_VALIDATED")
    _require_exact_keys(
        report,
        {"schema_version", "report_fingerprint", "identity"},
        "structural report",
    )
    identity = report["identity"]
    if not isinstance(identity, dict):
        raise IntegrityError("structural report identity is invalid")
    _verify_structural_report_identity(
        identity,
        consumer_manifest_sha256=consumer_manifest_sha256,
        consumer_manifest=consumer_manifest,
    )
    if (
        report["schema_version"] != STRUCTURAL_REPORT_SCHEMA_VERSION
        or report["report_fingerprint"] != report_fingerprint
        or _sha256_bytes(_canonical_bytes(identity)) != report_fingerprint
    ):
        raise IntegrityError("structural report is not the canonical G7 pass")
    if science in {report_fingerprint, report_sha256, consumer_manifest_sha256}:
        raise IntegrityError("scientific pass identity is not independent")
    return {
        **marker,
        "sha256": _sha256_bytes(content),
    }


def _verify_structural_report_identity(
    identity: Mapping[str, Any],
    *,
    consumer_manifest_sha256: str,
    consumer_manifest: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        identity,
        {
            "schema_id",
            "schema_version",
            "plan_fingerprint",
            "canonical_profile",
            "manifest_sha256",
            "total_rows",
            "total_values",
            "work_units",
            "csv_shards",
            "aggregation_values_checked",
            "roundtrip_values_checked",
            "max_abs_log_aggregation_error",
            "max_abs_simple_compounding_error",
            "max_abs_log_simple_roundtrip_error",
            "elapsed_seconds",
            "rows_per_second",
            "datasets",
            "shards",
            "hook_metrics",
        },
        "structural report identity",
    )
    plan = CANONICAL_STRUCTURAL_PLAN
    expected_aggregation_values = (
        plan.lf_paths * plan.years * 36 + plan.hf_paths * plan.years * 312
    )
    if (
        identity["schema_id"] != STRUCTURAL_REPORT_SCHEMA_ID
        or identity["schema_version"] != STRUCTURAL_REPORT_SCHEMA_VERSION
        or identity["plan_fingerprint"] != plan.fingerprint
        or identity["canonical_profile"] is not True
        or identity["manifest_sha256"] != consumer_manifest_sha256
        or identity["total_rows"] != plan.total_rows
        or identity["total_values"] != plan.total_values
        or identity["work_units"] != plan.work_unit_count
        or identity["csv_shards"] != plan.csv_shard_count
        or identity["aggregation_values_checked"] != expected_aggregation_values
        or identity["roundtrip_values_checked"] != plan.total_values
        or identity["datasets"] != _expected_datasets(plan)
    ):
        raise IntegrityError("structural report is not the canonical G7 geometry")
    for name, tolerance in (
        ("max_abs_log_aggregation_error", LOG_AGGREGATION_TOLERANCE),
        ("max_abs_simple_compounding_error", SIMPLE_RETURN_TOLERANCE),
        ("max_abs_log_simple_roundtrip_error", SIMPLE_RETURN_TOLERANCE),
    ):
        value = identity[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0.0 <= value <= tolerance
        ):
            raise IntegrityError("structural report tolerance result is invalid")
    elapsed = identity["elapsed_seconds"]
    throughput = identity["rows_per_second"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, int | float)
        or not math.isfinite(elapsed)
        or elapsed <= 0.0
        or isinstance(throughput, bool)
        or not isinstance(throughput, int | float)
        or not math.isfinite(throughput)
        or throughput <= 0.0
        or throughput != plan.total_rows / elapsed
    ):
        raise IntegrityError("structural report throughput result is invalid")
    expected_shards = [
        {
            "relative_path": record["relative_path"],
            "rows": record["rows"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        for record in cast(list[dict[str, Any]], consumer_manifest["shards"])
    ]
    if identity["shards"] != expected_shards or not isinstance(
        identity["hook_metrics"], dict
    ):
        raise IntegrityError("structural report shard or hook summary is invalid")


def _verify_evidence(root: Path, evidence: Sequence[ReleaseEvidence]) -> None:
    for item in evidence:
        digest, _ = _hash_regular_file(root / item.relative_path, "release evidence")
        if digest != item.sha256:
            raise IntegrityError("release evidence differs from finalization plan")


def _evidence_index_document(plan: ReleaseFinalizationPlan) -> dict[str, JSONValue]:
    return {
        "schema_id": EVIDENCE_INDEX_SCHEMA_ID,
        "schema_version": RELEASE_SCHEMA_VERSION,
        "run_id": plan.run_id,
        "entries": [item.as_dict() for item in plan.evidence],
    }


def _run_manifest_document(
    *,
    plan: ReleaseFinalizationPlan,
    data_validated: Mapping[str, Any],
    consumer_manifest_sha256: str,
    consumer_checksum_sha256: str,
    evidence_index_sha256: str,
    covered_files: tuple[str, ...],
) -> dict[str, JSONValue]:
    purposes = {item.purpose: item.sha256 for item in plan.evidence}
    return {
        "schema_id": RUN_MANIFEST_SCHEMA_ID,
        "schema_version": RELEASE_SCHEMA_VERSION,
        "run_id": plan.run_id,
        "lifecycle_state": "FINALIZATION_PENDING",
        "canonical_plan_fingerprint": CANONICAL_STRUCTURAL_PLAN.fingerprint,
        "finalization_plan_fingerprint": plan.fingerprint,
        "generation_commit": plan.generation_commit,
        "release_commit": plan.release_commit,
        "release_tag": plan.release_tag,
        "finalizer_identity": plan.finalizer_identity,
        "data_validated_sha256": cast(str, data_validated["sha256"]),
        "scientific_pass_fingerprint": cast(
            str, data_validated["scientific_pass_fingerprint"]
        ),
        "structural_report_fingerprint": cast(
            str, data_validated["structural_report_fingerprint"]
        ),
        "consumer_manifest_sha256": consumer_manifest_sha256,
        "consumer_sha256sums_sha256": consumer_checksum_sha256,
        "evidence_index_sha256": evidence_index_sha256,
        "clean_replay_fingerprint": purposes["clean_replay"],
        "release_authorization_fingerprint": purposes["release_authorization"],
        "covered_files": list(covered_files),
    }


def _verify_run_manifest(
    manifest: Mapping[str, Any], root_entries: Mapping[str, str]
) -> None:
    _require_exact_keys(
        manifest,
        {
            "schema_id",
            "schema_version",
            "run_id",
            "lifecycle_state",
            "canonical_plan_fingerprint",
            "finalization_plan_fingerprint",
            "generation_commit",
            "release_commit",
            "release_tag",
            "finalizer_identity",
            "data_validated_sha256",
            "scientific_pass_fingerprint",
            "structural_report_fingerprint",
            "consumer_manifest_sha256",
            "consumer_sha256sums_sha256",
            "evidence_index_sha256",
            "clean_replay_fingerprint",
            "release_authorization_fingerprint",
            "covered_files",
        },
        "final run manifest",
    )
    if (
        manifest["schema_id"] != RUN_MANIFEST_SCHEMA_ID
        or manifest["schema_version"] != RELEASE_SCHEMA_VERSION
        or manifest["lifecycle_state"] != "FINALIZATION_PENDING"
        or manifest["canonical_plan_fingerprint"]
        != CANONICAL_STRUCTURAL_PLAN.fingerprint
    ):
        raise IntegrityError("final run manifest schema or lifecycle is invalid")
    _require_token(manifest["run_id"], "final run ID", integrity=True)
    _require_token(manifest["release_tag"], "final release tag", integrity=True)
    _require_token(manifest["finalizer_identity"], "finalizer identity", integrity=True)
    for name in ("generation_commit", "release_commit"):
        value = manifest[name]
        if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
            raise IntegrityError(f"final {name} is invalid")
    if manifest["generation_commit"] == manifest["release_commit"]:
        raise IntegrityError("final generation and release commits are equal")
    for name in (
        "finalization_plan_fingerprint",
        "data_validated_sha256",
        "scientific_pass_fingerprint",
        "structural_report_fingerprint",
        "consumer_manifest_sha256",
        "consumer_sha256sums_sha256",
        "evidence_index_sha256",
        "clean_replay_fingerprint",
        "release_authorization_fingerprint",
    ):
        _require_fingerprint(manifest[name], name, integrity=True)
    covered = manifest["covered_files"]
    if (
        not isinstance(covered, list)
        or any(not isinstance(item, str) for item in covered)
        or covered != sorted(set(cast(list[str], covered)))
        or set(cast(list[str], covered)) != set(root_entries)
        or "run-manifest.json" not in covered
    ):
        raise IntegrityError("final run manifest covered-file allowlist is invalid")


def _verify_evidence_index(
    root: Path,
    manifest: Mapping[str, Any],
    root_entries: Mapping[str, str],
) -> None:
    content, index = _read_canonical_json(
        root / "evidence-index.json", "final evidence index"
    )
    if _sha256_bytes(content) != manifest["evidence_index_sha256"]:
        raise IntegrityError("evidence index differs from final run manifest")
    _require_exact_keys(
        index, {"schema_id", "schema_version", "run_id", "entries"}, "evidence index"
    )
    if (
        index["schema_id"] != EVIDENCE_INDEX_SCHEMA_ID
        or index["schema_version"] != RELEASE_SCHEMA_VERSION
        or index["run_id"] != manifest["run_id"]
    ):
        raise IntegrityError("evidence index schema or run identity is invalid")
    entries = index["entries"]
    if not isinstance(entries, list) or not entries:
        raise IntegrityError("evidence index entries are invalid")
    ids: list[str] = []
    paths: set[str] = set()
    purposes: dict[str, str] = {}
    covered_files = set(cast(list[str], manifest["covered_files"]))
    for value in entries:
        if not isinstance(value, dict):
            raise IntegrityError("evidence index entry is invalid")
        _require_exact_keys(
            value,
            {"evidence_id", "purpose", "relative_path", "sha256"},
            "evidence index entry",
        )
        evidence_id = value["evidence_id"]
        if (
            not isinstance(evidence_id, str)
            or _EVIDENCE_ID.fullmatch(evidence_id) is None
        ):
            raise IntegrityError("evidence index ID is invalid")
        purpose = value["purpose"]
        if purpose not in {"clean_replay", "release_authorization", "additional"}:
            raise IntegrityError("evidence index purpose is invalid")
        relative = _checked_relative_path(
            value["relative_path"], "evidence index path", integrity=True
        ).as_posix()
        digest = _require_fingerprint(
            value["sha256"], "evidence index SHA-256", integrity=True
        )
        if relative not in covered_files:
            raise IntegrityError("evidence index path is not root-covered")
        if root_entries.get(relative) != digest:
            raise IntegrityError("indexed evidence hash differs")
        ids.append(evidence_id)
        if relative in paths:
            raise IntegrityError("evidence index path is duplicated")
        paths.add(relative)
        if purpose != "additional":
            if purpose in purposes:
                raise IntegrityError("required evidence purpose is duplicated")
            purposes[purpose] = digest
    if ids != sorted(set(ids)) or set(purposes) != {
        "clean_replay",
        "release_authorization",
    }:
        raise IntegrityError("evidence index ordering or required purposes are invalid")
    if (
        purposes["clean_replay"] != manifest["clean_replay_fingerprint"]
        or purposes["release_authorization"]
        != manifest["release_authorization_fingerprint"]
    ):
        raise IntegrityError("required evidence differs from final run manifest")


def _verify_attestation(
    attestation: Mapping[str, Any], complete: CompleteVerification
) -> str:
    _require_exact_keys(
        attestation,
        {
            "schema_id",
            "schema_version",
            "run_id",
            "root_sha256sums_sha256",
            "complete_sha256",
            "run_manifest_sha256",
            "generation_commit",
            "release_commit",
            "release_tag",
            "finalizer_identity",
            "verifier_identity",
            "decision",
        },
        "external G8 attestation",
    )
    expected: Mapping[str, object] = {
        "schema_id": EXTERNAL_ATTESTATION_SCHEMA_ID,
        "schema_version": RELEASE_SCHEMA_VERSION,
        "run_id": complete.run_id,
        "root_sha256sums_sha256": complete.root_sha256sums_sha256,
        "complete_sha256": complete.complete_sha256,
        "run_manifest_sha256": complete.run_manifest_sha256,
        "generation_commit": complete.generation_commit,
        "release_commit": complete.release_commit,
        "release_tag": complete.release_tag,
        "finalizer_identity": complete.finalizer_identity,
        "decision": "release",
    }
    if any(attestation.get(name) != value for name, value in expected.items()):
        raise IntegrityError("external G8 attestation differs from COMPLETE")
    verifier = _require_token(
        attestation["verifier_identity"], "attestation verifier", integrity=True
    )
    if verifier == complete.finalizer_identity:
        raise IntegrityError("external G8 verifier is not independent")
    return verifier


def _checksum_bytes(entries: Mapping[str, str]) -> bytes:
    checked: dict[str, str] = {}
    for relative, digest in entries.items():
        path = _checked_relative_path(relative, "checksum path")
        checked[path.as_posix()] = _require_fingerprint(digest, "checksum digest")
    if len(checked) != len(entries):
        raise GenerationError("checksum paths are duplicated")
    return "".join(
        f"{checked[relative]}  {relative}\n" for relative in sorted(checked)
    ).encode("ascii")


def _parse_checksum_bytes(content: bytes, label: str) -> dict[str, str]:
    try:
        text = content.decode("ascii")
    except UnicodeDecodeError as error:
        raise IntegrityError(f"{label} is not ASCII") from error
    if not text or not text.endswith("\n"):
        raise IntegrityError(f"{label} is empty or unterminated")
    result: dict[str, str] = {}
    for line in text.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise IntegrityError(f"{label} line grammar is invalid")
        digest = line[:64]
        relative = line[66:]
        _require_fingerprint(digest, f"{label} digest", integrity=True)
        checked = _checked_relative_path(relative, f"{label} path", integrity=True)
        if checked.as_posix() in result:
            raise IntegrityError(f"{label} contains a duplicate path")
        result[checked.as_posix()] = digest
    if content != _checksum_bytes(result):
        raise IntegrityError(f"{label} is not in deterministic path order")
    return result


def _verify_checksum_entries(
    root: Path, entries: Mapping[str, str], *, base: Path
) -> None:
    del root  # documents that all paths are resolved below the explicit base
    for relative, expected in entries.items():
        actual, _ = _hash_regular_file(base / relative, "checksummed file")
        if actual != expected:
            raise IntegrityError("checksummed file hash differs")


def _verify_exact_tree(root: Path, expected_files: set[str]) -> None:
    actual_files, actual_directories = _scan_tree(root)
    expected_directories = _parent_directories(expected_files)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise IntegrityError("run tree differs from the finalization allowlist")


def _verify_exact_subtree(
    root: Path, relative_root: str, expected_files: set[str]
) -> None:
    subtree = root / relative_root
    files, directories = _scan_tree(subtree, relative_to=root)
    expected_directories = _parent_directories(expected_files)
    expected_directories = {
        value
        for value in expected_directories
        if value == relative_root or value.startswith(f"{relative_root}/")
    }
    if files != expected_files or directories != expected_directories:
        raise IntegrityError("consumer tree differs from its strict allowlist")


def _scan_tree(
    root: Path, *, relative_to: Path | None = None
) -> tuple[set[str], set[str]]:
    base = root if relative_to is None else relative_to
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError("tree root is missing or unsafe")
    files: set[str] = set()
    directories: set[str] = set()
    try:
        for directory, names, filenames in os.walk(
            root, topdown=True, followlinks=False
        ):
            current = Path(directory)
            if current != base:
                directories.add(current.relative_to(base).as_posix())
            for name in tuple(names):
                path = current / name
                metadata = path.lstat()
                if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
                    raise IntegrityError("run tree contains an unsafe directory")
            for name in filenames:
                path = current / name
                metadata = path.lstat()
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise IntegrityError("run tree contains an unsafe file")
                files.add(path.relative_to(base).as_posix())
    except (OSError, ValueError) as error:
        if isinstance(error, IntegrityError):  # pragma: no cover - type narrowing
            raise
        raise IntegrityError("run tree cannot be scanned") from error
    return files, directories


def _parent_directories(files: set[str]) -> set[str]:
    result: set[str] = set()
    for relative in files:
        current = PurePosixPath(relative).parent
        while current.as_posix() not in {".", ""}:
            result.add(current.as_posix())
            current = current.parent
    return result


def _current_marker_set(root: Path) -> set[str]:
    present: set[str] = set()
    for name in _LIFECYCLE_MARKERS:
        path = root / name
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise IntegrityError("run lifecycle marker is unsafe")
            present.add(name)
    return present


def _require_marker_set(root: Path, expected: set[str]) -> None:
    if _current_marker_set(root) != expected:
        raise ReleaseError("run lifecycle marker set is invalid for this transition")


def _hash_regular_file(path: Path, label: str) -> tuple[str, int]:
    descriptor, initial = _open_regular_file(path, label)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            while True:
                chunk = handle.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
            final = os.fstat(handle.fileno())
            current = path.stat(follow_symlinks=False)
            _require_unchanged(initial, final, current, label)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise
    return digest.hexdigest(), size


def _read_regular_file(path: Path, label: str, maximum: int) -> bytes:
    descriptor, initial = _open_regular_file(path, label)
    try:
        if initial.st_size > maximum:
            raise IntegrityError(f"{label} exceeds its size bound")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            content = handle.read(maximum + 1)
            if len(content) > maximum:
                raise IntegrityError(f"{label} exceeds its size bound")
            final = os.fstat(handle.fileno())
            current = path.stat(follow_symlinks=False)
            _require_unchanged(initial, final, current, label)
            return content
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise


def _open_regular_file(path: Path, label: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise IntegrityError(f"{label} cannot be opened") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise IntegrityError(f"{label} is not a regular single-link file")
    return descriptor, metadata


def _regular_file_size(path: Path, label: str) -> int:
    descriptor, metadata = _open_regular_file(path, label)
    os.close(descriptor)
    return metadata.st_size


def _require_unchanged(
    initial: os.stat_result,
    final: os.stat_result,
    current: os.stat_result,
    label: str,
) -> None:
    if _stat_identity(initial) != _stat_identity(final) or _stat_identity(
        initial
    ) != _stat_identity(current):
        raise IntegrityError(f"{label} changed during verification")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_nlink,
    )


def _read_canonical_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    content = _read_regular_file(path, label, MAX_JSON_BYTES)
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} is not a JSON object")
    try:
        canonical = _canonical_bytes(value)
    except (TypeError, ValueError) as error:
        raise IntegrityError(f"{label} contains invalid JSON values") from error
    if content != canonical:
        raise IntegrityError(f"{label} is not canonical JSON")
    return content, cast(dict[str, Any], value)


def _existing_safe_directory(value: str | Path, label: str) -> Path:
    if not isinstance(value, str | Path):
        raise GenerationError(f"{label} must be a path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise GenerationError(f"{label} must be absolute and traversal-free")
    try:
        if path.is_symlink() or not path.is_dir():
            raise IntegrityError(f"{label} is missing or unsafe")
        for component in (path, *path.parents):
            if component.is_symlink():
                raise IntegrityError(f"{label} contains a symbolic-link component")
    except OSError as error:
        raise IntegrityError(f"{label} cannot be inspected") from error
    return path


def _external_attestation_path(root: Path, value: str | Path) -> Path:
    if not isinstance(value, str | Path):
        raise GenerationError("attestation path must be a path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise GenerationError("attestation path must be absolute and traversal-free")
    try:
        if path.is_relative_to(root):
            raise ReleaseError("external G8 attestation must be outside the run root")
        for component in (path, *path.parents):
            if component.is_symlink():
                raise IntegrityError("attestation path contains a symbolic link")
    except OSError as error:
        raise IntegrityError("attestation path cannot be inspected") from error
    _open_and_close_regular(path, "external G8 attestation")
    return path


def _path_from_file_uri(root: Path, value: str) -> Path:
    parts = urlsplit(value)
    if parts.scheme != "file" or parts.netloc not in {"", "localhost"}:
        raise IntegrityError("RELEASED attestation URI is not a local file URI")
    return _external_attestation_path(root, Path(unquote(parts.path)))


def _open_and_close_regular(path: Path, label: str) -> None:
    descriptor, _ = _open_regular_file(path, label)
    os.close(descriptor)


def _publish_no_replace(path: Path, content: bytes) -> None:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise IntegrityError("publication directory is missing or unsafe")
    if path.exists() or path.is_symlink():
        raise ReleaseError("release publication destination already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            if handle.write(content) != len(content):
                raise IntegrityError("release publication write was incomplete")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReleaseError("release publication destination appeared") from error
        except OSError as error:
            raise IntegrityError(
                "release publication cannot be linked atomically"
            ) from error
        _fsync_directory(parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise IntegrityError(
            "release publication directory cannot be fsynced"
        ) from error


def _require_absent(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise ReleaseError(f"{label} already exists")


def _checked_relative_path(
    value: object, label: str, *, integrity: bool = False
) -> PurePosixPath:
    error_type = IntegrityError if integrity else GenerationError
    if not isinstance(value, str) or not value or "\\" in value:
        raise error_type(f"{label} is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise error_type(f"{label} is unsafe")
    if any(_PATH_COMPONENT.fullmatch(component) is None for component in path.parts):
        raise error_type(f"{label} has an invalid component")
    return path


def _require_fingerprint(value: object, label: str, *, integrity: bool = False) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        error_type = IntegrityError if integrity else GenerationError
        raise error_type(f"{label} is invalid")
    return value


def _require_token(value: object, label: str, *, integrity: bool = False) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        error_type = IntegrityError if integrity else GenerationError
        raise error_type(f"{label} is invalid")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IntegrityError(f"{label} must be a positive integer")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise IntegrityError(f"{label} keys are invalid")
