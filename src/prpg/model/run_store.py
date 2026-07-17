"""Crash-safe, resumable receipt store for long Phase-3 calibration runs.

Scientific artifacts are immutable and content addressed elsewhere.  This
store records operational progress before those artifacts exist.  Each task
receipt is canonical JSON, bound to the run identity, written atomically with
exclusive-create semantics, and safely reusable only when its complete bytes
match.  A COMPLETE marker can be created only after an explicit expected task
set has been verified.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from prpg.errors import IntegrityError, ModelError
from prpg.model.g3 import G3_CONTRACT_VERSION
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_BY_ID,
    G3_PLANNED_LOOK_REGISTRY,
    G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
    G3_TASK_BY_ID,
    G3_TASK_REGISTRY,
    G3_TASK_REGISTRY_FINGERPRINT,
    planned_look_spec,
    task_spec,
)

RUN_STORE_SCHEMA_VERSION = 3
RUN_MANIFEST_NAME = "run-manifest.json"
COMPLETE_MARKER_NAME = "COMPLETE.json"
LAUNCH_STATE_NAME = "launch-state.json"
LAUNCH_LOCK_NAME = ".launch.lock"
TASK_RESULT_DIRECTORY = "task-results"
PLANNED_LOOK_DIRECTORY = "planned-looks"
FAILED_PLANNED_LOOK_DISPOSITION_SCHEMA_ID = (
    "prpg-g3-failed-planned-look-dispositions-v1"
)

TaskResultStatus: TypeAlias = Literal["passed", "failed", "blocked"]
LookDecision: TypeAlias = Literal["continue", "pass", "fail"]
LaunchState: TypeAlias = Literal["planned", "running", "completed"]

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_COMPONENT = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


@dataclass(frozen=True, slots=True)
class CalibrationRunIdentity:
    """Scientific identity; worker count is deliberately operational only."""

    config_fingerprint: str
    processed_data_fingerprint: str
    calibration_input_fingerprint: str
    source_code_fingerprint: str
    dependency_lock_fingerprint: str
    contract_version: str
    task_registry_fingerprint: str = G3_TASK_REGISTRY_FINGERPRINT
    planned_look_registry_fingerprint: str = G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
    workers: int | None = None

    def as_dict(self) -> dict[str, str]:
        """Return only identity-bearing scientific fields."""

        return {
            "config_fingerprint": self.config_fingerprint,
            "processed_data_fingerprint": self.processed_data_fingerprint,
            "calibration_input_fingerprint": self.calibration_input_fingerprint,
            "source_code_fingerprint": self.source_code_fingerprint,
            "dependency_lock_fingerprint": self.dependency_lock_fingerprint,
            "contract_version": self.contract_version,
            "task_registry_fingerprint": self.task_registry_fingerprint,
            "planned_look_registry_fingerprint": (
                self.planned_look_registry_fingerprint
            ),
        }

    def execution_dict(self) -> dict[str, int]:
        """Return the non-scientific execution setting supplied at creation."""

        if self.workers is None:
            raise ModelError("calibration execution workers are not attached")
        return {"workers": self.workers}


@dataclass(frozen=True, slots=True)
class CalibrationRunReference:
    """Verified local operational run reference."""

    run_id: str
    path: Path
    identity: CalibrationRunIdentity
    reused: bool


@dataclass(frozen=True, slots=True)
class TaskReceipt:
    """Verified task output bound to one run/stage/task coordinate."""

    run_id: str
    stage: str
    task_id: str
    payload: Mapping[str, Any]
    fingerprint: str
    path: Path
    reused: bool


@dataclass(frozen=True, slots=True)
class CompletionMarker:
    """Verified terminal marker containing the exact expected receipt set."""

    run_id: str
    receipt_fingerprints: Mapping[str, str]
    fingerprint: str
    path: Path
    reused: bool


@dataclass(frozen=True, slots=True)
class G3TaskResult:
    """One immutable result in the exact 26-task dependency graph."""

    run_id: str
    task_id: str
    status: TaskResultStatus
    payload: Mapping[str, Any]
    dependency_fingerprints: Mapping[str, str]
    blocked_by: tuple[str, ...]
    fingerprint: str
    path: Path
    reused: bool


@dataclass(frozen=True, slots=True)
class PlannedLookReceipt:
    """One durable, hash-chained decision at a registered sequential look."""

    run_id: str
    family_id: str
    look_index: int
    sample_count: int
    decision: LookDecision
    observed: Mapping[str, Any]
    previous_fingerprint: str | None
    fingerprint: str
    path: Path
    reused: bool


@dataclass(frozen=True, slots=True)
class LaunchMarker:
    """Verified atomic lifecycle marker for exactly one G3 launch."""

    run_id: str
    state: LaunchState
    sequence: int
    preflight_fingerprint: str
    workers: int
    fingerprint: str
    path: Path
    reused: bool


@dataclass(slots=True)
class LaunchLease:
    """Process-held advisory lock required for every exact task write."""

    store: CalibrationRunStore
    run_id: str
    preflight_fingerprint: str
    workers: int
    _descriptor: int
    _closed: bool = False

    @property
    def active(self) -> bool:
        return not self._closed

    def close(self) -> None:
        if self._closed:
            return
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._closed = True

    def __enter__(self) -> LaunchLease:
        if self._closed:
            raise IntegrityError("calibration launch lease is already closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class CalibrationRunStore:
    """Atomic receipt store below ``<root>/calibration/<run-id>``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "calibration"

    def initialize(self, identity: CalibrationRunIdentity) -> CalibrationRunReference:
        """Create or verify a worker-count-independent scientific run."""

        checked = _validated_identity(identity)
        identity_dict = checked.as_dict()
        run_id = _sha256(_canonical_bytes(identity_dict))
        manifest = {
            "schema_version": RUN_STORE_SCHEMA_VERSION,
            "run_id": run_id,
            "identity": identity_dict,
        }
        _reject_symlink(self.root, "calibration run root")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_symlink(self.root, "calibration run root")
        run_path = self.root / run_id
        with suppress(FileExistsError):
            run_path.mkdir(mode=0o700)
        _reject_symlink(run_path, "calibration run directory")
        if not run_path.is_dir():
            raise IntegrityError("calibration run path is not a directory")
        manifest_path = run_path / RUN_MANIFEST_NAME
        reused = _write_or_verify_exact(manifest_path, _canonical_bytes(manifest))
        return CalibrationRunReference(run_id, run_path, checked, reused)

    def verify(self, run_id: str) -> CalibrationRunReference:
        """Verify a run manifest and its content-addressed directory name."""

        _require_fingerprint(run_id, "run ID")
        run_path = self.root / run_id
        _reject_symlink(run_path, "calibration run directory")
        if not run_path.is_dir():
            raise IntegrityError("calibration run directory is missing")
        manifest = _read_canonical_object(run_path / RUN_MANIFEST_NAME, "run manifest")
        if set(manifest) != {"schema_version", "run_id", "identity"}:
            raise IntegrityError("calibration run manifest keys are invalid")
        if manifest["schema_version"] != RUN_STORE_SCHEMA_VERSION:
            raise IntegrityError("calibration run schema version is unsupported")
        if manifest["run_id"] != run_id:
            raise IntegrityError("calibration run manifest ID is inconsistent")
        identity_value = manifest["identity"]
        if not isinstance(identity_value, dict):
            raise IntegrityError("calibration run identity is invalid")
        try:
            identity = CalibrationRunIdentity(**identity_value)
            checked = _validated_identity(identity)
        except (TypeError, ModelError) as error:
            raise IntegrityError("calibration run identity is invalid") from error
        if _sha256(_canonical_bytes(checked.as_dict())) != run_id:
            raise IntegrityError("calibration run identity hash is inconsistent")
        return CalibrationRunReference(run_id, run_path, checked, True)

    def plan_launch(
        self,
        run_id: str,
        *,
        preflight_fingerprint: str,
        workers: int,
    ) -> LaunchMarker:
        """Atomically create the only PLANNED marker for this scientific run."""

        reference = self.verify(run_id)
        _require_fingerprint(preflight_fingerprint, "preflight fingerprint")
        checked_workers = _positive_integer(workers, "launch workers")
        identity = _launch_identity(
            run_id=run_id,
            state="planned",
            sequence=0,
            preflight_fingerprint=preflight_fingerprint,
            workers=checked_workers,
        )
        fingerprint = _sha256(_canonical_bytes(identity))
        value = {**identity, "launch_fingerprint": fingerprint}
        path = reference.path / LAUNCH_STATE_NAME
        reused = _write_or_verify_exact(path, _canonical_bytes(value))
        return LaunchMarker(
            run_id,
            "planned",
            0,
            preflight_fingerprint,
            checked_workers,
            fingerprint,
            path,
            reused,
        )

    def read_launch(self, run_id: str) -> LaunchMarker:
        """Read and hash-verify the exact launch lifecycle marker."""

        reference = self.verify(run_id)
        return _read_launch_marker(reference.path / LAUNCH_STATE_NAME, run_id)

    def start_launch(
        self,
        run_id: str,
        *,
        preflight_fingerprint: str,
        workers: int,
        resume: bool = False,
    ) -> LaunchLease:
        """Acquire the exclusive lock and enter or resume RUNNING exactly once."""

        reference = self.verify(run_id)
        _require_fingerprint(preflight_fingerprint, "preflight fingerprint")
        checked_workers = _positive_integer(workers, "launch workers")
        descriptor = _acquire_launch_lock(reference.path, run_id)
        try:
            marker = _read_launch_marker(reference.path / LAUNCH_STATE_NAME, run_id)
            if (
                marker.preflight_fingerprint != preflight_fingerprint
                or marker.workers != checked_workers
            ):
                raise IntegrityError("launch identity differs from the planned launch")
            if marker.state == "completed":
                raise IntegrityError("completed calibration launch cannot be rerun")
            if marker.state == "planned" and resume:
                raise IntegrityError("resume requires a previously running launch")
            if marker.state == "running" and not resume:
                raise IntegrityError("calibration launch is already running")
            if marker.state == "running":
                self.verify_task_results(run_id, require_complete=False)
                self.verify_planned_look_receipts(run_id, require_terminal=False)
            running = _launch_identity(
                run_id=run_id,
                state="running",
                sequence=marker.sequence + 1,
                preflight_fingerprint=preflight_fingerprint,
                workers=checked_workers,
            )
            fingerprint = _sha256(_canonical_bytes(running))
            _atomic_replace(
                marker.path,
                _canonical_bytes({**running, "launch_fingerprint": fingerprint}),
            )
            return LaunchLease(
                self,
                run_id,
                preflight_fingerprint,
                checked_workers,
                descriptor,
            )
        except BaseException:
            _release_descriptor(descriptor)
            raise

    def complete_launch(self, lease: LaunchLease) -> LaunchMarker:
        """Verify all exact results/looks and atomically enter COMPLETED."""

        marker = self._assert_active_lease(lease)
        task_results = self.verify_task_results(lease.run_id, require_complete=True)
        self.verify_planned_look_receipts(
            lease.run_id,
            require_terminal=True,
            task_results=task_results,
        )
        identity = _launch_identity(
            run_id=lease.run_id,
            state="completed",
            sequence=marker.sequence + 1,
            preflight_fingerprint=lease.preflight_fingerprint,
            workers=lease.workers,
        )
        fingerprint = _sha256(_canonical_bytes(identity))
        _atomic_replace(
            marker.path,
            _canonical_bytes({**identity, "launch_fingerprint": fingerprint}),
        )
        return LaunchMarker(
            lease.run_id,
            "completed",
            marker.sequence + 1,
            lease.preflight_fingerprint,
            lease.workers,
            fingerprint,
            marker.path,
            False,
        )

    def write_task_result(
        self,
        lease: LaunchLease,
        *,
        task_id: str,
        status: TaskResultStatus,
        payload: Mapping[str, Any],
    ) -> G3TaskResult:
        """Atomically commit one graph-valid task result under the launch lock."""

        marker = self._assert_active_lease(lease)
        if marker.state != "running":  # pragma: no cover - helper enforces this
            raise IntegrityError("G3 task results require a running launch")
        spec = task_spec(task_id)
        if status not in {"passed", "failed", "blocked"}:
            raise ModelError("G3 task result status is invalid")
        normalized = _normalize_mapping(payload, "G3 task result payload")
        existing = self.verify_task_results(lease.run_id, require_complete=False)
        dependencies: dict[str, str] = {}
        blocked_by: list[str] = []
        for dependency in spec.dependencies:
            result = existing.get(dependency)
            if result is None:
                raise ModelError(
                    "G3 task dependencies are incomplete",
                    details={"task_id": task_id, "missing": dependency},
                )
            dependencies[dependency] = result.fingerprint
            if result.status != "passed":
                blocked_by.append(dependency)
        if blocked_by and status != "blocked":
            raise ModelError("a failed G3 dependency must propagate as blocked")
        if not blocked_by and status == "blocked":
            raise ModelError("a G3 task cannot be blocked when dependencies passed")
        owned_looks = tuple(
            look for look in G3_PLANNED_LOOK_REGISTRY if look.owner_task_id == task_id
        )
        if owned_looks:
            look_receipts = self.verify_planned_look_receipts(
                lease.run_id, require_terminal=False
            )
            if status == "blocked":
                if any(look.family_id in look_receipts for look in owned_looks):
                    raise ModelError("a blocked G3 task cannot own planned-look data")
                if "planned_look_dispositions" in normalized:
                    raise ModelError(
                        "a blocked G3 task cannot claim planned-look dispositions"
                    )
            elif status == "passed":
                if "planned_look_dispositions" in normalized:
                    raise ModelError(
                        "a passed G3 task cannot claim failure look dispositions"
                    )
                if any(
                    not look_receipts.get(look.family_id)
                    or look_receipts[look.family_id][-1].decision == "continue"
                    for look in owned_looks
                ):
                    raise ModelError(
                        "G3 task result requires every owned planned look to terminate"
                    )
            else:
                missing_terminal = any(
                    not look_receipts.get(look.family_id)
                    or look_receipts[look.family_id][-1].decision == "continue"
                    for look in owned_looks
                )
                if "planned_look_dispositions" in normalized or missing_terminal:
                    _validate_failed_planned_look_dispositions(
                        task_id=task_id,
                        payload=normalized,
                        receipts=look_receipts,
                    )
        identity = _task_result_identity(
            run_id=lease.run_id,
            task_id=task_id,
            status=status,
            payload=normalized,
            dependency_fingerprints=dependencies,
            blocked_by=tuple(blocked_by),
        )
        fingerprint = _sha256(_canonical_bytes(identity))
        value = {**identity, "result_fingerprint": fingerprint}
        directory = self.verify(lease.run_id).path / TASK_RESULT_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_symlink(directory, "G3 task-result directory")
        path = directory / f"{task_id}.json"
        reused = _write_or_verify_exact(path, _canonical_bytes(value))
        return G3TaskResult(
            lease.run_id,
            task_id,
            status,
            normalized,
            dependencies,
            tuple(blocked_by),
            fingerprint,
            path,
            reused,
        )

    def read_task_result(self, run_id: str, task_id: str) -> G3TaskResult:
        """Read one result only after verifying the complete existing DAG."""

        task_spec(task_id)
        results = self.verify_task_results(run_id, require_complete=False)
        try:
            return results[task_id]
        except KeyError as error:
            raise IntegrityError("G3 task result is missing") from error

    def verify_task_results(
        self,
        run_id: str,
        *,
        require_complete: bool,
    ) -> dict[str, G3TaskResult]:
        """Verify hashes, allowlist, dependencies, and failure propagation."""

        reference = self.verify(run_id)
        directory = reference.path / TASK_RESULT_DIRECTORY
        if not directory.exists():
            if require_complete:
                raise IntegrityError("the exact 26 G3 task results are incomplete")
            return {}
        _verify_flat_json_directory(
            directory,
            expected_stems=frozenset(G3_GATE_ORDER),
            label="G3 task-result directory",
        )
        results: dict[str, G3TaskResult] = {}
        for spec in G3_TASK_REGISTRY:
            path = directory / f"{spec.task_id}.json"
            if not path.exists():
                continue
            value = _read_canonical_object(path, "G3 task result")
            result = _validated_task_result(
                value,
                path=path,
                run_id=run_id,
                expected_task_id=spec.task_id,
                prior=results,
            )
            results[spec.task_id] = result
        if require_complete and tuple(results) != G3_GATE_ORDER:
            missing = [task_id for task_id in G3_GATE_ORDER if task_id not in results]
            raise IntegrityError(
                "the exact 26 G3 task results are incomplete",
                details={"missing": missing},
            )
        return results

    def write_planned_look_receipt(
        self,
        lease: LaunchLease,
        *,
        family_id: str,
        look_index: int,
        sample_count: int,
        decision: LookDecision,
        observed: Mapping[str, Any],
    ) -> PlannedLookReceipt:
        """Commit the next exact look in a durable hash chain."""

        self._assert_active_lease(lease)
        spec = planned_look_spec(family_id)
        if isinstance(look_index, bool) or not isinstance(look_index, int):
            raise ModelError("planned-look index must be an integer")
        if look_index < 1 or look_index > len(spec.sample_counts):
            raise ModelError("planned-look index is outside the closed schedule")
        expected_count = spec.sample_counts[look_index - 1]
        if sample_count != expected_count:
            raise ModelError("planned-look sample count differs from the registry")
        if decision not in {"continue", "pass", "fail"}:
            raise ModelError("planned-look decision is invalid")
        if look_index == len(spec.sample_counts) and decision == "continue":
            raise ModelError("the final planned look must fail closed or pass")
        normalized = _normalize_mapping(observed, "planned-look observations")
        task_results = self.verify_task_results(lease.run_id, require_complete=False)
        if spec.owner_task_id in task_results:
            raise ModelError("no planned look is allowed after its owner task result")
        owner = G3_TASK_BY_ID[spec.owner_task_id]
        unavailable = [
            dependency
            for dependency in owner.dependencies
            if dependency not in task_results
            or task_results[dependency].status != "passed"
        ]
        if unavailable:
            raise ModelError(
                "planned-look owner dependencies have not passed",
                details={"family_id": family_id, "unavailable": unavailable},
            )
        existing = self.verify_planned_look_receipts(
            lease.run_id, require_terminal=False
        )
        family = existing.get(family_id, ())
        if len(family) == look_index:
            previous = family[-2] if look_index > 1 else None
            previous_fingerprint = (
                previous.fingerprint if previous is not None else None
            )
            identity = _look_identity(
                run_id=lease.run_id,
                family_id=family_id,
                owner_task_id=spec.owner_task_id,
                look_index=look_index,
                sample_count=sample_count,
                decision=decision,
                observed=normalized,
                previous_fingerprint=previous_fingerprint,
            )
            if _sha256(_canonical_bytes(identity)) != family[-1].fingerprint:
                raise IntegrityError(
                    "existing planned-look receipt differs from requested bytes"
                )
            prior = family[-1]
            return PlannedLookReceipt(
                prior.run_id,
                prior.family_id,
                prior.look_index,
                prior.sample_count,
                prior.decision,
                prior.observed,
                prior.previous_fingerprint,
                prior.fingerprint,
                prior.path,
                True,
            )
        if len(family) != look_index - 1:
            raise ModelError("planned looks must be written sequentially without gaps")
        previous = family[-1] if family else None
        if previous is not None and previous.decision != "continue":
            raise ModelError("no look is allowed after a terminal decision")
        previous_fingerprint = previous.fingerprint if previous is not None else None
        identity = _look_identity(
            run_id=lease.run_id,
            family_id=family_id,
            owner_task_id=spec.owner_task_id,
            look_index=look_index,
            sample_count=sample_count,
            decision=decision,
            observed=normalized,
            previous_fingerprint=previous_fingerprint,
        )
        fingerprint = _sha256(_canonical_bytes(identity))
        value = {**identity, "look_fingerprint": fingerprint}
        directory = self.verify(lease.run_id).path / PLANNED_LOOK_DIRECTORY / family_id
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_symlink(directory, "planned-look family directory")
        path = directory / f"{look_index:02d}.json"
        reused = _write_or_verify_exact(path, _canonical_bytes(value))
        return PlannedLookReceipt(
            lease.run_id,
            family_id,
            look_index,
            sample_count,
            decision,
            normalized,
            previous_fingerprint,
            fingerprint,
            path,
            reused,
        )

    def verify_planned_look_receipts(
        self,
        run_id: str,
        *,
        require_terminal: bool,
        task_results: Mapping[str, G3TaskResult] | None = None,
    ) -> dict[str, tuple[PlannedLookReceipt, ...]]:
        """Verify every present look chain and optionally require final decisions."""

        reference = self.verify(run_id)
        root = reference.path / PLANNED_LOOK_DIRECTORY
        if not root.exists():
            if require_terminal:
                results = task_results or self.verify_task_results(
                    run_id, require_complete=True
                )
                required = [
                    item.family_id
                    for item in G3_PLANNED_LOOK_REGISTRY
                    if results[item.owner_task_id].status != "blocked"
                ]
                if required:
                    raise IntegrityError("required planned-look receipts are missing")
            return {}
        _verify_look_root(root)
        all_receipts: dict[str, tuple[PlannedLookReceipt, ...]] = {}
        for spec in G3_PLANNED_LOOK_REGISTRY:
            directory = root / spec.family_id
            if not directory.exists():
                continue
            expected_stems = frozenset(
                f"{index:02d}" for index in range(1, len(spec.sample_counts) + 1)
            )
            _verify_flat_json_directory(
                directory,
                expected_stems=expected_stems,
                label="planned-look family directory",
            )
            receipts: list[PlannedLookReceipt] = []
            for index, sample_count in enumerate(spec.sample_counts, start=1):
                path = directory / f"{index:02d}.json"
                if not path.exists():
                    break
                receipt = _validated_look_receipt(
                    _read_canonical_object(path, "planned-look receipt"),
                    path=path,
                    run_id=run_id,
                    spec=spec,
                    expected_index=index,
                    expected_count=sample_count,
                    previous=receipts[-1] if receipts else None,
                )
                receipts.append(receipt)
            actual_stems = tuple(sorted(entry.stem for entry in directory.iterdir()))
            expected_present = tuple(
                f"{index:02d}" for index in range(1, len(receipts) + 1)
            )
            if actual_stems != expected_present:
                raise IntegrityError("planned-look receipts contain a sequence gap")
            if receipts:
                all_receipts[spec.family_id] = tuple(receipts)
        if require_terminal:
            results = task_results or self.verify_task_results(
                run_id, require_complete=True
            )
            for spec in G3_PLANNED_LOOK_REGISTRY:
                owner_status = results[spec.owner_task_id].status
                family_receipts = all_receipts.get(spec.family_id, ())
                if owner_status == "blocked":
                    if family_receipts:
                        raise IntegrityError(
                            "blocked task cannot own planned-look receipts"
                        )
                    continue
                if owner_status == "failed" and (
                    "planned_look_dispositions" in results[spec.owner_task_id].payload
                ):
                    # Validate the complete per-owner document once.  Repeating
                    # this pure check for the second family is harmless and
                    # keeps this loop's family-local structure straightforward.
                    try:
                        _validate_failed_planned_look_dispositions(
                            task_id=spec.owner_task_id,
                            payload=results[spec.owner_task_id].payload,
                            receipts=all_receipts,
                        )
                    except ModelError as error:
                        raise IntegrityError(
                            "failed task planned-look dispositions are invalid"
                        ) from error
                    continue
                if not family_receipts or family_receipts[-1].decision == "continue":
                    raise IntegrityError(
                        "planned-look family lacks a durable terminal decision",
                        details={"family_id": spec.family_id},
                    )
        return all_receipts

    def _assert_active_lease(self, lease: LaunchLease) -> LaunchMarker:
        if (
            not isinstance(lease, LaunchLease)
            or lease.store is not self
            or not lease.active
        ):
            raise IntegrityError("an active launch lease from this store is required")
        marker = self.read_launch(lease.run_id)
        if (
            marker.state != "running"
            or marker.preflight_fingerprint != lease.preflight_fingerprint
            or marker.workers != lease.workers
        ):
            raise IntegrityError("launch lease does not match the running marker")
        return marker

    def write_receipt(
        self,
        run_id: str,
        *,
        stage: str,
        task_id: str,
        payload: Mapping[str, Any],
    ) -> TaskReceipt:
        """Atomically create or byte-verify one task receipt."""

        reference = self.verify(run_id)
        checked_stage = _component(stage, "stage")
        checked_task = _component(task_id, "task ID")
        normalized = _normalize_mapping(payload, "receipt payload")
        identity = {
            "schema_version": RUN_STORE_SCHEMA_VERSION,
            "run_id": run_id,
            "stage": checked_stage,
            "task_id": checked_task,
            "payload": normalized,
        }
        fingerprint = _sha256(_canonical_bytes(identity))
        receipt = {**identity, "receipt_fingerprint": fingerprint}
        stage_path = reference.path / "receipts" / checked_stage
        stage_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_symlink(stage_path, "calibration receipt stage")
        path = stage_path / f"{checked_task}.json"
        reused = _write_or_verify_exact(path, _canonical_bytes(receipt))
        return TaskReceipt(
            run_id,
            checked_stage,
            checked_task,
            normalized,
            fingerprint,
            path,
            reused,
        )

    def read_receipt(self, run_id: str, *, stage: str, task_id: str) -> TaskReceipt:
        """Read and verify one existing receipt."""

        reference = self.verify(run_id)
        checked_stage = _component(stage, "stage")
        checked_task = _component(task_id, "task ID")
        path = reference.path / "receipts" / checked_stage / f"{checked_task}.json"
        value = _read_canonical_object(path, "calibration task receipt")
        required = {
            "schema_version",
            "run_id",
            "stage",
            "task_id",
            "payload",
            "receipt_fingerprint",
        }
        if set(value) != required:
            raise IntegrityError("calibration task receipt keys are invalid")
        if (
            value["schema_version"] != RUN_STORE_SCHEMA_VERSION
            or value["run_id"] != run_id
            or value["stage"] != checked_stage
            or value["task_id"] != checked_task
        ):
            raise IntegrityError("calibration task receipt identity is inconsistent")
        payload = value["payload"]
        if not isinstance(payload, dict):
            raise IntegrityError("calibration task receipt payload is invalid")
        try:
            normalized = _normalize_mapping(payload, "receipt payload")
        except ModelError as error:
            raise IntegrityError(
                "calibration task receipt payload is invalid"
            ) from error
        identity = {
            "schema_version": RUN_STORE_SCHEMA_VERSION,
            "run_id": run_id,
            "stage": checked_stage,
            "task_id": checked_task,
            "payload": normalized,
        }
        fingerprint = _sha256(_canonical_bytes(identity))
        if value["receipt_fingerprint"] != fingerprint:
            raise IntegrityError("calibration task receipt hash is inconsistent")
        return TaskReceipt(
            run_id,
            checked_stage,
            checked_task,
            normalized,
            fingerprint,
            path,
            True,
        )

    def complete(
        self,
        run_id: str,
        *,
        expected_tasks: Mapping[str, Sequence[str]],
    ) -> CompletionMarker:
        """Create COMPLETE only after verifying every explicitly expected task."""

        reference = self.verify(run_id)
        if not expected_tasks:
            raise ModelError("completion requires a non-empty expected task mapping")
        fingerprints: dict[str, str] = {}
        for stage in sorted(expected_tasks):
            checked_stage = _component(stage, "stage")
            tasks = tuple(expected_tasks[stage])
            if not tasks or len(set(tasks)) != len(tasks):
                raise ModelError("expected task IDs must be non-empty and unique")
            for task_id in sorted(tasks):
                checked_task = _component(task_id, "task ID")
                receipt = self.read_receipt(
                    run_id, stage=checked_stage, task_id=checked_task
                )
                fingerprints[f"{checked_stage}/{checked_task}"] = receipt.fingerprint
        identity = {
            "schema_version": RUN_STORE_SCHEMA_VERSION,
            "run_id": run_id,
            "receipt_fingerprints": fingerprints,
        }
        fingerprint = _sha256(_canonical_bytes(identity))
        marker = {**identity, "completion_fingerprint": fingerprint}
        path = reference.path / COMPLETE_MARKER_NAME
        reused = _write_or_verify_exact(path, _canonical_bytes(marker))
        return CompletionMarker(run_id, fingerprints, fingerprint, path, reused)


def default_run_identity(
    *,
    config_fingerprint: str,
    processed_data_fingerprint: str,
    calibration_input_fingerprint: str,
    source_code_fingerprint: str,
    dependency_lock_fingerprint: str,
    workers: int,
) -> CalibrationRunIdentity:
    """Create the approved identity while keeping workers out of its hash."""

    return CalibrationRunIdentity(
        config_fingerprint=config_fingerprint,
        processed_data_fingerprint=processed_data_fingerprint,
        calibration_input_fingerprint=calibration_input_fingerprint,
        source_code_fingerprint=source_code_fingerprint,
        dependency_lock_fingerprint=dependency_lock_fingerprint,
        contract_version=G3_CONTRACT_VERSION,
        workers=workers,
    )


def _validated_identity(identity: CalibrationRunIdentity) -> CalibrationRunIdentity:
    if not isinstance(identity, CalibrationRunIdentity):
        raise ModelError("calibration run identity has the wrong type")
    _require_fingerprint(identity.config_fingerprint, "config fingerprint")
    _require_fingerprint(
        identity.processed_data_fingerprint, "processed data fingerprint"
    )
    _require_fingerprint(
        identity.calibration_input_fingerprint, "calibration input fingerprint"
    )
    _require_fingerprint(identity.source_code_fingerprint, "source code fingerprint")
    _require_fingerprint(
        identity.dependency_lock_fingerprint, "dependency lock fingerprint"
    )
    if identity.contract_version != G3_CONTRACT_VERSION:
        raise ModelError("calibration run contract version is unsupported")
    if identity.task_registry_fingerprint != G3_TASK_REGISTRY_FINGERPRINT:
        raise ModelError("calibration run task registry is unsupported")
    if (
        identity.planned_look_registry_fingerprint
        != G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
    ):
        raise ModelError("calibration run planned-look registry is unsupported")
    if identity.workers is not None:
        _positive_integer(identity.workers, "calibration run workers")
    return identity


def _launch_identity(
    *,
    run_id: str,
    state: LaunchState,
    sequence: int,
    preflight_fingerprint: str,
    workers: int,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_STORE_SCHEMA_VERSION,
        "run_id": run_id,
        "state": state,
        "sequence": sequence,
        "preflight_fingerprint": preflight_fingerprint,
        "workers": workers,
        "task_registry_fingerprint": G3_TASK_REGISTRY_FINGERPRINT,
        "planned_look_registry_fingerprint": (G3_PLANNED_LOOK_REGISTRY_FINGERPRINT),
    }


def _read_launch_marker(path: Path, run_id: str) -> LaunchMarker:
    value = _read_canonical_object(path, "calibration launch marker")
    required = {
        "schema_version",
        "run_id",
        "state",
        "sequence",
        "preflight_fingerprint",
        "workers",
        "task_registry_fingerprint",
        "planned_look_registry_fingerprint",
        "launch_fingerprint",
    }
    if set(value) != required:
        raise IntegrityError("calibration launch marker keys are invalid")
    state = value["state"]
    if state not in {"planned", "running", "completed"}:
        raise IntegrityError("calibration launch state is invalid")
    sequence = value["sequence"]
    workers = value["workers"]
    preflight = value["preflight_fingerprint"]
    try:
        checked_sequence = _nonnegative_integer(sequence, "launch sequence")
        checked_workers = _positive_integer(workers, "launch workers")
        _require_fingerprint(preflight, "preflight fingerprint")
    except ModelError as error:
        raise IntegrityError("calibration launch marker fields are invalid") from error
    if (
        value["schema_version"] != RUN_STORE_SCHEMA_VERSION
        or value["run_id"] != run_id
        or value["task_registry_fingerprint"] != G3_TASK_REGISTRY_FINGERPRINT
        or value["planned_look_registry_fingerprint"]
        != G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
    ):
        raise IntegrityError("calibration launch marker identity is inconsistent")
    identity = _launch_identity(
        run_id=run_id,
        state=state,
        sequence=checked_sequence,
        preflight_fingerprint=preflight,
        workers=checked_workers,
    )
    fingerprint = _sha256(_canonical_bytes(identity))
    if value["launch_fingerprint"] != fingerprint:
        raise IntegrityError("calibration launch marker hash is inconsistent")
    return LaunchMarker(
        run_id,
        state,
        checked_sequence,
        preflight,
        checked_workers,
        fingerprint,
        path,
        True,
    )


def _task_result_identity(
    *,
    run_id: str,
    task_id: str,
    status: TaskResultStatus,
    payload: Mapping[str, Any],
    dependency_fingerprints: Mapping[str, str],
    blocked_by: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": RUN_STORE_SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": task_id,
        "status": status,
        "payload": dict(payload),
        "dependency_fingerprints": dict(dependency_fingerprints),
        "blocked_by": list(blocked_by),
        "task_registry_fingerprint": G3_TASK_REGISTRY_FINGERPRINT,
    }


def _validated_task_result(
    value: dict[str, Any],
    *,
    path: Path,
    run_id: str,
    expected_task_id: str,
    prior: Mapping[str, G3TaskResult],
) -> G3TaskResult:
    required = {
        "schema_version",
        "run_id",
        "task_id",
        "status",
        "payload",
        "dependency_fingerprints",
        "blocked_by",
        "task_registry_fingerprint",
        "result_fingerprint",
    }
    if set(value) != required:
        raise IntegrityError("G3 task-result keys are invalid")
    task_id = value["task_id"]
    try:
        spec = task_spec(task_id)
    except ModelError as error:
        raise IntegrityError("G3 task result is outside the closed registry") from error
    status = value["status"]
    if status not in {"passed", "failed", "blocked"}:
        raise IntegrityError("G3 task-result status is invalid")
    payload = value["payload"]
    dependencies = value["dependency_fingerprints"]
    blocked_by = value["blocked_by"]
    if (
        value["schema_version"] != RUN_STORE_SCHEMA_VERSION
        or value["run_id"] != run_id
        or task_id != expected_task_id
        or value["task_registry_fingerprint"] != G3_TASK_REGISTRY_FINGERPRINT
        or not isinstance(payload, dict)
        or not isinstance(dependencies, dict)
        or not isinstance(blocked_by, list)
    ):
        raise IntegrityError("G3 task-result identity is inconsistent")
    try:
        normalized = _normalize_mapping(payload, "G3 task result payload")
    except ModelError as error:
        raise IntegrityError("G3 task-result payload is invalid") from error
    expected_dependencies: dict[str, str] = {}
    expected_blocked: list[str] = []
    for dependency in spec.dependencies:
        result = prior.get(dependency)
        if result is None:
            raise IntegrityError("G3 task result precedes a required dependency")
        expected_dependencies[dependency] = result.fingerprint
        if result.status != "passed":
            expected_blocked.append(dependency)
    if dependencies != expected_dependencies or blocked_by != expected_blocked:
        raise IntegrityError("G3 task-result dependency evidence is inconsistent")
    if expected_blocked and status != "blocked":
        raise IntegrityError("G3 dependency failure did not propagate")
    if not expected_blocked and status == "blocked":
        raise IntegrityError("G3 task is blocked without a failed dependency")
    identity = _task_result_identity(
        run_id=run_id,
        task_id=task_id,
        status=status,
        payload=normalized,
        dependency_fingerprints=expected_dependencies,
        blocked_by=tuple(expected_blocked),
    )
    fingerprint = _sha256(_canonical_bytes(identity))
    if value["result_fingerprint"] != fingerprint:
        raise IntegrityError("G3 task-result hash is inconsistent")
    return G3TaskResult(
        run_id,
        task_id,
        status,
        normalized,
        expected_dependencies,
        tuple(expected_blocked),
        fingerprint,
        path,
        True,
    )


def _look_identity(
    *,
    run_id: str,
    family_id: str,
    owner_task_id: str,
    look_index: int,
    sample_count: int,
    decision: LookDecision,
    observed: Mapping[str, Any],
    previous_fingerprint: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_STORE_SCHEMA_VERSION,
        "run_id": run_id,
        "family_id": family_id,
        "owner_task_id": owner_task_id,
        "look_index": look_index,
        "sample_count": sample_count,
        "decision": decision,
        "observed": dict(observed),
        "previous_fingerprint": previous_fingerprint,
        "planned_look_registry_fingerprint": (G3_PLANNED_LOOK_REGISTRY_FINGERPRINT),
    }


def _validated_look_receipt(
    value: dict[str, Any],
    *,
    path: Path,
    run_id: str,
    spec: Any,
    expected_index: int,
    expected_count: int,
    previous: PlannedLookReceipt | None,
) -> PlannedLookReceipt:
    required = {
        "schema_version",
        "run_id",
        "family_id",
        "owner_task_id",
        "look_index",
        "sample_count",
        "decision",
        "observed",
        "previous_fingerprint",
        "planned_look_registry_fingerprint",
        "look_fingerprint",
    }
    if set(value) != required:
        raise IntegrityError("planned-look receipt keys are invalid")
    decision = value["decision"]
    observed = value["observed"]
    prior_fingerprint = previous.fingerprint if previous is not None else None
    if (
        value["schema_version"] != RUN_STORE_SCHEMA_VERSION
        or value["run_id"] != run_id
        or value["family_id"] != spec.family_id
        or value["owner_task_id"] != spec.owner_task_id
        or value["look_index"] != expected_index
        or value["sample_count"] != expected_count
        or decision not in {"continue", "pass", "fail"}
        or not isinstance(observed, dict)
        or value["previous_fingerprint"] != prior_fingerprint
        or value["planned_look_registry_fingerprint"]
        != G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
    ):
        raise IntegrityError("planned-look receipt identity is inconsistent")
    if previous is not None and previous.decision != "continue":
        raise IntegrityError("planned-look receipt follows a terminal decision")
    if expected_index == len(spec.sample_counts) and decision == "continue":
        raise IntegrityError("final planned-look receipt did not fail closed")
    try:
        normalized = _normalize_mapping(observed, "planned-look observations")
    except ModelError as error:
        raise IntegrityError("planned-look observations are invalid") from error
    identity = _look_identity(
        run_id=run_id,
        family_id=spec.family_id,
        owner_task_id=spec.owner_task_id,
        look_index=expected_index,
        sample_count=expected_count,
        decision=decision,
        observed=normalized,
        previous_fingerprint=prior_fingerprint,
    )
    fingerprint = _sha256(_canonical_bytes(identity))
    if value["look_fingerprint"] != fingerprint:
        raise IntegrityError("planned-look receipt hash is inconsistent")
    return PlannedLookReceipt(
        run_id,
        spec.family_id,
        expected_index,
        expected_count,
        decision,
        normalized,
        prior_fingerprint,
        fingerprint,
        path,
        True,
    )


def _verify_flat_json_directory(
    directory: Path,
    *,
    expected_stems: frozenset[str],
    label: str,
) -> None:
    _reject_symlink(directory, label)
    if not directory.is_dir():
        raise IntegrityError(f"{label} is not a directory")
    try:
        entries = tuple(directory.iterdir())
    except OSError as error:
        raise IntegrityError(f"{label} cannot be inspected") from error
    for entry in entries:
        if (
            entry.is_symlink()
            or not entry.is_file()
            or entry.suffix != ".json"
            or entry.stem not in expected_stems
        ):
            raise IntegrityError(f"{label} contains an unregistered entry")


def _verify_look_root(root: Path) -> None:
    _reject_symlink(root, "planned-look root")
    if not root.is_dir():
        raise IntegrityError("planned-look root is not a directory")
    allowed = frozenset(G3_PLANNED_LOOK_BY_ID)
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise IntegrityError("planned-look root cannot be inspected") from error
    for entry in entries:
        if entry.is_symlink() or not entry.is_dir() or entry.name not in allowed:
            raise IntegrityError("planned-look root contains an unregistered entry")


def _validate_failed_planned_look_dispositions(
    *,
    task_id: str,
    payload: Mapping[str, Any],
    receipts: Mapping[str, tuple[PlannedLookReceipt, ...]],
) -> None:
    """Validate exact terminal-or-not-started evidence for a failed look owner.

    A scientific failure can occur before one or more registered experiments
    begin.  Such a task must not fabricate a planned-look receipt.  Instead its
    binding-aware failed result carries a sealed failure-certificate digest for
    each absent family.  A family that did start must retain its complete,
    terminal receipt chain.  This function validates only the durable JSON
    projection; the scientific publisher is responsible for deriving it from
    the producer-sealed failure object.
    """

    owned = tuple(
        item for item in G3_PLANNED_LOOK_REGISTRY if item.owner_task_id == task_id
    )
    if not owned:
        raise ModelError("failed look dispositions require a registered look owner")
    document = payload.get("planned_look_dispositions")
    if not isinstance(document, Mapping) or set(document) != {
        "families",
        "schema_id",
    }:
        raise ModelError(
            "failed G3 look owner requires exact planned-look dispositions"
        )
    if document["schema_id"] != FAILED_PLANNED_LOOK_DISPOSITION_SCHEMA_ID:
        raise ModelError("failed planned-look disposition schema is unsupported")
    families = document["families"]
    if not isinstance(families, Mapping) or set(families) != {
        item.family_id for item in owned
    }:
        raise ModelError("failed planned-look disposition families are incomplete")

    for spec in owned:
        entry = families[spec.family_id]
        if not isinstance(entry, Mapping):
            raise ModelError("failed planned-look family disposition is invalid")
        disposition = entry.get("disposition")
        family_receipts = receipts.get(spec.family_id, ())
        if disposition == "terminal_receipts":
            if set(entry) != {"disposition", "receipt_fingerprints"}:
                raise ModelError("terminal planned-look disposition keys are invalid")
            fingerprints = entry["receipt_fingerprints"]
            expected = [item.fingerprint for item in family_receipts]
            if (
                not isinstance(fingerprints, list)
                or fingerprints != expected
                or not family_receipts
                or family_receipts[-1].decision == "continue"
            ):
                raise ModelError(
                    "terminal planned-look disposition lacks its exact receipt chain"
                )
            continue
        if disposition == "not_started":
            if set(entry) != {
                "disposition",
                "failure_certificate_fingerprint",
            }:
                raise ModelError(
                    "not-started planned-look disposition keys are invalid"
                )
            certificate = entry["failure_certificate_fingerprint"]
            if (
                family_receipts
                or not isinstance(certificate, str)
                or _FINGERPRINT.fullmatch(certificate) is None
            ):
                raise ModelError(
                    "not-started planned-look disposition evidence is invalid"
                )
            continue
        raise ModelError("planned-look family disposition is unsupported")


def _acquire_launch_lock(run_path: Path, run_id: str) -> int:
    path = run_path / LAUNCH_LOCK_NAME
    _reject_symlink(path.parent, "calibration launch-lock parent")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise IntegrityError("calibration launch lock cannot be opened") from error
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode) or information.st_nlink != 1:
            raise IntegrityError("calibration launch lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise IntegrityError(
                "another calibration coordinator holds the lock"
            ) from error
        content = _canonical_bytes(
            {
                "schema_version": RUN_STORE_SCHEMA_VERSION,
                "run_id": run_id,
                "lock_scope": "process_advisory_exclusive",
            }
        )
        os.ftruncate(descriptor, 0)
        os.write(descriptor, content)
        os.fsync(descriptor)
        _fsync_directory(run_path)
        return descriptor
    except BaseException:
        _release_descriptor(descriptor)
        raise


def _release_descriptor(descriptor: int) -> None:
    with suppress(OSError):
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    with suppress(OSError):
        os.close(descriptor)


def _atomic_replace(path: Path, content: bytes) -> None:
    _reject_symlink(path.parent, "calibration atomic-record parent")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelError(f"{label} must be a non-negative integer")
    return value


def _write_or_verify_exact(path: Path, content: bytes) -> bool:
    _reject_symlink(path.parent, "calibration receipt parent")
    if path.exists() or path.is_symlink():
        existing = _read_regular(path, "existing calibration record")
        if existing != content:
            raise IntegrityError(
                "existing calibration record differs from requested bytes"
            )
        return True
    temporary_fd, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(temporary_fd, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            existing = _read_regular(path, "concurrent calibration record")
            if existing != content:
                raise IntegrityError(
                    "concurrent calibration record differs from requested bytes"
                ) from error
            return True
        _fsync_directory(path.parent)
        return False
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _read_canonical_object(path: Path, label: str) -> dict[str, Any]:
    content = _read_regular(path, label)
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IntegrityError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict) or content != _canonical_bytes(value):
        raise IntegrityError(f"{label} is not canonical JSON")
    return value


def _read_regular(path: Path, label: str) -> bytes:
    _reject_symlink(path, label)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise IntegrityError(f"{label} cannot be opened") from error
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode) or information.st_nlink != 1:
            raise IntegrityError(f"{label} must be a single-link regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _normalize_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ModelError(f"{label} must be a non-empty mapping")
    result = _normalize_json(dict(value), path=label)
    if not isinstance(result, dict):
        raise AssertionError("mapping normalization returned a non-mapping")
    return result


def _normalize_json(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelError(f"{path} contains a non-finite float")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ModelError(f"{path} contains an invalid key")
            if any(marker in key.casefold() for marker in _SECRET_MARKERS):
                raise ModelError(f"{path} contains a secret-like key")
            result[key] = _normalize_json(value[key], path=f"{path}.{key}")
        return result
    if isinstance(value, list | tuple):
        return [_normalize_json(item, path=f"{path}[]") for item in value]
    raise ModelError(f"{path} contains an unsupported JSON value")


def _component(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ModelError(f"{label} is not a safe canonical component")
    return value


def _require_fingerprint(value: str, label: str) -> None:
    if not isinstance(value, str) or not _FINGERPRINT.fullmatch(value):
        raise ModelError(f"{label} must be a lowercase SHA-256 fingerprint")


def _reject_symlink(path: Path, label: str) -> None:
    absolute = path.absolute()
    for candidate in reversed([absolute, *absolute.parents]):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise IntegrityError(f"{label} ancestors cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise IntegrityError(
                f"{label} contains a symbolic-link ancestor",
                details={"path": str(candidate)},
            )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
