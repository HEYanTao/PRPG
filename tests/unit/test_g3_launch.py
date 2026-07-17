from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from prpg.cli import app


def test_fresh_g3_launcher_and_rehearsal_import_closure_is_in_manifest() -> None:
    script = """
import json
import sys
from pathlib import Path
import prpg.g3_launch
import prpg.model.g3_rehearsal
from prpg.provenance import G3_EXECUTION_CLOSURE_MANIFEST

forbidden = (
    "prpg.model.g5_",
    "prpg.simulation.g4_reference",
    "prpg.simulation.production",
    "prpg.storage",
    "prpg.validation",
)
root = Path.cwd().resolve()
loaded = set()
for module in tuple(sys.modules.values()):
    raw = getattr(module, "__file__", None)
    if not raw:
        continue
    path = Path(raw).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        continue
    if relative.startswith("src/prpg/") and relative.endswith(".py"):
        loaded.add(relative)
manifest = set(G3_EXECUTION_CLOSURE_MANIFEST.module_paths)
print(json.dumps({
    "forbidden": sorted(
        name for name in sys.modules if name.startswith(forbidden)
    ),
    "loaded": sorted(loaded),
    "missing": sorted(loaded - manifest),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["loaded"]
    assert result["forbidden"] == []
    assert result["missing"] == []


def test_legacy_cli_cannot_preflight_or_initialize_a_g3_run(
    tmp_path: Path,
) -> None:
    config = tmp_path / "nonexistent.yaml"
    result = CliRunner().invoke(
        app,
        ["calibrate", "--preflight-only", "--config", str(config), "--json"],
    )

    assert result.exit_code == 2
    event = json.loads(result.stdout)
    assert event["event"] == "error"
    assert event["details"]["required_command"] == "python -m prpg.g3_launch"
    assert not (tmp_path / "runs").exists()
