from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED = [
    "AGENTS.md",
    "bootstrap/manifest.json",
    "harness/SCIENTIFIC_CONTRACT.md",
    "harness/VALIDATION_MATRIX.md",
    "harness/RUN_STATE.md",
    "harness/DECISION_LOG.md",
    "harness/FAILURE_LOG.md",
    ".agents/skills/mosaic-research-loop/SKILL.md",
    ".agents/skills/mosaic-coding-loop/SKILL.md",
    ".agents/skills/mosaic-topology-eess/SKILL.md",
    ".agents/skills/mosaic-homotopy-root/SKILL.md",
    ".agents/skills/mosaic-cuda-loop/SKILL.md",
    ".agents/skills/mosaic-online-learning/SKILL.md",
    ".agents/skills/mosaic-independent-review/SKILL.md",
]

missing = [item for item in REQUIRED if not (ROOT / item).is_file()]
if missing:
    raise SystemExit("missing required harness files: " + ", ".join(missing))
manifest = json.loads((ROOT / "bootstrap/manifest.json").read_text())
if manifest.get("project") != "MOSAIC-ODE":
    raise SystemExit("bootstrap manifest project mismatch")
if manifest.get("dae_scope") != "research":
    raise SystemExit("DAE scope must remain research in the bootstrap")
print(f"MOSAIC-ODE harness valid: {len(REQUIRED)} required files present")
