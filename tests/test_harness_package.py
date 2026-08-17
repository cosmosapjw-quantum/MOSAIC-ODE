import subprocess
from pathlib import Path
def test_integrated_harness_and_skills_validate():
 required={'.agents/skills/mosaic-research-loop/SKILL.md','.agents/skills/mosaic-coding-loop/SKILL.md','.agents/skills/mosaic-topology-eess/SKILL.md','.agents/skills/mosaic-homotopy-root/SKILL.md','.agents/skills/mosaic-cuda-loop/SKILL.md','.agents/skills/mosaic-online-learning/SKILL.md','.agents/skills/mosaic-independent-review/SKILL.md','harness/SCIENTIFIC_CONTRACT.md','harness/VALIDATION_MATRIX.md','harness/tools/validate_harness.py'}
 for item in required:assert Path(item).is_file()
 completed=subprocess.run(['python3','harness/tools/validate_harness.py'],capture_output=True,text=True);assert completed.returncode==0,completed.stdout+completed.stderr
