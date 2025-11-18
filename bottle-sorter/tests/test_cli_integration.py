import os
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION_TESTS", "0") != "1", reason="Integration tests disabled by default")
def test_cli_dry_run():
    # Run the CLI in dry-run mode. This should exit 0 quickly when integration tests are enabled.
    project_root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "bsort.cli", "train", "--config", "settings.yaml", "--dry-run"]
    proc = subprocess.run(cmd, cwd=str(project_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert proc.returncode == 0, f"CLI dry-run failed: {proc.stderr.decode()}"
