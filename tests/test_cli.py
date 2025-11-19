"""
Test CLI for bottle-sorter.
"""
from typer.testing import CliRunner
from bsort.cli import app

runner = CliRunner()

def test_cli_train():
    result = runner.invoke(app, ["train", "--config", "settings.yaml", "--dry-run"])
    assert result.exit_code == 0
