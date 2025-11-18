# Local pipeline runner

Use `scripts/run_local_pipeline.py` to run a quick prepare -> train (dry) -> export -> infer -> profile sequence on a small subset of images.

Usage:

```powershell
python scripts/run_local_pipeline.py settings.yaml
```

Notes:
- Training is run in `dry_run` mode to keep CI/local runs fast.
- The script attempts best-effort operations and will skip steps if required artifacts are missing (e.g., no checkpoint or TorchScript export).
- Ensure dependencies in `pyproject.toml` are installed (use `poetry install`).
