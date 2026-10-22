"""Repository paths and config files.

Paths in config files are relative to the repository root, so every command
works from any working directory. A path that is absolute is used as given.
"""
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve()
# src/gnn_gym/paths.py -> repository root, if run from a checkout
REPO_ROOT = _HERE.parents[2] if (_HERE.parents[2] / "pyproject.toml").exists() \
    else Path.cwd()


def resolve(path):
    """Absolute path; relative paths are taken from the repository root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(path):
    with open(resolve(path)) as fh:
        return yaml.safe_load(fh) or {}
