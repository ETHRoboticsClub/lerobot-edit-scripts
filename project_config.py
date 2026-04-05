from __future__ import annotations

import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config" / "project.yaml"


def load_project_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Project config not found: {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Project config must be a YAML mapping: {config_path}")

    return config


def get_dataset_repo_id(config_path: Path = CONFIG_PATH) -> str:
    config = load_project_config(config_path)
    repo_id = config.get("dataset_repo_id")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError(f"'dataset_repo_id' must be a non-empty string in {config_path}")
    return repo_id.strip()


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "dataset_repo_id"
    config = load_project_config()

    if key not in config:
        raise KeyError(f"Config key not found: {key}")

    value = config[key]
    if isinstance(value, (dict, list)):
        raise ValueError(f"Config key '{key}' must be a scalar value for CLI usage")

    print(value)


if __name__ == "__main__":
    main()
