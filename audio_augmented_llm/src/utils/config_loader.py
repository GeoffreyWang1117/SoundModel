"""
Configuration loader utility for loading YAML configs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load with OmegaConf for better YAML support
    config = OmegaConf.load(config_path)

    # Convert to regular dict
    return OmegaConf.to_container(config, resolve=True)


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    base_omega = OmegaConf.create(base_config)
    override_omega = OmegaConf.create(override_config)

    merged = OmegaConf.merge(base_omega, override_omega)

    return OmegaConf.to_container(merged, resolve=True)


if __name__ == "__main__":
    # Test loading
    config = load_config("audio_augmented_llm/configs/experiment_config.yaml")
    print(f"Loaded config with keys: {list(config.keys())}")
