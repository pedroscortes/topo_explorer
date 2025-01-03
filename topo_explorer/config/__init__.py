"""Configuration management for topo_explorer."""

import os
from pathlib import Path
import yaml
from typing import Dict, Any

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file. If None, loads default config.
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

__all__ = ["load_config", "save_config"]