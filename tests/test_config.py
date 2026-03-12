"""Test configuration parsing."""

import pytest
import tempfile
from pathlib import Path
from ufz.config.parser import Config, DataConfig


def test_config_defaults():
    """Test default configuration."""
    config = Config()
    assert config.seed == 42
    assert config.device == "auto"
    assert config.data.output_dir == "outputs"


def test_config_from_yaml():
    """Test YAML loading."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
data:
  shp_path: /path/to/buildings.shp
  poi_path: /path/to/poi.csv
  output_dir: /tmp/outputs

seed: 42
device: cuda
""")
        f.flush()
        
        config = Config.from_yaml(f.name)
        assert config.data.shp_path == "/path/to/buildings.shp"
        assert config.seed == 42
        assert config.device == "cuda"
        
        Path(f.name).unlink()


def test_config_to_yaml():
    """Test YAML saving."""
    config = Config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.to_yaml(f.name)
        
        # Reload and verify
        config2 = Config.from_yaml(f.name)
        assert config2.seed == config.seed
        
        Path(f.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
