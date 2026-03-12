"""Unified configuration parser for UFZ pipeline."""

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class DataConfig:
    """Data paths and settings."""
    shp_path: Optional[str] = None
    poi_path: Optional[str] = None
    raster_paths: List[str] = field(default_factory=list)
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    target_crs: str = "EPSG:32650"


@dataclass
class FeaturesConfig:
    """Feature computation settings."""
    groups: List[str] = field(default_factory=lambda: ["shape", "size", "orientation"])
    use_height: bool = True
    use_graphlet: bool = False
    graphlet_orca_path: str = "./orca"
    max_edge_length_m: float = 200.0


@dataclass
class SemanticConfig:
    """Semantic enhancement settings (Imputer + RefineNet)."""
    # IDW parameters
    idw_radius: float = 100.0
    idw_bandwidth: float = 30.0
    
    # POI matching
    poi_max_distance: float = 50.0
    poi_lon_col: str = "update_wgs84_lon"
    poi_lat_col: str = "update_wgs84_lat"
    poi_type_col: str = "poi类型"
    
    # Model architecture
    hidden_dim: int = 128
    heads: int = 4
    dropout: float = 0.3
    num_classes: int = 17
    
    # Training
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 4096
    eval_every: int = 10
    patience: int = 30


@dataclass
class ModelConfig:
    """MVCL model settings."""
    backbone: str = "gin"
    hidden_dim: int = 256
    repr_dim: int = 128
    proj_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    gat_heads: int = 4
    
    # Multi-tower settings
    geo_dim: int = 27
    topo_dim: int = 15
    sem_dim: int = 64
    tower_dim: int = 32
    fusion_dim: int = 128
    
    # Training
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 4096
    eval_every: int = 10
    patience: int = 30


@dataclass
class AnalysisConfig:
    """Clustering and analysis settings."""
    clustering_method: str = "hdbscan"
    hdbscan_min_cluster_size: int = 15
    dbscan_eps: float = 0.5
    kmeans_n_clusters: int = 10
    
    # Dimensionality reduction
    reducer_method: str = "umap"
    n_components: int = 2


@dataclass
class Config:
    """Unified configuration for entire pipeline."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    # Training control
    seed: int = 42
    device: str = "auto"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        
        # Handle _base_ inheritance
        base_path = data.pop("_base_", None)
        if base_path:
            base_config = cls.from_yaml(base_path)
            base_dict = asdict(base_config)
            cls._deep_update(base_dict, data)
            data = base_dict
        
        return cls(
            data=DataConfig(**data.get("data", {})),
            features=FeaturesConfig(**data.get("features", {})),
            semantic=SemanticConfig(**data.get("semantic", {})),
            model=ModelConfig(**data.get("model", {})),
            analysis=AnalysisConfig(**data.get("analysis", {})),
            seed=data.get("seed", 42),
            device=data.get("device", "auto"),
        )
    
    @staticmethod
    def _deep_update(base: Dict, update: Dict):
        """Recursively update nested dicts."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                Config._deep_update(base[key], value)
            else:
                base[key] = value
    
    def to_yaml(self, path: str):
        """Save to YAML file."""
        data = asdict(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
