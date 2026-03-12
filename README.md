# UFZ-MVGNN

**Multi-View Graph Neural Network for Urban Functional Zone Recognition**

A modular deep learning framework that identifies urban functional zones (e.g., residential, commercial, industrial) from building morphology data using multi-view contrastive learning on graphs.

> **Status**: Core framework complete. Experiments in progress — full benchmarks and paper forthcoming.

---

## Motivation

Urban functional zone (UFZ) recognition is essential for urban planning, resource allocation, and smart city development. Traditional approaches rely heavily on POI (Point of Interest) data, which suffers from **spatial sparsity** — many buildings lack nearby POI annotations.

UFZ-MVGNN addresses this by:
1. **Imputing** semantic information from physical building features via cross-view learning
2. **Diffusing** sparse POI signals across the spatial graph using IDW interpolation
3. **Contrasting** physical and semantic views to learn robust, transferable representations

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        UFZ-MVGNN Pipeline                          │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │  Data     │──▶│ Feature  │──▶│  Graph   │──▶│   Semantic     │  │
│  │  Loading  │   │ Engine   │   │ Builder  │   │   Enhancement  │  │
│  └──────────┘   └──────────┘   └──────────┘   └───────┬────────┘  │
│   Shapefile      27D Physical   Delaunay +      3-Stage Pipeline   │
│   POI / Raster   Features       Auto-pruning    (Impute→IDW→Refine)│
│                                                       │            │
│                                                       ▼            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │  Export   │◀──│ Cluster  │◀──│ Analysis │◀──│   Multi-View   │  │
│  │  GeoJSON  │   │ Viz      │   │ HDBSCAN  │   │   Contrastive  │  │
│  └──────────┘   └──────────┘   └──────────┘   │   Learning     │  │
│                                                └────────────────┘  │
│                                                 Physical ↔ Semantic │
│                                                 InfoNCE (τ=0.07)   │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Contributions

### 1. Three-Stage Semantic Enhancement

Addresses POI sparsity through a cascaded pipeline:

| Stage | Module | Description |
|-------|--------|-------------|
| **Imputation** | `CrossViewImputer` | GAT-based encoder predicts POI distribution (17-class) from 27D physical features |
| **Diffusion** | `GaussianIDW` | Inverse distance weighting with Gaussian kernel smooths predictions across spatial neighbors |
| **Refinement** | `RefineNet` | Dual-stream gated fusion network refines POI predictions using both physical and diffused semantic features |

### 2. Multi-View Contrastive Learning (MVCL)

Learns unified building representations by contrasting two complementary views:

```
Physical View (27D)                    Semantic View (17D)
  Building morphology                    Imputed POI distribution
        │                                       │
   GNN Encoder                              GNN Encoder
   (GIN/GAT/GCN)                           (GIN/GAT/GCN)
        │                                       │
   Projection Head                         Projection Head
        │                                       │
        └──────────── InfoNCE Loss ─────────────┘
                  Semantic-aware negative
                  sample reweighting
```

**Semantic-aware negative weighting**: Negative pairs with similar POI distributions receive lower weights, preventing the model from pushing semantically similar buildings apart.

### 3. Extensible Registry Pattern

New features and GNN backbones can be added with zero framework modification:

```python
# Add a custom feature — just register and configure
@FeatureRegistry.register('my_feature')
def compute_my_feature(gdf):
    gdf['custom'] = gdf.geometry.area / gdf.geometry.convex_hull.area
    return gdf

# Add a custom GNN backbone
@BackboneRegistry.register('sage')
class SAGEEncoder(BaseEncoder):
    ...
```

## Project Structure

```
ufz/
├── features/       # 27D physical features: shape(13) + size(2) + height(3) + orientation(4) + density
├── graph/          # Delaunay triangulation with quantile-based edge pruning
├── semantic/       # 3-stage enhancement: CrossViewImputer → IDW → RefineNet
├── models/         # MVCL model with pluggable GNN backbones (GIN, GAT, GCN)
├── analysis/       # Clustering (HDBSCAN/DBSCAN/KMeans/Leiden) + dimensionality reduction
├── visualization/  # Interactive Plotly visualizations
├── config/         # YAML-based config with inheritance (_base_) support
├── utils/          # Logging, caching, random seed utilities
└── cli.py          # CLI: train / cluster / export / visualize
```

> **Note**: Data loaders, training configs, export modules, and full pipeline scripts will be released upon paper publication.

## Installation

```bash
# Python 3.10+, CUDA 12.0+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirement.txt
```

**Core dependencies**: PyTorch, PyTorch Geometric, GeoPandas, Shapely, scikit-learn, HDBSCAN, Plotly, UMAP

## Usage

```bash
# Full pipeline
python -m ufz train --stage all --config configs/base.yaml

# Step by step
python -m ufz train --stage semantic --config configs/base.yaml   # Stage 1-3
python -m ufz train --stage mvcl --config configs/base.yaml       # Contrastive learning
python -m ufz cluster --config configs/base.yaml                  # Clustering
python -m ufz export --format map --config configs/base.yaml      # GeoJSON export
python -m ufz visualize --type embedding --config configs/base.yaml

# Override config on the fly
python -m ufz train --stage all --device cpu --epochs 30 --seed 42
```

### Configuration

YAML-based with inheritance support:

```yaml
# configs/base.yaml
data:
  shp_path: /path/to/buildings.shp
  poi_path: /path/to/poi.csv

features:
  groups: [shape, size, orientation]

semantic:
  epochs: 200
  hidden_dim: 128
  heads: 4              # GAT attention heads

model:
  backbone: gin          # gin | gat | gcn
  repr_dim: 128
  temperature: 0.07

analysis:
  clustering_method: hdbscan
  hdbscan_min_cluster_size: 15
```

```yaml
# configs/debug.yaml — inherit and override
_base_: base.yaml
data:
  max_buildings: 500
semantic:
  epochs: 30
device: cpu
```

## Feature Dimensions

| View | Features | Dim | Source |
|------|----------|-----|--------|
| **Physical** | Shape (area, perimeter, circularity, convexity, fractal dim, eccentricity, ...) | 13D | Building geometry |
| | Size (length, width) | 2D | Minimum bounding rectangle |
| | Height (min, mean, std) | 3D | Raster / attribute |
| | Orientation (mean direction, symmetry, ...) | 4D | Principal axis analysis |
| | Density | 5D | Spatial density metrics |
| **Topological** | Graphlet orbits (ORCA) | 15D | Delaunay graph structure |
| **Semantic** | POI category distribution | 17D | Imputed via 3-stage pipeline |
| **Final repr.** | Contrastive learning output | **128D** | MVCL fusion |

## Preliminary Results

> Experiments are actively running. Full quantitative benchmarks will be added upon completion.

<!-- TODO: Add result table and visualization figures -->
<!--
| Dataset | Method | NMI | ARI | Purity |
|---------|--------|-----|-----|--------|
| City A  | UFZ-MVGNN | - | - | - |
-->

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=ufz --cov-report=html
```

Covers: config parsing, model forward pass, semantic enhancement modules, clustering algorithms.

## Technical Highlights

- **Modular 9-layer pipeline** — each layer is independently testable and replaceable
- **Registry pattern** for features and GNN backbones — zero-modification extensibility
- **Automatic CRS handling** — detects WGS84 vs UTM and converts as needed
- **Config inheritance** — `_base_` support for environment-specific overrides (debug/server)
- **WeightedKLDivLoss** — upweights rare POI categories to handle long-tail distribution
- **4,200+ lines** of modular, typed Python across 57 files

## Roadmap

- [ ] Complete experiments on multiple city datasets
- [ ] Ablation studies (backbone, semantic stages, loss functions)
- [ ] Add pre-trained model checkpoints
- [ ] Paper submission

## License

This project is part of ongoing academic research. Code is provided for review purposes. Full release planned upon paper publication.

## Contact

Yongkang Sun — [GitHub](https://github.com/skysyk12) · syk940940@gmail.com
