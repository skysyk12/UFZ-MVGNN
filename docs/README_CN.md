# UFZ_MVGNN: 城市功能区多视图图学习框架

UFZ_MVGNN (Urban Functional Zone Multi-View Graph Neural Network) 是一个**完整的城市功能区识别深度学习流水线**，覆盖从空间数据加载、特征工程、语义增强、多视图对比学习、聚类分析到可视化与结果导出的全过程。

## ✨ 核心特性

### 🏗️ 完整的 10 层流水线架构

```
数据加载 → 特征工程 → 图构建 → 语义增强 → 对比学习 → 聚类 → 可视化 → 导出
  data/      features/   graph/   semantic/   models/  analysis/ visual/  export/
```

### 🚀 主要功能

| 功能 | 描述 |
|------|------|
| **数据加载** | Shapefile/POI/栅格数据自动加载 + CRS 自动检测修正 |
| **特征工程** | 13D 形状 + 2D 尺寸 + 4D 方向 + 15D Graphlet = 27D 物理特征 |
| **图构建** | Delaunay 三角剖分 + 分位数自动剪枝 + CRS 转换 |
| **语义增强** | 三阶段串联：CrossViewImputer → IDW 扩散 → RefineNet |
| **对比学习** | MVCL：物理视图 vs 语义视图 + 语义感知负样本加权 |
| **聚类分析** | HDBSCAN/DBSCAN/KMeans/Leiden 统一接口 |
| **可视化** | Plotly 交互式可视化：图、嵌入、聚类结果 |
| **结果导出** | GeoJSON + 聚类统计 + 模型检查点 |
| **LLM 查询** | 🆕 将聚类结果转化为知识图谱，支持自然语言查询 (Google/Deepseek) |

## 📋 目录结构

```
UFZ_MVGNN/
├── ufz/                          # 核心包
│   ├── __init__.py
│   ├── __main__.py               # 入口：python -m ufz
│   ├── cli.py                    # 4 个子命令：train/cluster/export/visualize
│   │
│   ├── config/                   # 配置系统
│   │   ├── __init__.py
│   │   └── parser.py             # YAML + dataclass，支持 _base_ 继承
│   │
│   ├── data/                     # 数据加载 (I/O 层)
│   │   ├── __init__.py
│   │   ├── loader.py             # Shapefile 加载 + CRS 自动检测
│   │   ├── poi.py                # POI 空间匹配 → GT 标签
│   │   ├── raster.py             # 栅格特征提取 (GDP/人口)
│   │   └── sampler.py            # 空间采样 (调试用)
│   │
│   ├── features/                 # 特征计算 (纯算法)
│   │   ├── __init__.py
│   │   ├── registry.py           # @FeatureRegistry 注册表
│   │   ├── manager.py            # 特征调度器
│   │   ├── shape.py              # 13D 形状特征
│   │   ├── size.py               # 2D 尺寸 + 3D 高度
│   │   ├── orientation.py        # 4D 方向特征
│   │   ├── density.py            # 密度特征
│   │   ├── graphlet.py           # 15D Graphlet Orbit (ORCA)
│   │   └── processing.py         # 归一化后处理
│   │
│   ├── graph/                    # 图构建
│   │   ├── __init__.py
│   │   └── builder.py            # Delaunay + 剪枝 + CRS 转换
│   │
│   ├── semantic/                 # 语义增强 (3 阶段)
│   │   ├── __init__.py
│   │   ├── trainer.py            # SemanticTrainer 统一训练器
│   │   ├── imputer.py            # CrossViewImputer (物理→POI)
│   │   ├── refine_net.py         # RefineNet (双流融合)
│   │   ├── idw.py                # Gaussian IDW 扩散
│   │   └── losses.py             # WeightedKLDivLoss
│   │
│   ├── models/                   # 对比学习模型
│   │   ├── __init__.py
│   │   ├── mvcl.py               # MVCLModel 主体
│   │   ├── tower.py              # MultiTowerEncoder (Geo/Topo/Sem 三塔)
│   │   ├── trainer.py            # MVCLTrainer 训练循环
│   │   ├── losses.py             # InfoNCE/NTXent/DGI 损失
│   │   └── backbones/            # GNN 编码器
│   │       ├── __init__.py
│   │       ├── registry.py       # @BackboneRegistry 注册表
│   │       ├── base.py           # BaseEncoder 抽象类
│   │       ├── gin.py            # Graph Isomorphism Network
│   │       ├── gat.py            # Graph Attention Network
│   │       └── gcn.py            # Graph Convolutional Network
│   │
│   ├── analysis/                 # 聚类 + 降维
│   │   ├── __init__.py
│   │   ├── clustering.py         # 4 种聚类算法统一接口
│   │   └── reducer.py            # PCA/UMAP 降维
│   │
│   ├── visualization/            # 可视化 (Plotly)
│   │   ├── __init__.py
│   │   ├── graph_viz.py          # Delaunay 图可视化
│   │   ├── embedding_viz.py      # UMAP 嵌入可视化
│   │   └── cluster_viz.py        # 聚类结果可视化
│   │
│   ├── export/                   # 结果导出
│   │   ├── __init__.py
│   │   ├── checkpoint.py         # 模型/嵌入/标签 保存加载
│   │   ├── maps.py               # GeoJSON + 聚类统计导出
│   │   ├── graphrag.py           # 🆕 层级知识图谱构建 (JSON序列化)
│   │   └── llm_interface.py       # 🆕 LLM查询引擎 (Google/Deepseek)
│   │
│   └── utils/                    # 工具库
│       ├── __init__.py
│       ├── logging.py            # 日志配置
│       ├── seed.py               # 随机种子固定
│       └── cache.py              # 统一缓存管理
│
├── configs/                      # 配置文件
│   └── base.yaml                 # 完整配置 (default/local/server 示例参考)
│
├── tests/                        # 单元测试
│   ├── __init__.py
│   ├── test_config.py            # 配置加载/保存
│   ├── test_models.py            # 编码器/MVCL
│   ├── test_analysis.py          # 聚类/降维
│   └── test_semantic.py          # 语义增强
│
├── data/                         # 原始数据 (不进 git)
│   ├── shp/                      # 建筑物 Shapefile
│   ├── poi/                      # POI 数据 (CSV)
│   └── raster/                   # 栅格数据 (GDP/人口)
│
├── cache/                        # 计算缓存 (不进 git)
├── outputs/                      # 训练输出 (不进 git)
├── orca_src/                     # ORCA C++ 源码 (第三方)
├── requirement.txt               # 依赖列表
├── demo_llm_query.py             # LLM 查询演示脚本
├── PROJECT_SUMMARY.md            # 项目详细总结
├── STRUCTURE.md                  # 架构设计文档
├── LLM_GUIDE.md                  # 🆕 LLM 集成完整指南
├── START.md                       # 快速启动 (3 步)
├── HOW_TO_RUN.md                 # 详细运行指南
├── QUICK_START.md                # 快速开始示例
└── README.md                     # 本文件
```

## 🛠️ 安装

### 环境要求
- Python 3.10+
- CUDA 12.0+ (可选，推荐用于加速训练)

### 步骤

```bash
# 克隆或进入项目目录
cd /path/to/UFZ_MVGNN

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 升级 pip 并安装依赖
pip install -U pip
pip install -r requirement.txt
```

### 依赖包

主要依赖包括：
- **数据处理**: NumPy, Pandas, GeoPandas, Shapely, SciPy
- **深度学习**: PyTorch, PyTorch Geometric
- **机器学习**: Scikit-learn, HDBSCAN, Scanpy
- **可视化**: Plotly, UMAP
- **工具**: PyYAML, Joblib, RasterStats
- **测试**: Pytest

详见 `requirement.txt`。

## 🚀 快速开始

### 基础命令

```bash
# 查看帮助
python -m ufz --help

# 查看子命令帮助
python -m ufz train --help
python -m ufz cluster --help
python -m ufz export --help
python -m ufz visualize --help
```

### 完整流水线

```bash
# 一键运行完整流水线（语义增强 + MVCL）
python -m ufz train --stage all --config configs/base.yaml

# 分步执行
python -m ufz train --stage semantic --config configs/base.yaml
python -m ufz train --stage mvcl --config configs/base.yaml

# 聚类分析
python -m ufz cluster --config configs/base.yaml

# 结果导出
python -m ufz export --format map --config configs/base.yaml

# 可视化
python -m ufz visualize --type embedding --config configs/base.yaml
```

### 参数 Override

在命令行直接覆盖配置参数：

```bash
# 调试模式（CPU + 30 个 epoch）
python -m ufz train --stage all --config configs/base.yaml \
    --device cpu --epochs 30 --seed 42

# 生产训练（GPU + 300 个 epoch）
python -m ufz train --stage mvcl --config configs/base.yaml \
    --device cuda --epochs 300 --seed 42
```

### 配置文件

配置采用 YAML 格式，支持 `_base_` 继承：

```yaml
# configs/base.yaml - 完整默认配置
data:
  shp_path: /path/to/buildings.shp
  poi_path: /path/to/poi.csv
  output_dir: outputs
  cache_dir: cache

features:
  groups: [shape, size, orientation]
  use_height: true
  use_graphlet: false
  max_edge_length_m: 200.0

semantic:
  idw_radius: 100.0
  idw_bandwidth: 30.0
  epochs: 200
  lr: 0.001
  batch_size: 4096

model:
  backbone: gin  # gin | gat | gcn
  repr_dim: 128
  proj_dim: 128
  epochs: 200

analysis:
  clustering_method: hdbscan  # hdbscan | dbscan | kmeans | leiden
  hdbscan_min_cluster_size: 15

seed: 42
device: auto  # auto | cuda | cpu
```

支持继承覆盖：

```yaml
# configs/local.yaml - 调试配置
_base_: base.yaml

data:
  max_buildings: 500  # 仅加载 500 栋建筑

semantic:
  epochs: 30  # 快速测试

model:
  epochs: 30

training:
  device: cpu
```

## 📊 特征维度统计

### 物理特征 (Geo): 27D

| 特征类型 | 维度 | 说明 |
|---------|------|------|
| **形状** | 13D | 面积、周长、圆形度、凸度、分形维数、离心率等 |
| **尺寸** | 2D | 长、宽 |
| **高度** | 3D | min/mean/std |
| **方向** | 4D | 主方向、对称性等 |
| **其他** | 5D | 密度等 |
| **总计** | **27D** | |

### 拓扑特征 (Topo): 15D
- Graphlet Orbit (ORCA C++ 计算)

### 语义特征 (Sem): 17D → 64D
- POI 17 类分布预测 (餐饮、购物、住宿、医疗等)
- 可编码为 64D 特征向量

### 最终表示: 128D
- 对比学习后的融合表示

## 🔬 核心算法

### 语义增强流程 (3 阶段)

```
Step 1: CrossViewImputer
┌─────────────────────────────────────┐
│ 物理特征 (27D)                       │
│   ↓                                  │
│ GAT (4 heads) + BatchNorm            │
│   ↓                                  │
│ GAT (1 head) + BatchNorm             │
│   ↓                                  │
│ MLP (2 层, ReLU)                     │
│   ↓                                  │
│ Softmax                              │
│   ↓                                  │
│ 初步 POI 分布预测 (17D)              │
└─────────────────────────────────────┘

Step 2: IDW 扩散
┌─────────────────────────────────────┐
│ 初步预测 (17D) + 图拓扑              │
│   ↓                                  │
│ Gaussian IDW: exp(-d²/σ²)           │
│   ↓                                  │
│ 邻域加权融合                         │
│   ↓                                  │
│ 平滑化 POI 分布 (17D)                │
└─────────────────────────────────────┘

Step 3: RefineNet
┌─────────────────────────────────────┐
│ 物理特征 (27D) + 扩散语义 (17D)      │
│   ↓                                  │
│ 双流编码 (分别处理)                  │
│   ↓                                  │
│ 拼接 (44D)                           │
│   ↓                                  │
│ 门控融合模块                         │
│   ↓                                  │
│ MLP (2 层, ReLU)                     │
│   ↓                                  │
│ 精细 POI 分布预测 (17D)              │
└─────────────────────────────────────┘

损失函数: WeightedKLDivLoss
  背景类权重: 低 (常见)
  长尾POI权重: 高 (稀有)
```

### 对比学习 (MVCL)

```
物理视图                    语义视图
├─ 建筑形态特征 (27D)      ├─ POI 分布预测 (17D)
├─ 空间邻接关系            ├─ 空间邻接关系
│                          │
↓ GNN 编码器               ↓ GNN 编码器
(GIN/GAT/GCN)             (GIN/GAT/GCN)
│                          │
↓ 表示 (128D)             ↓ 表示 (128D)
│                          │
↓ 投影头                   ↓ 投影头
│                          │
↓ 对比空间 (128D)         ↓ 对比空间 (128D)
└──────────┬────────────────┘
           ↓
    InfoNCE 对比损失
    (温度 τ = 0.07)
    语义感知负样本加权

融合策略:
  - 平均融合: (h_phys + h_sem) / 2
  - Tower 融合: Geo塔(27D) + Topo塔(15D) + Sem塔(64D)
```

### 聚类算法

支持 4 种聚类方法，统一接口：

```python
from ufz.analysis.clustering import cluster_embeddings

# HDBSCAN (推荐)
labels = cluster_embeddings(
    embeddings,
    method='hdbscan',
    min_cluster_size=15,
    min_samples=1
)

# DBSCAN
labels = cluster_embeddings(
    embeddings,
    method='dbscan',
    eps=0.5,
    min_samples=15
)

# KMeans
labels = cluster_embeddings(
    embeddings,
    method='kmeans',
    n_clusters=10
)

# Leiden (社区检测)
labels = cluster_embeddings(
    embeddings,
    method='leiden',
    resolution=1.0
)
```

## 🧪 测试

运行单元测试：

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_config.py -v
pytest tests/test_models.py -v
pytest tests/test_semantic.py -v
pytest tests/test_analysis.py -v

# 显示覆盖率
pytest tests/ --cov=ufz --cov-report=html
```

### 测试覆盖

- **test_config.py**: YAML 加载/保存、配置继承
- **test_models.py**: 编码器初始化、MVCL 前向传播
- **test_semantic.py**: CrossViewImputer、RefineNet
- **test_analysis.py**: 聚类、降维

## 💡 核心模块详解

### 特征注册机制

新增特征无需修改代码，仅需注册：

```python
# ufz/features/custom_feature.py
from .registry import FeatureRegistry

@FeatureRegistry.register('custom')
def my_custom_feature(gdf):
    """计算自定义特征."""
    gdf['my_feature'] = ...
    return gdf

# 配置中引用即可
# configs/base.yaml
# features:
#   groups: [shape, size, custom]
```

### 编码器注册机制

支持轻松添加新 GNN 编码器：

```python
# ufz/models/backbones/sage.py
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from .base import BaseEncoder
from .registry import BackboneRegistry

@BackboneRegistry.register('sage')
class SAGEEncoder(BaseEncoder):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, dropout)
        self.convs = nn.ModuleList([
            SAGEConv(in_dim, hidden_dim),
            *[SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)],
            SAGEConv(hidden_dim, out_dim)
        ])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        for conv, dropout in zip(self.convs[:-1], self.dropouts):
            x = conv(x, edge_index).relu()
            x = dropout(x)
        x = self.convs[-1](x, edge_index)
        return x

# 配置中使用
# configs/base.yaml
# model:
#   backbone: sage
```

## 📖 文档

- **PROJECT_SUMMARY.md**: 项目完整总结，包括统计数据和设计特性
- **STRUCTURE.md**: 架构设计文档，包括模块映射关系
- **START.md**: 最快速的 3 步启动指南
- **HOW_TO_RUN.md**: 详细的环境配置和运行指南
- **QUICK_START.md**: 详细的快速开始和代码示例
- **LLM_GUIDE.md**: 🆕 **LLM 查询系统完整指南** - 将聚类结果转化为知识图谱，支持自然语言查询
- **README.md**: 本文件（快速开始与使用指南）

## ⚙️ 高级配置

### 数据配置

```yaml
data:
  shp_path: /path/to/buildings.shp
  poi_path: /path/to/poi.csv
  raster_paths: [/path/to/gdp.tif, /path/to/population.tif]
  output_dir: outputs
  cache_dir: cache
  target_crs: EPSG:32650  # 目标投影
```

### 特征配置

```yaml
features:
  groups: [shape, size, orientation, density, graphlet]
  use_height: true
  use_graphlet: false  # ORCA 计算开销大，谨慎启用
  graphlet_orca_path: ./orca
  max_edge_length_m: 200.0  # Delaunay 边剪枝阈值
```

### 语义增强配置

```yaml
semantic:
  # IDW 参数
  idw_radius: 100.0
  idw_bandwidth: 30.0

  # POI 匹配
  poi_max_distance: 50.0
  poi_lon_col: update_wgs84_lon
  poi_lat_col: update_wgs84_lat
  poi_type_col: poi类型

  # 模型架构
  hidden_dim: 128
  heads: 4  # GAT heads
  dropout: 0.3
  num_classes: 17  # POI 类别数

  # 训练参数
  epochs: 200
  lr: 0.001
  weight_decay: 0.00001
  batch_size: 4096
  eval_every: 10
  patience: 30
```

### 对比学习配置

```yaml
model:
  backbone: gin  # gin | gat | gcn
  hidden_dim: 256
  repr_dim: 128
  proj_dim: 128
  num_layers: 2
  dropout: 0.5
  gat_heads: 4

  # Multi-tower 参数
  geo_dim: 27
  topo_dim: 15
  sem_dim: 64
  tower_dim: 32
  fusion_dim: 128

  # 训练参数
  epochs: 200
  lr: 0.001
  weight_decay: 0.00001
  batch_size: 4096
  eval_every: 10
  patience: 30
```

### 聚类配置

```yaml
analysis:
  clustering_method: hdbscan  # hdbscan | dbscan | kmeans | leiden
  hdbscan_min_cluster_size: 15
  dbscan_eps: 0.5
  kmeans_n_clusters: 10

  # 降维
  reducer_method: umap  # umap | pca
  n_components: 2
```

## 🔧 故障排除

### CRS 检测失败
如果 Shapefile 的 CRS 检测不正确，可手动指定：
```yaml
data:
  target_crs: EPSG:4326  # 或其他正确的 EPSG 代码
```

### 内存不足
减少批量大小或启用缓存：
```yaml
semantic:
  batch_size: 2048  # 从 4096 降低

model:
  batch_size: 2048
```

### CUDA 不可用
使用 CPU 训练（较慢）：
```bash
python -m ufz train --stage all --config configs/base.yaml --device cpu
```

### Graphlet 计算缓慢
禁用 Graphlet 特征：
```yaml
features:
  use_graphlet: false
```

## 📝 注意事项

1. **CLI 部分流程为 Stub 实现**: 当前 CLI 中大部分训练逻辑仅打印日志，核心算法模块完全实现，可独立调用。

2. **缺乏端到端集成测试**: 单元测试框架已准备，但缺乏完整的端到端流程测试。

3. **需要实际数据路径**: `configs/base.yaml` 中的 `shp_path` 和 `poi_path` 需要填充实际数据。

4. **ORCA 需要编译**: 如启用 Graphlet 特征，需预先编译 `orca_src/` 中的 C++ 代码。

## 🤝 扩展指南

### 添加新特征

1. 在 `ufz/features/` 中创建新文件
2. 使用 `@FeatureRegistry.register()` 装饰器
3. 在配置中添加到 `features.groups`

### 添加新编码器

1. 在 `ufz/models/backbones/` 中创建新文件
2. 继承 `BaseEncoder`
3. 使用 `@BackboneRegistry.register()` 装饰器
4. 在配置中指定 `model.backbone`

### 添加新聚类方法

1. 在 `ufz/analysis/clustering.py` 中添加新函数
2. 更新 `cluster_embeddings()` 的 method 选项
3. 在配置中指定 `analysis.clustering_method`

## 📜 许可证

（根据项目实际情况填写）

## 📞 联系方式

（根据项目实际情况填写）

---

**完整项目代码**: `/Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN`

**最后更新**: 2026-03-09
