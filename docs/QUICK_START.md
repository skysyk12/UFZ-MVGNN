# UFZ_MVGNN 快速运行指南

## 📋 目录

1. [环境准备](#环境准备)
2. [项目结构](#项目结构)
3. [运行示例](#运行示例)
4. [常见命令](#常见命令)
5. [调试技巧](#调试技巧)
6. [常见问题](#常见问题)

---

## 🔧 环境准备

### 前提条件
- ✅ Python 3.10+ (目前系统 Python 3.12.2 满足要求)
- ✅ 项目依赖已安装 (geopandas, torch, torch-geometric 等)

### 0️⃣ 初始化虚拟环境（如需要）

```bash
# 进入项目目录
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows

# 升级 pip
pip install -U pip

# 安装依赖
pip install -r requirement.txt
```

### ✅ 验证环装

```bash
# 测试 UFZ 包导入
python -c "import ufz; print('✓ UFZ 包已正确安装')"

# 测试 CLI 入口
python -m ufz --help
```

---

## 📁 项目结构速览

```
UFZ_MVGNN/
├── ufz/                    # 核心包 (可直接导入使用)
│   ├── cli.py              # CLI 命令行接口
│   ├── config/             # 配置系统
│   ├── data/               # 数据加载
│   ├── features/           # 特征计算
│   ├── graph/              # 图构建
│   ├── semantic/           # 语义增强
│   ├── models/             # 对比学习模型
│   ├── analysis/           # 聚类分析
│   ├── visualization/      # 可视化
│   ├── export/             # 结果导出
│   └── utils/              # 工具库
│
├── configs/                # 配置文件
│   └── base.yaml           # 完整配置示例
│
├── tests/                  # 单元测试
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_semantic.py
│   └── test_analysis.py
│
├── data/                   # 数据目录 (需填充实际数据)
│   ├── shp/                # 建筑 Shapefile
│   ├── poi/                # POI 数据
│   └── raster/             # 栅格数据
│
└── outputs/                # 输出目录 (自动生成)
```

---

## 🚀 运行示例

### 方式 1️⃣: 使用 CLI 命令行

最简单的方式是使用命令行工具。**注意**: 当前 CLI 实现中，训练逻辑是 stub（仅打日志），但核心算法模块完全实现。

#### 查看帮助

```bash
# 查看主帮助
python -m ufz --help

# 查看子命令帮助
python -m ufz train --help
python -m ufz cluster --help
python -m ufz export --help
python -m ufz visualize --help
```

#### 训练命令

```bash
# 训练语义增强模型
python -m ufz train --stage semantic --config configs/base.yaml

# 训练对比学习模型 (MVCL)
python -m ufz train --stage mvcl --config configs/base.yaml

# 完整流水线 (semantic + mvcl)
python -m ufz train --stage all --config configs/base.yaml

# 使用自定义参数
python -m ufz train --stage all --config configs/base.yaml \
    --device cpu --epochs 30 --seed 42
```

#### 聚类命令

```bash
python -m ufz cluster --config configs/base.yaml
```

#### 导出命令

```bash
# 导出为 GeoJSON 地图
python -m ufz export --format map --config configs/base.yaml

# 导出为知识图谱 (预留功能)
python -m ufz export --format graphrag --config configs/base.yaml
```

#### 可视化命令

```bash
# 可视化嵌入空间
python -m ufz visualize --type embedding --config configs/base.yaml

# 可视化 Delaunay 图
python -m ufz visualize --type graph --config configs/base.yaml

# 可视化聚类结果
python -m ufz visualize --type cluster --config configs/base.yaml
```

---

### 方式 2️⃣: 直接调用 Python API

更灵活的方式是在 Python 中直接导入并调用核心模块：

#### 例子 1: 加载配置

```python
from ufz.config.parser import Config

# 从 YAML 加载配置
config = Config.from_yaml('configs/base.yaml')

print(f"Device: {config.device}")
print(f"Clustering method: {config.analysis.clustering_method}")
print(f"Model backbone: {config.model.backbone}")
```

#### 例子 2: 加载数据

```python
from ufz.data.loader import load_shapefile

# 加载 Shapefile
gdf = load_shapefile('data/shp/buildings.shp')
print(f"加载了 {len(gdf)} 栋建筑物")
print(gdf.head())
```

#### 例子 3: 计算特征

```python
from ufz.features.manager import FeatureManager
from ufz.data.loader import load_shapefile

# 加载数据
gdf = load_shapefile('data/shp/buildings.shp')

# 创建特征管理器
feature_mgr = FeatureManager(groups=['shape', 'size', 'orientation'])

# 计算特征
gdf_with_features = feature_mgr.calculate_features(gdf)
print(f"计算了 {len(feature_mgr.get_feature_names())} 个特征")
print(gdf_with_features.columns)
```

#### 例子 4: 构建图

```python
from ufz.graph.builder import build_graph_from_gdf
from ufz.data.loader import load_shapefile

# 加载数据
gdf = load_shapefile('data/shp/buildings.shp')

# 构建 Delaunay 图
edge_index = build_graph_from_gdf(gdf, max_edge_length_m=200.0)
print(f"图包含 {edge_index.shape[1]} 条边")
```

#### 例子 5: 语义增强

```python
import torch
from ufz.semantic.imputer import CrossViewImputer
from ufz.semantic.trainer import SemanticTrainer
from ufz.semantic.losses import WeightedKLDivLoss

# 创建模型
model = CrossViewImputer(
    in_dim=27,           # 物理特征维度
    hidden_dim=128,
    num_classes=17,      # POI 类别数
    heads=4,
    dropout=0.3
)

# 创建损失函数
loss_fn = WeightedKLDivLoss()

# 创建训练器
trainer = SemanticTrainer(
    model=model,
    loss_fn=loss_fn,
    device='cpu',
    learning_rate=0.001
)

# 创建虚拟数据 (实际使用需要真实数据)
x = torch.randn(100, 27)              # 物理特征
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
y = torch.randint(0, 17, (100,))      # POI 标签
mask = torch.ones(100, dtype=torch.bool)

# 训练一个 epoch
loss = trainer.train_epoch([(x, edge_index, y, mask)])
print(f"Loss: {loss}")
```

#### 例子 6: 对比学习 (MVCL)

```python
import torch
from ufz.models.mvcl import MVCLModel

# 创建 MVCL 模型
model = MVCLModel(
    physical_dim=27,      # 物理特征维度
    semantic_dim=17,      # 语义特征维度
    hidden_dim=256,
    repr_dim=128,
    proj_dim=128,
    backbone='gin',       # gin | gat | gcn
    num_layers=2,
    dropout=0.5
)

# 虚拟前向传播
x_phys = torch.randn(100, 27)
x_sem = torch.randn(100, 17)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

h_phys, h_sem, z_phys, z_sem = model(x_phys, x_sem, edge_index)
print(f"物理表示: {h_phys.shape}")
print(f"语义表示: {h_sem.shape}")
print(f"对比空间: {z_phys.shape}")

# 计算对比损失
loss = model.compute_loss(z_phys, z_sem)
print(f"对比损失: {loss.item()}")
```

#### 例子 7: 聚类分析

```python
import numpy as np
from ufz.analysis.clustering import cluster_embeddings
from ufz.analysis.reducer import reduce_embeddings

# 创建虚拟嵌入
embeddings = np.random.randn(100, 128)

# HDBSCAN 聚类
labels = cluster_embeddings(
    embeddings,
    method='hdbscan',
    min_cluster_size=15
)
print(f"聚类结果: {len(set(labels))} 个簇")

# 降维到 2D
embeddings_2d = reduce_embeddings(
    embeddings,
    method='umap',
    n_components=2
)
print(f"降维后形状: {embeddings_2d.shape}")
```

#### 例子 8: 结果导出

```python
import numpy as np
from ufz.export.maps import export_geojson, export_cluster_summary

# 虚拟数据
positions = np.array([[120.0, 30.0], [120.1, 30.1], [120.2, 30.2]])
labels = np.array([0, 0, 1])
embeddings = np.random.randn(3, 128)

# 导出 GeoJSON
export_geojson(
    positions,
    labels,
    output_path='outputs/clusters.geojson',
    crs='EPSG:4326'
)

# 导出聚类统计
export_cluster_summary(
    labels,
    embeddings,
    output_path='outputs/cluster_summary.json'
)
```

---

## 📚 常见命令

### 快速参考

```bash
# 进入项目目录
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN

# 查看版本
python -c "import ufz; print(ufz.__file__)"

# 运行单元测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_config.py -v
pytest tests/test_models.py -v

# 进入 Python REPL 导入项目
python -c "from ufz.config.parser import Config; print('OK')"

# 创建新的 Python 脚本运行
cat > my_script.py << 'EOF'
from ufz.config.parser import Config
config = Config.from_yaml('configs/base.yaml')
print(config)
EOF
python my_script.py
```

### 配置覆盖

在不修改 YAML 的情况下，通过命令行参数覆盖配置：

```bash
# 设置设备和 epoch
python -m ufz train --stage semantic --config configs/base.yaml \
    --device cpu --epochs 50

# 设置随机种子
python -m ufz train --stage all --config configs/base.yaml --seed 123
```

---

## 🐛 调试技巧

### 1. 启用详细日志

```python
import logging

# 设置日志级别为 DEBUG
logging.basicConfig(level=logging.DEBUG)

from ufz.config.parser import Config
config = Config.from_yaml('configs/base.yaml')
```

### 2. 检查配置

```python
from ufz.config.parser import Config
from dataclasses import asdict

config = Config.from_yaml('configs/base.yaml')

# 打印所有配置
print(asdict(config))

# 打印特定子配置
print(config.model)
print(config.semantic)
```

### 3. 测试数据加载

```python
from ufz.data.loader import load_shapefile

try:
    gdf = load_shapefile('data/shp/buildings.shp')
    print(f"✓ 成功加载 {len(gdf)} 栋建筑")
    print(f"CRS: {gdf.crs}")
except FileNotFoundError:
    print("❌ 文件不存在，请检查路径")
except Exception as e:
    print(f"❌ 加载失败: {e}")
```

### 4. 测试导入

```bash
# 逐个测试导入
python -c "from ufz import cli; print('✓ cli')"
python -c "from ufz.config import parser; print('✓ config')"
python -c "from ufz.data import loader; print('✓ data')"
python -c "from ufz.features import manager; print('✓ features')"
python -c "from ufz.graph import builder; print('✓ graph')"
python -c "from ufz.semantic import imputer; print('✓ semantic')"
python -c "from ufz.models import mvcl; print('✓ models')"
python -c "from ufz.analysis import clustering; print('✓ analysis')"
```

---

## ❓ 常见问题

### Q1: "ModuleNotFoundError: No module named 'ufz'"

**A**: 确保你在项目根目录运行，或已将项目加入 PYTHONPATH：

```bash
# 方式 1: 使用 -m 标志运行 (推荐)
python -m ufz --help

# 方式 2: 在项目目录运行
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN
python -c "import ufz; print(ufz)"

# 方式 3: 设置环境变量
export PYTHONPATH=/Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN:$PYTHONPATH
python -c "import ufz; print(ufz)"
```

### Q2: "FileNotFoundError: configs/base.yaml"

**A**: 确保使用正确的配置文件路径：

```bash
# 检查文件是否存在
ls -la configs/

# 使用绝对路径
python -m ufz train --stage semantic \
    --config /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN/configs/base.yaml
```

### Q3: "torch.cuda.is_available() = False"

**A**: CUDA 不可用，使用 CPU 训练：

```bash
python -m ufz train --stage all --config configs/base.yaml --device cpu
```

### Q4: 内存不足

**A**: 减少批量大小或禁用某些特征：

```yaml
# configs/base.yaml
semantic:
  batch_size: 2048  # 从 4096 降低

model:
  batch_size: 2048

features:
  use_graphlet: false  # 禁用 Graphlet (计算开销大)
```

### Q5: "No data found" 错误

**A**: 需要在 `configs/base.yaml` 中指定实际的数据路径：

```yaml
data:
  shp_path: /path/to/your/buildings.shp
  poi_path: /path/to/your/poi.csv
  raster_paths:
    - /path/to/gdp.tif
    - /path/to/population.tif
```

### Q6: 如何只测试，不训练？

**A**: 运行单元测试：

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试模块
pytest tests/test_config.py -v
pytest tests/test_models.py -v
pytest tests/test_semantic.py -v
pytest tests/test_analysis.py -v
```

---

## 📖 下一步

1. **准备数据**: 将你的 Shapefile、POI 和栅格数据放入 `data/` 目录
2. **修改配置**: 编辑 `configs/base.yaml` 指定数据路径
3. **测试导入**: 运行 `python -m ufz --help` 验证安装
4. **运行训练**: 执行 `python -m ufz train --stage all --config configs/base.yaml`
5. **查看结果**: 检查 `outputs/` 和 `cache/` 目录

---

## 🔗 相关文档

- **README.md**: 完整使用指南
- **STRUCTURE.md**: 项目架构设计
- **PROJECT_SUMMARY.md**: 项目统计与总结

**最后更新**: 2026-03-09
