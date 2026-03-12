# 🚀 如何运行 UFZ_MVGNN

## 快速开始

你已经有一个 `gnn_research` 虚拟环境，里面包含了所有需要的包。直接使用它就可以运行这个新项目。

### 1️⃣ 激活虚拟环境

```bash
# 方法 1: 使用 conda (推荐，如果你用的是 conda 创建的)
conda activate gnn_research

# 方法 2: 使用绝对路径激活 (如果是 virtualenv)
source /path/to/gnn_research/bin/activate
```

激活后，你应该看到：
```
(gnn_research) sunyongkang@...
```

### 2️⃣ 进入项目目录

```bash
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN
```

### 3️⃣ 验证环境

```bash
# 检查 Python
python --version
# 应该输出: Python 3.10.x (或你 conda 创建时的版本)

# 检查关键包
python -c "import torch; print(torch.__version__)"
python -c "import torch_geometric; print(torch_geometric.__version__)"
python -c "import geopandas; print(geopandas.__version__)"
```

### 4️⃣ 验证项目导入

```bash
# 测试项目包导入
python -c "import ufz; print('✓ UFZ 包导入成功')"

# 测试 CLI
python -m ufz --help
```

### 5️⃣ 运行演示

最简单的方式 - 仅测试配置（无需任何依赖）：

```bash
python -c "
from ufz.config.parser import Config
config = Config.from_yaml('configs/base.yaml')
print('✓ 配置加载成功')
print(f'Device: {config.device}')
print(f'Backbone: {config.model.backbone}')
"
```

更完整的演示 - 展示所有模块（需要依赖）：

```bash
python demo.py
```

---

## 详细命令

### 查看帮助

```bash
# 主帮助
python -m ufz --help

# 子命令帮助
python -m ufz train --help
python -m ufz cluster --help
python -m ufz export --help
python -m ufz visualize --help
```

### 训练模型

注意：当前 CLI 中的训练逻辑是 stub 实现（仅打日志），但核心算法模块完全可用。

```bash
# 语义增强训练
python -m ufz train --stage semantic --config configs/base.yaml

# MVCL 对比学习训练
python -m ufz train --stage mvcl --config configs/base.yaml

# 完整流水线
python -m ufz train --stage all --config configs/base.yaml

# 使用 CPU（如果没有 GPU）
python -m ufz train --stage all --config configs/base.yaml --device cpu

# 自定义参数
python -m ufz train --stage mvcl --config configs/base.yaml \
    --epochs 300 --device cuda --seed 42
```

### 聚类

```bash
python -m ufz cluster --config configs/base.yaml
```

### 导出结果

```bash
# 导出为 GeoJSON
python -m ufz export --format map --config configs/base.yaml

# 导出为知识图谱 (预留功能)
python -m ufz export --format graphrag --config configs/base.yaml
```

### 可视化

```bash
# 嵌入可视化
python -m ufz visualize --type embedding --config configs/base.yaml

# Delaunay 图可视化
python -m ufz visualize --type graph --config configs/base.yaml

# 聚类结果可视化
python -m ufz visualize --type cluster --config configs/base.yaml
```

### 运行测试

```bash
# 全部测试
pytest tests/ -v

# 特定模块测试
pytest tests/test_config.py -v
pytest tests/test_models.py -v
pytest tests/test_semantic.py -v
pytest tests/test_analysis.py -v
```

---

## Python API 使用

在 Python 脚本中直接调用模块（更灵活）：

### 例子 1: 加载配置

```python
from ufz.config.parser import Config

config = Config.from_yaml('configs/base.yaml')
print(f"Device: {config.device}")
print(f"Model: {config.model.backbone}")
print(f"Clustering: {config.analysis.clustering_method}")
```

### 例子 2: 数据加载和特征计算

```python
from ufz.data.loader import load_shapefile
from ufz.features.manager import FeatureManager

# 加载 Shapefile
gdf = load_shapefile('data/shp/buildings.shp')
print(f"加载了 {len(gdf)} 栋建筑")

# 计算特征
mgr = FeatureManager(groups=['shape', 'size', 'orientation'])
gdf_featured = mgr.calculate_features(gdf)
print(f"计算了 {len(mgr.get_feature_names())} 个特征")
```

### 例子 3: 图构建

```python
from ufz.graph.builder import build_graph_from_gdf
from ufz.data.loader import load_shapefile

gdf = load_shapefile('data/shp/buildings.shp')
edge_index = build_graph_from_gdf(gdf, max_edge_length_m=200.0)
print(f"图包含 {edge_index.shape[1]} 条边")
```

### 例子 4: MVCL 模型

```python
import torch
from ufz.models.mvcl import MVCLModel

model = MVCLModel(
    physical_dim=27,
    semantic_dim=17,
    hidden_dim=256,
    repr_dim=128,
    proj_dim=128,
    backbone='gin',
    num_layers=2
)

# 虚拟数据
x_phys = torch.randn(100, 27)
x_sem = torch.randn(100, 17)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

# 前向传播
h_phys, h_sem, z_phys, z_sem = model(x_phys, x_sem, edge_index)
loss = model.compute_loss(z_phys, z_sem)
print(f"Loss: {loss.item():.4f}")
```

### 例子 5: 聚类分析

```python
import numpy as np
from ufz.analysis.clustering import cluster_embeddings
from ufz.analysis.reducer import reduce_embeddings

# 虚拟嵌入
embeddings = np.random.randn(1000, 128)

# 聚类
labels = cluster_embeddings(
    embeddings,
    method='hdbscan',
    min_cluster_size=15
)
print(f"聚类: {len(set(labels))} 个簇")

# 降维
embeddings_2d = reduce_embeddings(
    embeddings,
    method='umap',
    n_components=2
)
print(f"降维: {embeddings_2d.shape}")
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `README.md` | 完整项目文档 |
| `QUICK_START.md` | 详细快速开始指南 |
| `STRUCTURE.md` | 项目架构设计 |
| `PROJECT_SUMMARY.md` | 项目统计与总结 |
| `demo.py` | 演示脚本（展示各模块） |
| `configs/base.yaml` | 完整配置示例 |

---

## 常见问题

### Q: 虚拟环境失效怎么办？

**A**: 重新创建虚拟环境：

```bash
# 如果用 conda
conda create -n gnn_research python=3.10
conda activate gnn_research
pip install -r /Users/sunyongkang/Downloads/find_a_job/data_drive_internship/GNN/UFZ_MVGNN/requirements.txt

# 或者直接复用老项目的依赖
cd /Users/sunyongkang/Downloads/find_a_job/data_drive_internship/GNN/UFZ_MVGNN
pip install -r requirements.txt
```

### Q: 需要使用 GPU 吗？

**A**: 不必要。默认 `device: auto` 会自动检测。如果没有 GPU，使用 CPU 也可以（只是速度慢）：

```bash
python -m ufz train --stage all --config configs/base.yaml --device cpu
```

### Q: 需要准备实际数据吗？

**A**: 不需要。项目可以进行模块测试。但要完整运行流水线，需要在 `configs/base.yaml` 中指定：

```yaml
data:
  shp_path: /path/to/buildings.shp
  poi_path: /path/to/poi.csv
  raster_paths:
    - /path/to/gdp.tif
```

### Q: 如何自己写代码使用这个项目？

**A**: 参考上面的 Python API 示例，创建一个 `.py` 文件，导入需要的模块即可：

```python
# my_script.py
from ufz.config.parser import Config
from ufz.analysis.clustering import cluster_embeddings
import numpy as np

# 你的代码
```

然后运行：

```bash
python my_script.py
```

---

## 关键文件位置

```
/Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN/
├── ufz/                    # 核心包
├── configs/                # 配置文件
├── tests/                  # 测试
├── demo.py                 # 演示脚本
├── README.md               # 项目说明
├── QUICK_START.md          # 快速开始
└── configs/base.yaml       # 配置示例
```

---

## 推荐学习路径

### 快速了解 (5 分钟)

```bash
conda activate gnn_research
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN
python -c "from ufz.config.parser import Config; Config.from_yaml('configs/base.yaml'); print('✓ 项目正常')"
```

### 查看帮助 (5 分钟)

```bash
python -m ufz --help
python -m ufz train --help
```

### 运行演示 (5-10 分钟)

```bash
python demo.py
```

### 阅读文档 (15 分钟)

- README.md - 项目概览
- QUICK_START.md - 详细示例
- STRUCTURE.md - 架构设计

### 自己编写代码 (30+ 分钟)

根据需要编写 Python 脚本，利用各个模块。

---

## 快速参考

```bash
# 激活环境
conda activate gnn_research

# 进入项目
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN

# 验证
python -m ufz --help

# 演示
python demo.py

# 测试
pytest tests/ -v

# 查看配置
python -c "from ufz.config.parser import Config; import json; from dataclasses import asdict; print(json.dumps(asdict(Config.from_yaml('configs/base.yaml')), indent=2, default=str))"

# 测试聚类
python -c "import numpy as np; from ufz.analysis.clustering import cluster_embeddings; e = np.random.randn(100, 128); l = cluster_embeddings(e, method='kmeans', n_clusters=5); print(f'✓ 聚类: {len(set(l))} 簇')"
```

---

**最后更新**: 2026-03-09

**项目位置**: `/Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN`

**虚拟环境**: `gnn_research`
