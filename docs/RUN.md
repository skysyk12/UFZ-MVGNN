# 🚀 UFZ_MVGNN 运行指南

## 第 1 步：安装依赖

首先，你需要安装项目所有依赖。根据你的系统和需求，选择对应的安装方式。

### 方式 A: 完整安装（推荐）

```bash
# 进入项目目录
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN

# 方法 1: 直接安装（推荐）
pip install -r requirement.txt

# 方法 2: 创建虚拟环境后安装
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows

pip install -U pip
pip install -r requirement.txt
```

### 方式 B: 最小化安装（仅配置 + 聚类）

如果不需要深度学习功能，可以安装最小依赖：

```bash
pip install numpy pandas geopandas shapely scipy pyyaml joblib scikit-learn hdbscan pytest
```

---

## 第 2 步：验证安装

```bash
# 测试 UFZ 包
python -c "import ufz; print('✓ UFZ 包正常')"

# 测试 CLI 命令
python -m ufz --help

# 测试配置
python -c "from ufz.config.parser import Config; Config.from_yaml('configs/base.yaml'); print('✓ 配置正常')"
```

---

## 第 3 步：快速测试

有 4 种方式运行代码，根据安装情况选择：

### ✅ 方式 1: 配置系统测试（最简单，推荐）

```bash
python -c "
from ufz.config.parser import Config
config = Config.from_yaml('configs/base.yaml')
print('✓ 配置加载成功')
print(f'  Device: {config.device}')
print(f'  Seed: {config.seed}')
print(f'  Backbone: {config.model.backbone}')
"
```

**输出示例**:
```
✓ 配置加载成功
  Device: auto
  Seed: 42
  Backbone: gin
```

---

### ✅ 方式 2: 数据处理测试（需要 GeoPandas）

```bash
python -c "
from ufz.data.loader import load_shapefile

# 如果你有实际的 Shapefile 数据
try:
    gdf = load_shapefile('data/shp/buildings.shp')
    print(f'✓ 加载了 {len(gdf)} 栋建筑')
except FileNotFoundError:
    print('❌ 数据文件不存在，请先准备 Shapefile')
"
```

---

### ✅ 方式 3: 聚类测试（需要 scikit-learn）

```bash
python -c "
import numpy as np
from ufz.analysis.clustering import cluster_embeddings

# 创建虚拟嵌入数据
embeddings = np.random.randn(100, 128)

# DBSCAN 聚类
labels = cluster_embeddings(embeddings, method='dbscan', eps=0.5)
print(f'✓ DBSCAN 聚类完成')
print(f'  发现 {len(set(labels))} 个簇')
"
```

**输出示例**:
```
✓ DBSCAN 聚类完成
  发现 0 个簇
```

---

### ✅ 方式 4: 完整演示（需要完整依赖）

```bash
python demo.py
```

这会运行 6 个演示模块（配置、编码器、MVCL、聚类、语义增强、导出）。

---

## 第 4 步：使用 CLI 命令

安装完整依赖后，可以使用命令行工具：

```bash
# 查看帮助
python -m ufz --help

# 查看训练命令帮助
python -m ufz train --help

# 运行训练（需要实际数据）
python -m ufz train --stage all --config configs/base.yaml --device cpu --epochs 10

# 运行聚类
python -m ufz cluster --config configs/base.yaml

# 导出结果
python -m ufz export --format map --config configs/base.yaml

# 可视化
python -m ufz visualize --type embedding --config configs/base.yaml
```

---

## 第 5 步：使用 Python API

在 Python 脚本中直接调用模块（推荐用于开发）：

### 例子 1: 加载配置

```python
from ufz.config.parser import Config

config = Config.from_yaml('configs/base.yaml')
print(config.model.backbone)
print(config.analysis.clustering_method)
```

### 例子 2: 加载数据

```python
from ufz.data.loader import load_shapefile

gdf = load_shapefile('data/shp/buildings.shp')
print(f"加载了 {len(gdf)} 栋建筑")
print(gdf.crs)
```

### 例子 3: 计算特征

```python
from ufz.features.manager import FeatureManager
from ufz.data.loader import load_shapefile

gdf = load_shapefile('data/shp/buildings.shp')
mgr = FeatureManager(groups=['shape', 'size', 'orientation'])
gdf_featured = mgr.calculate_features(gdf)
print(f"计算了 {len(mgr.get_feature_names())} 个特征")
```

### 例子 4: 进行聚类

```python
import numpy as np
from ufz.analysis.clustering import cluster_embeddings
from ufz.analysis.reducer import reduce_embeddings

embeddings = np.random.randn(1000, 128)

# 聚类
labels = cluster_embeddings(embeddings, method='kmeans', n_clusters=10)

# 降维
embeddings_2d = reduce_embeddings(embeddings, method='umap', n_components=2)

print(f"聚类: {len(set(labels))} 个簇")
print(f"降维: {embeddings_2d.shape}")
```

---

## 🔧 常见问题

### Q1: "ModuleNotFoundError: No module named 'torch'"

**A**: 需要安装 PyTorch。根据你的系统选择：

```bash
# 仅 CPU 版本（推荐先用这个测试）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU 版本（需要 CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或使用 conda（推荐）
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

### Q2: "ModuleNotFoundError: No module named 'torch_geometric'"

**A**: 安装 PyTorch Geometric：

```bash
# 首先必须安装 PyTorch
pip install torch-geometric
```

### Q3: "ModuleNotFoundError: No module named 'geopandas'"

**A**: 安装地理数据库：

```bash
pip install geopandas shapely
```

### Q4: "ModuleNotFoundError: No module named 'hdbscan'"

**A**: 安装聚类库：

```bash
pip install hdbscan
```

### Q5: "FileNotFoundError: configs/base.yaml"

**A**: 确保在项目根目录运行：

```bash
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN
python -c "from ufz.config.parser import Config; Config.from_yaml('configs/base.yaml')"
```

### Q6: 需要实际数据如何处理？

**A**: 项目需要 3 类数据：

```
data/
├── shp/          # 建筑 Shapefile (必需)
│   └── buildings.shp
│       buildings.shx
│       buildings.dbf
│       buildings.prj
├── poi/          # POI CSV 数据 (可选)
│   └── poi.csv (包含: lon, lat, poi_type 等列)
└── raster/       # 栅格数据 (可选)
    ├── gdp.tif
    └── population.tif
```

在 `configs/base.yaml` 中指定路径：

```yaml
data:
  shp_path: /path/to/your/buildings.shp
  poi_path: /path/to/your/poi.csv
  raster_paths:
    - /path/to/gdp.tif
    - /path/to/population.tif
```

---

## 📋 安装检查清单

| 功能 | 最小依赖 | 完整依赖 |
|------|---------|---------|
| 配置系统 | ✓ | ✓ |
| 聚类分析 | ✓ (scikit-learn) | ✓ |
| 数据加载 | ✓ (geopandas) | ✓ |
| 特征计算 | ✓ (geopandas) | ✓ |
| 图构建 | ✓ (scipy) | ✓ |
| 语义增强 | ✗ (需要 PyTorch) | ✓ |
| 对比学习 | ✗ (需要 PyTorch) | ✓ |
| 结果导出 | ✓ | ✓ |
| 可视化 | ✓ (plotly) | ✓ |

---

## 🎯 推荐学习路径

### 初级用户（仅学习配置）

```bash
# 1. 安装最小依赖
pip install pyyaml numpy pandas geopandas scikit-learn

# 2. 测试配置
python -c "from ufz.config.parser import Config; Config.from_yaml('configs/base.yaml')"

# 3. 阅读 QUICK_START.md
```

### 中级用户（学习数据处理和聚类）

```bash
# 1. 安装中等依赖
pip install -r requirement.txt

# 2. 跳过深度学习部分，直接用聚类
python demo.py

# 3. 按照 QUICK_START.md 学习各模块
```

### 高级用户（完整功能）

```bash
# 1. 创建虚拟环境
python -m venv .venv && source .venv/bin/activate

# 2. 安装所有依赖
pip install -r requirement.txt

# 3. 运行完整演示
python demo.py

# 4. 准备数据并运行完整流水线
python -m ufz train --stage all --config configs/base.yaml
```

---

## 📖 相关文档

- **README.md**: 完整项目说明
- **QUICK_START.md**: 详细运行指南与示例
- **STRUCTURE.md**: 项目架构设计
- **PROJECT_SUMMARY.md**: 项目统计与总结

---

## 💡 快速参考

```bash
# 检查项目位置
pwd

# 检查 Python 版本
python --version

# 检查依赖
pip list | grep -E "numpy|pandas|torch|geopandas"

# 升级 pip
pip install -U pip

# 创建虚拟环境
python -m venv .venv && source .venv/bin/activate

# 安装依赖
pip install -r requirement.txt

# 测试导入
python -c "import ufz; print('OK')"

# 运行演示
python demo.py

# 运行测试
pytest tests/ -v

# 运行命令行工具
python -m ufz --help
python -m ufz train --help
```

---

**最后更新**: 2026-03-09
