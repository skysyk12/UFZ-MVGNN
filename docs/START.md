# ✅ UFZ_MVGNN 快速启动指南

> 这是对你老项目 `/Users/sunyongkang/Downloads/find_a_job/data_drive_internship/GNN/UFZ_MVGNN` 的完全重构版本

## 🎯 核心信息

| 项 | 值 |
|----|-----|
| **项目位置** | `/Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN` |
| **虚拟环境** | `gnn_research` (位置: `/opt/anaconda3/envs/gnn_research`) |
| **Python 版本** | 3.10 |
| **关键包** | torch, torch-geometric, geopandas, yaml 等 |

---

## 🚀 3 步快速启动

### 第 1 步: 激活虚拟环境

在你的 **Terminal** 或 **Shell** 中运行：

```bash
conda activate gnn_research
```

你应该看到命令行前面出现 `(gnn_research)` 标记。

### 第 2 步: 进入项目目录

```bash
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN
```

### 第 3 步: 验证一切正常

```bash
python -m ufz --help
```

你应该看到 CLI 帮助信息，包含 4 个子命令：`train`, `cluster`, `export`, `visualize`。

---

## ✨ 常用命令速查

### 查看帮助

```bash
python -m ufz --help              # 主帮助
python -m ufz train --help        # 训练命令帮助
python -m ufz cluster --help      # 聚类命令帮助
python -m ufz export --help       # 导出命令帮助
python -m ufz visualize --help    # 可视化命令帮助
```

### 运行演示

```bash
# 方式 1: 仅测试配置 (最简单)
python -c "
from ufz.config.parser import Config
config = Config.from_yaml('configs/base.yaml')
print('✓ 配置加载成功')
"

# 方式 2: 完整演示 (展示所有模块)
python demo.py
```

### 运行测试

```bash
pytest tests/ -v                  # 全部测试
pytest tests/test_config.py -v    # 仅配置测试
pytest tests/test_models.py -v    # 仅模型测试
pytest tests/test_analysis.py -v  # 仅聚类测试
```

### 运行训练（需要数据）

```bash
# 注意: 当前 CLI 实现中训练逻辑是 stub，核心算法模块完全可用

# 语义增强
python -m ufz train --stage semantic --config configs/base.yaml

# 对比学习
python -m ufz train --stage mvcl --config configs/base.yaml

# 完整流水线
python -m ufz train --stage all --config configs/base.yaml

# 使用 CPU
python -m ufz train --stage all --config configs/base.yaml --device cpu
```

---

## 📖 文档导航

| 文件 | 说明 | 建议阅读时间 |
|------|------|-----------|
| **HOW_TO_RUN.md** | 详细运行指南（本文件更简洁版） | 5 分钟 |
| **README.md** | 完整项目文档 | 15 分钟 |
| **QUICK_START.md** | 详细快速开始和代码示例 | 20 分钟 |
| **STRUCTURE.md** | 项目架构和设计 | 15 分钟 |
| **PROJECT_SUMMARY.md** | 统计数据和项目总结 | 10 分钟 |

---

## 💻 使用项目模块（推荐）

你不需要运行完整的 CLI，可以直接在 Python 中使用各个模块。这样更灵活，更适合开发。

### 最小化示例

```python
# 创建 test.py
from ufz.config.parser import Config
from ufz.analysis.clustering import cluster_embeddings
import numpy as np

# 1. 加载配置
config = Config.from_yaml('configs/base.yaml')
print(f"Device: {config.device}")
print(f"Clustering method: {config.analysis.clustering_method}")

# 2. 创建虚拟嵌入数据
embeddings = np.random.randn(1000, 128)

# 3. 进行聚类
labels = cluster_embeddings(embeddings, method='kmeans', n_clusters=10)
print(f"聚类完成: {len(set(labels))} 个簇")
```

运行：
```bash
python test.py
```

---

## 🔑 关键模块快览

| 模块 | 用途 | 导入示例 |
|------|------|--------|
| `config` | 配置管理 | `from ufz.config.parser import Config` |
| `data` | 数据加载 | `from ufz.data.loader import load_shapefile` |
| `features` | 特征计算 | `from ufz.features.manager import FeatureManager` |
| `graph` | 图构建 | `from ufz.graph.builder import build_graph_from_gdf` |
| `semantic` | 语义增强 | `from ufz.semantic.imputer import CrossViewImputer` |
| `models` | MVCL 模型 | `from ufz.models.mvcl import MVCLModel` |
| `analysis` | 聚类+降维 | `from ufz.analysis.clustering import cluster_embeddings` |
| `export` | 结果导出 | `from ufz.export.maps import export_geojson` |

---

## ❓ 常见问题

### Q1: 虚拟环境忘记激活了

**A**: 再运行一次：
```bash
conda activate gnn_research
```

### Q2: 虚拟环境报错或损坏

**A**: 从老项目的依赖重新安装：
```bash
conda activate gnn_research
cd /Users/sunyongkang/Downloads/find_a_job/data_drive_internship/GNN/UFZ_MVGNN
pip install -r requirements.txt
```

### Q3: 导入 ufz 时出错

**A**: 确保：
1. 虚拟环境已激活
2. 在项目目录中
3. 试试 `python -m ufz --help`

### Q4: 没有 GPU 怎么办？

**A**: 使用 CPU：
```bash
python -m ufz train --stage all --config configs/base.yaml --device cpu
```

### Q5: 需要准备数据吗？

**A**: 不必要。项目可以进行模块测试。完整流水线需要 Shapefile 数据。

---

## 📊 核心架构一览

```
数据加载 → 特征计算 → 图构建 → 语义增强 → 对比学习 → 聚类 → 导出
  data/      features/   graph/   semantic/   models/  analysis/ export/
```

**特征维度**:
- 物理特征 (Geo): **27D** (形状 13D + 尺寸 2D + 高度 3D + 方向 4D + 其他 5D)
- 拓扑特征 (Topo): **15D** (Graphlet Orbit)
- 语义特征 (Sem): **17D** (POI 类别)
- 最终表示: **128D** (对比学习后融合)

---

## 🎓 推荐学习路径

### 5 分钟快速体验
```bash
conda activate gnn_research
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN
python -m ufz --help
```

### 15 分钟深度了解
```bash
python demo.py                    # 演示各模块
python -c "from ufz.config.parser import Config; import json; from dataclasses import asdict; cfg = Config.from_yaml('configs/base.yaml'); print(json.dumps(asdict(cfg), indent=2, default=str))"  # 查看完整配置
```

### 30 分钟开发上手
```bash
# 创建 my_project.py
cat > my_project.py << 'EOF'
from ufz.config.parser import Config
from ufz.analysis.clustering import cluster_embeddings
import numpy as np

config = Config.from_yaml('configs/base.yaml')
embeddings = np.random.randn(5000, 128)
labels = cluster_embeddings(embeddings, method='hdbscan', min_cluster_size=15)
print(f"Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
EOF

python my_project.py
```

### 1 小时完整学习
- 阅读 README.md 了解项目
- 阅读 QUICK_START.md 学习各模块 API
- 运行 demo.py 看代码示例
- 自己编写一个完整脚本

---

## 🎯 你可以做的事

### 立即可做

- ✅ 加载和修改配置
- ✅ 进行聚类分析
- ✅ 降维（PCA/UMAP）
- ✅ 导出结果为 GeoJSON
- ✅ 运行单元测试

### 准备好数据后可做

- 🔜 加载 Shapefile 和计算特征
- 🔜 构建 Delaunay 图
- 🔜 运行语义增强训练
- 🔜 运行 MVCL 对比学习
- 🔜 完整的 end-to-end 流水线

### 高级扩展

- 🚀 添加新特征（通过 @FeatureRegistry）
- 🚀 添加新 GNN 编码器（通过 @BackboneRegistry）
- 🚀 添加新聚类算法
- 🚀 集成到自己的项目

---

## 📞 快速帮助

```bash
# 查看项目大小和行数
find . -name "*.py" -type f | xargs wc -l | tail -1

# 查看所有配置
python -c "from ufz.config.parser import Config; print(Config.from_yaml('configs/base.yaml'))"

# 检查导入
python -c "from ufz import *; print('All imports OK')"

# 列出测试
pytest tests/ --collect-only

# 清理缓存
rm -rf cache/* outputs/*
```

---

## 🔗 相关资源

- **老项目** (参考): `/Users/sunyongkang/Downloads/find_a_job/data_drive_internship/GNN/UFZ_MVGNN`
- **新项目** (当前): `/Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN`
- **虚拟环境**: `conda activate gnn_research`

---

## ⏰ 预计时间

| 任务 | 时间 |
|------|------|
| 激活环境 + 查看帮助 | 1 分钟 |
| 运行演示 | 5 分钟 |
| 阅读 README | 15 分钟 |
| 第一个脚本 | 10 分钟 |
| 深入学习 | 1-2 小时 |

---

**现在就开始吧！** 🚀

```bash
conda activate gnn_research
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN
python -m ufz --help
```

---

**最后更新**: 2026-03-09
