# UFZ_MVGNN 重构目录结构

## 设计原则

1. **单一入口**: `python -m ufz <command>` 统一所有操作
2. **按职责分模块**: 每个目录只做一件事
3. **流水线即配置**: YAML 驱动，代码不因实验而改动
4. **语义增强一体化**: Imputer + RefineNet 合并为 `semantic/` 模块
5. **预留 GraphRAG 出口**: `export/` 模块负责将结果转化为 LLM 可用格式

---

## 完整流程映射

```
数据加载        特征工程         语义增强              对比学习           聚类          导出
data/        → features/     → semantic/           → models/         → analysis/  → export/
 loader.py     shape.py        imputer.py            mvcl.py          cluster.py    graphrag.py
 raster.py     size.py         refine_net.py         tower.py         reducer.py    schema.py
               orientation.py  idw.py                losses.py
               graphlet.py                           trainer.py
               density.py
```

---

## 目录结构

```
UFZ_MVGNN/
│
├── ufz/                          # 核心包 (原 src/)
│   ├── __init__.py
│   ├── __main__.py               # python -m ufz 入口
│   ├── cli.py                    # CLI 子命令定义 (argparse/click)
│   │
│   ├── config/                   # 配置解析
│   │   ├── __init__.py
│   │   └── parser.py             # YAML → Dataclass, 统一配置模型
│   │
│   ├── data/                     # 数据加载 (I/O 层, 只负责读和预处理)
│   │   ├── __init__.py
│   │   ├── loader.py             # Shapefile 加载 + CRS 转换
│   │   ├── poi.py                # POI 空间匹配 → GT 标签 (sjoin)
│   │   ├── raster.py             # 栅格特征提取 (GDP/人口)
│   │   └── sampler.py            # 空间采样 (可选, 用于调试)
│   │
│   ├── features/                 # 特征计算 (纯计算, 无 I/O)
│   │   ├── __init__.py
│   │   ├── registry.py           # 特征注册表 (@register 装饰器)
│   │   ├── manager.py            # 特征管理器 (按 config 调度计算)
│   │   ├── shape.py              # 形状特征 (13D)
│   │   ├── size.py               # 尺寸特征 (2D+height)
│   │   ├── orientation.py        # 方向特征 (4D)
│   │   ├── density.py            # 密度特征
│   │   ├── graphlet.py           # Graphlet orbit (15D, ORCA)
│   │   └── processing.py         # 后处理: Z-score 归一化等
│   │
│   ├── graph/                    # 图构建
│   │   ├── __init__.py
│   │   └── builder.py            # Delaunay 三角剖分 + 分位数自动剪枝
│   │
│   ├── semantic/                 # 语义增强 (原 imputer + refine_net + idw)
│   │   ├── __init__.py
│   │   ├── idw.py                # Gaussian IDW 语义扩散
│   │   ├── imputer.py            # CrossViewImputer (物理→粗预测)
│   │   ├── refine_net.py         # RefineNet (IDW + 门控融合 → 密集预测)
│   │   ├── losses.py             # WeightedKLDivLoss (语义增强专用损失)
│   │   └── trainer.py            # 语义增强训练器 (Imputer+RefineNet 串联)
│   │
│   ├── models/                   # 对比学习模型
│   │   ├── __init__.py
│   │   ├── backbones/            # GNN 编码器
│   │   │   ├── __init__.py
│   │   │   ├── registry.py       # Backbone 注册表
│   │   │   ├── base.py           # BaseEncoder 抽象类
│   │   │   ├── gin.py
│   │   │   ├── gat.py
│   │   │   └── gcn.py
│   │   ├── mvcl.py               # MVCLModel (双编码器 + 投影头)
│   │   ├── tower.py              # MultiTowerEncoder (Geo/Topo/Sem 三塔)
│   │   ├── losses.py             # InfoNCE / DGI 对比损失
│   │   └── trainer.py            # MVCLTrainer (对比学习训练循环)
│   │
│   ├── analysis/                 # 聚类 + 降维
│   │   ├── __init__.py
│   │   ├── clustering.py         # PCA → L2 → HDBSCAN/DBSCAN/KMeans/Leiden
│   │   └── reducer.py            # PCA / UMAP 降维封装
│   │
│   ├── export/                   # 结果导出 + GraphRAG (未来)
│   │   ├── __init__.py
│   │   ├── checkpoint.py         # 模型/嵌入/标签保存加载
│   │   ├── maps.py               # 聚类地图导出 (原 export_cluster_maps.py)
│   │   ├── graphrag.py           # (预留) 图结构 → GraphRAG 知识图谱
│   │   └── schema.py             # (预留) 结构化描述 → LLM prompt 模板
│   │
│   ├── visualization/            # 可视化
│   │   ├── __init__.py
│   │   ├── graph_viz.py          # Delaunay 图可视化
│   │   ├── embedding_viz.py      # UMAP 嵌入可视化
│   │   └── cluster_viz.py        # 聚类结果可视化
│   │
│   └── utils/                    # 通用工具
│       ├── __init__.py
│       ├── logging.py            # 日志配置
│       ├── seed.py               # 随机种子
│       └── cache.py              # 统一缓存管理 (替代散落各处的缓存逻辑)
│
├── configs/                      # 配置文件 (精简)
│   ├── default.yaml              # 默认完整配置 (带注释, 原 base.yaml)
│   ├── local.yaml                # 本地测试 override (少量节点, CPU)
│   └── server.yaml               # 服务器生产 override (全量, GPU)
│
├── scripts/                      # 便捷脚本 (非核心, 可选)
│   └── run_pipeline.sh           # 一键串联三阶段 (shell wrapper)
│
├── tests/                        # 测试
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_graph.py
│   ├── test_semantic.py
│   └── test_models.py
│
├── data/                         # 原始数据 (不进 git)
│   ├── shp/                      # 建筑物 Shapefile
│   ├── poi/                      # POI 数据 (CSV 目录)
│   └── raster/                   # 栅格数据 (GDP/人口)
│
├── cache/                        # 计算缓存 (不进 git)
├── outputs/                      # 训练输出 (不进 git)
├── orca_src/                     # ORCA C++ 源码 (第三方)
│
├── pyproject.toml                # 项目元数据 + 依赖 (替代 requirements.txt)
├── .gitignore
└── README.md
```

---

## 与旧结构的映射关系

| 旧位置 | 新位置 | 变化说明 |
|---|---|---|
| `src/` | `ufz/` | 改为可安装包, 支持 `python -m ufz` |
| `src/config/parser.py` | `ufz/config/parser.py` | 合并双配置为统一配置模型 |
| `src/data/pipeline.py` | 删除 | 管线逻辑移入 `cli.py` 的子命令 |
| `src/data/views.py` | 删除 | 视图分离逻辑内化到 `models/mvcl.py` |
| `src/data/components/loader.py` | `ufz/data/loader.py` | 扁平化, 去掉 components 层 |
| `src/data/components/poi.py` | `ufz/data/poi.py` | 仅保留空间匹配, Bag-of-Tags 删除 |
| `src/data/components/idw.py` | `ufz/semantic/idw.py` | 移入语义增强模块 |
| `src/data/components/graph.py` | `ufz/graph/builder.py` | 与 graph/ 合并 |
| `src/data/components/processor.py` | `ufz/features/processing.py` | 归一化逻辑移入 features |
| `src/data/components/raster.py` | `ufz/data/raster.py` | 保持 |
| `src/data/components/base.py` | 删除 | 基类过度抽象, 内联到具体类 |
| `src/models/imputer.py` | `ufz/semantic/imputer.py` | 移入语义增强模块 |
| `src/models/refine_net.py` | `ufz/semantic/refine_net.py` | 移入语义增强模块 |
| `src/models/losses.py` | `ufz/models/losses.py` (InfoNCE) + `ufz/semantic/losses.py` (KL) | 按职责拆分 |
| `src/utils/checkpoint.py` | `ufz/export/checkpoint.py` | 移入导出模块 |
| `scripts/train.py` | 删除 | 原始 MVCL 入口, 功能合并到 cli.py |
| `scripts/train_imputer.py` | 删除 | 合并到 `python -m ufz train --stage semantic` |
| `scripts/train_refine.py` | 删除 | 同上 |
| `scripts/train_mvcl_refined.py` | 删除 | 合并到 `python -m ufz train --stage mvcl` |
| `scripts/export_cluster_maps.py` | `ufz/export/maps.py` | 移入导出模块 |
| `scripts/test.py` | `tests/` | 移入测试目录 |
| `poi_cut/` | 删除 | 一次性工具, 不属于核心代码 |
| `configs/*.yaml` (12个) | `configs/` (3个) | 精简为 default + local + server |
| `requirements.txt` | `pyproject.toml` | 现代化依赖管理 |
| `SKILLS.md` | 删除 | 重构后用 README 覆盖 |

---

## CLI 命令设计

```bash
# === 完整流水线 ===
python -m ufz train --stage all --config configs/server.yaml

# === 分步执行 ===
python -m ufz train --stage semantic --config configs/server.yaml   # Imputer + RefineNet
python -m ufz train --stage mvcl --config configs/server.yaml       # 对比学习
python -m ufz cluster --config configs/server.yaml                  # 聚类 (可单独重跑)

# === 导出 ===
python -m ufz export --format map                                   # 聚类地图
python -m ufz export --format graphrag                              # GraphRAG (未来)

# === 调试 ===
python -m ufz train --stage all --config configs/local.yaml         # 本地少量数据

# === 参数 override ===
python -m ufz train --stage mvcl --config configs/server.yaml \
    --epochs 300 --device cuda --seed 42
```

---

## GraphRAG 模块预留设计 (export/)

```
export/
├── graphrag.py    # 将聚类结果 + 图结构转为知识图谱
│                  #   - 节点: 功能区 (聚类标签 + 空间位置 + 特征摘要)
│                  #   - 边: 空间邻接 / 功能相似
│                  #   - 属性: POI 分布, 物理特征统计
│                  #
└── schema.py      # 结构化描述生成
                   #   - 功能区描述模板: "该区域以{主导POI}为主, 建筑{形态特征}..."
                   #   - LLM prompt 模板: 支持问答型查询
                   #   - 检索接口: 按空间 / 按功能 / 按相似度检索
```

预期使用方式:
```python
from ufz.export.graphrag import build_knowledge_graph
from ufz.export.schema import generate_zone_description

# 构建知识图谱
kg = build_knowledge_graph(
    embeddings="outputs/embeddings.npy",
    labels="outputs/cluster_labels.npy",
    graph="cache/edge_index.pt",
    features="outputs/features.csv"
)

# 生成功能区描述
desc = generate_zone_description(kg, zone_id=3)
# → "功能区3: 位于朝阳区东部, 以餐饮服务和购物为主导, 建筑形态以中高层商业综合体为主..."
```

---

## 配置精简策略

**旧 (12个文件)** → **新 (3个文件)**

### default.yaml
包含所有参数的完整定义和注释, 是唯一的参考文档

### local.yaml
仅 override 调试相关参数:
```yaml
_base_: default.yaml
data:
  max_buildings: 500
training:
  device: cpu
  epochs: 30
```

### server.yaml
仅 override 生产参数:
```yaml
_base_: default.yaml
training:
  device: cuda
  batch_size: 4096
  epochs: 200
```
