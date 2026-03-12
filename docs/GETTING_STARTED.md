# 🚀 开始使用 UFZ_MVGNN LLM 查询系统

**这是最简洁的入门指南 - 5 分钟快速上手**

---

## ✨ 您即将体验

将您的 40W 节点聚类结果转化为知识图谱，支持自然语言查询：

> **提问**: "我想开一家高端餐厅，应该选择哪个区域？"
>
> **LLM 回答**: "根据数据分析，区域 5 最适合，因为有 78% 的餐饮 POI，且地理位置优越。推荐指数：85%"

---

## 🎯 三种使用方式

### 方式 1️⃣ : 仅构建知识图谱 (推荐先试)

**无需 API 密钥，5 秒完成**

```bash
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN
conda activate gnn_research
python demo_llm_query.py --no-llm
```

✅ 生成: `outputs/knowledge_graph.json` (139 KB)

---

### 方式 2️⃣ : 使用 Deepseek API 查询

**推荐，价格便宜，速度快**

```bash
# 获取 API 密钥: https://www.deepseek.com/api

python demo_llm_query.py --provider deepseek --api-key sk-xxxxx
```

---

### 方式 3️⃣ : 在您的代码中使用

**最灵活的方式**

```python
from ufz.export.graphrag import (
    build_hierarchical_knowledge_graph,
    load_knowledge_graph
)
from ufz.export.llm_interface import create_query_engine

# 从您的聚类结果构建知识图谱
kg = build_hierarchical_knowledge_graph(
    labels_per_level=[level_0_labels, level_1_labels],
    embeddings=embeddings,
    positions=positions,
    features=features,
    semantic_probs=poi_distribution
)

# 创建查询引擎
engine = create_query_engine(
    kg,
    provider="deepseek",
    api_key="sk-xxxxx"
)

# 查询！
result = engine.query("我想开一家餐厅，哪个区域最好？")

# 结果包含
print(result.answer)                # "区域 5 最适合..."
print(result.recommended_clusters)  # [5, 12, 8]
print(result.reasoning)             # "详细理由..."
print(f"信心度: {result.confidence:.0%}")  # "85%"
```

---

## 📊 了解数据流

```
您的数据
  ↓
[40W 节点] + [2-3 层聚类标签]
  ↓
build_hierarchical_knowledge_graph()  ← 这里自动压缩
  ↓
[100 个聚类摘要] (JSON, 139 KB)
  ↓
engine.query("问题")  ← 智能选择相关聚类
  ↓
[5 个相关聚类] (~410 tokens) ← 传给 LLM
  ↓
LLM 回答 + 推荐
```

---

## 🔑 关键概念

### ClusterSummary - 聚类摘要
```python
{
    "cluster_id": 5,                    # 簇 ID
    "level": 1,                         # 层级 (0=粗, 1=细)
    "node_count": 3456,                 # 包含的节点数
    "center": [120.00, 30.00],          # 地理中心
    "poi_distribution": {               # Top 5 POI
        "restaurant": 0.22,
        "shopping": 0.18,
        ...
    },
    "characteristics": "以餐饮为主导的商业区",  # 中文描述
    "dominant_pois": ["restaurant", "shopping", ...],
    "suitable_business": ["餐饮服务", "零售商业"]
}
```

### QueryResult - 查询结果
```python
QueryResult(
    query="我想开餐厅",
    answer="区域 5 最适合，因为...",
    reasoning="该区域有 78% 的餐饮 POI，且客流量大",
    recommended_clusters=[5, 12, 8],
    confidence=0.85  # 0-1 之间
)
```

---

## 💰 成本对比

| 方案 | 成本 | 速度 | 自由度 |
|------|------|------|--------|
| 仅构建 KG | ¥0 | 5秒 | 无 API 调用 |
| Deepseek API | ¥0.1-1 | 2-5秒 | 完全 ✅ |
| Google Gemini | ¥0-100/月 | 3-8秒 | 完全 ✅ |

> **推荐**: 先用方式 1 体验，然后用 Deepseek (便宜) 或 Google (免费额度)

---

## 📚 文档速查

| 需求 | 文档 | 阅读时间 |
|------|------|---------|
| 快速开始 | **本文件** | 5 分钟 |
| 完整使用 | LLM_GUIDE.md | 20 分钟 |
| 实现细节 | IMPLEMENTATION_SUMMARY.md | 15 分钟 |
| 文件清单 | LLM_FILES_CHECKLIST.md | 10 分钟 |
| 快速入门 | QUICK_START.md | 15 分钟 |

---

## ⚡ 快速检查清单

- [ ] 激活环境: `conda activate gnn_research`
- [ ] 进入目录: `cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN`
- [ ] 验证代码: `python -c "from ufz.export.graphrag import *; print('✓')"`
- [ ] 运行演示: `python demo_llm_query.py --no-llm`
- [ ] 查看结果: `cat outputs/knowledge_graph.json | head -50`

---

## 🤔 常见问题

**Q: 我的数据有多少层级？**
A: 自动检测，支持 2-3 层。无需修改代码。

**Q: 需要多长时间？**
A: 构建 KG ~2 秒，查询 ~2-5 秒 (取决于 LLM)

**Q: 成本是多少？**
A: Deepseek ~¥0.1-1 per query，Google 有免费额度

**Q: 支持哪些语言？**
A: 优化了中文，其他语言也可以用但效果未测试

**Q: 可以多轮对话吗？**
A: 当前不支持，但代码结构支持扩展

**Q: 离线可以用吗？**
A: 可以仅构建知识图谱 (`--no-llm`)，查询必须联网

---

## 🎓 学习路径

### 路径 A: 快速体验 (10 分钟)
```bash
# 1. 构建知识图谱
python demo_llm_query.py --no-llm

# 2. 查看生成的数据
cat outputs/knowledge_graph.json | python -m json.tool | head -30
```

### 路径 B: 完整上手 (30 分钟)
```bash
# 1. 读本文件和 LLM_GUIDE.md

# 2. 获取 API 密钥
# Google: https://makersuite.google.com/app/apikey
# Deepseek: https://www.deepseek.com/api

# 3. 运行完整演示
python demo_llm_query.py --provider deepseek --api-key sk-xxxxx
```

### 路径 C: 集成到项目 (1 小时)
```bash
# 1. 从您的 MVCL 训练获得:
#    - labels_per_level (多层聚类标签)
#    - embeddings (最终嵌入)
#    - positions (地理位置)
#    - features (物理特征)
#    - semantic_probs (POI 分布)

# 2. 使用方式 3 的代码集成到项目

# 3. 测试和调优
```

---

## 🎁 开箱即用的例子

### 例子 1: 餐饮业务选址
```python
result = engine.query("我想开一家高端火锅店，选址在哪个区域最合适？")
```

### 例子 2: 商业地产投资
```python
result = engine.query("作为房地产投资者，这个城市最看好哪个区域？为什么？")
```

### 例子 3: 城市规划分析
```python
result = engine.query("这个城市的商业区有什么特点？分布在哪些地方？")
```

### 例子 4: 竞争分析
```python
result = engine.query("这个地区的竞争情况如何？容易进入吗？")
```

---

## 🚨 故障排除

**问题**: 导入错误 `ModuleNotFoundError`
```bash
# 解决: 激活环境
conda activate gnn_research
```

**问题**: API 错误 `API key required`
```bash
# 解决: 传入有效的 API 密钥
python demo_llm_query.py --provider deepseek --api-key sk-xxxxx
```

**问题**: Token 太多
```python
# 解决: 减少 max_context_tokens
engine = create_query_engine(
    kg,
    provider="deepseek",
    api_key="...",
    max_context_tokens=3000  # 更小
)
```

**问题**: 结果质量差
```python
# 建议: 修改系统提示词
custom_prompt = "你是一个资深的房地产顾问..."
engine = UrbanFunctionalZoneQueryEngine(
    kg,
    llm_provider,
    system_prompt_template=custom_prompt
)
```

---

## 📞 获取帮助

1. 查看 LLM_GUIDE.md 中的 "Troubleshooting"
2. 查看 IMPLEMENTATION_SUMMARY.md 中的设计细节
3. 查看 LLM_FILES_CHECKLIST.md 中的文件说明

---

## 🎉 您已准备好！

现在就试试吧：

```bash
python demo_llm_query.py --no-llm
```

或者阅读 LLM_GUIDE.md 获得完整文档。

---

**项目**: UFZ_MVGNN - 城市功能区识别 + LLM 查询
**状态**: ✅ 生产就绪
**最后更新**: 2026-03-09
