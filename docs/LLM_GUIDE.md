# 🤖 LLM Query Guide - Urban Functional Zone Analysis

This guide explains how to use the LLM integration to query urban functional zones based on hierarchical clustering results.

## Overview

The LLM query system transforms hierarchical clustering results into a **Knowledge Graph**, then enables natural language queries like:

> **User**: "我想开一家高端餐厅，应该选择哪个区域？"
> **LLM**: "根据分析，区域 5 最适合，因为有 78% 的餐饮相关 POI，且地理位置优越..."

### How It Works

```
Hierarchical Clustering Data (40W nodes, 2-3 levels)
           ↓
   Build Knowledge Graph (ClusterSummary objects)
           ↓
    Serialize to JSON (139 KB compressed)
           ↓
  LLM Query Pipeline (relevance scoring + context selection)
           ↓
  LLM Response (answer + reasoning + recommendations)
```

---

## Quick Start

### 1. Build a Knowledge Graph

```python
import numpy as np
from ufz.export.graphrag import build_hierarchical_knowledge_graph, save_knowledge_graph

# Your hierarchical clustering results
labels_per_level = [level_0_labels, level_1_labels, ...]  # List of np.ndarray
embeddings = np.random.randn(40000, 128)  # [N, D]
positions = np.random.randn(40000, 2) * 0.1 + [120.0, 30.0]  # [N, 2]
features = np.random.randn(40000, 27)  # [N, 27] - physical features
semantic_probs = np.random.dirichlet(np.ones(17), 40000)  # [N, 17] - POI distribution

# Build the knowledge graph
kg = build_hierarchical_knowledge_graph(
    labels_per_level=labels_per_level,
    embeddings=embeddings,
    positions=positions,
    features=features,
    semantic_probs=semantic_probs,
    poi_names=['restaurant', 'shopping', 'office', 'hotel', ...],
)

# Save for later use
save_knowledge_graph(kg, 'knowledge_graph.json')
```

### 2. Create a Query Engine

```python
from ufz.export.graphrag import load_knowledge_graph
from ufz.export.llm_interface import create_query_engine

# Load saved knowledge graph
kg = load_knowledge_graph('knowledge_graph.json')

# Create query engine with Google Gemini
engine = create_query_engine(
    kg,
    provider="google",
    api_key="YOUR_GOOGLE_API_KEY",
    max_context_tokens=6000
)

# Or with Deepseek
engine = create_query_engine(
    kg,
    provider="deepseek",
    api_key="YOUR_DEEPSEEK_API_KEY",
    max_context_tokens=6000
)
```

### 3. Query

```python
result = engine.query("我想开一家高端餐厅，应该选择哪个区域？")

print(result.answer)               # 推荐答案
print(result.recommended_clusters) # [5, 12, 8] - 推荐的簇 ID
print(result.reasoning)            # 详细理由说明
print(f"Confidence: {result.confidence:.0%}")
```

---

## Full Example

Here's a complete end-to-end example:

```python
import numpy as np
from ufz.export.graphrag import (
    build_hierarchical_knowledge_graph,
    save_knowledge_graph,
    load_knowledge_graph
)
from ufz.export.llm_interface import create_query_engine

# ============================================================================
# STEP 1: Create synthetic hierarchical clustering data
# ============================================================================
print("Step 1: Creating hierarchical clustering data...")

np.random.seed(42)
n_nodes = 40000

# Level 0: 10 coarse clusters
level_0_labels = np.random.randint(0, 10, n_nodes)

# Level 1: 100 fine clusters
level_1_labels = np.zeros(n_nodes, dtype=int)
for i in range(10):
    mask = level_0_labels == i
    level_1_labels[mask] = i * 10 + np.random.randint(0, 10, mask.sum())

# Create other data
embeddings = np.random.randn(n_nodes, 128)
positions = np.random.randn(n_nodes, 2) * 0.1 + np.array([120.0, 30.0])
features = np.random.randn(n_nodes, 27)
semantic_probs = np.random.dirichlet(np.ones(17), n_nodes)

poi_names = [
    'restaurant', 'shopping', 'office', 'hotel', 'hospital',
    'school', 'park', 'bank', 'supermarket', 'gym',
    'cafe', 'bar', 'cinema', 'library', 'museum', 'pharmacy', 'salon'
]

print(f"  ✓ Created {n_nodes} nodes with 2 hierarchy levels")

# ============================================================================
# STEP 2: Build and save knowledge graph
# ============================================================================
print("Step 2: Building knowledge graph...")

kg = build_hierarchical_knowledge_graph(
    labels_per_level=[level_0_labels, level_1_labels],
    embeddings=embeddings,
    positions=positions,
    features=features,
    semantic_probs=semantic_probs,
    poi_names=poi_names,
)

save_knowledge_graph(kg, 'my_knowledge_graph.json')
print(f"  ✓ Saved knowledge graph with {len(kg.clusters)} clusters")

# ============================================================================
# STEP 3: Create query engine and make queries
# ============================================================================
print("Step 3: Creating query engine...")

kg = load_knowledge_graph('my_knowledge_graph.json')

# Note: You need a real API key to make actual queries
# For testing, we'll just show the structure
engine = create_query_engine(
    kg,
    provider="deepseek",
    api_key="sk-xxx...",  # Your API key here
    max_context_tokens=6000
)

print("  ✓ Query engine ready")

# Example queries
queries = [
    "我想开一家高端餐厅，应该选择哪个区域？",
    "商业办公楼最适合在哪里建设？",
    "哪个区域适合开设购物中心？",
]

for query in queries:
    try:
        result = engine.query(query)
        print(f"\nQuery: {query}")
        print(f"Answer: {result.answer}")
        print(f"Recommended regions: {result.recommended_clusters}")
        print(f"Confidence: {result.confidence:.0%}")
    except Exception as e:
        print(f"Error: {e}")
```

---

## Architecture

### Knowledge Graph Structure

The knowledge graph is built from hierarchical clustering results and stored as compressed JSON:

```python
@dataclass
class ClusterSummary:
    cluster_id: int                                  # 聚类簇ID
    level: int                                       # 层级 (0=粗, 1=细)
    parent_id: Optional[int]                         # 父簇ID (层级链接)
    child_ids: List[int]                             # 子簇ID列表

    node_count: int                                  # 该簇包含的节点数
    center_lon: float                                # 中心经度
    center_lat: float                                # 中心纬度

    poi_distribution: Dict[str, float]               # Top 5 POI (压缩)
    physical_features: Dict[str, float]              # 物理特征统计 (mean/std)
    characteristics: str                             # 生成的特征描述 (中文)
    dominant_pois: List[str]                         # 主导 POI 列表
    suitable_business: List[str]                     # 适合的商业类型
    neighbor_clusters: List[Tuple[int, str, float]]  # 邻接关系 [(id, relation, weight)]
```

### Query Pipeline

```
Query: "我想开一家高端餐厅，应该选择哪个区域？"
        ↓
   1. Determine relevant hierarchy level
      → Heuristic: Keywords like "区域" → use coarse level
      → Keywords like "街道" → use fine level

        ↓
   2. Extract relevant clusters
      → Keyword matching: 搜索 "餐厅" 相关POI
      → Relevance scoring: +1.0 for exact match, +0.5 for business type
      → Top-5 selection: Keep top 5 by score

        ↓
   3. Build context
      → Format cluster info in Chinese (规模、位置、特征、POI分布等)
      → Estimate token count (~410 tokens for 5 clusters)
      → Truncate if exceeds max_context_tokens

        ↓
   4. Build prompt
      → Use system prompt template (Chinese urban planning consultant)
      → Inject context + question

        ↓
   5. Query LLM
      → Send to Google Gemini or Deepseek API

        ↓
   6. Parse response
      → Extract recommended clusters
      → Extract answer and reasoning
      → Calculate confidence score
      ↓
   QueryResult {
       answer: "区域 5 最适合...",
       reasoning: "该区域有 78% 的餐饮 POI...",
       recommended_clusters: [5, 12, 8],
       confidence: 0.85
   }
```

---

## Key Features

### 1. **Hierarchical Awareness**

The system understands cluster hierarchies and can:
- Navigate coarse-to-fine (start at level 0, drill down to level 1)
- Link parent-child relationships
- Understand cluster containment

```python
# Get parent cluster
parent_id = cluster.parent_id
parent_cluster = kg.get_cluster(parent_id)

# Get child clusters
child_ids = cluster.child_ids
child_clusters = [kg.get_cluster(cid) for cid in child_ids]
```

### 2. **Token Efficiency**

Instead of passing all 40W nodes:
- **Compress to clusters**: 40W nodes → 10-100 clusters
- **Compress cluster info**: Top-5 POIs only, key feature stats
- **Estimate tokens**: ~410 tokens for 5 clusters (vs 60K+ for all nodes)

### 3. **Multi-Provider Support**

Works with multiple LLM providers:

```python
# Google Gemini
engine = create_query_engine(kg, provider="google", api_key="...")

# Deepseek
engine = create_query_engine(kg, provider="deepseek", api_key="...")
```

Easy to extend for other providers by implementing `LLMProvider` interface.

### 4. **Dynamic Context Selection**

Intelligently selects relevant clusters based on:
- Question keywords
- POI matching (restaurant → 餐饮 clusters)
- Business type matching
- Relevance scoring

---

## Configuration

### System Prompt

Customize the system prompt to change LLM behavior:

```python
custom_prompt = """你是一个专业的房产投资顾问。
你的任务是基于城市功能区信息，给出最优的投资建议。

{context}

用户问题: {question}

请按以下格式回答:
## 投资建议
[具体建议]

## 推荐区域
[区域列表]

## 风险分析
[风险说明]"""

engine = UrbanFunctionalZoneQueryEngine(
    kg,
    llm_provider,
    system_prompt_template=custom_prompt,
    max_context_tokens=6000
)
```

### Max Context Tokens

Adjust the maximum context size:

```python
# Smaller context (faster, cheaper)
engine = create_query_engine(kg, provider="deepseek", api_key="...", max_context_tokens=3000)

# Larger context (more comprehensive)
engine = create_query_engine(kg, provider="deepseek", api_key="...", max_context_tokens=8000)
```

---

## Running the Demo

Complete demo with synthetic data:

```bash
# Build KG and skip LLM (no API key needed)
python demo_llm_query.py --no-llm

# Build KG and query with Deepseek
python demo_llm_query.py --provider deepseek --api-key sk-xxx...

# Build KG and query with Google Gemini
python demo_llm_query.py --provider google --api-key xxx...

# Custom parameters
python demo_llm_query.py --provider deepseek --api-key xxx... --n-nodes 20000 --n-levels 3
```

---

## Common Use Cases

### Use Case 1: Find Best Location for Restaurant

```python
result = engine.query("我想开一家中式餐厅，选择哪个区域最好？")
```

The system will:
1. Extract restaurant-focused clusters
2. Check POI distribution (餐饮 POI %)
3. Consider neighborhood characteristics
4. Recommend top clusters with reasoning

### Use Case 2: Analyze Business Potential

```python
result = engine.query("商业地产投资，最看好哪个区域？为什么？")
```

The system will:
1. Select business/commercial clusters
2. Consider size, connectivity, POI ecosystem
3. Provide investment analysis

### Use Case 3: Understand Zone Characteristics

```python
result = engine.query("这个城市的商业分布有什么特点？")
```

The system will:
1. Analyze all clusters
2. Identify patterns and characteristics
3. Provide overall city-level insights

---

## Troubleshooting

### Issue: "API key required for X"

**Solution**: Make sure to provide a valid API key:
```python
engine = create_query_engine(
    kg,
    provider="deepseek",
    api_key="sk-xxx..."  # Valid key
)
```

### Issue: Context too large

**Solution**: Reduce max_context_tokens:
```python
engine = create_query_engine(
    kg,
    provider="deepseek",
    api_key="...",
    max_context_tokens=3000  # Smaller context
)
```

### Issue: No relevant clusters found

This means the question doesn't match any clusters. Try rephrasing:
```python
# Before: Very specific
result = engine.query("北京朝阳区东三环的办公楼")  # Too specific

# After: More general
result = engine.query("办公楼区域在哪里？")  # Better
```

---

## Integration with Your Project

### From Training Pipeline

```python
from ufz.analysis.clustering import cluster_embeddings
from ufz.export.graphrag import build_hierarchical_knowledge_graph

# After training your MVCL model
final_embeddings = model(x_phys, x_sem, edge_index)

# Hierarchical clustering
level_0_labels = cluster_embeddings(final_embeddings, method='kmeans', n_clusters=10)
level_1_labels = cluster_embeddings(final_embeddings, method='kmeans', n_clusters=100)

# Build KG
kg = build_hierarchical_knowledge_graph(
    labels_per_level=[level_0_labels, level_1_labels],
    embeddings=model.get_embeddings(),
    positions=gdf[['lon', 'lat']].values,
    features=feature_matrix,
    semantic_probs=semantic_features
)
```

---

## Next Steps

1. **Prepare your data** - Get clustering results and embeddings from your MVCL pipeline
2. **Build KG** - Use `build_hierarchical_knowledge_graph()`
3. **Create engine** - Instantiate with your LLM provider and API key
4. **Query** - Use `engine.query()` for natural language analysis
5. **Extend** - Customize system prompts and context selection for your use case

---

**Last Updated**: 2026-03-09
