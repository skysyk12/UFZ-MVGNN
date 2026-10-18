# UFZ-MVGNN 两级聚类消融实验 — 最终报告

**日期**: 2026-06-04 | **数据**: exp_dedup (去重后 323,587 建筑)

---

## Level 1: 空间社区检测消融

### 实验设计
- **粗筛**: r_comm (200/300/400/500/700/1000) x leiden_resolution (0.01/0.05/0.1/0.2/0.5) = 30 配置
- **细筛**: top-3 粗筛 x k_repair (1/2/3) x sigma (r/2, r/3, r/4) = 27 配置
- **总计**: 57 配置

### 最优配置

| 参数 | 值 |
|------|-----|
| r_comm | **200m** |
| leiden_resolution | **0.01** |
| k_repair | **1** |
| sigma | **r/4 = 50m** |

### 结果

| 指标 | 值 |
|------|-----|
| 社区数 | 11,663 |
| 孤立社区率 | **2.6%** |
| 模块度 (modularity) | **0.884** |
| 大小 Gini | 0.285 |
| 平均内部余弦相似度 | 0.990 |

---

## Level 2: 社区类型聚类消融

### 实验设计
- **特征**: emb_mean(1.0) + emb_std(0.5) + sem_mean(1.0) + log_size(0.3) = 274D
- **PCA**: 16D (解释 87.7% 方差)
- **方法**: KMeans (多个 k), BGMM Dirichlet Process (全维度+PCA), HDBSCAN, DBSCAN, Agglomerative, OPTICS
- **总计**: 23 配置 (KMeans+BGMM) + 56 配置 (auto-k) = 79 配置

### 最终排名 (Top 10)

| Rank | Method | Config | types | largest% | gini | w_ent | Score |
|------|--------|--------|-------|----------|------|-------|-------|
| 1 | **BGMM DP** | PCA=16D, init=10 | 10 | **17.9%** | **0.292** | 2.514 | **0.6653** |
| 2 | KMeans | k=6, PCA=16D | 6 | 51.6% | 0.521 | 2.561 | 0.6622 |
| 3 | KMeans | k=8, PCA=16D | 8 | 38.5% | 0.515 | 2.547 | 0.6530 |
| 4 | KMeans | k=10, PCA=16D | 10 | 31.2% | 0.516 | 2.542 | 0.6478 |
| 5 | BGMM DP | full 274D, init=10 | 9 | 19.1% | 0.321 | 2.551 | 0.6470 |
| 6 | BGMM DP | full 274D, init=30 | 10 | 16.4% | 0.322 | 2.550 | 0.6384 |
| 7 | KMeans | k=12, PCA=16D | 12 | 26.7% | 0.529 | 2.537 | 0.6371 |
| 8 | KMeans | k=20, PCA=16D | 20 | 14.6% | 0.467 | 2.528 | 0.6356 |
| 9 | KMeans | k=15, PCA=16D | 15 | 17.5% | 0.501 | 2.533 | 0.6350 |
| 10 | BGMM DP | full 274D, init=15 | 12 | 17.7% | 0.340 | 2.548 | 0.6328 |

### Auto-k 方法结果 (全部失败)

| Method | Best Config | types | largest% | 问题 |
|--------|-------------|-------|----------|------|
| HDBSCAN | mcs=20 | 2 | **97.7%** | 退化为二分类 |
| DBSCAN | ms=3 eps=23.93 | 2-3 | — | 几乎全部在一个簇 |
| Agglomerative | single dt=31.94 | 5 | **100%** | 全部在一个簇 |
| OPTICS | ms=5 xi=0.01 | 93 | 0.1% | 676s/config, 噪声 93.7%, 不可用 |

### BGMM DP Prior 扫描

| prior | init | eff | pruned | 结果 |
|-------|------|-----|--------|------|
| 0.0001 | 15 | 15 | 0 | largest=18.2%, gini=0.452 |
| 0.001 | 15 | 15 | 0 | **完全一致** |
| 0.1 | 15 | 15 | 0 | **完全一致** |
| 1.0 | 15 | 15 | 0 | **完全一致** |
| 10.0 | 15 | 15 | 0 | **完全一致** |

**prior 参数完全无效** — 5 个 prior 值产生完全相同的结果。DP prior 对此数据不起作用。

---

## 核心发现

### 1. Auto-k 方法不适合此数据
社区 prototypes 在 274D 空间中高度集中。密度方法 (HDBSCAN/DBSCAN/OPTICS) 将此区域视为一个连续的大簇，无法区分功能子类型。层次聚类的 single/average linkage 同样退化为一个主导簇 + 少量碎片。

### 2. BGMM DP 有弱剪枝但 prior 无效
- full 274D: pruned 1-20 个分量 (取决于 init)
- PCA=16D: pruned 0 (完全不剪枝)
- prior 参数扫描 5 个值全部一致
- **init 参数才是真正决定 k 的因素**

### 3. PCA=16D 优于 full 274D
- BGMM: PCA=16D score 0.6653 > full 274D score 0.6470
- KMeans: PCA=16D score 0.6478 > full 274D score 0.6250
- 降维不仅加速 (3-8s vs 4-13s)，还提升了聚类质量

### 4. 最优选择: BGMM DP PCA=16D init=10
- 10 个类型，最大簇仅 17.9% (最接近 18% 目标)
- Gini=0.292 (最接近 0.3 目标)
- 加权熵 2.514 (功能区分度好)
- 3s 完成

---

## 推荐生产配置

```yaml
Level 1 (空间社区):
  r_comm: 200
  leiden_resolution: 0.01
  k_repair: 1
  sigma_comm: 50  # r_comm / 4
  use_cpm: true

Level 2 (社区类型):
  feature_weights: [1.0, 0.5, 1.0, 0.3]  # emb_mean, emb_std, sem_mean, log_size
  pca_dim: 16
  algorithm: BGMM_DP  # BayesianGaussianMixture with Dirichlet Process
  init_components: 10
  covariance_type: diag
  distance: euclidean
```

### 备选: KMeans k=10

如果追求稳定性和一致性 (跨不同 L1 partition 表现一致)，KMeans k=10 PCA=16D 是实用最优:
- 10 types, largest=31.2%, gini=0.516
- 3s, 确定性结果
- 生产环境当前使用的就是此配置

---

## 与旧测试的对比

旧测试在 production community_id (33,248 社区, avg ~10 建筑/社区) 上运行，新测试在 best L1 partition (11,663 社区, avg ~28 建筑/社区) 上:

| Partition | KMeans k=10 largest% | BGMM init=10 largest% | BGMM pruning |
|-----------|---------------------|----------------------|--------------|
| Production (33k) | 21.8% | 18.0% | 0 |
| Best L1 (11.7k) | 31.2% | 17.9% (PCA16) / 19.1% (274D) | 0-1 |

Best L1 partition 产生了更少但更大的社区，导致 KMeans 的最大簇比例上升 (21.8% → 31.2%)，但 BGMM DP 的最大簇比例保持稳定 (~18%)。BGMM 对不同 L1 partition 更鲁棒。

---

## 文件清单

| 文件 | 位置 |
|------|------|
| Level 1 消融脚本 | `/tmp/ablation_level1.py` |
| Level 2 消融脚本 (原始) | `/tmp/ablation_level2.py` |
| Level 2 最终脚本 | `/tmp/ablation_l2_final.py` |
| Level 1 结果 JSON | `analysis/ablation_level1_results.json` |
| Level 2 结果 JSON | `analysis/ablation_level2_final.json` |
| 日志 (run_all) | `/tmp/run_all_outer.log` |
| 日志 (Level 2 final) | `/tmp/l2_final.log` |
| 最优 community_id | `analysis/best_community_id.npy` |
