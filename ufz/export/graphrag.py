"""GraphRAG: Hierarchical Knowledge Graph for LLM-based Urban Analysis.

This module transforms hierarchical clustering results into a structured
knowledge graph that can be efficiently queried and summarized for LLM input.

Key features:
- Automatic hierarchy detection (2-3 levels)
- Cluster summarization to reduce token usage
- Dynamic context selection based on user queries
- Multi-API support (Google, Deepseek)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class ClusterSummary:
    """Summary of a cluster at any hierarchy level."""

    cluster_id: int
    level: int
    parent_id: Optional[int] = None
    child_ids: List[int] = field(default_factory=list)

    # Statistics
    node_count: int = 0
    center_lon: float = 0.0
    center_lat: float = 0.0

    # Feature summaries (compressed)
    poi_distribution: Dict[str, float] = field(default_factory=dict)  # Top 5 POIs
    physical_features: Dict[str, float] = field(default_factory=dict)  # Key stats

    # Human-readable characteristics
    characteristics: str = ""
    dominant_pois: List[str] = field(default_factory=list)
    suitable_business: List[str] = field(default_factory=list)

    # Relations
    neighbor_clusters: List[Tuple[int, str, float]] = field(default_factory=list)  # (cluster_id, relation_type, weight)

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding expensive fields for transmission."""
        return {
            'cluster_id': self.cluster_id,
            'level': self.level,
            'parent_id': self.parent_id,
            'child_ids': self.child_ids,
            'node_count': self.node_count,
            'center': [self.center_lon, self.center_lat],
            'poi_distribution': self.poi_distribution,
            'physical_features': {k: round(v, 2) for k, v in self.physical_features.items()},
            'characteristics': self.characteristics,
            'dominant_pois': self.dominant_pois,
            'suitable_business': self.suitable_business,
            'neighbors': [
                {'cluster_id': cid, 'relation': rel, 'weight': round(w, 2)}
                for cid, rel, w in self.neighbor_clusters
            ]
        }


class HierarchicalKnowledgeGraph:
    """Knowledge graph for hierarchical urban functional zones."""

    def __init__(self, max_levels: Optional[int] = None):
        """Initialize hierarchical knowledge graph.

        Args:
            max_levels: Maximum hierarchy levels to build (auto-detect if None)
        """
        self.clusters: Dict[int, ClusterSummary] = {}  # cluster_id -> summary
        self.hierarchy_levels: Dict[int, List[int]] = {}  # level -> [cluster_ids]
        self.max_levels = max_levels
        self.metadata: Dict = {}

    def add_cluster(self, summary: ClusterSummary) -> None:
        """Add a cluster to the graph."""
        self.clusters[summary.cluster_id] = summary

        if summary.level not in self.hierarchy_levels:
            self.hierarchy_levels[summary.level] = []
        self.hierarchy_levels[summary.level].append(summary.cluster_id)

    def get_cluster(self, cluster_id: int) -> Optional[ClusterSummary]:
        """Get a cluster summary by ID."""
        return self.clusters.get(cluster_id)

    def get_level_clusters(self, level: int) -> List[ClusterSummary]:
        """Get all clusters at a specific level."""
        cluster_ids = self.hierarchy_levels.get(level, [])
        return [self.clusters[cid] for cid in cluster_ids if cid in self.clusters]

    def get_hierarchy_depth(self) -> int:
        """Get the depth of the hierarchy."""
        return max(self.hierarchy_levels.keys()) + 1 if self.hierarchy_levels else 0

    def to_dict(self, compress: bool = True) -> Dict:
        """Convert to dictionary for serialization and LLM input.

        Args:
            compress: Whether to compress for LLM (remove less important fields)
        """
        hierarchy = {}
        for level in sorted(self.hierarchy_levels.keys()):
            cluster_ids = self.hierarchy_levels[level]
            clusters_data = [
                self.clusters[cid].to_dict()
                for cid in cluster_ids if cid in self.clusters
            ]
            hierarchy[f'level_{level}'] = {
                'level_description': self._get_level_description(level),
                'num_clusters': len(clusters_data),
                'clusters': clusters_data
            }

        return {
            'metadata': self.metadata,
            'hierarchy': hierarchy
        }

    @staticmethod
    def _get_level_description(level: int) -> str:
        """Get human-readable description for hierarchy level."""
        descriptions = {
            0: "城市宏观商业分布",
            1: "区域商业功能细分",
            2: "街区微观功能特征",
        }
        return descriptions.get(level, f"层级 {level} 功能区划")


def build_hierarchical_knowledge_graph(
    labels_per_level: List[np.ndarray],  # [L1_labels, L2_labels, ...]
    embeddings: np.ndarray,  # [N, D]
    positions: np.ndarray,  # [N, 2]
    features: Optional[np.ndarray] = None,  # [N, F]
    semantic_probs: Optional[np.ndarray] = None,  # [N, num_pois]
    poi_names: Optional[List[str]] = None,
    edge_index: Optional[np.ndarray] = None,  # [2, E]
) -> HierarchicalKnowledgeGraph:
    """Build hierarchical knowledge graph from multi-level clustering.

    Args:
        labels_per_level: List of label arrays for each hierarchy level
        embeddings: Node embeddings
        positions: Node positions (lon, lat)
        features: Node features
        semantic_probs: POI probability distributions
        poi_names: List of POI type names
        edge_index: Graph edges

    Returns:
        HierarchicalKnowledgeGraph object
    """
    kg = HierarchicalKnowledgeGraph()
    num_levels = len(labels_per_level)

    logger.info(f"Building hierarchical KG with {num_levels} levels from 40W nodes")

    # Build bottom-up from finest to coarsest
    for level_idx in range(num_levels):
        labels = labels_per_level[level_idx]
        unique_clusters = sorted(np.unique(labels[labels != -1]))

        logger.info(f"  Level {level_idx}: {len(unique_clusters)} clusters")

        # Build summary for each cluster
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            node_indices = np.where(cluster_mask)[0]

            if cluster_mask.sum() == 0:
                continue

            # Calculate center
            center_pos = positions[cluster_mask].mean(axis=0)

            # Extract dominant POIs
            dominant_pois = []
            poi_distribution = {}
            if semantic_probs is not None:
                cluster_semantic = semantic_probs[cluster_mask].mean(axis=0)
                top_indices = np.argsort(cluster_semantic)[-5:][::-1]

                if poi_names is not None:
                    dominant_pois = [
                        poi_names[i] for i in top_indices
                        if i < len(poi_names)
                    ]

                poi_distribution = {
                    poi_names[i]: float(cluster_semantic[i])
                    for i in top_indices
                    if i < len(poi_names)
                }

            # Extract key physical features
            physical_features = {}
            if features is not None:
                cluster_features = features[cluster_mask]
                # Keep only key statistics (mean, std of top features)
                for feat_idx in range(min(5, cluster_features.shape[1])):
                    physical_features[f'feature_{feat_idx}_mean'] = float(
                        cluster_features[:, feat_idx].mean()
                    )
                    physical_features[f'feature_{feat_idx}_std'] = float(
                        cluster_features[:, feat_idx].std()
                    )

            # Generate characteristics
            characteristics = _generate_cluster_characteristics(
                cluster_id, dominant_pois, poi_distribution,
                cluster_mask.sum(), physical_features
            )

            # Determine suitable business
            suitable_business = _determine_suitable_business(
                dominant_pois, poi_distribution
            )

            # Create cluster summary
            summary = ClusterSummary(
                cluster_id=int(cluster_id),
                level=level_idx,
                parent_id=None,  # Will be set in hierarchy linking
                child_ids=[],
                node_count=int(cluster_mask.sum()),
                center_lon=float(center_pos[0]),
                center_lat=float(center_pos[1]),
                poi_distribution=poi_distribution,
                physical_features=physical_features,
                characteristics=characteristics,
                dominant_pois=dominant_pois,
                suitable_business=suitable_business,
            )

            kg.add_cluster(summary)

        logger.info(f"    ✓ Level {level_idx} processed")

    # Add neighbor relations (spatial adjacency for same level)
    if edge_index is not None:
        _add_spatial_relations(kg, labels_per_level, edge_index)

    # Link hierarchy (parent-child relationships)
    _link_hierarchy_levels(kg, labels_per_level)

    # Compute semantic similarity between clusters
    _add_semantic_relations(kg, embeddings, labels_per_level)

    kg.metadata = {
        'total_nodes': len(positions),
        'hierarchy_depth': num_levels,
        'created_at': pd.Timestamp.now().isoformat(),
    }

    logger.info(f"✓ Hierarchical KG built successfully")
    return kg


def _generate_cluster_characteristics(
    cluster_id: int,
    dominant_pois: List[str],
    poi_dist: Dict[str, float],
    node_count: int,
    features: Dict[str, float],
) -> str:
    """Generate human-readable characteristics for a cluster."""
    parts = []

    # Size and POI dominance
    if dominant_pois:
        poi_str = "、".join(dominant_pois[:3])
        parts.append(f"以{poi_str}为主导")

    # Density and scale
    if node_count > 10000:
        parts.append("高密度商业区")
    elif node_count > 1000:
        parts.append("中等商业密度")
    else:
        parts.append("低密度社区")

    return "，".join(parts) if parts else "功能混合区域"


def _determine_suitable_business(
    dominant_pois: List[str],
    poi_dist: Dict[str, float],
) -> List[str]:
    """Determine suitable business types based on POI distribution."""
    suitable = []

    # Simple heuristics
    if 'restaurant' in dominant_pois or any('food' in p.lower() for p in dominant_pois):
        suitable.append("餐饮服务")

    if 'shopping' in dominant_pois:
        suitable.append("零售商业")

    if 'office' in dominant_pois or 'business' in dominant_pois:
        suitable.append("办公楼")

    if 'hotel' in dominant_pois:
        suitable.append("酒店服务")

    if not suitable:
        suitable.append("混合商业")

    return suitable


def _add_spatial_relations(
    kg: HierarchicalKnowledgeGraph,
    labels_per_level: List[np.ndarray],
    edge_index: np.ndarray,
) -> None:
    """Add spatial adjacency relations between clusters."""
    # Use finest level for spatial relations
    finest_level_labels = labels_per_level[-1]

    spatial_edges = set()
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[0, i], edge_index[1, i]

        src_label = finest_level_labels[src]
        tgt_label = finest_level_labels[tgt]

        if src_label == tgt_label or src_label == -1 or tgt_label == -1:
            continue

        edge = tuple(sorted([src_label, tgt_label]))
        spatial_edges.add(edge)

    # Add edges to clusters
    for src_id, tgt_id in spatial_edges:
        src_cluster = kg.get_cluster(src_id)
        if src_cluster:
            src_cluster.neighbor_clusters.append((tgt_id, 'spatial', 1.0))


def _link_hierarchy_levels(
    kg: HierarchicalKnowledgeGraph,
    labels_per_level: List[np.ndarray],
) -> None:
    """Link parent-child relationships between hierarchy levels."""
    for level in range(len(labels_per_level) - 1):
        current_labels = labels_per_level[level]
        next_labels = labels_per_level[level + 1]

        # Map children to parents
        for child_cluster_id in kg.hierarchy_levels.get(level + 1, []):
            child_mask = next_labels == child_cluster_id

            # Find parent (most common label in parent level)
            parent_labels = current_labels[child_mask]
            parent_labels = parent_labels[parent_labels != -1]

            if len(parent_labels) > 0:
                parent_id = int(np.bincount(parent_labels).argmax())

                child = kg.get_cluster(child_cluster_id)
                parent = kg.get_cluster(parent_id)

                if child and parent:
                    child.parent_id = parent_id
                    parent.child_ids.append(child_cluster_id)


def _add_semantic_relations(
    kg: HierarchicalKnowledgeGraph,
    embeddings: np.ndarray,
    labels_per_level: List[np.ndarray],
    similarity_threshold: float = 0.6,
) -> None:
    """Add semantic similarity relations between clusters."""
    from sklearn.metrics.pairwise import cosine_similarity

    # Use coarsest level for efficiency
    coarsest_labels = labels_per_level[0]
    unique_clusters = sorted(np.unique(coarsest_labels[coarsest_labels != -1]))

    if len(unique_clusters) < 2:
        return

    # Compute cluster embeddings (mean of node embeddings)
    cluster_embeddings = {}
    for cluster_id in unique_clusters:
        mask = coarsest_labels == cluster_id
        if mask.sum() > 0:
            cluster_embeddings[cluster_id] = embeddings[mask].mean(axis=0)

    # Compute pairwise similarities
    cluster_ids = list(cluster_embeddings.keys())
    cluster_embs = np.array([cluster_embeddings[cid] for cid in cluster_ids])
    similarities = cosine_similarity(cluster_embs)

    # Add similarity edges
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            sim = similarities[i, j]
            if sim > similarity_threshold:
                src_id = cluster_ids[i]
                tgt_id = cluster_ids[j]

                src_cluster = kg.get_cluster(src_id)
                if src_cluster:
                    src_cluster.neighbor_clusters.append((tgt_id, 'semantic', float(sim)))


def save_knowledge_graph(kg: HierarchicalKnowledgeGraph, output_path: str) -> None:
    """Save knowledge graph to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = kg.to_dict(compress=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Knowledge graph saved to {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def load_knowledge_graph(input_path: str) -> HierarchicalKnowledgeGraph:
    """Load knowledge graph from JSON."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    kg = HierarchicalKnowledgeGraph()

    # Reconstruct clusters from hierarchy
    for level_key, level_data in data['hierarchy'].items():
        level_idx = int(level_key.split('_')[1])

        for cluster_data in level_data.get('clusters', []):
            # Reconstruct neighbor relations
            neighbors = [
                (n['cluster_id'], n['relation'], n['weight'])
                for n in cluster_data.get('neighbors', [])
            ]

            summary = ClusterSummary(
                cluster_id=cluster_data['cluster_id'],
                level=level_idx,
                parent_id=cluster_data.get('parent_id'),
                child_ids=cluster_data.get('child_ids', []),
                node_count=cluster_data['node_count'],
                center_lon=cluster_data['center'][0],
                center_lat=cluster_data['center'][1],
                poi_distribution=cluster_data.get('poi_distribution', {}),
                physical_features=cluster_data.get('physical_features', {}),
                characteristics=cluster_data.get('characteristics', ''),
                dominant_pois=cluster_data.get('dominant_pois', []),
                suitable_business=cluster_data.get('suitable_business', []),
                neighbor_clusters=neighbors,
            )

            kg.add_cluster(summary)

    kg.metadata = data.get('metadata', {})
    logger.info(f"✓ Knowledge graph loaded from {input_path}")

    return kg
