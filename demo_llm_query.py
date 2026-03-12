#!/usr/bin/env python
"""
End-to-End Demo: Urban Functional Zone LLM Query

This demo shows how to:
1. Build a hierarchical knowledge graph from clustering results
2. Query the knowledge graph using LLM
3. Get recommendations and reasoning

Usage:
    python demo_llm_query.py --provider deepseek --api-key YOUR_KEY
"""

import argparse
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_synthetic_data(
    n_nodes: int = 40000,
    n_levels: int = 2,
) -> tuple:
    """Create synthetic hierarchical clustering data for demo.

    Args:
        n_nodes: Number of nodes (40W in real scenario)
        n_levels: Number of hierarchy levels (2-3)

    Returns:
        Tuple of (labels_per_level, embeddings, positions, features, semantic_probs)
    """
    logger.info(f"Creating synthetic data: {n_nodes} nodes, {n_levels} levels")

    np.random.seed(42)

    # Level 0: 10 coarse clusters
    level_0_labels = np.random.randint(0, 10, n_nodes)

    # Level 1: 100 fine clusters (each level 0 cluster has ~10 level 1 clusters)
    level_1_labels = np.zeros(n_nodes, dtype=int)
    for i in range(10):
        mask = level_0_labels == i
        level_1_labels[mask] = i * 10 + np.random.randint(0, 10, mask.sum())

    # Level 2 (if applicable): 300 micro clusters
    labels_per_level = [level_0_labels, level_1_labels]
    if n_levels >= 3:
        level_2_labels = np.zeros(n_nodes, dtype=int)
        for i in range(100):
            mask = level_1_labels == i
            level_2_labels[mask] = i * 3 + np.random.randint(0, 3, mask.sum())
        labels_per_level.append(level_2_labels)

    # Embeddings (128 dimensions)
    embeddings = np.random.randn(n_nodes, 128)

    # Positions (lon, lat around 120.0, 30.0 for demo)
    positions = np.random.randn(n_nodes, 2) * 0.1 + np.array([120.0, 30.0])

    # Features (27 dimensions - physical features)
    features = np.random.randn(n_nodes, 27)

    # Semantic probabilities (POI distribution - 17 dimensions)
    semantic_probs = np.random.dirichlet(np.ones(17), n_nodes)

    logger.info(f"✓ Synthetic data created")
    return labels_per_level, embeddings, positions, features, semantic_probs


def build_kg_demo(
    labels_per_level,
    embeddings,
    positions,
    features,
    semantic_probs,
):
    """Build knowledge graph from clustering results."""
    from ufz.export.graphrag import (
        build_hierarchical_knowledge_graph,
        save_knowledge_graph,
    )

    logger.info("Building hierarchical knowledge graph...")

    poi_names = [
        'restaurant', 'shopping', 'office', 'hotel', 'hospital',
        'school', 'park', 'bank', 'supermarket', 'gym',
        'cafe', 'bar', 'cinema', 'library', 'museum',
        'pharmacy', 'salon'
    ]

    kg = build_hierarchical_knowledge_graph(
        labels_per_level=labels_per_level,
        embeddings=embeddings,
        positions=positions,
        features=features,
        semantic_probs=semantic_probs,
        poi_names=poi_names,
    )

    # Save for later use
    output_path = Path("outputs/knowledge_graph.json")
    save_knowledge_graph(kg, str(output_path))

    logger.info(f"✓ Knowledge graph saved to {output_path}")
    return kg


def query_kg_demo(kg, api_key: str, provider: str = "deepseek"):
    """Query knowledge graph using LLM."""
    from ufz.export.llm_interface import create_query_engine

    logger.info(f"Creating query engine ({provider})...")

    try:
        engine = create_query_engine(
            kg,
            provider=provider,
            api_key=api_key,
            max_context_tokens=6000,
        )
    except Exception as e:
        logger.error(f"Failed to create query engine: {e}")
        logger.info("To use LLM queries, install required packages:")
        if provider == "google":
            logger.info("  pip install google-generativeai")
        elif provider == "deepseek":
            logger.info("  pip install openai")
        return None

    # Example queries
    queries = [
        "我想开一家高端餐厅，应该选择哪个区域？",
        "商业办公楼最适合在哪里建设？",
        "哪个区域适合开设购物中心？",
    ]

    logger.info("\n" + "="*80)
    logger.info("QUERIES AND RESULTS")
    logger.info("="*80)

    results = []
    for query in queries:
        logger.info(f"\n【用户问题】: {query}\n")

        try:
            result = engine.query(query)

            logger.info(f"【答案】\n{result.answer}\n")
            logger.info(f"【推荐区域】: {result.recommended_clusters}\n")
            if result.reasoning:
                logger.info(f"【理由】\n{result.reasoning}\n")
            logger.info(f"【置信度】: {result.confidence:.1%}\n")

            results.append(result)

        except Exception as e:
            logger.error(f"Query failed: {e}")

        logger.info("-" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="UFZ LLM Query Demo")
    parser.add_argument(
        "--provider",
        choices=["google", "deepseek"],
        default="deepseek",
        help="LLM provider"
    )
    parser.add_argument(
        "--api-key",
        help="API key for the LLM provider"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM queries (only build KG)"
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=40000,
        help="Number of nodes (default: 40000)"
    )
    parser.add_argument(
        "--n-levels",
        type=int,
        default=2,
        help="Number of hierarchy levels (default: 2)"
    )
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("UFZ LLM Query Demo")
    logger.info("="*80)

    # Step 1: Create synthetic data
    logger.info("\n【STEP 1】Creating synthetic hierarchical clustering data...")
    labels_per_level, embeddings, positions, features, semantic_probs = create_synthetic_data(
        n_nodes=args.n_nodes,
        n_levels=args.n_levels,
    )

    # Step 2: Build knowledge graph
    logger.info("\n【STEP 2】Building hierarchical knowledge graph...")
    kg = build_kg_demo(
        labels_per_level,
        embeddings,
        positions,
        features,
        semantic_probs,
    )

    logger.info(f"\nKnowledge Graph Summary:")
    logger.info(f"  - Hierarchy depth: {kg.get_hierarchy_depth()} levels")
    logger.info(f"  - Total clusters: {len(kg.clusters)}")
    for level in sorted(kg.hierarchy_levels.keys()):
        logger.info(f"    - Level {level}: {len(kg.hierarchy_levels[level])} clusters")

    # Step 3: Query with LLM (optional)
    if not args.no_llm:
        logger.info("\n【STEP 3】Querying with LLM...")

        if not args.api_key:
            logger.warning(f"⚠️  No API key provided for {args.provider}")
            logger.info(f"Usage: python demo_llm_query.py --provider {args.provider} --api-key YOUR_KEY")
            logger.info("\nSkipping LLM queries. To enable:")
            if args.provider == "google":
                logger.info("  1. Get API key from https://makersuite.google.com/app/apikey")
                logger.info("  2. Install: pip install google-generativeai")
            else:
                logger.info("  1. Get API key from https://www.deepseek.com/api")
                logger.info("  2. Install: pip install openai")
            return

        results = query_kg_demo(kg, args.api_key, args.provider)

        if results:
            logger.info(f"\n✓ Successfully processed {len(results)} queries")
    else:
        logger.info("\n【STEP 3】Skipping LLM queries (--no-llm flag set)")

    logger.info("\n" + "="*80)
    logger.info("Demo completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
