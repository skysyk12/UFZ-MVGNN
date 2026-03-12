"""CLI: Command-line interface for UFZ pipeline."""

import logging
import argparse
from pathlib import Path
from ufz.config.parser import Config

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def cmd_train(args):
    """Train semantic enhancement or MVCL models."""
    config = Config.from_yaml(args.config)
    
    if args.device != "auto":
        config.device = args.device
    if args.seed:
        config.seed = args.seed
    
    logger.info(f"Training stage: {args.stage}")
    logger.info(f"Config: {config}")
    
    if args.stage in ["semantic", "all"]:
        logger.info("Training semantic enhancement (Imputer + RefineNet)...")
        from ufz.utils.seed import set_seed
        set_seed(config.seed)
        logger.info("✓ Semantic training complete (stub)")
    
    if args.stage in ["mvcl", "all"]:
        logger.info("Training MVCL (multi-view contrastive learning)...")
        logger.info("✓ MVCL training complete (stub)")


def cmd_cluster(args):
    """Run clustering on embeddings."""
    config = Config.from_yaml(args.config)
    logger.info(f"Clustering method: {config.analysis.clustering_method}")
    logger.info("✓ Clustering complete (stub)")


def cmd_export(args):
    """Export results (maps, graphrag, etc)."""
    config = Config.from_yaml(args.config)
    logger.info(f"Export format: {args.format}")
    
    if args.format == "map":
        logger.info("Exporting cluster maps as GeoJSON...")
        logger.info("✓ Maps exported (stub)")
    elif args.format == "graphrag":
        logger.info("Building knowledge graph...")
        logger.info("✓ Knowledge graph built (stub)")


def cmd_visualize(args):
    """Visualize graphs and embeddings."""
    config = Config.from_yaml(args.config)
    logger.info(f"Visualization type: {args.type}")
    
    if args.type == "graph":
        logger.info("Visualizing graph structure...")
    elif args.type == "embedding":
        logger.info("Visualizing embeddings...")
    elif args.type == "cluster":
        logger.info("Visualizing clusters...")
    
    logger.info("✓ Visualization complete (stub)")


def main():
    """Main CLI entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="UFZ (Urban Functional Zone) Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ufz train --stage all --config configs/server.yaml
  python -m ufz train --stage semantic --config configs/local.yaml --epochs 100
  python -m ufz cluster --config configs/server.yaml
  python -m ufz export --format map --config configs/server.yaml
  python -m ufz visualize --type embedding --config configs/server.yaml
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument(
        "--stage",
        choices=["semantic", "mvcl", "all"],
        default="all",
        help="Training stage",
    )
    train_parser.add_argument(
        "--config",
        required=True,
        help="Config YAML file",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="Override epochs",
    )
    train_parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    train_parser.set_defaults(func=cmd_train)
    
    # cluster command
    cluster_parser = subparsers.add_parser("cluster", help="Run clustering")
    cluster_parser.add_argument(
        "--config",
        required=True,
        help="Config YAML file",
    )
    cluster_parser.add_argument(
        "--embeddings",
        help="Path to embeddings file",
    )
    cluster_parser.set_defaults(func=cmd_cluster)
    
    # export command
    export_parser = subparsers.add_parser("export", help="Export results")
    export_parser.add_argument(
        "--format",
        choices=["map", "graphrag"],
        default="map",
        help="Export format",
    )
    export_parser.add_argument(
        "--config",
        required=True,
        help="Config YAML file",
    )
    export_parser.set_defaults(func=cmd_export)
    
    # visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize results")
    viz_parser.add_argument(
        "--type",
        choices=["graph", "embedding", "cluster"],
        default="embedding",
        help="Visualization type",
    )
    viz_parser.add_argument(
        "--config",
        required=True,
        help="Config YAML file",
    )
    viz_parser.set_defaults(func=cmd_visualize)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args) or 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
