import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ufz.cli import setup_logging, cmd_train, cmd_cluster, cmd_export, cmd_visualize
from ufz.config.parser import Config


def resolve_config_path(raw_path: str, root_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = root_dir / path
    return candidate.resolve()


def run_step(title: str, func, args: argparse.Namespace):
    print(f"\n========== {title} ==========")
    print(f"args={args}")
    func(args)


def main():
    parser = argparse.ArgumentParser(description="UFZ 全流程运行器（Python）")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="配置文件路径（支持相对或绝对路径）",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="训练设备，auto 表示不覆盖配置文件",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="随机种子，提供后覆盖配置中的 seed",
    )
    args = parser.parse_args()

    root_dir = ROOT_DIR
    config_path = resolve_config_path(args.config, root_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    Config.from_yaml(str(config_path))
    setup_logging()

    train_args = argparse.Namespace(
        stage="all",
        config=str(config_path),
        epochs=None,
        device=args.device,
        seed=args.seed,
    )
    cluster_args = argparse.Namespace(
        config=str(config_path),
        embeddings=None,
    )
    export_map_args = argparse.Namespace(
        format="map",
        config=str(config_path),
    )
    export_graphrag_args = argparse.Namespace(
        format="graphrag",
        config=str(config_path),
    )
    viz_graph_args = argparse.Namespace(
        type="graph",
        config=str(config_path),
    )
    viz_embedding_args = argparse.Namespace(
        type="embedding",
        config=str(config_path),
    )
    viz_cluster_args = argparse.Namespace(
        type="cluster",
        config=str(config_path),
    )

    print(f"Project root: {root_dir}")
    print(f"Using config: {config_path}")
    print(f"Device override: {args.device}")
    print(f"Seed override: {args.seed}")

    run_step("Train (semantic + mvcl)", cmd_train, train_args)
    run_step("Cluster", cmd_cluster, cluster_args)
    run_step("Export map", cmd_export, export_map_args)
    run_step("Export graphrag", cmd_export, export_graphrag_args)
    run_step("Visualize graph", cmd_visualize, viz_graph_args)
    run_step("Visualize embedding", cmd_visualize, viz_embedding_args)
    run_step("Visualize cluster", cmd_visualize, viz_cluster_args)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
