#!/usr/bin/env python
"""
UFZ_MVGNN 演示脚本
展示如何直接调用核心模块

运行方法:
  python demo.py
"""

import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_config():
    """演示 1: 配置加载"""
    logger.info("=" * 60)
    logger.info("演示 1: 配置加载与管理")
    logger.info("=" * 60)

    try:
        from ufz.config.parser import Config
        from dataclasses import asdict

        # 从 YAML 加载配置
        config = Config.from_yaml('configs/base.yaml')

        logger.info(f"✓ 配置加载成功")
        logger.info(f"  - Device: {config.device}")
        logger.info(f"  - Seed: {config.seed}")
        logger.info(f"  - 模型 Backbone: {config.model.backbone}")
        logger.info(f"  - 聚类方法: {config.analysis.clustering_method}")
        logger.info(f"  - 物理特征: {config.features.groups}")

        print("\n📋 完整配置:")
        import json
        print(json.dumps(asdict(config), indent=2, default=str))

    except Exception as e:
        logger.error(f"❌ 配置加载失败: {e}")
        return False

    return True


def demo_encoding():
    """演示 2: 编码器注册机制"""
    logger.info("\n" + "=" * 60)
    logger.info("演示 2: GNN 编码器注册机制")
    logger.info("=" * 60)

    try:
        from ufz.models.backbones.registry import BackboneRegistry
        import torch

        # 获取所有注册的编码器
        encoders = BackboneRegistry.list_backbones()
        logger.info(f"✓ 已注册 {len(encoders)} 个编码器: {encoders}")

        # 创建 GIN 编码器
        GINEncoder = BackboneRegistry.get('gin')
        encoder = GINEncoder(
            in_dim=27,
            hidden_dim=256,
            out_dim=128,
            num_layers=2,
            dropout=0.5
        )

        logger.info(f"✓ 创建 GIN 编码器成功")
        logger.info(f"  - 输入维度: 27")
        logger.info(f"  - 隐层维度: 256")
        logger.info(f"  - 输出维度: 128")

        # 虚拟前向传播
        x = torch.randn(100, 27)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = encoder(x, edge_index)

        logger.info(f"✓ 前向传播成功")
        logger.info(f"  - 输出形状: {out.shape}")

    except Exception as e:
        logger.error(f"❌ 编码器演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def demo_mvcl():
    """演示 3: MVCL 对比学习模型"""
    logger.info("\n" + "=" * 60)
    logger.info("演示 3: MVCL 多视图对比学习")
    logger.info("=" * 60)

    try:
        from ufz.models.mvcl import MVCLModel
        import torch

        # 创建模型
        model = MVCLModel(
            physical_dim=27,
            semantic_dim=17,
            hidden_dim=256,
            repr_dim=128,
            proj_dim=128,
            backbone='gin',
            num_layers=2,
            dropout=0.5
        )

        logger.info(f"✓ MVCL 模型创建成功")
        logger.info(f"  - 物理视图维度: 27")
        logger.info(f"  - 语义视图维度: 17")
        logger.info(f"  - 表示维度: 128")
        logger.info(f"  - 对比空间维度: 128")

        # 虚拟数据
        x_phys = torch.randn(100, 27)
        x_sem = torch.randn(100, 17)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        # 前向传播
        h_phys, h_sem, z_phys, z_sem = model(x_phys, x_sem, edge_index)

        logger.info(f"✓ 前向传播成功")
        logger.info(f"  - 物理表示: {h_phys.shape}")
        logger.info(f"  - 语义表示: {h_sem.shape}")
        logger.info(f"  - 物理对比: {z_phys.shape}")
        logger.info(f"  - 语义对比: {z_sem.shape}")

        # 计算对比损失
        loss = model.compute_loss(z_phys, z_sem)
        logger.info(f"✓ 对比损失: {loss.item():.4f}")

        # 获取融合表示
        embeddings = model.get_embeddings(x_phys, x_sem, edge_index)
        logger.info(f"✓ 融合表示: {embeddings.shape}")

    except Exception as e:
        logger.error(f"❌ MVCL 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def demo_clustering():
    """演示 4: 聚类分析"""
    logger.info("\n" + "=" * 60)
    logger.info("演示 4: 聚类分析 (4 种方法)")
    logger.info("=" * 60)

    try:
        from ufz.analysis.clustering import cluster_embeddings
        import numpy as np

        # 创建虚拟嵌入
        embeddings = np.random.randn(100, 128)

        methods = ['hdbscan', 'dbscan', 'kmeans', 'leiden']

        for method in methods:
            try:
                if method == 'hdbscan':
                    labels = cluster_embeddings(
                        embeddings,
                        method=method,
                        min_cluster_size=15
                    )
                elif method == 'dbscan':
                    labels = cluster_embeddings(
                        embeddings,
                        method=method,
                        eps=0.5,
                        min_samples=15
                    )
                elif method == 'kmeans':
                    labels = cluster_embeddings(
                        embeddings,
                        method=method,
                        n_clusters=10
                    )
                elif method == 'leiden':
                    labels = cluster_embeddings(
                        embeddings,
                        method=method,
                        resolution=1.0
                    )

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = (labels == -1).sum()

                logger.info(f"✓ {method.upper()}: {n_clusters} 个簇"
                           f" ({n_noise} 噪声点)")

            except ImportError as e:
                logger.warning(f"⚠ {method.upper()} 依赖缺失: {e}")
            except Exception as e:
                logger.warning(f"⚠ {method.upper()} 执行失败: {e}")

    except Exception as e:
        logger.error(f"❌ 聚类演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def demo_semantic_model():
    """演示 5: 语义增强模型"""
    logger.info("\n" + "=" * 60)
    logger.info("演示 5: 语义增强 - CrossViewImputer")
    logger.info("=" * 60)

    try:
        from ufz.semantic.imputer import CrossViewImputer
        import torch

        # 创建模型
        model = CrossViewImputer(
            in_dim=27,
            hidden_dim=128,
            num_classes=17,
            heads=4,
            dropout=0.3
        )

        logger.info(f"✓ CrossViewImputer 创建成功")
        logger.info(f"  - 输入维度: 27 (物理特征)")
        logger.info(f"  - 隐层维度: 128")
        logger.info(f"  - 输出维度: 17 (POI 类别)")
        logger.info(f"  - 注意力头数: 4")

        # 虚拟数据
        x = torch.randn(100, 27)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        # 前向传播
        logits = model(x, edge_index)

        logger.info(f"✓ 前向传播成功")
        logger.info(f"  - 输出形状: {logits.shape}")
        logger.info(f"  - POI 概率范围: [{logits.min():.4f}, {logits.max():.4f}]")

    except Exception as e:
        logger.error(f"❌ 语义增强演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def demo_export():
    """演示 6: 结果导出"""
    logger.info("\n" + "=" * 60)
    logger.info("演示 6: 结果导出")
    logger.info("=" * 60)

    try:
        from ufz.export.maps import export_geojson, export_cluster_summary
        import numpy as np

        # 创建虚拟数据
        positions = np.array([
            [120.0, 30.0],
            [120.1, 30.1],
            [120.2, 30.2],
            [120.3, 30.3],
        ])
        labels = np.array([0, 0, 1, 1])
        embeddings = np.random.randn(4, 128)

        # 创建输出目录
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)

        # 导出 GeoJSON
        geojson_path = output_dir / 'demo_clusters.geojson'
        export_geojson(
            positions,
            labels,
            output_path=str(geojson_path),
            crs='EPSG:4326'
        )
        logger.info(f"✓ GeoJSON 导出成功: {geojson_path}")

        # 导出聚类统计
        summary_path = output_dir / 'demo_cluster_summary.json'
        export_cluster_summary(
            labels,
            embeddings,
            output_path=str(summary_path)
        )
        logger.info(f"✓ 聚类统计导出成功: {summary_path}")

    except Exception as e:
        logger.error(f"❌ 导出演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """运行所有演示"""
    logger.info("\n")
    logger.info("🚀 UFZ_MVGNN 核心模块演示")
    logger.info("=" * 60)
    logger.info("本脚本展示如何直接调用 UFZ 的核心模块")
    logger.info("=" * 60)

    demos = [
        ("配置系统", demo_config),
        ("编码器注册", demo_encoding),
        ("MVCL 对比学习", demo_mvcl),
        ("聚类分析", demo_clustering),
        ("语义增强", demo_semantic_model),
        ("结果导出", demo_export),
    ]

    results = {}

    for name, func in demos:
        try:
            results[name] = func()
        except Exception as e:
            logger.error(f"演示失败: {name}")
            results[name] = False

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("📊 演示结果汇总")
    logger.info("=" * 60)

    for name, success in results.items():
        status = "✓" if success else "❌"
        logger.info(f"{status} {name}")

    success_count = sum(results.values())
    total_count = len(results)

    logger.info("\n" + "=" * 60)
    if success_count == total_count:
        logger.info(f"🎉 全部演示成功! ({success_count}/{total_count})")
    else:
        logger.warning(f"⚠ 部分演示失败 ({success_count}/{total_count})")
    logger.info("=" * 60)

    return 0 if success_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
