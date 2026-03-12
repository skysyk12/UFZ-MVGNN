#!/bin/bash

# 激活 gnn_research 虚拟环境并进入项目目录

echo "🔧 激活 gnn_research 虚拟环境..."
source /opt/anaconda3/envs/gnn_research/bin/activate

echo "✓ 虚拟环境已激活"
echo "$(python --version)"

echo ""
echo "📁 进入项目目录..."
cd /Users/sunyongkang/Downloads/UFZ_all/coding/UFZ_MVGNN
pwd

echo ""
echo "✅ 环境就绪！"
echo ""
echo "现在可以运行以下命令："
echo "  python demo.py                          # 运行演示"
echo "  python -m ufz --help                    # 查看帮助"
echo "  python -m ufz train --help              # 查看训练命令"
echo "  pytest tests/ -v                        # 运行测试"
echo ""
echo "或直接编写 Python 脚本使用项目模块"
echo ""
