# 快速开始指南

## 1分钟了解

本项目实现了"专家ODE策略 → 条件扩散模型 → 先验引导RL"的疫苗分配优化管线。

## 环境配置（5分钟）

```bash
# 安装依赖
pip install -r requirements.txt

# 验证安装
python test_modules.py
```

## 快速测试（10分钟）

```bash
# 测试专家轨迹生成（10条轨迹）
python test_m1_quick.py

# 预期输出：
# ✓ Saved 10 trajectories to data/macro_expert/test_trajectories.pkl
```

## 完整运行（3-4小时）

```bash
# 运行完整管线
python run_pipeline.py --all

# 或分步运行
python run_pipeline.py --m1  # 生成专家轨迹（~20分钟）
python run_pipeline.py --m2  # 微观重放（~15分钟）
python run_pipeline.py --m3  # 训练扩散+PPO（~2-3小时）

# 查看结果
python run_pipeline.py --report
```

## 关键输出

1. **专家轨迹**: `data/macro_expert/expert_trajectories.pkl`
2. **微观数据**: `data/micro_replay/micro_trajectories.pkl`
3. **扩散模型**: `logs/diffusion/diffusion_model_final.pt`
4. **PPO模型**: `logs/ppo/*_final.pt`
5. **对比图**: `logs/ppo/comparison_plot.png`

## 查看训练进度

```bash
# TensorBoard（如果可用）
tensorboard --logdir=logs/

# 或查看日志文件
tail -f logs/ppo/runs/*/events.*
```

## 常见问题

**Q: 内存不足？**
A: 减少批次大小，修改 `diffusion/train.py` 和 `rl/train_ppo.py` 中的 `batch_size` 参数。

**Q: 训练太慢？**
A: 减少训练迭代次数，修改 `rl/train_ppo.py` 中的 `n_iterations` 从 200 降到 50。

**Q: GPU支持？**
A: 如果有GPU，安装 CUDA 版本的 PyTorch：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Q: 快速测试整个流程？**
A: 创建小规模测试配置（待实现）

## 详细文档

- 完整项目说明：[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- 实施计划：[README_dev.md](README_dev.md)
- 原始说明：[README.md](README.md)

## 下一步

阅读 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) 了解：
- 详细技术架构
- 各模块功能说明
- 预期结果和验收标准
- 未来扩展方向
