# Wibawa MultiAttentionMIL

[论文](https://arxiv.org/html/2209.03041)给出的作者代码为
[mdsatria/MultiAttentionMIL](https://github.com/mdsatria/MultiAttentionMIL)，固定提交
`fae5840f801c0485eb22f2e7119d392f2fa975f3`。公开仓库是 MNIST 补充实现；
本适配完整保留其聚合器和训练循环，将输入明确换为已准备好的冻结 WSI patch 特征。
论文中的私有肿瘤筛选模型、权重和原始病例流程没有被公开材料覆盖，未声称已经复现。

## 输入和运行

```bash
milplus setup --method msattention
milplus run --data /path/to/task_bundle --method msattention \
  --config configs/msattention.yaml --output /path/to/new_run
```

`msattention` 默认解析为 `msattention_repaired`；显式 `msattention_source` 保留作者推理时的 Dropout。
两条路线是一条方法记录的两个执行配置。支持单张切片、任意非空长度的冻结特征 bag，
输入宽度从 FM 数据包读取；支持二分类或多分类。患者评价可聚合一人多片，训练仍保留单片 bag。

原模型在同一组特征上依次使用三个 128 维投影与 tanh 注意力，
将各层加权池化向量通过残差累加后分类。这里的深度表征不能解释成多个光学倍率：
不要求虚构坐标网格，也不把重复同尺度特征当成多倍率输入。
历史目录的 D 组归类仍保留，实际输入要求在独立规格中列出。

## 完整训练与选择

采用补充 notebook 的 batch size 1、随机打乱整袋顺序、完整 bag CE，
以及 Adam（学习率 1e-4、betas 0.9/0.999、eps 1e-8、weight decay 0）。
默认 10 个 epoch、三层隐藏宽度均为 128；没有截取实例、额外损失、调度器或梯度裁剪。
源码每个 epoch 内重新置零 `best_acc`，并用训练准确率保存文件；
基准改用独立验证集上的 log-loss 严格改善选择最佳模型。
评价单位遵循任务数据包中的 `evaluation_unit`，可为 slide 或 patient；日志明确记录该单位。
测试结果仅在独立 `evaluate` 生成。

## 源码行为与命名修复

作者在 `forward` 里临时创建 `nn.Dropout`，因此 `eval()` 后仍随机丢弃特征。

- `msattention_source` 保留这一行为。推理使用与标签无关的固定切片随机流；
  `prediction_samples` 规定抽样数，平均概率，并恢复后续训练随机状态。
- `msattention_repaired` 只将 Dropout 改为遵循 `train/eval` 模式；
  训练时与作者前向、梯度、Adam 更新及随机抽样一致，推理时关闭 Dropout。
  该路线要求 `prediction_samples: 1`，不会把重复确定性计算记作随机采样预算。

两条路线都保留原始三层残差注意力及完整 CE/Adam 训练。
模型 checkpoint 与统一层保存的 Adam、随机状态、任务和特征签名共同支持精确续训。
三层注意力以独立的 `[instances, 3]` 数组导出，并保留原始坐标行顺序；
每一列分别沿实例归一化，不虚构合成注意力或因果解释。

## 验证与分发

```bash
python scripts/validate_msattention.py --device cpu --output /path/to/new_native_cpu
python scripts/validate_msattention.py --device cuda --output /path/to/new_native_cuda
```

专项测试使用未改写的作者模型独立比较原生 512/1280 维输入、三层注意力、所有梯度及随机状态；
并执行原 notebook 的完整外层训练循环，逐次比较梯度、参数与 Adam 状态。
同时检查源码／修正推理、患者／切片评价、精确续训、最佳模型重载、重复预测和对齐的注意力导出。
独立安装包验证运行两条路线与两种原生输入宽度，保留默认 10 epoch 及 128 隐藏宽度。
实际结果在 `docs/validation` 单独记录；CPU 合成验证不等于 GPU、真实 WSI 或论文数值复现。

固定作者源码树中未发现 LICENSE，台账保留该状态；本项目仅提供适配代码和获取入口，
不将其源码打包或重新声明为本项目许可证。
