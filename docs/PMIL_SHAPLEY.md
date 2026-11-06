# PMIL：Shapley 伪包增强

来源：[RenaoYan/PMIL](https://github.com/RenaoYan/PMIL)，MIT，固定
`a5df40276fa5848056a89c0cde221500fad4099d`。
[论文 v4](https://arxiv.org/html/2312.05490v4)。这是 Shapley/progressive pseudo-bag 工作，
与原型 PMIL、PAMIL 分别登记。

## 执行配置

| 骨干 | 源码兼容路线 | 命名修正版 |
|---|---|---|
| 作者 ABMIL | `pmil_shapley_abmil_source` | `pmil_shapley_abmil_repaired` |
| 作者 CLAM-SB | `pmil_shapley_clam_sb_source` | `pmil_shapley_clam_sb_repaired` |
| 作者 CLAM-MB | `pmil_shapley_clam_mb_source` | `pmil_shapley_clam_mb_repaired` |

六条配置计为一个方法家族。Mean、Max、DSMIL 虽出现在原仓库选项中，
其公开 Shapley 路径不完整，因此没有借用其他模型把它们计为已实现路线。

```bash
milplus setup --method pmil_shapley
milplus run --data /path/to/task_bundle --method pmil_shapley_abmil_repaired \
  --config configs/pmil_shapley_abmil.yaml --output /path/to/new_run
```

使用既有分类任务数据包、冻结 patch 特征、固定患者划分和单张切片 bag。
切片须有足够真实实例用于最大伪包数、CLAM 的 k 个实例和残余 coalition；
不足会提前报错，不补假实例或悄悄减少伪包。

## 完整训练机制

保留原始网络及 CLAM 实例约束。训练包括：多轮重新初始化、验证收敛时的学习率变化、
逐渐增加伪包数、训练切片的 Shapley 重划分，以及上一轮最佳网络指导下一轮初始化划分。
后续轮次从新的模型参数和 Adam 状态开始训练，不能简单改成上一轮权重继续微调。

Shapley 估计保留作者的注意力候选、随机残余 coalition、重复排列和原始类别 logit 边际贡献。
保留对残余 coalition 求和的源码缩放，随后按分数和余下注意力排序，交错分配到伪包。
达到最大伪包数后仍可能重新估计划分，这也是公开循环的行为。

选择器保留原来的双条件：验证 bag CE 变小，或指定验证指标持平/改善，均可保存。
源码多分类 AUC 使用展平 one-hot 的 micro ROC；它与统一报告的 macro AUC 分开记录。
每轮最佳网络独立保存于 `rounds/round_N.pt`，最终测试使用最后一轮选出的网络。
训练过程中不读取测试结果选模；原始程序每轮的测试日志改为独立 `evaluate`。

checkpoint 保存轮次、轮内 epoch、待执行重划分、完整训练索引、选择器快照及哈希、
Adam、收敛计数和随机状态。中断在轮次或伪包切换前后时，都能从 epoch 边界恢复。

## 源码与修正边界

- `source` 保留一个原切片内多个伪包之间的旧梯度累积；`repaired` 每次伪包更新前清零梯度。
- CLAM-MB 源码候选始终取第 0 分支；修正版按训练切片的目标类别选择分支。
- 原 Shapley 的零分数范围会产生 NaN。源码路线明确报错；修正版返回等分数并保留可复查排序。
- 两类路线都保留原来的轮次、完整目标、学习率/选择器规则；没有改成单阶段 CE。
- 训练日志记录实际优化目标的均值；原打印代码额外除以伪包数的统计缩放不用于优化。
- 通用 FM 输入宽度是显式适配；默认隐藏宽度 512，ABMIL/CLAM 注意力宽度分别为 128/256。
- 公开代码默认学习率 1e-3、伪包步长 2、上限 10，与论文设置有差异。
  另提供 BRCA 参数配置：3e-4、步长 4、上限 6。配置名称不代表已完成论文数值复现。

`model.rounds` 和 `model.round_epochs` 定义总预算；默认上限为 10×200 个 epoch，
收敛可提前结束某轮。`model.patience` 是轮内耐心值，统一层不会截断必要轮次。
默认 FP32；GPU、真实 WSI 和论文分数验证单独记录。

## 验证入口

```bash
python scripts/validate_pmil_shapley.py --device cpu --output /path/to/new_native_cpu
python scripts/validate_pmil_shapley.py --device cuda --output /path/to/new_native_cuda
```

专项检查执行作者完整主程序，并比较两轮选出的网络、分区和最终随机状态；
另逐步比较完整训练函数的梯度、Adam 权重和动量，验证全部六条路线的中断恢复。
强制收敛观察值仅用于切换逻辑的软件测试，不能作为模型性能结果。
