# Loss-Attention：完整特征配方与时序一致性

本适配对应 AAAI 2020 的 Loss-Based Attention for Deep Multiple Instance Learning。来源为[作者仓库](https://github.com/xsshi2015/Loss-Attention)，固定提交 `71926177d84fc1859b70d135090b163ea6bd58bf`；已读取仓库所附论文及补充材料，特别是 Eq.(5–7)、Table A2/A4/A9/A13。根目录未发现许可文件，原始网络/损失/优化器代码保持在外部 Git，通过固定对象加载，不打包复制。

该条目属于原计划 K 组的通用 MIL 向 WSI 迁移。作者公开演示面向 MNIST-bags 和 CIFAR 图像；本库没有声称取得原生 WSI 实现，也没有宣称复现原论文分数。

## 分别登记三套配方

| 配置 | 来源与输入边界 | 损失与原始优化设置 |
|---|---|---|
| `loss_attention_classical` | 论文 A2/A4 的预计算特征路线；256→128→64，每层 ReLU/dropout | bag CE + 0.1 × **sum**(alpha × instance CE)；Adam，lr1e-4、wd5e-4、100 epochs；omega=0 |
| `loss_attention_mil_source` | `MIL/main.py` 与 `model.py`；FM 替代 MNIST CNN 输出，保留单层500维投影、dropout/ReLU、原始共享头 | bag CE + 2 × **mean**(alpha × instance CE)；Adam，lr5e-4、wd1e-4、50 epochs；omega=0 |
| `loss_attention_temporal` | `Image classification/main.py` 的完整时序配方向变长 WSI bag 迁移；默认512维投影 | bag CE + **mean**(alpha × instance CE) + ramp-up加权的一致性；源自定义 Adam、每轮重置状态、原始 lr/beta 调度；200 epochs |

`--method loss_attention` 或 `--method 'Loss-based Attention'` 默认选择 classical 路线。三配置只计一个家族。默认 dropout=.5：MNIST源码用 PyTorch 默认值；论文 A2 只给出 dropout 层，未披露具体概率，classical 中的 .5 是本库明确采用的值，不能当成已核实的历史参数。

论文补充 A4/A9 明确写传统特征和 MNIST 的 omega=0；这两条完整配方不要求时序项。第三条必须标为从图像配方迁移，不能将其冒称为作者发布的 WSI 训练代码。不同 FM 输入维度、类别数和投影宽度均记录在配置中；未人为补入实例真值或额外标注。

## 保留的中心机制

对 patch 表征 H，最终分类器参数 W、b 同时产生实例 logits 和注意力。注意力与各类指数得分之和成正比，在整个 bag 内归一化；bag 表征为 alpha 加权的 H，再由**同一个** W、b 分类。注意力路径保留梯度，没有另造独立 attention MLP 或 detach alpha。

原代码先乘 N×alpha 再求 mean，等于加权 sum。本库保留原计算次序。源码 `AttentionLayer.linear` 是未被 forward 使用的参数，保留并披露；它没有梯度，不会被描述为有效训练模块。

实例分类弱目标来自 bag label 的广播，与 PAMIL 需要的额外训练实例标注不同。source-MIL 的加权 CE 用 mean 归约，存在额外 1/N；论文 Eq.(6) 用 sum。重复相同实例时，bag 预测不变，source 实例项会按长度缩小，二者不能混写成一个公式。

## 时序状态与优化器

temporal 配置在训练集原始 patch 行顺序上保存 attention EMA。每轮结束先做 `Z=.6Z+.4Z_now`，下一轮以 `Z/(1-.6**(epoch+1))` 去偏目标计算 MSE。当前 epoch=0 的一致性系数为0；后续按作者 Gaussian ramp-up 提升。没有将验证或测试 attention 写入训练 bank。

原 image loop 每轮新建 Adam；本适配每轮清空全部动量状态，并应用相同 lr、beta 和 epsilon 位置。该旧 Adam 的 epsilon 放置与现代 `torch.optim.Adam` 存在差异，不能随意替换。80轮附近改变 beta2，150轮开始 ramp-down，200轮为原预算边界；具体端点以固定源码函数为准。

EMA 保留 CPU 上的 Z，当前 epoch 的预测另外暂存；变长 bag 没有补成相同长度。每个训练 slide 必须每轮完整访问一次，缺行、重复访问或 patch 数变化会拒绝提交 EMA。默认两份 attention 缓存上限512MiB，在分配前检查；扩大该资源限额必须明确配置。原图分支的固定4×4位置平均转为 WSI 的变长 bag 平均，不解释为真实空间网格；float32 归约布局的微小差异在独立数值对照中记录。

## 选模与兼容修正

- 原公开代码会在 test 上逐轮比较并选最好结果。本库固定使用 validation slide 错误率，test 只在显式 evaluate 时读取。classical 按错误率、再 bag CE 的字典序选模；论文仅写最低验证错误与损失，未提供联合比较代码，该字典序是明确的协议补充。两个 source 迁移配置在错误率相同时保留先出现的最佳权重。
- 保留固定训练预算，不用通用早停跳过后续 temporal epoch。temporal 至少两轮，确保非零一致性项实际进入训练。
- 原 singleton 的 alpha squeeze 和 log_sum_exp squeeze 会损坏维度；修正为保持 `[N]` 和 `[N,1]`。移除旧 NumPy 标量别名和旧 Adam 调用语法，数值步骤另行对照。
- 原 robust Adam 会将 NaN 参数/梯度/状态置零。本适配在 forward、loss 和梯度检查中明确拒绝非有限值，不把自动置零当作训练成功。
- 不允许通用 max_instances 截断或额外梯度裁剪，不合并不同 slide；患者级预测仍可按公共评估协议聚合。

## 使用

只需标准分类任务包中的患者、切片、划分、标签及 FM 特征，不需要实例标签。坐标不是聚合输入；已有真实坐标时，实例输出会保留其原始精度和行顺序。

```bash
milplus setup --method loss_attention
milplus run --data /data/brca_task --method loss_attention \
  --config configs/loss_attention_classical.yaml --output runs/brca_loss_attention
milplus evaluate --run runs/brca_loss_attention
```

软件演示可使用 `milplus demo` 的合成数据；classical 宽度覆盖应指定完整三层 `widths`，另两条用单层 `hidden`。模型不能接收不完整的多尺度、图文或多标签输入后静默退回其他方法。

## 检查范围

独立参考直接读取固定 Git 的原始训练函数、共享头、加权损失和旧优化器。MNIST 路线在明确移除图像编码器边界后比较完整训练 epoch；temporal 则执行原图像训练函数的完整数值循环，包含按样本 EMA、ramp、每轮新 Adam，未以适配器自身冒充参考。classical 另按论文 Eq.(5–6) 对输出及完整梯度进行独立验证。

检查包括 singleton、不同长度和 dropout、逐步梯度/参数/Adam 状态、归约差异、共享头梯度、调度边界、EMA 输入契约、精确续训、选模、标签独立性与实例 HDF5。当前只有 CPU 合成验证；GPU、真实 WSI、原论文数值复现分别保持独立状态，最终证据见 [VALIDATION.md](VALIDATION.md)。

## BRCA 显式冻结 UNI 条件完成（2026-11-06）

上述“只有CPU合成验证”为早期软件检查范围。现有节点的`loss_attention_repair1`
已实际完成100epoch、74200次连续Adam更新，无优化器重置；每轮使用全部742个TRAIN
完整袋，max_instances为None。原256/128/64投影、dropout.5、共享bag/instance头、
instance weight.1及sum归约保持。该条件是明确的frozen FM/WSI适配，不声称原图像编码器复现。
登记的singleton/现代设备及语法兼容端口保留，选模只用VAL slide error、再bag CE，
不用原源码的TEST选模；不是未声明地改成按TEST或AUC挑选。

锁定45患者TEST于01:40:02Z独立完成，AUC.9892241379、accuracy.9111111111、
balanced accuracy.9170258621、macro-F1.9054621849。61项完成方法的患者身份、
概率及指标独立重算零错误。原拟合status保留test_evaluated=false，后续评价另有receipt。
选中checkpoint SHA256为`3f594f8ed6d62b9a47fb5906d2f724e0a75d2bc81f1191a5a8d8d9e4c8e530bd`。
完整request/status/inputs/protocol/environment/100轮history原字节保存在
`results/loss_attention/source_run_evidence_r1/`及`audit_receipt.json`。
这是当前真实BRCA实验完成，不自动推广为全部profile、论文指标或全部正式纳入已经验收。
