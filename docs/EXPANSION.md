# 0.2 方法扩展与实现边界

本轮把原来 17 个执行配置扩展为 43 个，涉及 37 个清单条目。全部 164 条计划均保留独立的 `method_specs/<id>.json`；其中 127 条尚无执行适配器。规格文件、可运行配置、独立论文和原论文数值复现是四种不同统计对象。

本版本提供**机制明确的统一协议适配**。共同优化器、验证选模和训练预算不等于所有作者的默认训练配方；下面明确记录差异。正式 benchmark 前仍需完成真实参考实验与预算冻结。

## 新增执行配置

| ID | 已保留的核心机制 | 来源与适配边界 |
|---|---|---|
| `frmil` | 正负 bag 配对；特征重校准；bag CE、实例最大 BCE、特征幅度损失 | 动态加载固定作者代码。保留训练集最大 bag 填充及原 BCE 标量权重；使用独立 val。二分类；margin 需针对 FM 在验证侧确定 |
| `focusmil` | 变分实例编码、重参数采样、最大实例分类、KL | 作者 Camelyon16 路线的重写；测试使用后验均值。二分类；共享优化设置未复制作者 AdamW/plateau 默认值 |
| `dgrmil` | 真实作者全局 lesion token、Nystrom、交叉注意力、类别条件中心 EMA、warmup、BCE、logdet 多样性和 triplet | 动态加载作者代码；正确解释整数类别，logdet 使用 float64 与 jitter；二分类，共享优化器而非原 Lookahead/scheduler |
| `rrt` | 区域 MSA、跨区域 MSA、EPEG | 动态加载作者重构版；隔离 imports、替换等价 DropPath、修复公开 `mlp_dim` 参数与硬编码 512 冲突。作者重构版不能据此声称精确复现历史论文 |
| `chowder` | 每类实例评分、top/bottom extremes、共享 sigmoid MLP | Owkin 后续实现适配；至少 2×extremes 个实例；数值对照作者模块 |
| `varpool` / `varpool_cox` | 注意力均值、可学习投影上的注意力加权方差、联合预测头 | 分类 CE / 全队列 Cox 是两个明确配置；sqrt 加数值下限，原框架其他生存配方未涵盖 |
| `h2t` | 原型硬分配、按簇平均归一化 patch 表征 | PANTHER 作者提供的 H2T 对照版本；训练集球面聚类初始化、线性分类头；不代表原 H2T 全部描述符和协议 |
| `panther` | 固定原型、对角 NIW MAP-EM、比例/均值/方差 allcat 表征 | 训练集 reservoir + sklearn KMeans；MAP-EM 数值对照作者模块；仅当前线性分类头路线 |
| `ibmil` | 基础聚合器监督 warmup、训练 bag 表征聚类、混杂注意力和拼接分类 | 采用 warmup 最后 checkpoint，sklearn 替代原聚类后端；当前 concat 版本；两阶段状态和优化器重置可恢复 |
| `additive_mil` | 实例级加性贡献、注意力与预测 MLP | 固定 MIL_BASELINE 中基于官方补充材料的适配；未独立重跑原论文 |
| `linmil` | ReLU kernel 线性注意力、CLS token、DropPath | 固定 MIL_BASELINE 适配；去除仅用于 DropPath 的 timm 依赖 |
| `fouriermil` | 自适应 Fourier 过滤块与 CLS 聚合 | 固定 MIL_BASELINE 适配；尚未完成独立作者源逐层数值对照 |
| `retmil` | 局部 chunk retention、全局 retention、门控池化 | 固定 MIL_BASELINE 适配；`chunk_len` 是公开配置名称 |
| `pneamil` | 非负证据、类别最大池化、L1 证据稀疏 | 稀疏目标实际加入 loss；不能关闭后仍当成完整 PNEA 路线 |
| `mrblock` | 冻结 affine 映射、非线性低秩残差与门控聚合 | 当前 MR-AB-MIL 配置；不是任意基础聚合器的所有组合 |
| `carmil` | factual CE、counterfactual effect CE、attention 正则 | 三项均参与反向传播；默认权重可在配置中明示 |
| `fcmil` | 空间/频域注意力与 drop/replace 因果正则 | 因果正则实际参与训练；保留选取比例与混合强度配置 |
| `asmil` | 多 token/多头注意力、token drop、EMA anchor 一致性 | 原训练文件中 token CE/diversity 已注释，当前未私自加入；修复原 EMA 条件导致正常参数不更新的问题，anchor 固定 eval |
| `attrimil` | 类别属性分数、物理近邻约束、跨切片排序队列 | 冻结特征路线；需要真实坐标。队列在 checkpoint 保存。不是包含原图编码器更新的全流程 |
| `nnmil` | 训练集 bag 长度规划、类别平衡批次、随机特征子空间、padding mask、概率 ensemble | 当前分类路线；不是 nnMIL 的完整任务自配置框架；回归、生存及作者全部超参选择尚未接入 |
| `madmil` | 特征分头、独立门控注意力、各头池化后拼接 | 对照作者推理和含 dropout 的训练前向，记录数值误差 |
| `litemil` | 作者原始 query cross-attention | 保存最佳状态，严格 val 选模；没有原脚本中按外层 test 准确率选择折权重的流程 |
| `psemix_abmil` | ProtoDiv 余弦表型细化、固定伪包划分、content 比例混合、soft BCE | PULS ABMIL 基座；分区仅在训练集建立并保存。默认 30 伪包，小 bag 必须显式降低；末尾不足完整批次仍训练 |
| `prodiv_abmil` | 同样的表型分区，训练时采样单个伪包 | 使用 PseMix 作者提供的 ProtoDiv 对照路线；不宣称已独立验证 ProtoDiv 原论文完整设置 |
| `remix_abmil` | 训练 bag 原型和协方差扰动；同类 donor；replace/append/interpolate/cov 联合增强；原 ReLU ABMIL + BCE | sklearn KMeans 替代 GPU FAISS；零均值高斯用中心化样本矩阵作为协方差因子，单实例簇使用零扰动；推理读全 bag |

## 使用与运行条件

```bash
milplus methods                         # 全部 164 条计划
milplus methods --runnable              # 当前 43 个执行配置
milplus coverage --json                 # 按分组统计实际适配覆盖
milplus inspect --method frmil
```

RRT、FRMIL、DGR-MIL 的作者代码在联网 CPU 端准备，GPU 端必须能读同一目录：

```bash
export MILPLUS_SOURCE_ROOT=/your/persistent/path/milplus-sources
milplus setup --method rrt
milplus setup --method frmil
milplus setup --method dgrmil
```

运行时验证固定 Git commit 与 tracked 文件状态，再从 Git 对象读取源码。不会使用未跟踪文件替换模型，也不会因依赖缺失自动换成另一个模型。源代码身份进入运行指纹。

```bash
milplus run --data /path/to/brca_task --method psemix --device cuda
milplus run --data /path/to/coad_task --method dgrmil --device cuda
```

`data` 必须是完成了患者标签、固定划分和 FM 特征登记的任务包。上述命令不是对当前共享盘数据已完成准备的声明。

需要坐标的 AttriMIL 不接受把多张切片直接拼成同一坐标平面。FRMIL、FocusMIL、DGR 当前只接受二分类；其他任务会在兼容性检查时被拒绝。ProtoDiv/PseMix 要保留已登记的 patch 顺序，不能再叠加 `max_instances` 随机截取。

## 验证范围

每个新增配置均检查训练、保存、重新构建模型和预测。对包含 anchor、FIFO 队列、两阶段初始化、批次采样、原型缓存和伪包分区的配置，额外比较中断恢复与不中断运行的逐项模型状态和完整训练历史。

`scripts/validate_author_equivalence.py` 从锁定的原始仓库提取指定定义做独立数值对照，覆盖 CHOWDER、MAD-MIL、PANTHER MAP-EM、PseMix 分区及 ReMix 五种增强。它不会运行作者的训练脚本。通过这些对照也不等于已复现作者的真实数据指标。

机器可读证据和未完成范围分别见 [VALIDATION.md](VALIDATION.md) 与 [ADAPTATION_STATUS.md](ADAPTATION_STATUS.md)。所有 164 条的独立规格文件位于 [method_specs](../method_specs)，待接入条目没有伪造的 Python 模型入口。
