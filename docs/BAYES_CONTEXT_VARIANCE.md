# Bayes-MIL、CAPRMIL 与 VarMIL

0.9 增加 3 个家族、7 个执行配置，均使用固定作者源码。模型、目标、随机推理、调度与恢复分别测试。软件验证使用 CPU 合成特征，不等于真实病理实验或论文数值复现。

## 获取源码和运行

```bash
milplus setup --method bayesmil
milplus setup --method caprmil
milplus setup --method varmil

milplus check --data /path/to/task --method bayesmil_apcrf --deep
milplus run --data /path/to/task --method bayesmil_apcrf --config configs/bayesmil.yaml --output runs/bayesmil
milplus run --data /path/to/task --method caprmil_mean --config configs/caprmil.yaml --output runs/caprmil
milplus run --data /path/to/task --method varmil --config configs/varmil.yaml --output runs/varmil
milplus evaluate --run runs/varmil
```

`Bayes-MIL` 默认解析到完整空间配置 `bayesmil_apcrf`。缺少坐标或真实画布信息会拒绝运行，不自动降级。显式选择 `bayesmil_vis` 或 `bayesmil_sdpr` 可运行作者的非空间配置。`CAPRMIL` 默认解析为 `caprmil_mean`。所有来源单独获取，包中不分发作者代码或模型权重。

## Bayes-MIL

来源：[ralphc1212/bayes-mil](https://github.com/ralphc1212/bayes-mil)，commit `43bf913e8f6ddda53e3c21de07cb46bce7564d62`。根许可证为 GPL-3.0，README 另有非商业学术用途说明；两者均保留供使用者核查。

| 配置 | 实际过程 |
|---|---|
| `bayesmil_vis` | 变分权重、logistic-normal 注意力采样、CE 与权重正则 |
| `bayesmil_sdpr` | 上述过程加作者按 slide 类别选择先验的 patch 正则 |
| `bayesmil_apcrf` | SDPR 加真实全切片画布中的空间高斯平滑与采样 |

保留作者 LinearVDO 局部重参数化、门控注意力与变分分类器。默认输入隐藏宽度为 512、注意力宽度为 256；FM 输入维度由数据包声明。先验与二分类头按作者定义保留，三个配置均限定二分类。目标为 `CE + 1e-8 × weight_regularizer + 1e-6 × mean(patch_regularizer)`；vis 没有 patch 项。可选择按训练类别频数倒数、有放回地抽取训练 bags。

训练使用 Adam；ReduceLROnPlateau 根据 PULS 声明评价单位上的验证 MC 概率 log-loss 调度。验证选模指标仍由任务协议决定。优化器、调度器及 RNG 一起保存，恢复到完整 epoch 边界。

### APCRF 数据

在 `slides.csv` 每行提供实际 `width_level0,height_level0`，均为正整数。对应 encoder 的 `spatial` 必须声明 `unit: level0_pixels`、`origin: top_left`、正整数 `patch_size` 和来源记录；完整格式见 [空间协议](SPATIAL.md)。尺寸应来自 WSI 元数据，不从已有 patches 的边界推断。

坐标是与特征同行的真实 level-0 左上角像素；必须非负、在画布内，且每个栅格单元最多一条 patch 特征。不同 WSI 不能拼为同一个坐标平面。实现保留作者 `floor(width/patch_size)+1`、`floor(height/patch_size)+1` 的画布分配规则及包含空白格的 patch 正则；默认 `max_grid_cells: 4000000` 是单个画布的格数限制，不是总内存保证。

### 明确披露的修正和公式边界

1. **空间索引修正**：作者扁平化 stride 使用 `width//patch_size`，与分配画布的宽度差 1。改为实际分配宽度，二维 `[y,x]` 独立栅格测试验证了输出、目标及梯度；第一行输入仍与原代码完全一致。
2. **零方差反向修正**：原 LinearVDO 对全零输入计算 `sqrt(0)`，产生 `inf × 0` 的 NaN 梯度。修正仅在方差恰为零时选择零次梯度；前向值仍为零，所有正方差处的值和导数保持原样。不以取消 dropout 或跳过坏 batch 隐藏问题。
3. **原样保留 patch 正则**：作者函数直接平方 log-variance 参数，而不是将其指数化成 Gaussian 方差。当前配置保留并命名为 `literal author log-variance formula`，未声称它等于标准 Gaussian KL。未来若修正公式，必须增加不同配置并单独比较。
4. **MC 推理修正**：原验证代码生成多次预测后，部分指标仍读取最后一次抽样。PULS 明确平均各次概率，默认 16 次；不是平均 logits 后做 softmax。保存预测熵、平均条件熵和两者差值。固定隔离 RNG 保证重复评价可复现，且不消耗后续训练的随机状态。
5. 注册先验、温度和固定参数为 buffer，随设备移动并进入 checkpoint；不更改其数值。验证关闭 dropout，变分权重及注意力仍采样。

这些配置是带明示修正的作者实现适配，不表示完整复现作者数据、训练划分、早停规则或论文结果。小 bag 的 CPU 测试可缩减隐藏维度与 MC 次数；这些配置写入 request，不代表正式实验预算。

## CAPRMIL

来源：[mandlos/CAPRMIL](https://github.com/mandlos/CAPRMIL)，commit `b1f21a25e21d889a90bd604d29887abb3c1dbeeb`，修改版 MIT，含非商业限制。

使用原作者 CAPRMIL、TransolverBlock 和 TransolverAttention：实例到软上下文簇、上下文 token 自注意力、按软分配回写到实例，以及残差 MLP。`caprmil_mean`、`caprmil_attention`、`caprmil_gated` 分别使用作者 mean、attention、gated-attention 聚合器。默认 hidden=128、heads=8、clusters=4、depth=1、MLP ratio=4、dropout=0.1；无需物理坐标。

目标为 CE；AdamW 与余弦调度来自作者默认配置。也支持 `scheduler: linearcosine`，使用作者 LinearWarmupCosineAnnealingLR；要求 `2 <= warmup_epochs < epochs`。调度状态进入 checkpoint。作者传入但未生效的 `head_dim` 不作为有效超参暴露，实际每头宽度为 `hidden//heads`。

固定源码的 Lightning 入口引用未定义的 `TSMIL`。PULS 直接调用该仓库真实存在的 `CAPRMIL` 类，避免训练入口错误，未以其他模型替换。保留作者可训练温度及 attention 池化分支硬编码的 dropout 行为；三个配置分别与原网络对照 logits、CE 和全部参数梯度。两种 FM 宽度和实例顺序不变性均有测试。

## VarMIL / DeepSMILE

来源：[NKI-AI/dlup-lightning-mil](https://github.com/NKI-AI/dlup-lightning-mil)，commit `f33d6c0eb504d73bd822909f29e548bc186e5306`，Apache-2.0。它是 [DeepSMILE 论文](https://arxiv.org/abs/2107.09405) 指向的分类实现来源。

接入作者 `VarAttention`：对原始 FM 特征学习注意力，分别计算加权均值和加权方差，拼接成 `2D` 后线性分类。没有把它替换为投影方差的 VarPool。方差修正是作者的 `N_nonzero/(N_nonzero-1)`；单实例或只有一个非零注意力权重时明确拒绝。

原模型仅移除旧 Lightning/torchmetrics 的日志初始化；注意力、均值、方差和分类公式保持原样。默认 attention hidden=128，CE、Adam；`configs/varmil.yaml` 显式使用学习率 0.001、weight decay 0。

本条目覆盖**冻结特征 VarMIL 分类器**，不包括 DeepSMILE 的 SimCLR 编码器预训练。固定作者仓库中的训练复现 shell 文件为空，因此没有把取得模型定义等同于取得完整论文复现流程。患者划分、采样、验证选模遵循登记的 PULS 受控协议。

## 验证范围

专项检查涵盖三种 Bayes 网络与作者原始输出/损失/梯度对照、二维 APCRF 栅格、零输入数值边界、MC 均值与 RNG 隔离、三种 CAPR 聚合器及其调度、VarMIL 加权方差，以及七个配置的精确断点恢复、重载评价和画布元数据变更检测。实际结果与环境指纹见 [验证记录](VALIDATION.md)。GPU 与真实 WSI 验证尚待执行。
