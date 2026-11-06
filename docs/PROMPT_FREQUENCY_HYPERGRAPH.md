# S-DFP、MFC-MIL 与 DyHG

本轮增加 3 个候选家族、7 个执行配置。作者网络、必要训练流程、PULS 修正与验证范围分别记录；没有把作者论文的指标当作当前实现的结果。

## 固定来源

| 方法 | 作者仓库 | commit |
|---|---|---|
| S-DFP | [thomascong121/S-DFP](https://github.com/thomascong121/S-DFP) | `22ef367ee3c99ca2dbd1ed70d13130c1c354623e` |
| MFC-MIL | [WissingChen/MFC-MIL](https://github.com/WissingChen/MFC-MIL) | `145494ec0f221c8b1cc32843efeafb28426634c0` |
| DyHG | [Yuxuan-Chen7/DyHG](https://github.com/Yuxuan-Chen7/DyHG) | `dd41a86d76d41c69ae010f97c653877058b0619c` |

以上版本均未发现根目录许可证；许可状态保留待核。源码通过 `milplus setup` 单独取得，运行时检查提交和工作区，读取固定 Git 对象中的定义；本发布包不重新分发这些作者代码。

## S-DFP：两阶段提示训练

四配置：`sdfp_abmil`、`sdfp_gated`、`sdfp_dsmil`、`sdfp_transmil`。`S-DFP` 默认解析为 `sdfp_abmil`。输入是单张切片的冻结 `[N,D]` 特征，不要求文本或原图。患者多片可在评价时聚合概率，不把多张切片合成一个切片提示上下文。

接入作者 `model/s_dfp.py`，并使用作者仓库中的对应 MIL 网络：

1. 前 `warmup_epochs` 个 epoch：共享提示与 MIL 联合训练，调制 MLP 不参与前向，因此不更新。
2. 后续 epoch：以当前切片 patch 特征均值为输入，经两层 MLP 生成偏移，加到全部共享提示上，再联合训练提示、MLP 和 MIL。
3. `multiply` / `add` 对多个提示取平均；`prepend` 将提示作为额外实例附加在特征之后。具体顺序保留源代码。
4. 推理始终使用切片调制，遵循作者 runner。为避免把尚未训练调制器的 warm-up 模型当成最终模型，只允许第二阶段 checkpoint 参与最终选模；训练必须包含两个阶段。

| 配置 | 作者骨干与目标 |
|---|---|
| `sdfp_abmil` | 直接对原始提示后特征做非门控注意力；bag CE |
| `sdfp_gated` | 原始提示后特征的门控注意力；bag CE |
| `sdfp_dsmil` | 作者 DSMIL：线性 query、value 投影、关键实例与卷积 bag 头；所有实例的 slide-label CE 与 bag CE 各占一半 |
| `sdfp_transmil` | 作者 Nystrom、PPEG 和 CLS；bag CE |

DSMIL 路线不是原始 DSMIL 的 max-instance BCE 配方。其输入宽度就是 FM 特征宽度，不提供一个实际无效的 `hidden` 投影参数。ABMIL / TransMIL 的默认宽度沿用作者尺寸映射中 192→128、384→192、其余→512 的已接入约定；未提供的 FM 输入维度通过显式尺寸适配，不根据模型名称猜测维度。

优化器使用作者默认 Adam；提示和 MIL 有独立 weight decay，默认均为 1e-5。提示默认学习率 0.002。作者每步调度将**全部参数组**的学习率设为同一个经调度的 `prompt_lr`，即使初始 MIL lr 不同，本实现仍保留此行为。前五分之一训练步线性 warmup，之后 cosine；总步数、阶段、已完成步数及优化器状态均可恢复。当前采用每张切片一次更新，等于作者 `accumulate_grad_batches=1` 默认路线。

源码观察与修正：

- 作者 runner 按每轮 test accuracy / F1 保存模型；这里使用固定验证集选模，训练循环不评价测试集。
- 门控注意力前向残留 RGB 卷积及 reshape，二维特征会直接报错，而且这段 `H` 结果没有用于注意力计算。删除这段无效调用；两种注意力模型中无关的 RGB extractor 声明也移除，避免保存大量不参与预测的参数。
- `SDFP` 声明的 dropout 未在 forward 调用；保留此实际行为，不额外插入 dropout。
- 当前 S-DFP 与 MIL 联合更新，不能解释为冻结已有 MIL 权重的仅提示训练。
- 原仓库示例脚本有旧入口和旧参数，发布配置按当前 `train.py`、builder 和 runner 核对。CLAM、FRMIL、ACMIL 提示配方尚未被这四配置覆盖。

## MFC-MIL：多分辨率、频域与记忆干预

两配置：`mfcmil_abmil`、`mfcmil_transmil`；`MFC-MIL` 默认解析为前者。使用作者完整 ABMIL / TransMIL 路径及 `DownSample`、最后定义的 `FeatureFreProcessing`、`InstanceMemory` 和 `InterventionOnline`。

- 多分辨率来自序列方形重排、深度卷积与三个不同 dilation 的 stride-16 卷积及 max-pool。它是作者特征重组操作，不能声称读取了真实不同光学倍率的 WSI。
- 输入顺序被网络使用；同一切片、不同 FM 应保留一致的 patch 顺序。当前每个 bag 至少 16 个实例，不伪造小包实例来绕过卷积要求。
- ABMIL 路线先产生高低分辨率特征，再分别做频域投影与记忆读写；TransMIL 路线的先后次序及共享投影按其源码保留。
- `InstanceMemory` 使用可训练槽位，当前 bag 通过 softmax 写权重构造临时槽位、再读出。forward 不原地修改跨 bag 缓冲；训练通过梯度更新槽位。
- 两个尺度都调用 `self.causal`，作者另声明的 `causal_ll` 与 `fuse` 没有进入实际前向；本实现保留并披露，未自动改接成另一个方法。
- `FeatureFreProcessing` 在同一文件定义了两次，后者覆盖前者。实际 forward 是线性投影 → Hilbert 解析信号的实部 → 输出投影与残差。对实数输入，取实部基本恢复输入，已用数值和梯度检查确认；注释里的 phase-only、DCT、DWT 并未启用。

作者默认隐藏宽度 512。PULS 将 ABMIL 中硬编码的 reshape 改成实际隐藏宽度，将二分类头改成任务类别数；二分类原宽度路径另做完整作者对照。TransMIL 保留 Nystrom 库路径，设备固定的 CLS 操作改为跟随模型设备。

记忆参数创建有显式修正：作者先创建 `nn.Parameter` 再调用 `.cuda()`，设备转换可能使它成为未注册 Tensor，影响优化器和 checkpoint。这里先创建并注册 Parameter，再整体移动模型。当前配置因此是**记忆可训练且可保存的修正版**，没有声称与作者原 GPU 运行逐位一致；两个记忆槽均有有效梯度并实际更新。

| 配置 | lr | weight decay | 其他 |
|---|---:|---:|---|
| `mfcmil_abmil` | 0.0005 | 0.0001 | Adam，betas=(0.95,0.99)，AMSGrad |
| `mfcmil_transmil` | 0.0002 | 0.00001 | 同上 |

两者默认 100 epochs、10 epochs 学习率 warmup。尽管作者 helper 名称包含 linear，实际返回的是 warmup 后 cosine 到基础 lr 的 0.001 倍；这里按实际实现，逐步数值对照作者调度器。统一训练引擎另有显式梯度裁剪参数。作者每轮 test 日志不沿用；仍只用验证集选模。

## DyHG：动态超图

`dyhg` 使用作者 `DyHG.py`，输入为单 bag 的 patch 特征；无需物理坐标，超图由特征计算。

流程：特征投影 → 低秩超边 incidence logits → soft Gumbel-Softmax → 节点到超边与超边到节点传递 → 按节点维的 L2 归一化 → 与原始投影相加并除以二 → LayerNorm → 门控注意力池化与分类。

默认采用作者训练入口的 hidden=512、4 个超边、温度 0.05、dropout=0.25；CE、Adam、lr=1e-4、weight decay=1e-5。`num_layers` 在当前作者实现中仅改变缩放因子，并未建立多层循环，所以本配置固定为真实的一次传递，不声称具有可配置堆叠深度。`fc2` 未参与前向，保留为已披露的无梯度参数。

作者 Gumbel 采样在 eval 中仍然执行。PULS 保留一次真实 Gumbel 抽样，并用 `inference_seed` 隔离推理随机状态，使重复测试一致、验证不消耗后续训练随机数；没有替换成普通 softmax，也没有额外冒称 MC ensemble。单节点时去除错误的 `.squeeze(1)`，保留节点轴，正常多节点路径与作者逐位对照。

## 使用与验证

```bash
milplus setup --method sdfp
milplus setup --method mfcmil
milplus setup --method dyhg

milplus run --data /path/to/classification_bundle --method sdfp_gated \
  --config configs/sdfp_gated.yaml --device cuda --output runs/sdfp_gated
milplus run --data /path/to/classification_bundle --method mfcmil_abmil \
  --config configs/mfcmil_abmil.yaml --device cuda --output runs/mfcmil_abmil
milplus run --data /path/to/classification_bundle --method dyhg \
  --config configs/dyhg.yaml --device cuda --output runs/dyhg
milplus evaluate --run runs/dyhg --device cuda
```

TransMIL 路线需要 `prompt` extra 中锁定的 `nystrom-attention`。验证包含完整作者输出/梯度、两种 FM 输入宽度、多类别、Hilbert 实部语义、功能式记忆写入、超图手算与 singleton、提示三种组合、两个阶段参数更新、作者学习率对照、测试输入变更拒绝和七配置精确恢复。当前均为 CPU 合成验证；GPU、真实 WSI 与论文数值复现仍未完成。
