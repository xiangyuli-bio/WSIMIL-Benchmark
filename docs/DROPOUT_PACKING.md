# MIL-Dropout 与 PackMIL 接入记录

本轮增加两个候选家族、六个执行配置。保留训练机制、独立验证集和源码锁；软件测试与真实 WSI / 论文数值复现分别记录。

## 来源与版本

| 方法 | 作者代码 | 固定提交 | 许可证 |
|---|---|---|---|
| MIL-Dropout | [ChongQingNoSubway/MILDropout](https://github.com/ChongQingNoSubway/MILDropout) | `61af86f56a24685a916d0e3e4b416cb4751323bb` | MIT |
| PackMIL | [FangHeng/PackMIL](https://github.com/FangHeng/PackMIL) | `b261cf0abd9c5ce384927722061e15a93b3489f7` | Apache-2.0 |

作者源码通过 `milplus setup` 单独获取，运行时检查提交和工作区，再从 Git 对象加载所需定义；不在本工程中重新分发。PackMIL 仅加载当前 ABMIL 路线所需定义，未选择的骨干及 asymmetric loss 分支会明确失败。

## MIL-Dropout

四个配置为 `mil_dropout_abmil`、`mil_dropout_gated`、`mil_dropout_dsmil`、`mil_dropout_transmil`，均调用该仓库的真实模型。`MIL-Dropout` / `mil_dropout` 默认解析为非门控 ABMIL。

保留的操作顺序为：三个浅层 Linear/ReLU，每层 ReLU 后计算实例通道均值的 sigmoid，以此选择 top-k；再对剩余实例取每个 top-k 实例的余弦近邻并求并集；屏蔽这些行，乘以实例总数 / 保留行数。行数不变，被置零的位置仍参与后续网络。推理关闭 MIL-Dropout，单实例 bag 按作者代码原样返回。

参数默认 `topk=3, kernel=4`，对应作者模型文件；README 中独立 dropout 类的默认 kernel 是 7，不能混为一套实验参数。`topk=0` 属于无 dropout 消融，当前方法配置明确拒绝；需消融时应另行登记。

| 配置 | 浅层宽度与聚合 | 训练设置 |
|---|---|---|
| `mil_dropout_abmil` | D→D//2→256→128，非门控注意力 | bag CE、AdamW、Lookahead |
| `mil_dropout_gated` | 同上，门控注意力 | bag CE、AdamW、Lookahead |
| `mil_dropout_dsmil` | D→D//2→256→hidden；默认 hidden=256，作者 FCLayer+ReLU 选择关键实例 | bag CE、AdamW、epoch cosine |
| `mil_dropout_transmil` | 同上，默认 hidden=512；作者自带 Nystrom、PPEG 和 CLS | bag CE、RAdam、Lookahead |

[论文 Table 3](https://arxiv.org/html/2504.14783v1) 列出上述优化器类别和 CE。ABMIL 两路线默认学习率 / weight decay 均为 1e-4；DSMIL 与 TransMIL 默认 lr=1e-4、weight decay=5e-3。Lookahead 的间隔 5、插值系数 0.5 是显式 PULS 默认值，已核材料未给出这两个数；两者均可配置。慢权重、更新计数、fast optimizer 动量与 cosine 状态全部保存。Lookahead 评估使用当前 fast 权重。

作者 DSMIL 文件只返回 bag logits，实例头用于离散关键实例排序。此路线的实例分类头不获得 bag CE 梯度；本工程保留这一源码行为并作测试。它与原始 DSMIL 的双分支 BCE 是不同训练配置，不会额外补入未声明的实例损失。

两个兼容修正：

- 创建 mask 的 `.cuda()` 改为跟随输入设备和 dtype；TransMIL 的 CLS 跟随输入设备。
- 全部实例被屏蔽时明确报错，要求用户降低 `topk/kernel` 或增加 bag 大小。作者代码在此处为除零；本工程不会暗中减少 top-k 或留回某一实例。

原论文附录描述的部分浅层宽度与公开模型文件不同，本配置按固定代码执行并记录实际宽度。没有声称已复现论文全部设置，DTFD 的 MIL-Dropout 扩展尚未提供。

## PackMIL

配置为 `packmil_abmil`（分类）与 `packmil_survival`（患者生存），都保留作者 PackMIL、`get_packs`、ADS、带 mask 的 ABMIL、hyperslide 标签和辅助损失。`PackMIL` 默认解析为分类路线。

训练过程：

1. 对一个批次的变长 bags 进行作者随机分支划分。源码把 `int(N*token_dropout)` 个实例放到主分支，这一变量名不能直接理解为“主分支丢弃比例”。
2. 每个分支在包长预算内截取，并执行 ADS。ADS 保留注意力加权残差、嵌入、token 分组和随机通道选择，两个分支共享该模块。
3. 主分支按原始顺序组合成包，通过实例 ID 与 padding mask 隔离各 bag 的注意力；每个原始 bag 得到一个预测。
4. 剩余分支把包当作 hyperslide，使用同一个聚合网络和分类头，计算作者定义的混合标签辅助目标。
5. 优化主损失与加权辅助损失。包含不足整批的尾批；所有训练数据只来自训练集。

短于 `min_sequence * downsample_ratio` 的 bag，源码将完整 bag 送入两个分支，因此这时两个分支并非互斥。较大 bag 的两个分支按随机划分互补；超出包预算仍会截断。`max_instances` 若另行设置，会在打包前执行明确的 PULS 采样。

| 设置 | 分类默认 | 生存默认 |
|---|---:|---:|
| batch_size | 32 | 32 |
| pack_length | 2400 | 3200 |
| min_sequence | 128 | 256 |
| downsample_ratio | 4 | 3 |
| token_dropout | 0.5 | 0.5 |
| auxiliary_weight | 0.2 | 0.5 |
| lr / weight_decay | 0.0005 / 0.00001 | 0.001 / 0.00001 |

网络使用作者 GELU 投影、512 隐藏维度、0.25 dropout 和带 bias 的 ABMIL。默认参数参照作者 BRCA / survival 配置及命令；优化器为 Adam，学习率为无 warmup 的 epoch cosine，下限 1e-7。所选 epoch 数由运行配置保存，不根据测试表现调整。

分类主损失是 CE。辅助标签是包内类别计数除以最大类别计数，例如 `[0,0,1]` 生成 `[1,0.5,0]`，各维之和不必为 1。辅助损失保留作者 `0.25*(1-exp(-BCE))^2*BCE` 均值，不能当作普通 softmax mixup。

生存主损失是患者离散删失 NLL，分箱仅依据唯一训练患者的观察事件时间拟合。同一患者的切片先合成一个无空间关系的 bag，再进入批次。hyperslide 的时间、事件与删失混合规则直接使用作者实现：最早事件/删失分箱、最早观察事件及删失比例共同决定辅助目标。风险为负的离散生存概率和。测试患者不参与分箱，也不会与训练患者拼包。

推理走作者全量 ADS 路径：保留注意力和 MLP，关闭训练通道随机采样与 token 压缩，因此推理并不是简单地将实例数除以 r。推理无随机分支划分。

当前两配置不代表 PackMIL 的所有可替换骨干。PANDA 的特定分级映射、asymmetric single-label 损失及 BN 设置也尚未纳入；通用分类配置拒绝 ordinal 任务，避免错误套用分级标签混合。可以继续在同一家族下扩展这些明确命名的配置。

## 运行

```bash
milplus setup --method mil_dropout
milplus setup --method packmil

milplus run --data /path/to/classification_bundle --method mil_dropout_abmil \
  --config configs/mil_dropout_abmil.yaml --device cuda --output runs/drop_abmil
milplus run --data /path/to/classification_bundle --method packmil_abmil \
  --config configs/packmil_abmil.yaml --device cuda --output runs/pack_abmil
milplus run --data /path/to/survival_bundle --method packmil_survival \
  --config configs/packmil_survival.yaml --device cuda --output runs/pack_survival
milplus evaluate --run runs/pack_survival --device cuda
```

测试覆盖作者模型输出与梯度、mask 及缩放、全屏蔽拒绝、Lookahead 参数更新、两种任务的完整打包损失、主分支独立性、hyperslide 标签/删失公式、ADS 梯度、短包复用、尾批、训练专用分箱和六配置精确断点恢复。实际运行记录见 `docs/validation`。所有本轮测试使用 CPU 合成特征；没有据此宣称 GPU、真实 WSI 或论文指标复现完成。
