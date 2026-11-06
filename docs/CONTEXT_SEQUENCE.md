# MiCo 与 EfficientMIL

0.10 增加 MiCo 分类/生存和 EfficientMIL GRU/LSTM 四个执行配置。它们使用固定作者定义，目标、初始化、模型状态和训练恢复分别验证。方法家族增加 2 个，不把四个配置统计成四篇论文。

```bash
python -m pip install -e '.[clustering]'
milplus setup --method mico
milplus setup --method efficientmil
milplus run --data /path/to/task --method mico --config configs/mico.yaml --output runs/mico
milplus run --data /path/to/survival_task --method mico_survival --config configs/mico_survival.yaml --output runs/mico_survival
milplus run --data /path/to/task --method efficientmil_gru --config configs/efficientmil.yaml --output runs/efficientmil_gru
milplus run --data /path/to/task --method efficientmil_lstm --config configs/efficientmil.yaml --output runs/efficientmil_lstm
```

源码按锁定提交单独获取，没有分发未核许可的第三方源码。两个仓库均未找到根许可证，许可证复核仍待完成。

## MiCo

来源：[junjianli106/MiCo](https://github.com/junjianli106/MiCo)，commit `d679f352567ec9eb15a3bd7ac2026e448f4ced79`。论文：[MiCo: Multiple Instance Learning with Context-Aware Clustering for Whole Slide Image Analysis](https://arxiv.org/abs/2506.18028)。

使用作者真实的 patch/cluster 投影、距离或点积路由、soft/straight-through hard 分配、上下文残差增强、逐层 cluster reducer、门控注意力和最终分类器。默认 64 个可学习初始中心、512 维隐藏层、3 次增强、dropout 0.25。每个增强器只在当前切片内工作；生存路线先分别增强患者的多张切片，再汇合做一次患者级注意力池化。没有把患者切片预先拼成一个 bag，也没有用物理多尺度数据替代作者的上下文层级。

初始化沿用作者 `FaissKmeans` 的 L2 聚类。PULS 按 `slide_id` 排序，选取前 `cluster_slides` 个**训练切片**，默认 20；聚合这些切片的特征后计算中心。FAISS 内部的默认采样规则保留。原脚本遍历目录，不能识别 train/validation/test；这里用任务划分限制为训练集，保存实际参与初始化的 slide ID。未知来源的仓库预计算中心不会自动载入。

运行依赖固定 `faiss-cpu==1.12.0`，使用作者提供的 CPU 分支、单线程、显式 seed 1234、20 次迭代。作者 requirements 列的是旧版 GPU FAISS，因此本次 CPU 后端结果不被宣称为该旧环境逐位复现。`max_clustering_bytes` 限制聚类输入特征字节数，默认 1 GiB；它不保证总峰值 RAM。超限明确停止，可以预先登记更小的初始化切片集合，不能静默改用随机中心。训练开始后中心继续获得梯度。

分类目标为 CE，分类训练单位限定为单张切片；任务仍可按预先声明规则进行患者级评价。生存使用作者离散时间 NLL，默认四个区间、alpha=0.15，事件定义为 1=观察到事件。时间边界仅由唯一训练患者的观察事件拟合，风险为负的生存概率和。为稳定计算 NLL，作者生存输出增加原始 hazard logits，未改变网络计算；保存每个区间的 hazard 与 survival。

固定来源未提供 README 中引用的 YAML 训练配置。PULS 明确选择 Adam、保留作者的 bias/一维参数零 weight decay 分组；示例学习率和轮数是受控配置，不冒充论文精确配方。作者 `similarity_bias` 定义后未使用，保留其状态并明确记录；可学习 `similarity_scale` 的原始参数化保持不变，零值或非有限值会报错。

## EfficientMIL

来源：[chengyingshe/EfficientMIL](https://github.com/chengyingshe/EfficientMIL)，commit `28f71ab2bffa8ff5c159354ad1b59dbba7624490`。论文：[EfficientMIL](https://arxiv.org/abs/2509.23640)。

目前覆盖作者公开的 GRU 和 LSTM 变体。实例分类头计算每个 patch 的类别 logits；APS 使用最大类别 logit，加 0.3 倍余弦多样性和 0.3 倍预测熵选取 top-k patches。选中序列进入对应双向循环网络、残差/归一化和均值池化，再计算 bag logits。默认选择 64 个 patches、隐藏宽度 768、两层、dropout 0.1；输入特征维度从 FM 数据包读取。

配方对应固定源码 `train.py` 的多类别路径：

- 独立实例最大分支与 bag 分支，各占一半 one-hot BCEWithLogits。
- 附加 `1e-4 × sum(parameter²)`，同时保留 Adam 的配置 weight decay。
- Adam betas=(0.5,0.9)，默认 lr=0.0002，weight decay=0.00001。
- 50 轮 cosine schedule，最低 lr=0.000005；状态随 checkpoint 恢复。
- Linear 使用作者 `apply_sparse_init` 的正交初始化；循环层保留 PyTorch 默认初始化。
- 互斥类别评价使用 bag logits 的 softmax；当前接口不提供旧脚本的单 logit 阈值调优路线。

源码类构造器的默认 `hybrid` 与外层支持策略不匹配，PULS 和作者训练入口一样显式选择 `aps`。可显式使用 `top-k`、`random-k` 作为作者变体；后者在评价时使用隔离固定 RNG，重复推理不改变训练随机状态。

**APS 的公开实现分配 N×N 相似度矩阵。** 默认 `max_similarity_bytes=536870912` 仅限制该矩阵，不能当作总显存上限，也不能声称本适配实现是线性内存。APS 的多样性分母是 N-1，因此拒绝单实例 bag；减小 bag 长度必须登记为实验采样预算。

EfficientMIL 的 Mamba 路径在导入失败时会用普通 MLP 替代。当前库没有开放这条回退；自定义 BiMamba 后端尚待单独接入和验证。`EfficientMIL` 别名指向明确的 `efficientmil_gru`，结果中保留配置 ID，不能将其报告为 Mamba 结果。

## 同批核查、仍待接入

| 条目 | 固定来源中的具体观察 | 后续要求 |
|---|---|---|
| AGDR-MIL | `8e830683c0a695e1e1983f01d35dc85471bf327c`；`model.py` 的评分网络只生成布尔索引，真实 CE 反向没有评分参数梯度；头部 Sigmoid 后又传入 CE；低分样本在 eval 仍随机丢弃 | 0.50 开发分支已接入完整来源配方、YAML/TCGA 明确配置、原始验证随机状态与逐行输出；保留无梯度边界，退化输入明确拒绝。正式验收进度见 [AGDR-MIL](AGDRMIL.md) |
| PAMIL prototype | `7fe90096f64697bd6b68d4d7209e31f437a6b08f`；多分类版本的 active `push` 使用实例标签筛选，并替换 prototype Parameter；warm/joint 冻结原型；CE 接收 softmax 概率 | 0.33 已接入两阶段聚类、warm/joint/push/last、训练实例标注契约及独立多标签路线；源码行为与显式修正分别登记，见 [PAMIL](PAMIL.md) |

来源分别为 [AGDR-MIL](https://github.com/shuiguoc/AGDR-MIL) 与 [PAMIL](https://github.com/Jiashuai-Liu/PAMIL)。AGDR 的无梯度观察已有针对固定作者代码的自动化回归检查。PAMIL 的多分类 push 有实际实例标签依赖，不能把它当作纯 slide 标签流程。这些观察只针对固定提交，不推断历史论文结果。

## 验证边界

测试覆盖两种 FM 宽度下的作者输出、目标与梯度，MiCo 生存的多切片处理和删失目标、训练专用 FAISS 中心、患者总采样预算，EfficientMIL APS 手算值/顺序、双分支目标、L2、调度及随机推理隔离。四个配置均比较完整训练与中断恢复的模型和历史，重载后重新评价。

结果在 [VALIDATION.md](VALIDATION.md) 和机器可读证据中登记。CPU 合成特征检查不等于 GPU 验证、真实 BRCA/COAD 训练或原论文数值复现。
