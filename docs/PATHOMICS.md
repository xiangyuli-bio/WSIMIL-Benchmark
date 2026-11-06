# PathOmics：两阶段跨模态预训练与生存微调

接入的是 [作者仓库](https://github.com/Cassie07/PathOmics) 的 `9705d6137f2974923c978597c18e7198bf888e23`，论文为 [Pathology-and-genomics Multimodal Transformer for Survival Outcome Prediction](https://arxiv.org/abs/2307.11952)，MICCAI 2023。作者代码在独立缓存中按 Git blob 读取，不随本仓库分发；没有发现根目录许可证。

## 三个执行配置

| ID | 生存预测使用的表征 |
|---|---|
| `pathomics_concat_source`（别名 `pathomics`） | 图像和组学，作者 concat MLP |
| `pathomics_image_source` | 图像分支 |
| `pathomics_omics_source` | 组学分支 |

三者都先进行成对图像/组学的无监督 MSE 对齐预训练。一个前向模型不能代替该阶段。`_source` 指作者网络和默认参数冻结行为；以下修正仍然明确存在，不能宣称运行环境和数据协议完全等同原论文。

数据包需要患者生存时间、事件（1=观察到事件、0=删失）、冻结 patch 特征，以及 `omics` 表和有序 gene-group/signature 定义。使用现有 [组学数据协议](SURVIVAL.md)。按患者合并多张切片，不把切片计作独立事件。每个患者至少有 gene-group 数量那么多的 patch；不足时拒绝空组，而不制造补充实例。

作者示例使用八个 gene family；适配器由数据包明确给出数量和成员，避免猜测基因归属。固定 FM 输入维度由数据包读取，作者原生为 1024；隐藏维度默认 256，可显式设为其他 8 的倍数。不同 FM 或宽度是新的实验条件。

## 完整配方

1. 对每个患者随机打乱 patch 索引，按 gene-group 数量均分。每组经过作者 DimReduction 和门控 attention，得到图像 token；组学每组经过两层 SNN，得到分子 token。
2. 两个独立的两层 Transformer 分别建模图像和分子 token。默认使用作者 README 的 gated attention pooling（`use_gap: true`），再通过各自 projection，计算配对 MSE。
3. 预训练默认 25 epochs、Adam `lr=1e-4`、weight decay 0、每 32 名患者累积一次。**作者预训练累积是损失之和，没有除以 32**，这里保留。预训练验证只计算 alignment MSE，不调用生存分类头或用生存标签选预训练权重。
4. 恢复验证 alignment MSE 最低的不可变权重快照，进行 25 epochs 生存微调。新的 Adam 默认 `lr=5e-5`、weight decay 0，没有继承预训练 Adam 动量；不使用额外 scheduler。微调使用作者 sigmoid hazard、累积生存概率和删失 NLL（默认 alpha=0），每个累积窗口平均患者损失。
5. 以独立 validation 的 C-index 选最终权重（可显式改为 `survival_loss`）；显式 `evaluate` 才评价 test。预训练阶段不会生成 `best.pt` 生存模型或触发最终选模 patience。

支持作者的 `SNN`、`Attention`、`SNN_Attention` 分子编码路线、GAP/逐 token 对齐、projection ratio。`Attention` 对单个分子向量的通道做 softmax；`SNN_Attention` 的各组引用**同一个** MHA。保留这些源码行为，不将其描述成独立 gene-token attention。`dropout` 控制作者相应层，SNN 和 gated-head 中固定的 0.25 仍按源码保留。

## 参数冻结与源码问题

`freeze_policy: source_zip` 是默认值。作者 `bash_main.py:349` 将 `named_parameters()` 与 `children()` 用 zip 遍历，遍历长度以直接子模块数为限。于是只冻结早期部分参数，后续 backbone 参数仍可训练。适配器逐字执行等价选择；协议记录**实际所有可训练参数名**，测试独立执行作者这段循环对照。

显式设置 `freeze_policy: heads` 可使用修正版：完整遍历全部参数，只训练源码白名单中的 attention head、rho、fusion、classifier、projection 和 logit_scale。此选项应单独登记实验条件；它与默认源码行为不同。冻结参数后，训练模式 dropout 仍然开启，与作者训练模式一致。

其他明确修正与边界：

- 作者 `best_pretrain_model = model` 是可变别名；这里保存张量 clone 并在阶段切换时恢复真正的最佳预训练权重。
- 作者每 32 个患者才 step，末尾不足一个窗口的梯度不会 flush。这里完成尾窗口：预训练仍求和，微调按尾窗口实际人数平均，随后清空梯度。
- 作者时间分箱在划分前完成；这里只使用训练患者的观察事件时间分位数。分子标准化按数据包选择 `none`（已定义的输入 z-score）或 `train_standard`（仅训练患者拟合）。预计算 z-score 的来源与潜在队列暴露仍需数据提供者登记。
- 作者脚本 `omic_modal=miRNA` 实际读取 `data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt`；不要将这份数据标成 microRNA 检测。
- 源码 bilinear 分支引用 `BilinearFusion`，本仓库版本没有该类定义，因此**不提供可运行的 bilinear 配置**。不以别的方法融合层悄悄替代。
- 主脚本向 `kmeans_cluster` 分支传空 cluster 列表；这里只接入完整的 random-group 路线，不声称已实现该实验。
- Cosine 预训练分支缺少所需 target，未接入；未声明支持参数重置、COAD→READ 外部预训练权重加载和独立少样本切分实验。
- 图像或组学单分支输出对另一模态的数值独立，但**当前 CLI 数据包仍要求成对模态**，尚非允许病例缺失另一模态的部署接口。训练预训练阶段本来也需要成对输入。
- 推理的随机分组使用隔离的固定 seed，重复评价一致且不消耗下一轮训练随机数；不沿用原源码每次评价重新随机分组的波动。
- 索引跟随输入 device；增加 hazard logits 返回项用于统一输出，不改变作者 hazard/NLL。没有修改宿主驱动或申请 GPU。

## 使用

```bash
milplus setup --method pathomics
milplus check --data /path/to/survival_bundle --method pathomics --deep
milplus run --data /path/to/survival_bundle --method pathomics \
  --config configs/pathomics_source.yaml --output runs/pathomics_concat
milplus evaluate --run runs/pathomics_concat
```

总 epochs 必须大于 `model.pretrain_epochs`。默认配置为 25+25；小型软件检查可显式设置总 epochs=3、pretrain_epochs=1、hidden=16、accumulation_steps=3，不能把该结果作为论文复现。

## 验证范围

36 项专项检查覆盖三配置 × 两种冻结策略的独立作者完整训练循环（共 48 次连续 Adam 更新，逐项比较参数及梯度），三配置在预训练中、阶段边界及微调中的精确续训（9 组），三种组学编码 × 两种 pooling 的原生 1024/256 网络梯度对照，以及快照别名、尾批、训练专用拟合、无标签对齐、RNG 隔离和输入失败处理。机器证据见 `validation/pathomics.json`。

所有证据属于 CPU 合成软件验证。真实 TCGA COAD/READ/BRCA 多模态病例、CUDA 环境及原论文指标尚未验证。
