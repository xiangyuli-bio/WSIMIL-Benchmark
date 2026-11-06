# 方法实现与配方差异

这里区分：算法/模型来源、实际训练配置、软件测试、真实任务验证、论文数值复现。最后两项当前未完成。

## 基础配置

- `mean_linear`：原始特征均值 + train-only StandardScaler + LogisticRegression。与非线性实例分类后平均 logits 的 `mean_mil` 不同。
- `max_mil`：每个类别分别对实例 logits 取最大值。
- ABMIL / Gated ABMIL：冻结 WSI 特征适配；原始 [AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) 的 MNIST 图像编码器没有被当作病理 FM。attention 在实例维归一化。

## CLAM

模型来自 [mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM)，固定 commit 在 sources.lock.json。保留 SB/MB 分支与实例 top/bottom 分类器。训练目标为 `bag_weight × bag CE + (1-bag_weight) × instance CE`；支持 subtyping 的类外实例监督。

这是一组官方可选 CE 实例损失配置，没有把 smooth top-1 SVM 默认为已实现。小于 `2*k_sample` 的训练 bag 被拒绝，避免 top/bottom 实例重叠。默认 hidden 512 与官方 small 模型一致。

## DSMIL

来源：[binli123/dsmil-wsi](https://github.com/binli123/dsmil-wsi)。冻结特征的双流模型保留关键实例查询、bag 分支和实例最大分支，两个 BCE 损失等权。

本仓库的互斥多分类报告将两分支 sigmoid 平均后再次按类别归一化。该概率适配必须在与原结果比较时披露。没有执行其上游自监督编码器训练。

## TransMIL

模型来自固定 [MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE) 适配，使用 Nystrom 近似、两层注意力和 PPEG。序列填充形成的方格属于该架构操作，不会被登记成真实组织坐标。没有用普通 Transformer 替换 Nystrom。

## DTFD

依据 [DTFD-MIL](https://github.com/hrzhang1123/DTFD-MIL) 固定源码的两级训练。tier 1 更新维度压缩、注意力和组分类器；tier 2 更新蒸馏 bag 的注意力与分类器。

原训练先更新 tier 1，再对保留的计算图 backward tier 2。PULS 在 tier 1 更新前保存并 detach 其蒸馏特征：保持两个优化器的有效参数归属，避免现代 autograd 对已修改参数的错误。它不允许第二级损失意外更新第一级。

AFS 可用于多分类；MaxS/MaxMinS 保留正类 CAM 排序，只开放二分类。默认验证按输入序列确定性分组；原代码的多次随机分组均值推理尚未复现。本配置与原论文超参/多次推理的差异不能省略。

## ILRA

依据 [low_rank_wsi](https://github.com/jinxixiang/low_rank_wsi) 的双向低秩投影、门控残差与非局部池化实现。默认一个 pooling seed；`rank` 控制中间低秩潜变量。该配置不是原论文全量调参结果。

## WiKG

依据 [WiKG](https://github.com/WonderLandxD/WiKG) 的 feature-derived 图。没有强制要求物理坐标。保留官方 `einsum('ijkl,ijkm->ijk')` 的独立求和语义，而不是擅自将其改为点积。当前使用 bi-interaction、attention readout。

邻居分数按 query 分块，降低前向临时分数矩阵大小；依然存在特征和选中邻居的内存成本。未覆盖官方其他聚合器/读出变体。

## ACMIL-GA 与 AEM

[ACMIL](https://github.com/dazhangyu123/ACMIL) 的 GA 变体保留多个 gated attention 分支、top-k 中随机掩码、各分支 CE、bag CE 和分支 attention 余弦相似度惩罚。MHA 变体没有被冒充为同一个实现。

[AEM](https://github.com/dazhangyu123/AEM) 被接入为 ABMIL + 全 bag 注意力负熵正则，权重必须大于零。源码锁固定来源；没有启用已经被原 README 划去的 top-100 实验方案。

## MHIM v1 注意力路线

依据 [MHIM-MIL v1](https://github.com/DearCaat/MHIM-MIL/tree/v1)，没有把当前主分支 MHIM v2 与 v1 混用。

当前 recipe 的监督 warmup 在同一 train split 上完成，随后整模型初始化学生/教师、重置优化器动量并进入 masked training。它保留 high/low/random mask、high 候选随机比例、high mask cosine decay、teacher feature soft-target cross entropy 与 EMA。教师评估模式固定，不接收梯度。

这是显式的 PULS warm-start 配方，不是作者发布的初始化 checkpoint；没有实现 v1 的所有 attention backbone 和所有选项。warmup 阶段不能成为最终 MHIM 最佳 checkpoint。恢复文件包括教师状态、初始化标志、epoch 和 RNG。

## 生存基线

`abmil_cox`、`gated_abmil_cox` 是明确命名的 PULS 适配。患者多片拼成一个 bag；完整训练队列风险集使用 Breslow ties。两遍前向重放 RNG，使每个患者的梯度与第一遍风险估计一致。它们不等于 Patch-GCN、MCAT 或其他生存论文方法。

## 离散生存与多模态

`survpath`、`mcat`、`porpoise` 使用作者多模态模型，`patchgcn` 使用作者 GENConv 模型。四者共同使用患者级离散生存 NLL，保留事件/删失方向、训练患者时间分箱和逐档生存概率。模型、组学、图输入和配方差异见 [SURVIVAL.md](SURVIVAL.md)。这些正式命名适配具有各自的实际结构，仍与论文数值复现分开登记。

## 待接入方法

来源台账可查不等于实现可运行。图/多尺度、原图训练、文本/组学、持续学习、预训练 slide 模型必须接入各自的输入与完整过程。缺依赖、权重或必要训练阶段会阻断，不可自动降级为 ABMIL、普通 Mamba 或虚构坐标。

## 0.4：GP、分组与提示聚类

`sgpmil`、`mmil`、`mmil_coords`、`ptcmil` 的固定源码、实际损失、训练状态与适配差异见 [GP_PROMPT_MULTILEVEL.md](GP_PROMPT_MULTILEVEL.md)。四个配置覆盖三个新增家族，PTCMIL 生存路径没有计入本轮实现。

## 0.5：CBMIL

`cbmil` 的双视图训练、CLAM 实例损失、聚类输入、FM 维度适配、尾批处理及原实现差异见 [CBMIL.md](CBMIL.md)。

## 0.6：CMTA 与 MOTCat

`cmta` 保留跨模态编解码、翻译与停止目标梯度的双向 L1 对齐。`motcat`/`motcat_micro` 保留最优传输聚合及各自整包/小包训练目标。固定作者网络、POT 版本适配、输入要求及受控训练协议见 [CROSSMODAL.md](CROSSMODAL.md)。

## 0.7：CS-MIL 与 HAG-MIL

`csmil` 保留三尺度原始评分、簇采样和掩码注意力；`hagmil_ce` 保留四层 IAM、三层选区域与独立优化步骤。真实尺度对应、CE 配方范围、依赖版本和源码修复见 [MULTISCALE.md](MULTISCALE.md)。

## 空间平滑、概率空间注意力和 SlideGraph+

0.8 的 SmMIL early/mid/late/Transformer、PSA-MIL 和 SlideGraph+ 使用固定作者实现及各自训练目标。空间单位、算法参数约定、训练校准和图退化处理等差异见 [SPATIAL.md](SPATIAL.md)。这些是 3 个家族的 6 个配置，不是 6 篇独立论文。

## 0.9：Bayes-MIL、CAPRMIL 与 VarMIL

三种变分/空间 Bayes 配置、三种 CAPR 池化配置及冻结特征 VarMIL 使用固定作者模型。MC 平均、空间索引、零方差梯度、原样正则、调度与 DeepSMILE 范围见 [BAYES_CONTEXT_VARIANCE.md](BAYES_CONTEXT_VARIANCE.md)。

## 0.10：MiCo 与 EfficientMIL

`mico` / `mico_survival` 保留作者的训练专用 FAISS 初始中心、逐切片上下文增强与分类/患者生存目标。`efficientmil_gru` / `efficientmil_lstm` 保留 APS、真实循环网络、双分支 BCE、L2 和 cosine 调度。适配范围、原始输入、后端与尚未开放的 Mamba 变体见 [CONTEXT_SEQUENCE.md](CONTEXT_SEQUENCE.md)。

## 0.11：连续分布与表型簇生存

`scmil` 保留稀疏上下文和连续混合密度删失似然，`deepattnmisl` 保留表型共享卷积/池化、批内 Cox 与 L1。聚类后端、患者多片扩展、数值修正和评价边界见 [CONTINUOUS_SURVIVAL.md](CONTINUOUS_SURVIVAL.md)。

## 0.12：MIL-Dropout 与 PackMIL

MIL-Dropout 新增非门控 ABMIL、门控 ABMIL、DSMIL 和 TransMIL 四条作者模型路线，保留 CE、Lookahead/epoch cosine 及显式全屏蔽拒绝。PackMIL 新增 ABMIL 分类与患者生存路线，保留真实打包、剩余 hyperslide 分支、ADS、混合标签和专用辅助目标。固定来源、输入、优化配方与未覆盖变体见 [DROPOUT_PACKING.md](DROPOUT_PACKING.md)。

## 0.13：S-DFP、MFC-MIL 与 DyHG

新增三家族七配置，均有完整作者网络、训练机制与恢复检查。S-DFP 保留两阶段联合训练，MFC-MIL 提供明确的记忆可训练修正版，DyHG 保留动态 Gumbel incidence；关键代码与论文语义边界、输入和适配差异见 [PROMPT_FREQUENCY_HYPERGRAPH.md](PROMPT_FREQUENCY_HYPERGRAPH.md)。

## 0.14：两个同名 CAMIL

`camil_channel` 保留四层 Nystrom/MCAB、通道门控、RAdam/Lookahead 和梯度累积；`camil_context` 为官方 TensorFlow 邻域上下文模型的独立方程移植，保留真实坐标构图、字面卷积轴和平方 sigmoid 融合，并与作者输出、目标、梯度及优化器比较。固定版本、兼容修正、缓存和不同 FM 适配见 [CAMIL.md](CAMIL.md)。
