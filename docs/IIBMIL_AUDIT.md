# IIB-MIL 未准入训练的具体证据

核查日期：2026-10-16。固定来源：
[TencentAILabHealthcare/IIB-MIL](https://github.com/TencentAILabHealthcare/IIB-MIL/tree/a354ec6ea2b9ea2159e6784d54a6afb355c04b50)，
commit `a354ec6ea2b9ea2159e6784d54a6afb355c04b50`。
该条始终保留在 164 条计划中。下文保留 2026-10-16 的历史审查；
2026-10-26 的新核查与明确命名的完整训练重建见文末及 [IIBMIL.md](IIBMIL.md)。

已取得：残差实例编码器、bag transformer decoder、类原型更新、PartialLoss
置信度 EMA、数据读取及推理脚本。`main/engine.py` 只有 evaluate，README 的命令是 test。
本次也核查了 [官方 issues](https://github.com/TencentAILabHealthcare/IIB-MIL/issues)：
开放问题 1、2、3 均询问训练脚本或自定义数据训练。此页面是检索快照，不代表永久没有更新。

| 已核查位置 | 实际行为 | 尚需落实 |
|---|---|---|
| `main/util/loss.py` | PartialLoss 用 confidence 做软目标；原型 argmax 的 one-hot 用 EMA 写回；EMA 调度硬编码 200 轮 | 置信度初始化、训练/更新先后、辅助损失权重和真实调度调用 |
| `main/datasets/wsi_feat_dataset.py` | `batchY` 全一，允许全部类别；training 分支索引 `self.confidence_dir` | 该映射未在构造器创建，需确认原始建立与 checkpoint 恢复方法 |
| 同上 | mask 为 True 表示有效 patch；padding 为零且复用首个 patch 名 | 重复 confidence index 的写回冲突及 padding 排除策略 |
| `main/models/iibmil.py` | encoder 读取 mask 但未传入注意力；prototype 更新选择有效 patch 的 top10% | 小包 top-k 为零时的处理；模型掩码是否需作为修正版独立配置 |
| `update_prototypes` | 利用 `wsi_idx.size` 修正 EMA，并逐特征更新原型 | 索引对象应为何种类型、训练期调用时点与 no_grad 范围 |
| `configs/TCGA-NSCLC.yaml` | 300 轮、warmup10、EMA .99、AdamW、多个历史配置字段 | 这些字段不是完整 trainer 的替代，不能猜出未公开 loss 路径 |
| `main/engine.py` | bag softmax 后又传给 CrossEntropyLoss | 训练目标尚未取得；推理 loss 行为不应反推原始训练 |

[MICCAI 官方记录](https://conferences.miccai.org/2023/papers/327-Paper1110.html)
确认了原型和 confidence bank 是方法的核心；因此只保留 bag CE 的 wrapper 不作为 IIB-MIL 准入。
作者主页另有 CIB-MIL 后续工作，但不同论文/版本的代码不能无说明代替 IIB-MIL。

下一步是取得足够的原始训练定义，或明确实现并命名可追溯的重建配方，逐项校验
confidence bank、原型、梯度路径和断点恢复。当时并未完成该方法的训练接入或数值复现。

## 2026-10-26 更新

重新逐行检查发现，`confidence_dir` 与初始置信度是在该文件的顶层
`build()`（147–210 行）中创建，不在 Dataset 构造器内创建。
历史表格的构造器观察不足以判断初始化缺失；现已找到可直接复用和逐值校验的定义：
弱标签类为 `1-1e-6`，其余类合计 `1e-6`，银行只包含原始 TRAIN patch。

新适配保留原始完整网络、500 行采样、PartialLoss、200 分母的置信度 EMA 和
类比例修正的 top10% 原型更新。缺失的完整 trainer 仍没有取得。
因此，新三配置明确命名 `*_complete_reference`，披露损失系数、预热解释、更新顺序、
有效实例掩码和验证选模等重建决定；不能作为作者原始完整配方或论文指标已复现的证据。
