# 源码检查记录

本轮固定 MIL_BASELINE commit `e1c5d9fcd771f30f94437e993ecf24c0535df991`。
当前工作区没有旧报告提到的 Starter ZIP、186 条原始 Excel 或 29 项审计附件。
不能把那些叙述当作本轮已经重新验证的事实。

## 本轮直接读取的证据

| 对象 | 观察与工程处理 |
|---|---|
| `modules/TRANS_MIL/trans_mil.py` | Nystrom、CLS、PPEG 已存在；本工程保留该固定适配并记录来源，没有重新归为普通 Transformer |
| 官方 CLAM `models/model_clam.py` | 实例分类头、top/bottom 选择、subtyping 分支均接入；不只使用 logits |
| 官方 DTFD `Main_DTFD_MIL.py` | 两个优化器、AFS/MaxS/MaxMinS；本工程修订保留计算图在优化器 step 后反传的方式，并记录推理分组差异 |
| 官方 MHIM 主分支与 `v1` 分支 | 当前主分支为 v2；当前配置固定 v1 并记录 warm-start 训练配方，保留教师、feature distillation、掩码与 EMA |
| 官方 ACMIL `Step3_WSI_classification_ACMIL.py` | 总损失包括 slide、各分支以及注意力多样性；GA 变体完整保留这些损失 |
| 官方 AEM `main.py` | attention negative entropy 与 bag loss 相加；本配置保留全 bag 路线 |
| 官方 WiKG `model.py` | knowledge weight 的 einsum 对两个特征轴独立求和；本适配明确保留而非偷偷改式 |
| 上游 `modules/REMIX_MIL/remix_mil.py` | 观察到 `Label mixing (simplified to uniform mixing)` 及固定 0.5；未将其作为完整 ReMix 方法启用 |
| 上游 `process/AC_MIL/process_ac_mil.py` | 常规 train/val/test 分支每 epoch 计算 test；PULS 不继承该流程，改为独立显式测试入口 |

未逐条完成源码审查的方法维持 pending_adapter。来源锁中“git_remote_resolved_only”表示 HEAD 可解析，不表示论文身份、许可或训练机制已全部审核。

## 软件验证

测试涵盖病例划分、全局身份别名、H5/NPY/PT、坐标长度、空包、非有限值、CLAM 实例头更新、DTFD 两级更新、MHIM 教师与恢复、训练集标准化、完整 Cox 风险集、预测配对、患者 bootstrap、CLI 与批量调度。

当前软件验证使用合成特征。真实切片解码、特征提取速度、H200 训练、作者报告指标误差、跨 FM 适用性均未由这些测试证明。
