# 方法完整名称与执行配置

0.121.0 候选修正了方法清单中的 11 个完整显示名称：它们现在沿用既有方法家族的默认执行配置。没有新增算法、默认架构、权重、训练配方或方法数量。

| 方法清单名称 | 既有执行配置 |
|---|---|
| FRMIL MICCAI 2022 | `frmil` |
| VarMIL / DeepSMILE | `varmil` |
| CAMIL channel attention | `camil_channel` |
| SmMIL / Smooth Operator | `smmil_early` |
| CAMIL context aware | `camil_context` |
| MR-Block / MR-AB-MIL | `mrblock` |
| R2T-MIL / RRT-MIL | `rrt` |
| Campanella MIL-RNN | `campanella_rnn` |
| SCMIL Sparse Context | `scmil` |
| ProtoSurv heterogeneous graph | `protosurv_repaired` |
| MGGMMIL autofluorescence | `mggmmil_uaf_original_graph_patient_reference` |

完整名称包含空格或斜线时，在命令行中加引号：

```bash
milplus inspect --method 'FRMIL MICCAI 2022'
milplus check --data /path/to/prepared_task --method 'FRMIL MICCAI 2022' --deep
milplus run --data /path/to/prepared_task --method 'FRMIL MICCAI 2022'
```

数据包、依赖和原始输入必须满足所选方法的要求。名称识别成功不代表原始权重、患者标签或空间信息已经准备好，也不代表论文实验已经复现。

显式执行配置 ID 保留原有含义；分类、生存、类别数量及专用输入条件仍由原有规则选择对应配置。技术家族 `icmil_coupled` 的查看入口继续列出三种架构，执行入口继续使用原来登记的完整 ABMIL 默认流程。

DyKo、DSPA-MIL／DSPA-U-MIL、FEATHER／MIL-Lab 仍是待完成条目，没有通过名称映射获得可运行资格。未知或有歧义的名称会报错。

验证保留实际安装的 0.120.0 父版本路由快照，比较所有 164 个家族、352 个显式配置及五类任务条件。此前已能识别的 150 个显示名称、全部显式配置的完整登记和既有家族默认选择保持一致；11 个问题名称改为既有默认流程。

候选保留父版本全部 267 份测试输入，另加 43 项路由回归。源码侧 581 项相关检查通过，包含只读路由、源码准备、实际公开 CLI 数据检查、线性演示运行及目录冷构建。这些检查没有执行全部方法的真实 WSI 训练；完整验收状态见根目录 `CURRENT_VALIDATION.json`。
