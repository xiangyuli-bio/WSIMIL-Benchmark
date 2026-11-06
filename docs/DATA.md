# 数据协议 v1

患者组学、通路定义、离散生存和 Patch-GCN 图输入详见 [生存与多模态接口](SURVIVAL.md)。

## 完整配置

```yaml
schema_version: 1
task_id: tcga_brca_idc_ilc
task_type: classification  # classification / ordinal / survival / regression / multilabel
synthetic: false
classes: [IDC, ILC]         # 类别顺序固定；有序任务须按实际等级排序
evaluation_unit: patient
training_unit: slide       # patient 时拼接患者所有切片，按病例更新
label_provenance: 具体来源和整理规则
slides: slides.csv
defaults:
  fm: uni_v1
  split: fold0
splits:
  fold0: split.csv
encoders:
  uni_v1:
    dimension: 1024
    index: features_uni_v1.csv
    weights: 实际权重哈希
    preprocessing: 实际 TRIDENT 版本、MPP、切块、QC、输入变换记录
```

`slides.csv`：

```csv
slide_id,patient_id,global_patient_id,label,wsi_path
TCGA-XX-0001-DX1,TCGA-XX-0001,TCGA:TCGA-XX-0001,IDC,/raw/example.svs
```

`split.csv` 每个 patient_id 只出现一次。若不同 patient_id 对应同一 global_patient_id，所有别名仍必须位于同一划分。

`features_uni_v1.csv` 每个 slide_id 只出现一次，特征路径也不能重复引用同一物理文件。相对路径相对于任务包，不相对于启动目录。

```csv
slide_id,path,coords_path,sha256,coords_sha256
TCGA-XX-0001-DX1,/features/example.h5,,实际SHA256,
```

缺失类别、空 bag、NaN/Inf、维度错误、患者泄漏、缺特征和坐标长度错误会停止训练。`--deep` 读取数组并计算内容哈希。坐标长度相同只是必要条件；行顺序仍须由同一次 TRIDENT 提取及有序坐标文件保证，不能随意拼接两次提取产物。

## 不同任务

- classification：互斥类别，默认 validation macro AUROC 选模。
- ordinal：类别顺序来自 `classes`，默认 validation QWK 选模；模型仍为多分类头，不能据此称为专门有序损失方法。
- survival：`time` 为正数，`event` 为 1=事件发生、0=删失。同一患者的所有切片必须有一致终点。当前 Cox 适配支持单终点、无左截断；风险值越大表示更早发生事件。默认 validation Harrell C-index 选模。

患者评价先对切片概率求平均，再算指标。患者 bootstrap 与患者概率融合是两个独立操作。切片标签可不同的任务使用 slide 评价，并保持患者聚类抽样。多个来源的病例身份需在数据准备阶段映射。

## 跨 FM

每个 encoder index 必须与 slides.csv 一一对应。比较前固定有序 patch 集和坐标、实际 MPP、视野、组织 QC 及模型输入变换；仅维度或像素尺寸相同不能证明输入等价。主研究与 native preprocessing 实验应使用不同 task/encoder 版本。

权重许可、预训练队列暴露、未知信息和输入缺项需要记录到外部研究台账。程序不把未披露自动认作无重叠。

## 成本与限制

默认读取完整 bag，不自动截断；`max_instances` 是显式实验条件。训练截断使用 seeded 随机采样，验证/测试使用确定性均匀索引。Cox 用两遍前向构建完整风险集的梯度，GPU 图内存按患者控制；患者数的风险矩阵仍是 O(P²)。超大队列需要另外实现经过验证的排序风险集版本。

`inputs.json` 的哈希检查包含全体特征内容，启动会产生一次完整读取成本。当前训练逐 bag 读取 HDF5，不提供无限缓存，也没有多 GPU DDP；多 GPU 用于并行独立实验。

## 外部实例聚类标签

特征索引可添加 `cluster_labels_path` 和可选 `cluster_labels_sha256`。数组必须是逐实例对齐的非负整数 `[N]` 或 `[N,1]`；HDF5 / tensor dictionary 的键为 `cluster_labels`。内容哈希进入输入清单，修改后不能复用旧运行。CBMIL 的两种聚类模式、簇数要求与 FM 关系见 [CBMIL.md](CBMIL.md)。

## 多尺度

CS-MIL 使用三尺度逐行中心对齐，HAG-MIL 使用三层四叉树。encoder.multiscale 定义物理视野和各层索引，父索引和中心几何必须通过检查；格式见 [MULTISCALE.md](MULTISCALE.md)。普通 TRIDENT 单尺度特征不能自动替代这些输入。
## 基因表达回归补充

`task_type: regression` 的患者目标表、project 分层、显式 `log10_1p`/`none` 变换及 HE2RNA 空间 supertile 输入见 [分子预测与融合](MOLECULAR_REGRESSION.md)。RNA 目标与 `omics` 模型输入分别声明，未检测的 assay 不会自动填零。

## 多标签与SMMILe空间资产

`task_type: multilabel` 使用声明顺序的端点列表及显式0/1列，不把缺失或未测视为阴性。患者身份、划分和患者级目标一致性均校验。SMMILe以真实坐标、物理步长和冻结superpixel资产训练，含两阶段和实例输出；完整schema、三条路线、泛化名称解析及配置见 [SMMILe](SMMILE.md)。独立概率不要求行和为1；其他不支持多标签的配置会明确阻止运行。


HDF5 frozen features must be embedded datasets backed by the hashed container. External links, soft aliases, virtual datasets and external raw-data storage are rejected: the container checksum alone cannot lock their computation payload. Convert such inputs to an embedded feature file and register its new checksum.
