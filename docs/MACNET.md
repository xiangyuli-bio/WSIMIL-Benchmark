# MacNet：原图、区域引导与完整训练

本条目属于原图编码器训练组。它需要真实 RGB patch、区域标注引导和完整 Swin-Tiny 编码器训练，不能将普通冻结 FM 特征送入另一聚合器后称为原始 MacNet。当前为候选接入；六配置的软件检查已经在前一冻结版本通过，当前版本正在加强数据约束，完整预算、独立安装包和整库准入分别验收。没有真实 WSI 或论文数值复现结果。

## 来源与运行环境

- 原始代码：<https://github.com/Prince-Lee-PathAI/MacNet>，固定提交 `7610d7a534cd833ba0b139136ed9af9b41f59f48`。
- 固定源码没有 LICENSE。本项目保存来源锁和适配层，不分发该第三方源码；用户在获准使用后配置私有源码缓存。
- 实际原始计算使用 Torch 2.5.1、torchvision 0.20.1、timm 0.9.12、CUDA；使用独立环境安装 `.[macnet]`。不应将不兼容的全部方法 extra 装入同一个环境。
- 原始脚本加载 Swin-Tiny ImageNet-22K 权重。使用微软模型库公开的 `swin_tiny_patch4_window7_224_22k.pth`，SHA256 为 `108a66d1241f874f8328dbb2876b320e8f541618a7ad9ebe8102f7f18dd4396e`。这不证明它与作者未公开的私有文件逐字节相同。
- 完整特征编码器为 27,499,482 参数、输出 768 维，读取真实 RGB 后按原始流程 Resize 到 96×96、ToTensor、Normalize(.5,.5)。原始物理视野、提取倍率、切块和筛选规则仍需登记。

## 六种配置

| 条件 | 类别 | 未标注 patch | 每包附加区域引导 | 原始训练 batch |
|---|---:|---:|---:|---:|
| public | 2 | 3–1995 | 5 | 1 |
| lscc81 | 3 | 961 | 81 | 2 |
| cscc64 | 3 | 961 | 64 | 2 |

每种条件分别提供 `source_cached` 和 `refresh_repaired`，构成六个明确命名的配置：`macnet_{条件}_{类型}_patient_reference`。`MacNet` 家族名根据准备后的实际条件选择 `source_cached`。不得通过补零或重复 patch 达到输入计数。

`source_cached` 保留源码初始化时缓存的自适应对角矩阵数值，并把它加入检查点，恢复其原始导数连接。原始 `state_dict` 遗漏该矩阵，直接重建模型会改变预测。`refresh_repaired` 每次前向重新构建对角矩阵，是明确命名的修正条件，不是对论文 PSD 度量的复原。两者都保留原始 GFK/PCA/GSVD/核平方根、三个簇、proxy 权重、CE + 最小距离 − 最大距离目标。

## 准备真实数据

先按公共任务协议准备 `task.yaml`、`slides.csv`、患者 TRAIN/VAL/TEST 划分，以及 `raw_index.csv`。编码器登记为原图接口、维度 768；它的 index 指向每张 slide 的 HDF5，而不是冻结特征。临床标签和划分需要先核查；引导区域不是由 slide 标签自动推断。

可以用 `pack-macnet` 将已有 RGB patch 无损打包。输入 CSV 必须且仅包含：

```text
slide_id,patch_id,source_slide_id,global_patient_id,role,image_path
```

每个 bag 的未标注行必须在前，`role=unlabelled`；实际肿瘤区域引导行在后，`role=prior_tumor`。`image_path` 相对 CSV 目录解析。必须是真实 uint8 RGB、至少 96×96，包内尺寸一致。打包不缩放、不选 patch、不改变通道、不推断标签、不隐式排序。

```bash
milplus pack-macnet --table /data/macnet/patches.csv --output /data/macnet/packed
```

输出 `raw_index.csv` 以及 HDF5 的 `rgb`、`patch_ids`、`source_slide_ids`、`global_patient_ids`、`roles` 五个数据集；打包记录保留所有源图像和归档的校验码。将输出 index 绑定到任务编码器。

区域标注 JSONL 每行包括：

```json
{"patch_id":"actual_patch_id","source_slide_id":"actual_source_slide","global_patient_id":"actual_patient","region_label":"tumor","image":"/data/macnet/actual_patch.png","annotator":"actual_annotation_source","provenance":"actual_region_annotation_version"}
```

`image` 也可用 `{"path":"/data/origin.h5","dataset":"rgb","index":0}` 明确引用实际源图像。系统逐像素核对引导图像和该源图像，校验患者、patch 身份和顺序。不能用合成标注冒充临床标注。

默认建议 `train_bank`：全部 bag 共享同一有序 TRAIN 区域引导库，不能按 query 的真实类别选引导。另一种 `per_slide_annotated` 要求每张 query 都有其自身真实区域标注，属于额外推理监督条件，必须单独报告。二者均不能据此声称仅需 slide 标签。

准备 schema，例如：

```json
{
  "schema_version":1,
  "condition":"public",
  "guidance_scope":"train_bank",
  "annotation_index":"/data/macnet/region_annotations.jsonl",
  "pretrained_weights":{
    "path":"/models/swin_tiny_patch4_window7_224_22k.pth",
    "sha256":"108a66d1241f874f8328dbb2876b320e8f541618a7ad9ebe8102f7f18dd4396e"
  },
  "provenance":"Actual WSI extraction and annotation version, physical field of view, sampling protocol"
}
```

```bash
milplus prepare-macnet --data /data/macnet/task --schema /data/macnet/schema.json --output /data/macnet/macnet_manifest.json
```

将任务相应编码器的 `macnet.manifest` 设置为输出路径。准备和重新加载都会核对相同的标注语义、原始图像、完整文件校验码、固定 TRAIN 引导库和患者划分；仅重新计算 manifest 校验码不能绕过这些关系检查。

```bash
milplus check --data /data/macnet/task --method MacNet --deep
milplus run --data /data/macnet/task --method MacNet --output /runs/macnet_001 --device cuda:0
milplus evaluate --run /runs/macnet_001 --device cuda:0
```

## 原始训练语义与修正

默认 100 轮，完整原始 RMSprop 每轮重建，CE 和几何目标全部保留。源码复用参数组字典，因此请求的学习率虽在第 50、75 轮变为 5e-6、1e-6，实际参数组仍保留 1e-5。默认 `source_mutated_group` 保留该行为；配置 `lr_policy: intended_schedule_repaired` 才显式实施请求的学习率调度，并作为不同实验条件记录。

默认 FP32。完整单设备计算替代源码四 GPU DataParallel 的执行包装，保留完整网络和输入。每轮记录实际 LR、所有 batch loss、真实数据顺序、验证平均 batch 准确率；首次有效验证用于初始化选模。TEST 仅在明确 `evaluate` 后前向。保存模型、原始缓存矩阵、梯度、RMSprop、scaler、随机状态和历史，可通过同一命令追加 `--resume` 恢复。

## 验收边界

两轮软件测试用于检查代码、训练机制和恢复，不等于原始 100 轮条件。`scripts/validate_macnet_native.py` 要求每个配置实际 100 轮、完整 public2000/private1042/1025 RGB bag，并在第 1、50、75 轮真实中断恢复，逐轮比较全部状态和两次新进程选模重载。源码和独立安装包分别运行。多程序共用 GPU 时，耗时不当作独占效率结果。

不将该原图及区域监督条件混入同一冻结 FM 的纯 slide 标签主比较。真实队列、临床标注、跨来源性能和论文数值复现仍需单独验收。
