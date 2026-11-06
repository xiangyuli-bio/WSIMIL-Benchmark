# LRMIL

完整接入候选：HR 教师 MIL → 低倍率编码器蒸馏 → LR 学生 MIL。正式准入、完整预算安装包验证和真实 WSI 实验仍分别验收。

使用 [作者源码](https://github.com/hvcl/LRMIL) `66ea1c5a447ed2a830bb1af6739ab9261e61000d`；采用完整原生 CONCH 视觉主干，768维、12层、12头、86,250,240参数，实际读取全部150个预训练视觉张量。没有用较小编码器替代。固定依赖为 timm0.9.12、torchvision0.20.1；安装 `pip install -e '.[lrmil]'`，按实际设备选择 PyTorch。

## 输入准备

LRMIL 需要原图和中间层，普通512维 CONCH 对比特征不能替代这些输入。一个标准任务包需要同时登记：

- HR 最终视觉 trunk 的768维 CLS 特征，作为主特征索引。
- 同序的 layer0、layer1、layer10 三层 HR CLS 特征。
- 同一 WSI 的5x RGB256低倍率 patch，以及20x高倍率 patch 的实测坐标。
- 每张切片的 level0/high/low MPP；低倍率 MPP 为高倍率的4倍。
- 真实初始化权重及其来源。当前固定 CONCH 权重 SHA256 为 `40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c`。

CSV 索引列为 `slide_id,path`，覆盖任务的全部切片。每个 HDF5 文件提供 schema 指定的 RGB、粗细坐标、最终特征和三个中间层；特征与坐标逐行对应。数据登记会验证文件、患者、划分、标签、范围、层宽及真实跨尺度匹配。局部组织缺失可以保留，系统不补造缺失 patch。

```bash
milplus setup --method LRMIL
milplus prepare-lrmil --data /path/to/task --index lrmil_index.csv \
  --schema lrmil_schema.json --output lrmil/manifest.json
```

准备后，在所选 encoder 的 `task.yaml` 中添加：

```yaml
lrmil:
  manifest: lrmil/manifest.json
```

```bash
milplus check --data /path/to/task --method LRMIL --deep
milplus run --data /path/to/task --method LRMIL --device cuda --output /path/to/new_run
milplus evaluate --run /path/to/new_run --device cuda
```

`milplus demo --lrmil` 只生成明确标记的合成 RGB/特征和完整物理网格，供软件检查；它使用真实编码器初始化，但不代表实际 WSI 特征提取。原始权重保留其原始使用条款。

## 实际配方

三段均使用 Adam，默认学习率0.0001、weight decay0。教师和学生按完整验证集 AUROC 选取最早最高值，patience20；编码器按 TRAIN 平均批损失选取最早最低值。每段分别以同一运行种子重新初始化，使用真正选中的教师与编码器传递到下一段。

Patch 蒸馏保留三层区域 MSE 和注意力加权特征 MSE。学生保留 CE、bag KL、soft attention KL、top/bottom 实例分类四项，权重分别为1、0.5、0.5、0.3；包含原始温度平方、batchmean 缩放和 raw attention 门槛。低倍率特征由实际蒸馏后的编码器生成，并绑定所选编码器、原图和输入版本的哈希；学生推理使用 LR 分支。

源仓库没有提供最大轮数、patch batch、patch损失系数等参数的完整默认配置。本库明确登记参考上限200/200/200、patch batch64、patch系数1/1和温度1/1；这些是补充的工程配置，不声称是原论文历史值。`configs/lrmil.yaml` 提供完整配置。短训练软件检查另行标记，不替代完整预算。

断点包含三份实际 Adam、各阶段状态与种子、真实采样顺序、选模历史、教师和编码器快照及传递哈希。已跑通完整原生编码器的三段软件训练；教师及 patch 训练循环分别与作者真实循环的权重、全部 Adam 状态和随机状态逐位对照通过。连续训练与跨阶段中断、多分类、完整当前源码检查及安装包验证继续单独登记。

源码布局、NHWC、教师元数据返回顺序和论文/代码差异见 [审计](LRMIL_AUDIT.md)。目前不报告真实 WSI 指标或论文数值复现。
