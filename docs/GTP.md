# GTP：完整图分类配方与原生 GraphCAM

配置为 `gtp_source` 和 `gtp_repaired`，别名 `GTP` 默认选择修正版。两条路线属于一个方法家族。

## 固定官方来源

采用 [Kolachalama 实验室的官方 TMI2022 仓库](https://github.com/vkola-lab/tmi2022)，提交 `0d395a160a969578fa5ce66a8a7f0fd0817712b4`，MIT。外部获取并校验 Git 对象后加载 `GCNBlock`、Classifier、原生 ViT 和完整 LRP 层；不采用另外的推理 fork 或普通 Transformer 替身。

原论文使用预训练 SimCLR 产生的 patch 特征。本仓库明确适配冻结 FM 特征输入，保留图分类器的完整结构和训练目标。使用其他 FM 不等于复现了原始 NLST 对比预训练流程；原论文数值、原始队列与原始编码器预训练均未完成验证。

## 输入和图

需要标准二/多分类任务包、患者与切片映射、固定划分、真实二维坐标和物理元数据。独立切片作为训练单位；患者评价通过已声明规则融合，不拼接不同切片坐标系。

`encoder.spatial.patch_stride` 表示坐标单位中的相邻 patch 网格步长；未单独提供时沿用 `patch_size`，适用于已登记的无重叠网格。坐标必须能还原该真实网格。图精确使用源码文件名索引定义的 Chebyshev 距离：两个网格轴的距离都不超过1，且节点不等于自身。不用一般欧氏半径替代，也不生成序列伪网格。输入邻接矩阵不补自环；GCN 自己添加原始节点的残差分支。

读取保留 float64 坐标精度；图按原始特征行序生成，保留真实缺口。重复、非有限、非网格对齐坐标被拒绝。图和池化中间量按 dense 计算，超过已登记的 `max_dense_bytes` 时在分配前停止，不偷偷裁剪切片。

## 完整模型和训练

保留 GCN 邻接聚合、自分支、L2 特征归一化、仅有效节点的 BN；可学习 assignment；PyG2.6.1 的 min-cut 池化；CLS token；作者三层、八头、宽度64的 ViT，默认池化节点100。没有添加源码不存在的位置向量或另一套聚合器。默认分类头为作者的线性分类头。

训练损失完整为 CE + min-cut + assignment 正交正则，两种正则都参与反向传播。默认使用 scripts/train.sh 的 batch8 与 drop_last；CLI 原始默认 batch6 是另一参数，二者差异不合并隐藏。默认120轮、Adam lr=1e-3、weight_decay=5e-4、无梯度裁剪。MultiStepLR 的20/100里程碑、gamma=.1，以及每个 batch 开始前调用 `step(epoch)` 的源码顺序均保留。固定声明轮数预算，用验证切片 accuracy 保存第一份最优权重；测试标签不参与训练、选模或 query 前向。

Adam、BN 统计、CLS/池化参数、学习率状态、随机数和训练历史全部进入 checkpoint。测试只在显式 evaluate 时运行。

## 来源兼容与显式修复

两条配置共同处理设备分配并暴露实际 logits，不将原接口的硬预测当作 logits。LRP 和数学前向仍使用官方实现；不导入会解析参数或创建文件的主程序。源码的原生图条件保留在 `gtp_source`：无边图会使 min-cut 出现0/0，单个训练观察也无法执行原 BN，所以这些条件明确报错。

`gtp_repaired` 另外定义：无边图的 min-cut 项为0，其他图的项按原 batch 分母平均；其他图仍完全使用原 min-cut 路径，所有图的正交正则保留。仅一个有效训练节点时，BN 使用已有 running statistics，不更新其计数；其余 BN 操作与源码逐位相同。没有默认添加额外图边来改变上述目标。

## 原生 GraphCAM

评价导出的是作者真实的 attention-gradient/LRP rollout。逐类执行无标签前向和原始 logit 加权反向、调用 ViT 的 `relprop(..., method='transformer_attribution')`，再按原可视化脚本，用 softmax assignment 将池化节点相关性映射回 patch。

HDF5 包括 `graphcam_raw [N,C]`、`pooled_graphcam [K,C]`、`graphcam_display [N,C]`、`graphcam_display_defined [C]` 和真实坐标。display 按原可视化脚本的 min-max 与类别概率0.4下限缩放，不是 patch 癌症概率。原脚本对常量图的 min-max 未定义；两条路线都用0占位并明确标记 defined=false，原始相关性保留，不能把占位显示解释为有效阴性定位。

解释过程保留既有参数梯度、不更新 BN 或模型，不写官方脚本的固定 `graphcam/` 文件。逐类重新前向避免 LRP 缓存相互干扰；多个类别的结果已独立与原脚本逐类输出对照。未评估临床定位准确度。

## 使用和验证边界

```bash
python -m pip install -e '.[graph]'
milplus setup --method GTP
milplus run --data /absolute/task_bundle --method GTP \
  --output /absolute/new_run --config configs/gtp_repaired.yaml
milplus evaluate --run /absolute/new_run --subset test
```

专项检查覆盖原生宽度的完整官方网络和所有参数梯度、直接执行 main.py 原始训练内循环、BN/Adam/调度/RNG 状态、独立 min-cut/正交公式和梯度、真实 GraphCAM 数值、无边/单节点处理、精确续训、标签独立性、内存与输入边界。全库最终结果见 `docs/validation/summary.json`。

CPU 合成验证、GPU、真实 WSI、论文分数和定位指标分别登记；后四项尚未完成。

## BRCA UNI 完整输入的实际阻塞（2026-11-06）

2026-11-05T22:00Z，既有`gtp_repair2`尝试在原生1GiB dense保护处拒绝，未产生best/last或训练历史；
失败目录与原始训练台账保留。这不是已完成训练，也不能直接据此断言GPU显存不足。

22:09Z独立读取全部856袋真实坐标/2422295行，并逐袋校验原始坐标SHA、调用未改的`lattice_edges`：
628袋不满足已声明的物理1024px网格，228袋通过。仅增大内存保护仍会被原生几何契约拒绝。
原batch8最大TRAIN袋16848行的源码预估为46091373568字节，属于保护公式估计而非实测峰值。
没有降低batch、丢弃病例、取整坐标或改标stride1，也未启动一次已知仍会失败的重试。

共享凭据`gtp_real_native_prerequisites_diagnostic_r1.json`记录完整逐袋结果和原失败来源。
汇总将该原失败归为待真实物理输入准备的blocked，原失败台账不改；不是新增用户排除项。
诊断脚本初次因把NumPy坐标直接传给Tensor接口而失败，随后只修正调用为`torch.from_numpy`；
未修改活动方法代码，成功诊断没有构造网络或分配dense邻接矩阵。
