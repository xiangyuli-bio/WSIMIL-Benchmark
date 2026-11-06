# DGMIL：动态实例选择和负类特征分布

来源：[作者论文](https://arxiv.org/abs/2206.08861)、[官方代码](https://github.com/miccaiif/DGMIL)，固定提交 `a80e5fe51c7e5a3879d4a9b060e49c6b431f70e1`。作者定义从外部已校验 Git blob 加载；没有发现根目录许可证，本库不重新分发其代码。

## 实际接入范围

- `dgmil`：稳定排序选出不重复的极端实例，使用训练集负类分布进行推理。
- `dgmil_source_extremes`：保留作者 heapq + list.index 的实例选择方式，包括距离并列时重复选择首个实例的行为。推理同样仅使用训练集分布。

两配置属于一个方法家族，均为**冻结特征的二分类**。类别顺序必须明确：0 是用于分布拟合的负类，1 是正类。没有默认多分类或生存扩展。原论文的自监督 MAE 图像预训练是外部特征提取步骤，本次接入不是从原始 WSI 重训 MAE 的实现；已有 MAE 或其他 FM 特征必须登记权重和预处理版本。更换 FM 属于新的实验条件。

## 保留的训练路径

公开 `dynamic_MAE_fintuning_addslidelevel.py` 的有效训练过程为：

1. 从全部负类训练切片的 patch 特征建立分布，在当前空间计算负类和正类实例的距离。初始化使用原始冻结特征，后续使用学习后的线性投影。
2. 负类取距离最低的 10%，正类取最高的 10%，赋予 0/1 伪标签；始终取回**原始 FM 特征**训练，不将上轮投影再次输入下一轮投影器。
3. 按原代码固定 random_state=12345，将被选实例的 10% 留出；剩余实例做一次完整批次的 CE + Adam 更新。该内部实例划分不是患者验证集，也不用于最终选模。
4. 使用更新后的 `projection_head` 重新映射训练特征，重新拟合负类分布、更新下一轮选择。此处只取线性投影，不经过 BatchNorm、ReLU 或类别头。

默认 50 epochs、Adam lr=0.01、weight decay=0、无调度、无梯度裁剪。模型为作者的 `Linear_projection_MAE`：D→D 线性层、BatchNorm、ReLU、D→2 分类头，原生 D=768；输入维度适配实际 FM。不是把一个普通 bag CE 模型改名为 DGMIL。

分布步骤按源码执行：行 L2 归一化（eps=1e-10），用负类训练均值/标准差变换，Faiss KMeans（默认 10 clusters、100 iterations，作者默认聚类配置），每簇经验协方差的伪逆，以及各簇平方 Mahalanobis 距离的最小值。协方差、归一化参数、簇分配、极端实例索引、投影器、Adam 和 RNG 状态都进入 checkpoint。

每轮对原始训练实例重新计算；程序不会悄悄采样一个更小的特征库。所有训练特征及投影数据需要 CPU 内存，选中的完整批次需要设备内存。投影按 `projection_batch`（默认 65536）分块传输，协方差评分分块计算；这不是降低拟合样本量。需要缩减数据时应在实验协议中定义新的数据包并单独登记。

## 对公开脚本的修正

- 原切片测试段将测试特征同时传为 `get_score_and_dis_feats` 的拟合和评分输入。基准不采用这一条路径：验证/测试只读取训练负类得到的固定分布，不参与拟合。
- 删除每轮访问测试实例真值、计算诊断 AUC 的流程。由统一的患者验证集选择模型，显式 `evaluate` 才访问测试集。由于原诊断路径还消耗随机数，移除后聚类随机数流与完整历史脚本可能不同；训练函数数值对照限定在明确记录的训练路径。
- 作者对实例诊断 AUC 使用 argmax 类别；这里保留连续距离用于排名指标。
- `dgmil_source_extremes` 保留重复索引；`dgmil` 的并列项用稳定排序选择唯一索引。这一修正不会被标成逐字原实现。
- 拒绝单例簇、全零协方差、非有限特征、类别缺失和不足的极端实例，给出失败原因，不自动缩小 clusters 或退回 MeanMIL。奇异但非零协方差仍按原代码使用 pinv。
- 模型和工具的旧版 NumPy、无用 `models`/`data` 顶层导入不被执行；只读取需要的固定定义。源码中的 512 维 README 示例与主模型 768 维不同，运行协议记录实际维度。

## 距离分数与概率各自的用途

原方法的 bag 输出是平均 patch 距离，并不是已经校准的概率。预测 CSV 因此额外保留 `ranking_score`：

- AUROC/AUPRC 使用原始 `ranking_score`，不经 sigmoid 或概率校准，避免数值饱和或反转改变排名。
- 为统一导出 accuracy/F1/log-loss，每轮仅在训练 bag 的距离均值上拟合标准化的一维 LogisticRegression（C=1），将结果保存为 `p_0`/`p_1`。这是额外的报告层，不是论文原始模型或其概率复现，也不反向更新投影器。
- 若评价单位是 patient，`ranking_score` 在患者内按切片均值汇总，概率列独立取均值。若任务设置患者级训练，则先合并该患者所有切片的实例，再求 bag 均值。两种协议要预先选择，不能混用。
- 成对患者 bootstrap 同样使用原始排名分数。没有 `ranking_score` 的其他方法保持既有概率排名规则。

## 使用

```bash
python -m pip install -e '.[clustering]'
milplus setup --method dgmil
milplus check --data /path/to/binary_task --method dgmil --deep
milplus run --data /path/to/binary_task --method dgmil \
  --config configs/dgmil.yaml --output runs/dgmil
milplus evaluate --run runs/dgmil
```

使用 `--method dgmil_source_extremes` 对照作者实例选择行为。`model.clusters` 应按预注册配置设置；分类数据必须包含足够的负类和正类训练实例。不支持全局 `max_instances` 截断，避免训练分布与评价输入无声不一致。

## 验证

27 项专项检查包括：4 组独立作者分布公式对照（含原生 768 维），4 组连续三步作者完整训练函数/动态筛选对照，逐步核对梯度、参数、BatchNorm、Adam 和分布评分；原生 768 维网络更新；两配置精确续训；并列项反例；训练专用拟合；投影边界；推理排列/RNG/状态不变性；原始排名和患者 bootstrap；输入变化及退化分布拒绝。机器记录见 `validation/dgmil.json`。

证据仅为 CPU 合成软件检查。真实 BRCA/COAD、MAE 图像预训练、GPU 和论文指标复现尚未运行。不能将模拟数据上的指标作为研究结果。
