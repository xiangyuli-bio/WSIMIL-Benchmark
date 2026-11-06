# PIBD：公开源码生存配方

`pibd` 是 `pibd_source` 的别名。此配置接入患者级图像 + RNA 通路生存模型，不能仅凭 WSI 特征运行。它不是普通 WSI 分类聚合器。

来源：[作者仓库](https://github.com/zylbuaa/PIBD)，固定提交 `bd5bd94e6f8d48e7679c6e68209a3a65c9e56a78`；[ICLR 2024 论文](https://arxiv.org/html/2401.01646v2)。固定 README 声明 Apache-2.0，源码树没有单独 LICENSE；作者代码独立获取，不打包重发。默认外部目录为 `~/.cache/milplus/sources/pibd`。

## 输入与最短启动

任务必须是 `survival`，患者主表明确 `time`、`event`（1=发生事件，0=删失）、时间单位、终点定义、患者划分；所有患者必须有已对齐的 RNA 表和显式通路。DSS、OS、PFI 不能混用。论文使用 DSS，不会据数据集名自动推断终点。

```bash
milplus setup --method pibd
milplus check --data /path/to/survival_bundle --method pibd --deep
milplus run --data /path/to/survival_bundle --method pibd \
  --config configs/pibd_source.yaml --device cuda --output runs/pibd_fold0
milplus evaluate --run runs/pibd_fold0 --device cuda
```

数据包沿用 [组学/生存接口](SURVIVAL.md)。`omics.columns` 明确 RNA 特征顺序，`omics.pathways` 指向有序 `name/features` JSON 列表，`omics.preprocessing` 说明上游转换。归一化选择必须在任务文件中登记：

```yaml
omics:
  table: omics.csv
  columns: [GENE_A, GENE_B, GENE_C]  # 实际全部 RNA 列
  pathways: pathways.json
  normalization: train_global_minmax_preserve_zero
  preprocessing: 请替换为实际 RNA 转换与原始零值的含义
```

`train_global_minmax_preserve_zero` 忠实于作者 RNA 路径：将唯一训练患者的整个 RNA 数值表展平成一列，拟合单个 `MinMaxScaler(-1,1)`，变换后将原始零值恢复成零。它不是每基因标准化；验证/测试极值不参与拟合，不裁剪超出训练范围的数值。全零/常数训练表按 sklearn 的常数区间规则处理。该模式当前仅允许 PIBD，避免其他模型误用。

已有的 `train_standard` 或 `none` 可以用于显式受控比较，但属于不同的预处理条件。不会根据方法名称悄悄覆盖数据包的归一化声明。不隐式填补缺失检测，不构造虚假通路，不从基因名自动猜测临床标签。

## 保留的完整机制

- 每条 RNA 通路独立两层 SNN（Linear/ELU/AlphaDropout），WSI 与通路各有一个三层编码器和 8 个高斯原型。
- 四个离散时间区间 × 两种删失状态定义原型身份；训练以结局确定原型，推理仅按特征相似度选原型。
- Gaussian Monte Carlo 抽样、原型解码生存损失、KL 先验约束、正负原型对比及实例筛选。
- PoE 产生公共 token；解耦 Transformer 保留图像自注意力、通路自注意力和公共 token 跨模态注意力。
- 两个 CLUBMean 估计器分别估计模态间、特异/公共表征之间的互信息，并包含估计器学习损失。
- 单个 Adam 联合更新全部参数，包括 CLUB；公开训练器没有交替估计器优化或 detach，PULS 不擅自添加。

模型输出四区间 hazards 与 survival，风险为 `-sum(survival)`。患者多切片合并，验证只用患者风险/结局评价，不向网络传入结局。

## 默认配方与采样

作者公开路径默认：隐藏维度 256，4 个时间区间，患者批大小 32，训练外层 patch 预算 4096，内部 micro-bag 512，原型 Monte Carlo 抽样 50，WSI 保留比 0.5、通路保留比 0.8。支持明确覆盖隐藏维度、批大小、预算与抽样数；四区间原型字典保持原代码，其他 bins 直接拒绝。

每名训练患者先按 NumPy 不放回抽样、索引排序，再零填充至 4096。模型内部独立抽取 8 次至多 512 个位置；不同 micro-bag 可以重叠，同批患者共用位置索引。原代码传给模型的输入没有 padding mask，故零填充参与计算。验证/测试使用全部患者特征，仅执行模型内部抽样，不做外层填充。过短输入无法满足原型-实例 top-k 时明确报错。

默认训练 sampler 按训练患者的 `(time bin, event)` 组合逆频率、有放回抽取与患者数相同的样本。空类别不参与分母计算。可以显式设 `weighted_sampling: false` 使用无放回随机排列。保留尾批；尾批大小 1 时 CLUB 的正负样本互信息估计为零，估计器拟合项仍有梯度。

优化器为 Adam，学习率 `5e-4`、有效 weight decay `0`、30 epoch、无 scheduler、无默认梯度裁剪。作者参数表虽有 `reg=1e-3`，Adam 分支未传入它，故本配置不把它当成有效默认。PULS 在验证 C-index 严格改善时保存，默认 patience 等于 epoch 数；如显式调整参数，保存到 request 文件。

完整训练目标：

```text
(sum survival NLL
 + gamma * mean prototype contrast
 + alpha * sum prototype decoder NLL
 + beta  * mean prototype KL
 + sigma * (mean MI estimate + mean estimator fitting loss)) / actual_batch_size
```

默认 `alpha=.1, beta=.01, gamma=1, sigma=.1`；主 NLL 和原型解码 NLL 均使用作者删失权重 0.5。原型项按作者内部 micro-bag 平均。整个目标最终除以批大小，已经取均值的辅助项也再除，因此改变 batch size 会改变其相对权重。历史记录里的 `loss` 是更新批目标的均值，尾批权重也需结合 `batch_patients` 解读。所有分项和梯度范数都记录。

## 与论文、源码及统一协议的差异

1. 公开代码的 PoE 将原型 `sigma` 直接用于精度分母，再把所得倒精度直接作为噪声乘数，没有平方/平方根转换，也没有论文正文式 12 的单位高斯先验专家。`pibd_source` 保留该公开计算，不声称它与论文公式完全一致，不静默换成另一种 PoE。
2. 全部原型-实例对的 top-k 可重复选中同一实例，保留原行为。此处的有效实例数不一定等于 top-k 数。
3. 作者数据准备在分折前用全队列事件时间建立区间；PULS 仅用唯一训练患者已发生事件的时间计算四等分点。至少需要四个不同事件时间。留出集不拟合区间。
4. 推理保留源码内部固定 CPU 高斯种子，但在隔离的 CPU 随机流内完成整次前向，且首轮采样固定种子。不推进训练 RNG，也不重置其他 CUDA 设备。训练所有随机状态随 checkpoint 保存。
5. 设备硬编码 `.cuda()` 改为当前模块所在设备；默认支持 CPU 验证、CUDA 需真实设备验证。作者输入/目标不依赖 CUDA 索引副作用。
6. 使用 PULS 预先固定的 train/val/test、严格验证选模和独立测试；不提供从测试集选模型路径。未复制源码全队列临床指标/IPCW 拟合流程；当前公共评价是患者 Harrell C-index 和离散生存损失。

## 验证与边界

`tests/test_pibd.py` 对照独立读取的固定作者网络和完整 `_train_loop_survival`，比较三次连续 Adam 更新的参数与梯度，并检查所有核心分支、字面 PoE、归一化、输入拒绝、采样/尾批、随机流隔离和精确续训。具体当前结果见 [验证记录](VALIDATION.md) 及 `validation/pibd.json`。

所有本轮实验是 CPU 合成数据软件验证。真实 BRCA/COAD 的 RNA、终点、患者划分、真实 WSI/FM 特征、GPU 性能和论文指标复现尚未执行。该条目进入统一库，并不意味着可以用“WSI 文件夹”替代必要的分子输入。
