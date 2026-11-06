# HEALNet 与 HE2RNA

两条记录属于原定 164 条范围。`healnet` 是有组学输入的生存融合；`he2rna` 是图像预测基因表达，RNA **只作为目标**。二者不能在相同输入声明下视为两个普通分类聚合器。

## 来源与接入范围

| 配置 | 作者来源与提交 | 许可 | 接入流程 |
|---|---|---|---|
| `healnet` | [konst-int-i/healnet](https://github.com/konst-int-i/healnet/tree/90459b65a3d4a4ef9fd405671c457b5a1163cc7d) | Apache-2.0 | 两模态共享 latent 融合、加权离散生存 NLL、L1、Adam / OneCycle |
| `he2rna` | [owkin/HE2RNA_code](https://github.com/owkin/HE2RNA_code/tree/d47f481fbcaab41a49635e4a35e810bf30eae0e0) | GPL-3.0-or-later | 空间 supertiles / 原始特征、随机 top-k、表达回归、训练均值初始化、验证相关性选模 |

通过 `milplus setup --method healnet` 或 `milplus setup --method he2rna` 获取固定来源。代码从核验后的 Git blob 提取作者定义，源码树存在受跟踪修改或提交不符时拒绝加载。第三方代码不随本仓库重新分发。

这两条配置的证据是 CPU 合成数据软件检查。真实 TCGA 训练、GPU 训练和论文数值复现尚未完成。HE2RNA 的 MSI 迁移分类器、论文统计显著性分析和特定基因预训练权重微调不是本配置已经完成的内容。

## HEALNet

数据包沿用 [生存和组学协议](SURVIVAL.md)：`time,event` 为患者终点；`omics` 为患者对齐的输入，所有分区按 `global_patient_id` 隔离。每个患者的切片在稳定 manifest 顺序下拼接，这是明确的 PULS 患者适配。Fourier 位置编码表示**序列位置**，不能宣称使用了真实组织坐标。

默认网络取作者 `config/hyperparameters.yml` 的通用结构：2 层、4 个 latent、宽度 16、两个 Fourier 频段、cross head 1、latent heads 8、SELU 门控前馈，dropout 0.5。作者另有每癌种调参配置；本库默认参数不冒充这些调参结果。默认 `omics_layout: vector` 与作者 `omic_attention: True` 的实际张量相同，即 `[B,1,G]`。可显式选择 `gene_tokens`，得到 `[B,G,1]`，对应另一条作者输入组织路线。

每层依次用组学、图像更新共享 latent；开启一次 self-attention 时，在每个有效模态更新后执行，保留作者的循环位置。`self_per_cross_attn` 只接受 0 或 1：固定源码中大于 1 会解包失败。

训练保留以下定义：

- 训练患者的观测事件时间决定分箱；组学标准化仅使用独立训练患者。可用 `normalization: none` 保留已登记的上游值。
- 分箱权重默认反频率，并在损失函数中归一化。作者实际公式为 `(1-alpha) * weight * (event_nll + censor_nll) + alpha * event_nll`；最后的事件项没有乘分箱权重。默认 `alpha=0.4`。
- 默认每 4 个患者更新一次，NLL 除以 16，再加全模型 L1。这里 16 是作者源码的损失除数：源码每批都调用 `step()`，并非跨 16 批累积。`loss_divisor` 可显式修改，协议记录其值。
- Adam 无 L2，OneCycleLR 每个批次更新。统一配置的 `lr` 指该 cycle 的 `max_lr`，默认 0.008。使用 Torch 默认 cycle 参数，起始值为 max_lr/25，beta1 也由调度器管理。
- 变长患者包逐患者计算并积累批次均值梯度，尾批按实际患者数归一化。无 padding。与无 dropout 的作者整批更新做数值对照；有 dropout 时串行变长读取不保证同于上游批张量的随机数消耗顺序。
- 验证按作者的批均值再平均计算未加 L1 的加权生存损失，选择最小值；不查看测试结果。模型、Adam、OneCycle 和全部随机状态一并保存恢复。

修正公开源码中的三处问题：移除跨模态计算的裸 `except: pass`；将缺失模态的 `continue` 移出 `if verbose`；缺失模态仍保留原 `[omics,wsi]` 槽位。`eval_missing: omics` 或 `wsi` 是明确的评价消融：训练仍使用两种模态，验证/测试跳过所选模态。它不是缺失临床检测的自动填补功能；标准数据包仍要求完整患者对齐组学。两模态同时缺失会报错。

没有复制作者在划分之前对全队列均值填补和分箱的预处理。PULS 使用训练侧拟合，并拒绝隐式缺失值填补。

```bash
milplus run --data /path/to/survival_with_omics --method healnet \
  --config configs/healnet.yaml --output runs/healnet
milplus evaluate --run runs/healnet
```

## HE2RNA 数据包

这是独立的 `task_type: regression`。`slides.csv` 保留 `slide_id,patient_id` 和可选全局身份；不要求分类 `label`。特征索引与固定患者划分沿用公共协议。增加以下内容：

```yaml
task_type: regression
training_unit: slide
evaluation_unit: patient
regression_targets:
  table: expression.csv
  columns: [ENSG00000198851, ENSG00000167286]  # 示例；必须换成已审核的目标列
  transform: log10_1p
  preprocessing: "实际 assay 与定量单位，例如 GDC FPKM-UQ；保留上游处理版本"
  label_provenance: "实际病例—RNA 样本匹配表与数据版本"
```

`expression.csv` 示例：

```csv
global_patient_id,project,ENSG00000198851,ENSG00000167286
TCGA-AA-0001,TCGA-BRCA,12.4,6.8
TCGA-AA-0002,TCGA-BRCA,4.1,2.7
```

示例行并非真实患者数据。每个全局患者仅一行，患者集必须与任务完全一致，所有目标非缺失且有限；未检测不能填为零。`log10_1p` 对非负 assay 值执行作者的 `log10(1+x)`；数据已变换时显式选择 `none` 并说明。不能对已经取对数的数据重复变换。目标、基因顺序、project 和变换配置都进入输入哈希。

目标基因列必须事先明确，不按测试队列表达分布筛选。`project` 控制分层评价，按原始数据组织或预先批准的队列合并规则填写；本库不自动把 LUAD/LUSC 等项目合并。缺失癌种或 assay 对齐需要先在数据准备阶段处理，不能由训练器猜测。

## HE2RNA 输入与训练

默认 `input_mode: supertiles`，按每张切片的真实二维坐标 KMeans 分为 100 组，再对组内 patch 特征取均值。原实现用 libKMCUDA，本实现显式使用 sklearn Lloyd、10 次初始化和固定种子；算法后端变化已经登记，不能声称逐位复刻原作者预处理。可给 `input_mode: tiles` 使用原始或已经登记的 supertile 特征，系统不再二次聚类。

supertiles 逐片缓存于 `~/.cache/milplus/he2rna_supertiles`，可由 `MILPLUS_HE2RNA_CACHE` 改位置。缓存键包含特征、坐标、配置、sklearn 版本及适配代码哈希；缓存内容另有校验值。缺少坐标、可用不同坐标数少于分组数、或总实例数小于 `max(ks)` 会报错。不得用序列方格伪造空间坐标。通用 `max_instances` 不适用此配置，应在数据准备和本方法参数中明确实例预算。

模型仍使用作者共享 Conv1D MLP，默认隐藏层 `[1024,1024]` 和 dropout 0.25。训练每批共享一个随机 `k`，按基因独立选择最高分实例后平均；默认 ks 为 `[1,2,5,10,20,50,100]`。推理分别计算所有 k，再平均预测。

原作者以输入通道最大值 `>0` 判断 padding，不适用于任意 FM 的有符号特征。本库不接受隐式 padding，所有已登记实例都是有效的；因此有效负值特征不会被删除，负实例得分也不会被伪零值覆盖。原始非负、无 padding 条件下，输出、MSE 和梯度与作者定义对照。

最终偏置初始化为**训练单位**目标均值：默认 slide 训练会重复计入多片患者，与原 slide 训练一致；若显式设 `training_unit: patient`，先逐片聚类再拼接，同一患者只计一次。MSE 使用原始网络输出，ReLU 仅用于导出的表达预测和相关性。Adam 默认 lr=0.0003、无 weight decay，batch size=16；变长包串行累积批均值梯度，尾批用实际大小。选模默认验证 project × gene 的平均 Pearson。

输出 CSV 包含 `target_i,raw_i,prediction_i` 及 project、患者、切片身份；`method_protocol.json` 保存对应目标基因顺序。MSE 比较 raw 与目标，MAE 比较 ReLU 后的表达预测与目标。患者评价先平均患者切片预测，再分 project、gene 计算相关性。目标或预测为常数、样本不足时相关性为 `null`，报告可定义/不可定义数量，不用 0 冒充相关性；若全部相关性不可定义，默认选模会明确失败，可预先指定 `selection_metric: mse`。

```bash
milplus run --data /path/to/tcga_expression --method he2rna \
  --config configs/he2rna.yaml --output runs/he2rna
milplus evaluate --run runs/he2rna

# 软件演示：合成包实例少于默认 100，使用明确的小配置。
milplus demo --regression --output /tmp/expression_demo
milplus run --data /tmp/expression_demo --method he2rna \
  --config configs/he2rna_demo.yaml --output /tmp/expression_run
milplus evaluate --run /tmp/expression_run
```

## 验证与边界

`tests/test_healnet.py`：4 组原网络输出/输入和参数梯度；3 种权重 NLL 手算；缺失模态槽位与异常传播；训练专用预处理；完整批次优化与调度对照；3 种缺失消融精确续训。

`tests/test_he2rna.py`：作者训练/推理输出和梯度；有符号特征；真实坐标聚类与缓存校验；训练均值偏置；共享 k 的批次更新；无效标签拒绝；标签哈希；分队列和患者评价；两种输入模式完整训练、续训、重载与测试；方法/任务不兼容拒绝。

统计工具 `milplus compare` 当前仍是分类的配对 bootstrap，不能直接对回归或生存输出使用。回归相关性的患者 bootstrap、HE2RNA 论文中的基因显著性检验与 RNA assay 跨队列标准化需独立研究协议，不属于这里的软件检查结果。
