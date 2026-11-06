# HMIL 层级分类接入与 SCL-WC 审计

0.19 新增 `hmil_source_repaired`，属于原 164 条清单中的 HMIL。`--method HMIL` 是该配置的别名，实际请求和结果记录完整配置名。它执行修复后的作者公开训练路径；不声称已复现论文指标，也不将源码与论文之间的差异隐藏在统一 CE 训练中。

## 来源与修复依据

- [HMIL 作者仓库](https://github.com/ChengJin-git/HMIL)，当前锁定 `d4f5ad18d0258a8e1bebe2c5a37e909cc349a946`，根目录 GPL-3.0。模型和损失从独立下载的 Git 对象载入。
- 历史来源 `af6f85ae38303fa0afb14ea479c47355fb3eebbe` 独立登记为 `hmil_prior`，仅用于核对模型修复与验证，不增加方法数。
- [HMIL 论文](https://arxiv.org/html/2411.07660v1) 提供层级注意力、细到粗映射及课程训练的定义。

当前 `src/models.py` 的 class-wise pooling 产生 C×D 特征，但分类器为 `Linear(D,C)`，返回 C×C；`src/trainer.py` 却用一张切片的单个类别训练。原始形状无法完成该 CE 调用。历史版本采用共享 `Linear(D,1)`，转置后返回 1×C。本配置恢复这一结构，保留当前版本的其余网络。没有任意取矩阵对角线或平均矩阵。

另将无条件 `squeeze()` 改为明确的实例×特征形状，使单实例切片可以运行；CUDA 定值分配随输入设备；默认粗特征维数等于 FM 实际维数，细特征宽度为其 1/4（取整、至少 1），细注意力宽度再为其 1/4。原默认 1024→256、注意力 64 可原样运行，其他宽度明确记录在配置中。

作者当前指标收集器还将已 squeeze 成向量的 logits 按 dim=1 求 argmax。PULS 使用统一的患者/切片评价实现；独立作者训练对照仅修正该指标收集的向量与标量形状，不修改其损失、反向传播或优化器步骤。

## 必须提供层级标签

已有 `classes` 是有序的细类名称。在任务 `task.yaml` 中增加以下结构；这里仅用虚构类别展示格式，不能作为 BRCA 或 COAD 临床标签定义：

```yaml
classes: [fine_A1, fine_A2, fine_B1]
hierarchy:
  coarse_classes: [coarse_A, coarse_B]
  fine_to_coarse:
    fine_A1: coarse_A
    fine_A2: coarse_A
    fine_B1: coarse_B
  label_provenance: "请填写预先确定的分组定义与来源版本"
```

每个细类必须且只能映射到一个粗类，全部粗类必须被覆盖。映射、类别顺序和来源随 task.yaml 进入数据哈希；变更映射会阻止旧 checkpoint 续训。如果 `slides.csv` 提供 `coarse_label`，它必须与映射一致。模型不从数据文件名、癌种名称或测试表现猜测映射。

训练输入为单张切片的变长 bag，同一批次保留独立 bag；不允许把患者多张切片拼接后当作原方法。可以按已声明的患者 ID 对预测概率求均值，进行患者级评价。原方法所需的层级关系属于额外任务信息；没有合理层级定义的 BRCA/COAD 任务应使用其他兼容方法，不能为了运行 HMIL 人为编造粗类。

## 完整公开训练路径

网络保留特征再嵌入、两级门控 class-wise attention、两级分类头和归一化的细级 class-wise 表征。默认 Adam lr=1e-4、weight_decay=0、batch_size=64、200 epoch、无调度器、无默认梯度裁剪。最后不足一个 batch 的切片仍参加训练。每个 epoch 打乱训练切片；批内联合对比损失需要同时保留这些切片的计算图，内存随 batch 和 bag 长度增长。

源代码目标为：

```
alpha = 1 - (epoch / epochs)^2
L = alpha*(CE_coarse + attention_matching)
    + CE_fine + CE(mapped_fine_probabilities)
    + (1-alpha)*SupCon(normalized_fine_embeddings, labels)/batch_size
```

`epoch` 从 0 开始；实际最后批次使用其实际 bag 数作 SupCon 分母。attention matching 对同粗类的细类注意力求和，再按每实例的粗类向量计算 cosine。SupCon 把 class-wise 表征作为多个 views、使用全部 anchors。温度随 epoch 按完整 cosine 周期在 0.1–1.0 间变化。

这是公开代码路径，存在以下明确的论文差异：

- 论文 Eq.4 是映射后粗类概率的负对数；源码把概率再次送入 CrossEntropyLoss。本配置保留源码行为。
- 论文 Eq.6 的动态权重为线性 beta，且 bag alignment 在加权组中；代码使用平方进度 alpha，bag alignment 不乘 alpha。
- 论文正文写温度 0.1；代码使用 0.1–1.0 的周期温度，并额外按批次大小除 SupCon。

这些差异不能被一次软件测试消除。后续论文配方实验需单独命名和验证。

验证默认按细分类 `macro_auc` 选模；不查询测试集。默认 patience 等于 epoch 上限，因此按固定 epoch 完成；用户显式缩短 patience 属于配置变更。保存粗头概率 `coarse_p_*`、细头概率 `p_*`、细概率映射的 `mapped_coarse_p_*` 及两级目标。分别报告粗头、细头与映射粗概率的指标，不强制它们的 argmax 一致。

```bash
milplus setup --method HMIL
milplus check --data /path/to/hierarchical_task --method HMIL --deep
milplus run --data /path/to/hierarchical_task --method HMIL \
  --config configs/hmil_source_repaired.yaml --device cuda --output runs/hmil
milplus evaluate --run runs/hmil --device cuda
```

setup 同时下载当前与历史固定源码；不更改数据、申请 GPU 或启动训练。

## 验证范围

29 项专项测试涵盖：原始分类头形状错误、历史网络逐张量对照；6 组作者完整 epoch 的损失/梯度/Adam 参数对照（含 dropout、训练首/中/末阶段和最后单 bag 批次）；5 个必要损失项的有效梯度；课程与 bag alignment 语义；三个 FM 宽度下单实例训练；层级输入拒绝规则；切片/患者两种评价及精确断点续训。全库、独立配置与安装包验证结果见 [验证记录](VALIDATION.md)。

验证使用合成特征。GPU、真实 WSI 和原论文数值复现均未运行。

## SCL-WC 的历史0.19审计（不代表当前接入状态）

以下保留最初审计发现，不作为当前版本的接入结论。后续完整论文参考重建、
字面源码对照及参考终点角色条件见`SCLWC.md`；当前代码可用状态见根目录
`USABLE_CODE_STATUS.json`，不能仅依据本节历史标题判断是否已有实现。

[作者仓库](https://github.com/Xiyue-Wang/SCL-WC) 固定在 `d8d0ff44e887cadeeaff199cd771af9ee23e5855`。本次静态审查发现：

1. `training_methods/SCL.py::train_loop` 未把 epoch 传给模型；`models/public_models/SCL.py` 默认 epoch=0，而对比项仅在 epoch>4 时执行。
2. 即使补传 epoch，对比损失仅使用三个 memory queue 取出的 buffer 特征。入队由 `no_grad` 包装，队列不连接当前网络图；该对比项没有网络梯度。
3. hard/easy/positive 类别分支只处理二分类，queue 尚未填满时也可能抽到随机初始化条目。不能直接推广为任意多类配置。

因此未把仅能运行的 CLAM 式路径登记为完整 SCL-WC。仍保留原清单条目与具体缺项；后续需要依据论文明确当前特征与历史 queue 的配对和梯度方向，补齐完整训练后再验收。静态代码问题不等同于对原论文实验结果的判断。

## BRCA 单一二分类的实际层级阻塞（2026-11-06）

以实际856患者、UNI1024、IDC/ILC两类的未改slide任务包调用原生HMIL构造器，
因没有登记hierarchy而拒绝；TaxoMIL原生输入接口也在GPT2加载前拒绝同一缺项。
没有构造模型或运行GPU，也没有修改任何真实标签/划分。

仅在内存中使用明确标注的虚构父类作反事实接口控制：
一个共享父类被现有`2..fine_count`粗类约束拒绝；两个父类映射被接口接受，但两细类
覆盖两粗类只能是一一映射。这不等于用户已经提供真实临床层级，不能为开跑而自动采用。
非平凡层级需另外的真实标签/任务定义；不是补一份特征文件或下载语言模型能自行解决。
这些方法仍记输入/科学终点blocked，没有擅自扩充原25项任务排除。
共享证据`brca_hierarchy_and_ahdmil_actual_scope_r1.json`。
