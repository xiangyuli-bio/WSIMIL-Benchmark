# 连续生存分布与表型簇生存模型

0.11 接入 `scmil` 和 `deepattnmisl` 两个家族/运行配置。它们都有专用生存训练流程；没有改成普通 CE 或统一的全队列 Cox。原始作者来源单独获取，不随本仓库分发。

```bash
milplus setup --method scmil
milplus setup --method deepattnmisl
milplus run --data /path/to/survival_task --method scmil --config configs/scmil.yaml --output runs/scmil
milplus run --data /path/to/survival_task --method deepattnmisl --config configs/deepattnmisl.yaml --output runs/deepattnmisl
milplus evaluate --run runs/scmil
```

任务仍使用患者级 `time,event`，其中 event=1 表示观察到终点、0 表示删失；同一患者的标签必须一致，划分不能跨患者。没有正式标签的数据目录不构成一个可直接训练的生存任务。

## SCMIL：Sparse Context-aware MIL

来源：[yang-ze-kang/SCMIL](https://github.com/yang-ze-kang/SCMIL)，commit `ec6759a11ab19e68d5ee4d45df22ab31e470b7dc`，MIT。论文与源码对应关系见 [MICCAI 2024 页面](https://papers.miccai.org/miccai-2024/671-Paper2991.html)。这不是同名的 Sparsely Coded MIL。

### 实际网络与输入

使用作者 SoftFilter、ClusterLocalAttention、AMIL_layer 和 SurvivalMDN。SoftFilter 对原特征乘以可学习 gate，gate>0.5 的特征进入局部上下文；其他特征保留到最终聚合。局部组同时考虑特征和物理坐标，作者 scalar mean/std 归一化与加权方式保持原样。大簇按作者规则拆成较小窗口，在窗口内进行八头注意力。所有特征最终通过门控注意力聚合，进入混合密度头。

每张切片都必须有与特征同行的真实二维坐标，以及 `encoder.spatial` 中的单位、原点、patch_size 和来源说明。物理坐标格式见 [SPATIAL.md](SPATIAL.md)。特征维度须能被 8 整除。固定源码的 SoftFilter 路线跳过 fc1，因此保持 backbone 宽度等于 FM 输入宽度；不使用额外隐式投影改变该路线。`hidden` 仅控制 MDN 的隐藏宽度，默认 256。

保留作者 YAML 中的一层上下文、默认 cluster_size=64、feature_weight=0.8、dropout=0.25、100 个分布成分，以及共享可训练的 mu/sigma registers。源码中未使用的 `backbone.fc1`、`backbone.attens.0.atten` 仍保留，并在协议中标明；不能根据参数存在就声称它们被训练。

### 聚类后端与患者多片

原代码使用 cuML KMeans。当前默认 `clustering_backend: sklearn` 是**显式 CPU 适配后端**：保留 L2 聚类问题、k-means++、max_iter=5、tol=1e-4、random_state=1，设置 n_init=1。它不是 cuML 数值等价声明。作者网络的对照测试使用相同 CPU 聚类标签；GPU/cuML 路线尚未运行验证。选择 `cuml` 时必须实际安装依赖，不自动回退。

聚类针对当前切片的 gate 筛选结果逐次计算，不使用其他患者或测试队列拟合的代码本。零方差造成的非有限归一化明确报错，不能用虚构坐标补齐。只有一个选中 patch 时，沿用作者直接跳过局部注意力的行为。

原数据加载器以 WSI 为样本。PULS 的患者多片适配为：每张切片独立产生分布，患者分布是这些分布的等权混合，训练用这个患者混合分布的删失似然。**不同切片不合并坐标，也不是平均各片 NLL。** 一位患者只有一片时与作者网络/目标对应。该患者规则属于明示的 PULS 协议扩展，不宣称是作者原始实验的多片处理。

设置 `max_instances` 时，这是患者所有切片的总预算；每张片至少保留一个 patch，剩余额度按各片容量比例分配。训练随机采样，评价确定性采样；特征与坐标同步选择。

### 时间、似然与数值处理

任务必须声明：

```yaml
time_unit: months
```

作者模型使用月。若原表记录天，任务写 `time_unit: days`，同时在 model 配置显式给出 `time_divisor`（每模型月对应的天数），不能猜测采用 30 天还是其他换算。时间转换与换算值写入协议。NLL 的密度单位是模型月；跨不同时间尺度直接比较原始 NLL 没有统一含义。

MDN 对高斯随机变量做 softplus 变换，以预测正的生存时间。保留作者的权重、mu 截断、sigma softplus/截断、logsumexp 内 epsilon 和外层似然 epsilon。删失样本贡献 `-log(S(t)+1e-6)`；观察事件的贡献等价于 `-log(pdf(t)+1e-6)`。

作者直接计算 `log(exp(t)-1+eps)` 及其 Jacobian，float32 长随访时间会溢出。此处改成代数相等的 `t+log(-expm1(-t)+eps*exp(-t))` 和相应 Jacobian。中等时间范围的值、目标与梯度对照通过；100、500、1000 个月下检查有限前向和梯度。生存概率在浮点误差边界夹到 [0,1]，未移除原有 epsilon 或更换分布。

默认 Adam lr=0.0002、weight_decay=0.0001；沿用作者 bias/一维参数不衰减分组。提供梯度累积，最后不足整组的患者仍更新。示例 20 轮；主程序的早停与验证选模仍遵循登记的 PULS 配置。

### 输出与评价范围

保存从 0 到**训练集最大随访时间**的固定月数网格，默认 64 点；该网格进入 checkpoint。预测 CSV 输出 `survival_time_i`、`survival_i`、`survival_loss`。每个患者的标量风险定义为此网格上梯形积分 RMST 的负值，并据此计算患者 Harrell C-index。

该固定网格/标量风险是 PULS 的受控评价约定。**没有把它称为作者的 time-dependent C-index、IPCW C-index 或 Brier Score 复现。** 作者原评估协议与这些专门指标仍需独立对接；当前曲线与分布参数的学习过程已经接入。

## DeepAttnMISL

来源：[uta-smile/DeepAttnMISL](https://github.com/uta-smile/DeepAttnMISL)，commit `d7099ed88c452aec37f16fc8e93cc0c068794c2a`。这是作者 MedIA 2020 的 DeepAttnMISL 来源，不能同时当作独立 MICCAI 2019 原版的完整实现。固定仓库根目录未找到许可证，许可复核待完成。

### 表型聚合

使用作者 `DeepAttnMIL_Surv`：各表型共享 1×1 Conv、ReLU 和平均池化，表型级注意力汇合，再经含 dropout=0.5 的 MLP 输出风险。原始 4096 维 VGG 输入只在首层 Conv 的输入通道上推广到数据包声明的 FM 维度；默认 64/32 隐藏宽度保持原样。空表型以作者 dummy zero bag 表示，注意力 mask 阻止其贡献。

有两种明确输入方式：

- `clustering: kmeans`（默认）：PULS 补充的每患者特征聚类，合并该患者所有切片特征后用 sklearn KMeans 分配表型；固定 seed，n_init=10，默认 100 次迭代、10 簇。训练患者的分配进入 checkpoint，验证/测试仅对当前患者自己的特征聚类，且不修改训练缓存。
- `clustering: provided`：读取每片特征索引中的 `cluster_labels_path`；它必须是对齐的非负整数 [N]，范围在配置簇数内。标签来自同一患者级聚类，不能把各片独立产生的局部 cluster ID 直接混作同一表型。元数据示例如下。

```yaml
encoders:
  uni:
    # dimension/index/weights 等其他字段照任务数据包填写
    phenotype_clusters:
      scope: patient
      count: 10
      provenance: "真实的患者表型聚类生成流程、参数和版本"
```

作者仓库要求预先提供聚类标签，没有完整发布其特征提取/聚类流程。因此默认 KMeans 是明确的 PULS 预处理扩展，不声称已经复现原论文表型生成。坐标不会进入这个模型的聚类或池化。

### 专用训练配方

保留作者每批最多 16 位患者的 Cox 目标，分母风险集只来自该批；Cox 项按**所有患者数**平均，不改成按观察事件数平均。目标加 `1e-5 × L1(all parameters)`，Adam 默认 lr=0.0001、weight_decay=0.0005。全删失 batch 按作者行为跳过；末尾不足 16 人但有事件的 batch 保留。全 epoch 无观察事件会明确报错。

验证损失使用整个验证患者集的 Cox+L1；ReduceLROnPlateau 根据它调度，factor=0.5、patience=20。默认也以 `survival_loss` 选模，较低更优。checkpoint 包含优化器、调度器、聚类缓存和随机状态。原论文原数据划分、旧环境及表征仍未复现。

风险集计算使用稳定 logsumexp，保留作者 `T_j >= T_i` 的并列时间处理。作者 mask 在指数运算后乘零，极大 masked logit 可能出现 inf×0；修正为在指数前将无效项置为负无穷。常规输入输出/梯度与作者一致，溢出边界单独测试。

## AdvMIL 源码审计

[liupei101/AdvMIL](https://github.com/liupei101/AdvMIL)，commit `6853786624bbab36c2a08b35dcf20c7bdff1a105`，MIT。本轮固定并读取了生成器、判别器、RLIP、训练阶段、损失和输入配置，尚未添加可运行适配。

当前 `loss/utils.py` 的 `bce` 判别器假样本项实际为 `mean(log(sigmoid(fake)+1e-8))-1`，其梯度不同于标准 fake BCE。已有针对固定代码的数值检查；这只陈述代码公式，不推断论文结果。原样参考配方与修正版必须分别命名。RLIP 还要求真实 4×4 区域对应，不能随意把连续 16 个无空间顺序的特征当作一个区域。监督/半监督状态和交替优化流程仍待实现。

## 验证

专项检查涵盖 SCMIL 参数/似然/梯度对照、长时间数值边界、训练专用时间网格、多片分布混合、总采样预算、单位检查，以及 DeepAttnMISL 表型池化、空簇、患者均值 Cox、L1 批次更新、尾批、生成/提供聚类两条路线的精确恢复。实际结果见 [VALIDATION.md](VALIDATION.md)。CPU 合成验证、GPU、真实 WSI 与论文数值复现分别登记。

0.20 更新：AdvMIL 已完成连续 ESAT/ABMIL 两骨干与源码/修正目标四配置的适配，以上审计保留为历史发现；当前输入、配方和范围见 [ADVERSARIAL_SURVIVAL.md](ADVERSARIAL_SURVIVAL.md)。
