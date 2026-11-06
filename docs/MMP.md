# MMP：多模态原型与生存训练

来源：[ICML 2024 论文](https://proceedings.mlr.press/v235/song24b.html)、[作者仓库](https://github.com/mahmoodlab/MMP)，固定提交 `1fd75e37592f7ab47b787f737733ba3c5a7e714c`。作者 LICENSE.md 为 CC-BY-NC-SA-4.0；通过外部独立 checkout 加载固定 Git 对象，不在本包复制作者实现。

## 四个配置

| 配置 | 图像原型 | 多模态融合 |
|---|---|---|
| `mmp_panther_source`（`mmp` 别名） | PANTHER 的混合概率、均值、对角协方差 | 作者 full attention |
| `mmp_panther_uot_source` | 同上 | 双向可微 UOT + 模态内 attention |
| `mmp_ot_source` | 固定原型的 OT 均值 token | 作者 full attention |
| `mmp_ot_uot_source` | OT 均值 token | 双向可微 UOT + 模态内 attention |

四个配置属于一个 MMP 方法家族。使用源码计算的配置不是论文数值复现认证。当前不暴露作者排名损失、单模态消融或任意叠层数；不把额外控制实验计成新的独立方法。

## 数据与运行

需要患者级生存数据包、冻结 patch 特征、患者对齐的分子表和显式有序通路。任务声明 `time`、`event`（1=事件，0=删失）、时间单位、结局定义、固定患者划分和 RNA 上游处理。见 [生存接口](SURVIVAL.md)。BRCA/COAD 的 WSI 文件夹本身不能替代 RNA 和生存标签。

推荐按作者流程在数据包声明 `omics.normalization: train_standard`，标准化仅拟合唯一训练患者的每个基因。`none` 可用于明确标注的受控条件；不会默默改写数据包。PIBD 的全局展平 minmax 不适用于 MMP。

```bash
python -m pip install -e '.[transport]'
milplus setup --method mmp
milplus check --data /path/to/survival_bundle --method mmp --deep
milplus run --data /path/to/survival_bundle --method mmp \
  --config configs/mmp_source.yaml --device cuda --output runs/mmp_fold0
milplus evaluate --run runs/mmp_fold0 --device cuda
```

换融合配置只需修改 `--method`。POT 固定为 `0.9.6.post1`，显式选择与作者 0.9.4 对应的 entropy 正则，避免新默认 KL 改变方法。

## 预处理与完整训练

1. 从当前训练划分的 WSI 按每张相同 quota 抽 patch，预算为 `prototypes * prototype_patches`。只保留实际收集的行，采用局部随机流，保存抽样来源、数量、种子和原型内容哈希。
2. 使用作者提供的 sklearn KMeans 路线拟合中心；显式设置 5 次初始化、最多 50 次迭代及 seed，区别于作者 FAISS 默认和 sklearn 版本相关的隐式初始化次数。验证/测试不会参与拟合。
3. 全部患者切片特征按患者合并，用冻结 PANTHER MAP-EM 或源码 OT 核转换成固定数量的 token。默认 PANTHER 为 16 原型、一次 EM、tau=1、初始协方差参数=1。没有标签输入该步骤。
4. 每条通路两层 SNN，图像线性投影；默认隐藏宽度 256，每个图像/通路原型附加 32 维可训练身份嵌入。保留作者每 token FFN、融合注意力、模态内均值以及分类/风险投影。
5. 默认患者批大小 64，batch Cox 目标，每个 minibatch 更新一次 AdamW。50 epoch、学习率 `1e-4`、decay `1e-5`；一维参数及名称包含 bn/ln/bias/logit_scale 的参数不做 decay。默认无梯度裁剪。
6. 学习率高于 `5e-5` 时默认一个 epoch 线性 warmup，然后 cosine；更低学习率默认 constant，匹配公开启动脚本。显式 `warmup_epochs` 可覆盖，设 0 为无 warmup 的 cosine。初始 warmup LR 为零，先 optimizer.step，再 scheduler.step。

数据包/配置、原型、归一化、scheduler、optimizer 与随机流均进入运行或 checkpoint 记录。原型 token 使用内容寻址、容量有界的内存缓存；缓存不进入 checkpoint，重载后由已保存原型确定性重算。缓存不节约重复特征文件读取，主要节约 EM/OT 计算。预处理、聚类和训练均计入运行耗时。

保留不足整批的患者，包括单人尾批。无事件 Cox 批次返回可微零损失，但 AdamW 动量、decay 与 scheduler 仍按源码前进。Cox 风险集局限于当前 minibatch；这不是全训练队列 Cox。改变 batch size 会改变目标，必须记录。

## 损失和模型选择

默认 `model.loss: cox_source` 使用作者按时间降序的 logcumsumexp 部分似然，按事件数归一化。源码没有正确合并同时间风险集，因此它的 tied-time 结果可能依赖排序。显式 `cox_breslow` 使用并列时间完整风险集修正版，作为不同配置记录，不静默替换。

可选 `model.loss: nll` 保留作者离散 NLL，默认四个区间、删失权重 alpha=.5。区间由唯一训练患者的观测事件时间拟合；要求至少 bins 个不同观测时间。Cox 不计算这些区间，不因事件数量不足四个而拒绝任务。

验证 Cox 在整个验证患者集合上重新计算，不以单患者 Cox 的零损失做选模。默认按验证 survival_loss 严格改善保存 best.pt，并单独保留 last.pt；这是 PULS 的统一选模协议，与作者启动脚本默认最后 epoch 不同。用户显式选 C-index 时同样只用验证集。测试仅由 `evaluate` 单独启动。输出 Cox log-risk，避免源码 exp 溢出；排序相同，但近似并列风险容差是在 log-risk 尺度解释。离散 NLL 输出 hazards、survival 和 `-sum(survival)` 风险。

## 明确修复与保留的源码行为

- OT token 没有协方差，公开 trainer 却无条件构造 `Tensor(cov)`；本适配器传入模型声明所需的 D 维均值 token，不拼接不存在的协方差。
- 原型采样以独立索引替代对 Torch Tensor 的 NumPy 原地 shuffle，避免行交换别名；聚类只接收实际采样行，避免 FAISS 路径使用未填充预分配内存。
- PANTHER 协方差在 float32 相减后可能非正，明确加 `1e-7` 下界；普通数值输入与源码一致。常数/大值特征应额外检查。
- 图像 OT 保留源码 **正距离** 指数（`exp(+distance/eps)`），没有悄悄取负。默认采用作者已有的 log-Sinkhorn 等价稳定路径。它倾向较远的原型，不应直接当成标准负代价 OT 的实现。`ot_log_domain: false` 可作源码普通指数对照，溢出时报错。
- full attention 保留图像到通路块先 softmax、再与图像自注意力 logits 拼接后再次 softmax 的非对称计算。不是静默替换成标准 Transformer。
- 跨模态 UOT 的代价 **不 detach**，梯度经过双向 solver。零距离时明确定义零代价，并保留可微零梯度；使用原 entropy 正则。与 MOTCat 的 detached transport 不同。
- `.cuda()` 改为当前输入设备；通路网络硬编码 256 改为显式 hidden，native 256 不变。单个 attention block、两层 SNN 和 token FFN 保持源码。
- 不使用原 validation 的无条件 attention-dump 分支，因为 UOT 不返回这些 attention keys。当前不声称已实现论文可视化、IPCW 或解释性分析。

## 验证范围

`tests/test_mmp.py` 独立读取作者 Git 定义和完整训练循环，对照四配置 × 两种损失的连续优化更新；覆盖原型、UOT 数值及梯度、同时间风险集、无事件批次、训练专用拟合、缓存失效、无标签推理、所有分支梯度、患者尾批、native 256 宽度和精确续训。当前机器结果以 [VALIDATION.md](VALIDATION.md) 和 `validation/mmp.json` 为准。

所有软件验证使用 CPU 合成数据；GPU、真实 WSI/RNA、生存终点的临床核验与论文数值复现仍待运行。
