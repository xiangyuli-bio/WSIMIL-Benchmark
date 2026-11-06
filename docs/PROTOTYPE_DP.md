# TPMIL、cDP-MIL 与 INS 来源审查

本轮新增 `tpmil` 与 `cdpmil_source` 两个运行入口。它们分别属于既有 164 条清单中的 TPMIL 和 cDP-MIL，不增加计划分母。INS 固定来源并登记缺项，尚未提供运行入口。所有运行验证限于 CPU 合成特征；GPU、真实 WSI 和论文指标复现均未运行。

## 固定来源

| 方法 | 作者来源 | 提交 |
|---|---|---|
| TPMIL | https://github.com/LitaoYang-Jet/TPMIL | `0d89a1f98f164803d92c6f6c686451711873eeca` |
| cDP-MIL | https://github.com/HKU-MedAI/cDPMIL | `83f7ed596618266589745c5b1e977894958c85df` |
| INS | https://github.com/miccaiif/INS | `cd97ba2fa7df2a951f218e6de405f664129931c8` |

三个来源在所核提交中均未发现根目录许可证；本仓库不分发其原始代码。`milplus setup` 获取固定源码，运行时从经过版本和工作区检查的 Git 对象加载指定定义。来源公开不代表可以任意再许可。

TPMIL 论文：https://proceedings.mlr.press/v227/yang24d.html 。会议 MIDL 2023，PMLR 出版年为 2024。cDP-MIL 论文：https://arxiv.org/abs/2407.11448 。INS 论文：https://arxiv.org/abs/2307.02249 。源码身份与论文数值复现是不同证据。

## TPMIL 完整目标

`model_tpmil.py::CLAM_SB` 在输入投影后创建 K+1 个可训练原型；K 为分类数，额外原型表示背景。实例到原型的分数为负平方欧氏距离除以投影维数。未经 softmax 的原始注意力 detach 后做 min/max 归一化，生成两个非零目标：切片类别为 p，背景为 1-p。KL 使用 `batchmean`，训练目标严格保留 `0.5 CE + 0.5 KL`。

`train_utils.py::train_loop_tpmil` 明确替换 CLAM 实例聚类损失。适配保留作者模型中的实例头参数，但不错误地将这些未使用头的损失加入目标。默认门控注意力、p=.25 dropout、512 投影宽度、256 注意力宽度；只将固定输入 1024 改为任务 FM 的实际维度。非默认宽度会写入运行配置，不宣称为作者原宽度实验。

默认使用每个训练 slide 的逆类别频数有放回采样，采样数等于训练 slide 数；类别频数不使用验证/测试集。优化器为 Adam，lr=2e-4、weight_decay=1e-5，无默认裁剪。配置文件提供 200 epoch 上限；验证 `log_loss` 选模、patience=20。PLUS 的统一 early stopping 没有照搬 CLAM 的额外最低停止 epoch 规则；这是公开的评测层差异。

源码差异：

- 原始注意力相同或只有一个实例时，min/max 分母为零。适配仅在完全相等时赋予 p=.5；其余输入保留作者公式及 clamp。该中性退化策略是本工程修正，不能据此推断原论文如何处理。
- 作者 forward 即使推理也需要 label 来构造未使用的辅助损失。适配推理只调用同一 attention/project/classifier，完全不接收标签。
- `CLAM_MB` 源码并不包含 TPMIL 原型学习，不将它登记为另一种 TPMIL。
- 任务必须以单个 slide bag 训练；患者级预测融合仍由任务协议控制。

## cDP-MIL：公开训练路径的明确边界

`DP_feats_aggr.py::get_feats` 对每张 slide 独立拟合 `BayesianGaussianMixture`，默认 10 分量、浓度 .1、完整协方差、random_state=0、max_iter=30；对最终非空硬分配计算特征均值。适配按完全相同算法生成中心，没有用 KMeans 替代。所需 N>=max(2,clusters)，不足时直接拒绝而非悄悄减少分量。

`main.py::train` 对这些中心调用 `DP_Classifier.forward`，后者只调用 `DirichletProcess_VI.infer`。每个类别有可训练 Gaussian 均值和 rho，标准差为 log(1+exp(rho))；Beta stick 权重由 SciPy 抽样，分类 logits 为各中心的 log density + log stick weight 均值，目标仅为 CE。实际主程序将命令行 lr 覆盖为 .01，Adam betas=(.5,.9)、weight_decay=.005，epoch cosine 到 5e-6；本入口保留这些实际设置。

**公开主程序不调用 `DirichletProcess_VI.forward/update_gamma`。gamma1=1、gamma2=eta 保持固定，也没有附加变分似然训练项。** 因此入口明确命名 `cdpmil_source`，`cDP-MIL` 别名解析到这一公开源码路径；不能把它报告成论文全部变分机制已复现。源文件还存在 HDP/EM 等类，本入口没有擅自将这些未被主程序调用的分支拼成新算法。

适配差异：

- 去掉硬编码 `.cuda()`，使用模型参数所在设备；未在 GPU 上验证。
- 每张 slide 的聚类均独立，不使用其标签，也不跨 slide 拟合。验证/测试时会计算该 slide 的无监督局部聚类，这是方法的推理预处理，不是训练集全局统计。
- 使用按特征内容哈希索引、默认最多 128 bags 的内存缓存；缓存不进入模型权重，重载后确定性重算。大 FM 的完整协方差拟合可能昂贵，应记录在端到端成本中。此实现尚未提供持久磁盘聚类缓存。
- 作者推理每次抽样；统一评价采用隔离的固定 NumPy seed 的一次抽样，保留概率分类器并避免验证消耗训练随机状态，不额外声称采用多次 MC ensemble。
- 作者 `main.py` 每个 epoch 访问 test，且没有实现选择并保存最佳 checkpoint。PLUS 只在 validation 上选择/保存，test 需显式 evaluate。
- 使用统一训练 bag 随机顺序和安全数据读取。源码数值对照固定一个训练步骤，不宣称整个 epoch 排序逐位一致。

## 使用

在有网络的 CPU 开发机准备作者源码，再向训练环境传递固定源缓存：

```bash
milplus setup --method tpmil
milplus setup --method cdpmil
milplus inspect --method TPMIL
milplus check --data /path/to/task --method TPMIL --deep
milplus run --data /path/to/task --method TPMIL --config configs/tpmil.yaml --output runs/tpmil
milplus run --data /path/to/task --method cdpmil_source --config configs/cdpmil_source.yaml --output runs/cdpmil
milplus evaluate --run runs/tpmil
```

任务路径指向含患者、标签、固定划分及 FM 特征索引的 task.yaml 数据包，不是只有 SVS 文件的目录。默认训练参数和模型参数均记录在 request.json；实际适配范围写入 method_protocol.json。

## INS 尚待实现的具体内容

`model_feat.py` 包含 momentum encoder、prototypes、queue/queue_pseudo/queue_partial、DDP shuffle 和 teacher attention head。`train_bag_TCGA.py` 包含预训练、partial-label confidence EMA、原型伪目标、SupCon 和 bag loss。它们必须一并接入，不能用一个注意力分类器代表 INS。

`train_onlybag_feat_by_bagcls.py` 虽名为 feature 路径，实际仍导入 `model.INS`，且导入当前源码树中不存在的 `dataset_CervicalCancer_feat`。`dataset_CAMELYON16_BasedOnFeat.py` 另有成对特征和 patch/slide 标签组织，须核清哪些 patch 标签仅用于监控、哪些进入 loss 或 confidence，再建立冻结 FM 的完整数据协议。DDP collectives、queue 的 batch 整除要求、instance index 与 confidence 恢复也需要独立机制验证。当前不提供静默替代路线。

## 验证依据

`tests/test_prototype_dp.py` 对照作者完整 TPMIL 训练步（2/3 类 × 门控/非门控 × dropout 开/关），检查目标值、所有有效梯度与 Adam 参数更新；另外直接执行作者 cDP 预处理和训练函数，分别比较二/三类输出、梯度和 Adam 更新。包含原型分支 detach、常量注意力修正、逆频率采样、推理随机状态、内容缓存、患者边界、非法输入与两方法二/三类精确断点续训。SciPy 版本纳入运行软件身份；更换其版本会拒绝旧 checkpoint 的评价。

实际通过数与软件哈希见 `docs/validation/prototype_dp.json`，全库运行证据见 `profiles_cpu.json`；这些是软件正确性证据，不是临床有效性结论。
