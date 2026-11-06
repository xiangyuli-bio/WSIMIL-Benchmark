# 当前0.74候选验证说明

当前128/164条已写入、246个配置；正式验收117条/219配置。KMG-Mamba完整实现已写入，专项重验、全库及独立包完整预算仍待完成。以下历次版本记录保留原口径，不能当成当前候选通过证据。

# 验证结果，2026-10-22

| 检查 | 实际结果 |
|---|---|
| 自动化测试 | 2361 passed、0 failed、0 errors、0 skipped，当前冻结0.55源码完整检查 |
| 独立 profile 演示 | 210配置完成CPU合成训练、保存、实际重载与预测 |
| 作者源数值对照 | 10 项通过：CHOWDER、MAD-MIL 训练/推理、PANTHER MAP-EM、PseMix 分区、ReMix 五种增强 |
| 静态检查 | Ruff 通过 |
| GP / 分组模型对照 | pytest 新增 2 项 GP 抽样路线及 3 项 MMIL 分组路线的作者输出数值对照 |
| CBMIL 对照 | 两种输入宽度对照作者完整目标/输出；三种批大小对照 NT-Xent 手算值与梯度 |
| 跨模态机制检查 | 19 项专项测试通过，含 CMTA/MOTCat 完整作者网络输出与目标梯度对照、旧熵核数值对照、小包流式梯度对照 |
| 生存损失对照 | pytest 中额外 3 项对照 SurvPath 作者 NLL 值与梯度；覆盖 alpha=0、0.15、0.5 |
| wheel 独立导入 | 0.55 C²MIL两组完整原生128隐藏/四层GCN/512到1024缩略图网络，1024及1536输入，每组完成原始100轮；最终独立包实际重载，详见wheel.json |
| 多尺度机制检查 | 21 项通过：尺度/父索引完整性、CS-MIL 输出/损失/梯度、HAG-MIL 原模型对照与三步参数更新、精确续训 |
| 空间机制检查 | 25 项通过：SmMIL 四配置作者对照、PSA-MIL 目标及梯度、SlideGraph 聚类/排序/校准与 6 配置精确恢复 |
| Bayes/上下文/方差机制检查 | 27 项通过：Bayes 三配置与 MC、二维 APCRF、零方差梯度，CAPR 三聚合器/调度，VarMIL 加权方差，七配置精确恢复 |
| MiCo / EfficientMIL 机制检查 | 20 项通过：作者输出/目标/梯度、训练专用 FAISS 中心、患者分片、APS、双分支损失、调度、四配置精确恢复及 AGDR 无梯度审计 |
| 连续生存 / 表型簇机制检查 | 16 项通过：SCMIL 完整网络、删失分布和梯度，长随访数值稳定性，患者多切片混合，DeepAttnMISL 簇池化、Cox/L1、调度、三种精确恢复路线及 AdvMIL 目标审计 |
| Dropout / PackMIL 机制检查 | 24 项通过：十二组完整作者输出/目标/梯度对照、掩码缩放、Lookahead、打包掩码、残差标签与生存混合损失、六配置精确恢复 |
| 提示 / 频域 / 超图机制检查 | 33 项通过：十五组完整作者输出/梯度对照，提示两阶段、Hilbert 实部/记忆写入、超图手算、源码错误审计、两种作者调度与七配置精确恢复 |
| 两个 CAMIL 机制检查 | 20 项通过：四组完整通道模型对照、13 步作者 RAdam/Lookahead、梯度累积、邻域缓存/构图、三组独立 TensorFlow 输出/损失/梯度/Adam/调度对照、两方法精确恢复 |
| 分子融合 / 表达回归 | 29 项通过：6 组作者网络输出/梯度、3 种加权 NLL、两个批次优化配方、缺失模态、表达目标/缓存/分层指标和 5 路精确恢复 |
| 视觉语言机制检查 | 26 项通过：6 组作者完整输出/目标/梯度、三配置精确恢复、真实 BPE、文本及双尺度身份/权重校验、EOT 修正和编码器冻结 |
| RankMix 机制检查 | 31 项通过：8 组作者完整单步目标/梯度/Adam 更新逐位对照、原索引反例、padding 排除、两阶段选模/调度、四配置各两处精确恢复 |
| 方法覆盖 | 164条独立规格，110条有执行适配，54条待适配；变体单独计数 |
| 源码版本 | sources.lock.json 固定 commit；解析远程 HEAD、源码审查、数值对照和真实复现分别记录 |
| H GPU tools 镜像 | 短作业确认默认 Python 缺少训练依赖，作业 Succeeded 后退出 |
| GPU 神经训练 | 0.55整包未新增GPU验证；保留0.52 PathRWKV实际H200报告及原始环境身份，不改称当前全库GPU通过 |
| BRCA / COAD 真实 WSI 训练 | 未执行；当前工作区未提供提取后的正式特征包 |
| 原论文数值复现 | 未执行 |

机器可读证据位于 [validation/](validation/)：pytest XML、逐配置结果、作者数值对照、覆盖统计、wheel 验证、H 镜像检查和汇总。历史0.41的168配置完整运行目录为 `/data/milplus_campanella_v41_r2/profiles`；0.26 的历史演示运行目录为 `/root/MIL/validation/milplus_mamba_validation_v26/final_profiles_r2`，其中包含配置、checkpoint、预测和指标；演示权重未放入源码包。此前的 `/root/milplus_prompt_frequency_hypergraph_validation_v13`（89 配置）、`/root/milplus_dropout_packing_validation_v12`（82 配置）、`/root/milplus_continuous_survival_validation_v11`（76 配置）、`/root/milplus_context_sequence_validation_v10`（74 配置）、`/root/milplus_bayes_context_validation_v9`（70 配置）、`/root/milplus_spatial_validation_v8`（63 配置）、`/root/milplus_multiscale_validation_v7`（57 配置）、`/root/milplus_crossmodal_validation_v6`（55 配置）、`/root/milplus_cluster_validation_v5`（52 配置）、`/root/milplus_gp_prompt_validation_v4`（51 配置）、`/root/milplus_survival_validation_v3`（47 配置）、`/root/milplus_expansion_validation_v2`（43 配置）和 `/root/milplus_validation_20261015`（17 配置）保留作历史记录，不能作为当前源码验证证据。

重要行为检查包括：

- 训练使用验证集选模；测试指标只在显式 evaluate 中生成。
- CPU 断点恢复与不中断运行得到相同模型状态及训练历史。
- MHIM 恢复包含教师，监督 warmup 不能被选为最终 masked-training checkpoint。
- CLAM 实例头确实更新；DTFD 两级参数分别更新。
- Cox 两遍前向在包含 dropout 与采样的条件下，与完整计算图的风险集梯度一致。
- HDF5/NPY/PT、两种特征维度、二/多分类、有序 QWK、生存、患者 bootstrap 均经过相应检查。
- 特征变更会阻断复用旧 checkpoint 的评价；患者别名跨划分会被拒绝。
- ASMIL anchor、AttriMIL 排序队列、IBMIL 两阶段状态、nnMIL 批次采样、FRMIL 配对、DGR 中心、PseMix 分区、ReMix 缓存均经过中断恢复一致性检查。
- 原型和伪包准备只读取训练集；多类别输入会拒绝二分类专用方法，完整矩阵保留未适配记录。
- CHOWDER / MAD-MIL / PseMix / ReMix 指定数值对照误差为 0；PANTHER MAP-EM 最大绝对误差约 1.49e-6。
- SurvPath、MCAT、PORPOISE 和 Patch-GCN 使用真实作者网络，完成患者级训练、重载及逐位断点恢复；两模态和 GENConv 参数均有有效梯度。
- 分箱与分子标准化只根据唯一训练患者拟合，组学/通路内容变化会阻断旧 checkpoint 评价。模型不会自动补全未知分子数据。
- Patch-GCN 不跨切片坐标系连接，患者级采样预算、单节点图、重复坐标拒绝和末尾不足整批的梯度累积均有检查。

- SGPMIL 的诱导点、变分均值和 Cholesky 参数实际更新；KL 按训练单位缩放，抽样语义与固定作者代码对照。
- MMIL 组间消息层有有效梯度；坐标/嵌入聚类退化、singleton 和全组掩码受到检查。
- PTCMIL 提示和 token 合并层有效更新，移动上下文在 epoch 间重置，验证不修改训练状态。
- 四个新配置均通过逐位断点恢复；GP / MMIL 固定种子推理保留训练随机状态，即使推理上下文异常退出。

- CBMIL 聚类选择、实例头、簇融合、对比头和非 512 维输入投影均有有效梯度并更新；提供/生成聚类两条路线均通过精确续训。
- 聚类数组对齐、类型、簇数及内容变化检查通过；验证不会修改训练缓存或消耗训练随机状态，单例尾批会并入前批。

- CMTA 的对齐目标停止梯度，生存与翻译路径保留梯度；作者完整网络输出和目标梯度对照通过。
- MOTCat 显式使用旧式 entropy 正则，退化代价与 singleton 输入有效；micro-batch 均值目标的流式梯度与联合反向一致，推理保留训练 RNG。
- 三个新增配置均通过患者级精确续训、重复预测和生存曲线/风险一致性检查。

- CS-MIL 保留原始尺度评分、簇采样和空簇掩码；推理不改变训练缓存/RNG，末尾梯度累积得到更新。
- HAG-MIL 的三个逐层优化步骤与作者顺序的最终参数逐位一致，支持三类标签和不同 FM 宽度；实例头与注意力均更新。
- 多尺度任一层特征变化阻止旧 checkpoint 评价，错误中心、重复象限、父索引错误和假尺度复用都会拒绝。

- SmMIL 四个平滑位置的输出、损失和梯度与固定作者定义一致；padding 不改变有效 bag 结果，exact 与 eval dropout 修正受到检查。
- PSA-MIL 的距离衰减参数有有效梯度，独立学习率组、类别权重和分段调度通过检查；变长批次与逐 bag 推理一致。
- SlideGraph 聚类与非退化 Delaunay 对照通过；成对 hinge 梯度、每轮样本覆盖、训练专用标准化及 Platt 校准均验证。
- 六个空间配置通过逐位中断恢复与重载评价，独立 wheel 也完成相同配置的两轮训练。

- Bayes-MIL 三种作者网络的输出/目标/梯度对照通过，真实二维栅格映射、零方差有限梯度、MC 概率平均和不确定性分解分别检查。
- CAPRMIL 三聚合器在两种 FM 宽度下与作者输出/梯度一致，上下文参数更新、实例顺序不变性和两种调度状态通过检查。
- VarMIL 两种输入宽度下的原始特征加权均值/方差、CE 及梯度与作者模型一致；退化方差被拒绝。
- 七配置均完成逐位中断恢复与独立 wheel 训练，APCRF 画布字段变化会阻止旧模型评价。

- MiCo 的中心由训练切片专用 FAISS 初始化；分类路由、患者分片增强、生存目标和梯度与固定作者定义对照。
- EfficientMIL 的 GRU/LSTM 输出、双分支 BCE、L2 梯度、APS 和 cosine 调度经过检查；自定义 Mamba 未用普通 MLP 替代。
- 四配置均通过精确恢复和独立 wheel 训练；AGDR 评分器无梯度的源码观察已有实际反向测试。

- SCMIL 的两种 FM 宽度与事件/删失条件下的参数、目标和梯度与固定作者代码对照；长随访逆 softplus 使用代数等价稳定公式。
- 患者多切片按分布混合，保留每张切片的坐标系；评价时间网格仅从训练患者确定，导出曲线对应的时间。
- DeepAttnMISL 的簇均值、注意力、Cox 按患者数平均、L1 和批内更新与作者定义对照；空簇和极端风险值经过检查。
- SCMIL、DeepAttnMISL 自算簇和提供簇三路线均通过精确续训；外部簇数组变化阻断旧 checkpoint。这一历史阶段的 AdvMIL 只完成目标函数审计；0.20 已接入并另行验证。

- MIL-Dropout 四骨干在两种 FM 宽度下与作者完整网络一致；重要实例与相似邻居掩码、缩放、Lookahead 快慢权重及优化器状态经过检查。全部掩掉会明确报错。
- MIL-Dropout DSMIL 保留作者 bag CE，离散 argmax 实例头的无梯度行为已披露；未用双分支 BCE 冒充该配方。
- PackMIL 分类/生存两路线在两种 FM 宽度下与作者输出、主/辅助目标和梯度一致；ADS 两分支实际更新，主分支 bag 掩码与残差 hyperslide 标签分别核对。
- 六配置完成精确续训与独立 wheel 验证；小包双分支复用、非整批尾部、训练专用时间分箱、评价保留全部实例以及特征变更拒绝均经过检查。

- S-DFP 四骨干两种输入宽度下的两阶段输出、CE 和梯度与作者网络对照；三个提示组合方式另做手算。调制器仅第二阶段更新，warm-up 不产生最终 checkpoint。
- MFC-MIL 包含作者原宽度二分类和两种输入宽度的多分类对照；两个记忆槽、三种 stride-16 卷积和频域投影均有梯度与更新，保留共享干预模块。
- DyHG 保留作者 Gumbel 超图抽样、节点维归一化和一轮消息传递；正常 bag 与作者逐位一致，singleton 修复及推理 RNG 隔离经过检查。
- 七配置完成精确恢复，包含模型、优化器、训练历史及预测；S-DFP 和 MFC-MIL 的步级调度与作者独立 helper 对照。

这些检查证明当前实现的软件行为，不证明真实任务性能或对作者论文的数值复现。部分方法存在已披露的适配差异，详见 METHODS.md。

可重复安装 CPU 依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.lock
python -m pip install -e . --no-deps
milplus setup --method frmil
milplus setup --method rrt
milplus setup --method dgrmil
milplus setup --method survpath
milplus setup --method mcat
milplus setup --method porpoise
milplus setup --method patchgcn
milplus setup --method sgpmil
milplus setup --method mmil
milplus setup --method ptcmil
milplus setup --method cbmil
milplus setup --method cmta
milplus setup --method motcat
milplus setup --method csmil
milplus setup --method hagmil_ce
milplus setup --method smmil
milplus setup --method psamil
milplus setup --method slidegraph
milplus setup --method bayesmil
milplus setup --method caprmil
milplus setup --method varmil
milplus setup --method mico
milplus setup --method efficientmil
milplus setup --method agdrmil
milplus setup --method scmil
milplus setup --method deepattnmisl
milplus setup --method advmil
milplus setup --method mil_dropout
milplus setup --method packmil
milplus setup --method sdfp
milplus setup --method mfcmil
milplus setup --method dyhg
pytest --junitxml=docs/validation/pytest_cpu.xml
```

未准备测试所需的固定外部源码时，对应集成测试会明确 skip，而不是声称通过。未安装 graph/gp/prompt/transport/multiscale/clustering extra 时相应测试也会明确 skip；本机完整验证没有 skip。GPU 环境需要 CUDA 版 torch，不能照搬 CPU lock。CI 配置已提供，但没有宣称远端 GitHub Actions 已运行。

## 0.14 CAMIL 对照证据

通道版和上下文版分别登记为两个来源。`validation/camil_tensorflow.json` 保存三组固定输入、参考输出哈希及各参数梯度误差；完整临时输入/输出位于 `/root/milplus_camil_validation_v14/pytest/test_context_full_tensorflow_a{0,1,2}`。这些结果对照 TensorFlow 2.15.1 / Addons 0.23.0 的兼容执行，不冒称作者 TensorFlow 2.6 原环境或真实临床结果。

原生通道模型在两种输入宽度、两种池化下与作者输出/梯度逐位一致。上下文图在随机坐标与规则网格上对照作者预处理；N=19/256/257 覆盖 leading padding 与整除路径。两个方法均完成独立 wheel 的训练、重载、预测和三轮精确中断恢复，特征或坐标修改会拒绝旧 checkpoint 评价。

复跑独立参考前按 [CAMIL.md](CAMIL.md) 创建 TensorFlow 验证环境，并设置 `MILPLUS_TF_PYTHON`。该环境不改变主训练环境的 NumPy/PyTorch 依赖。

- 0.16 曾独立验证 CATE、ViLa 修正版、ViLa 源码行为三个完整合成训练/重载流程；本版重新运行全部图文测试与 profile。`validation/wheel_cpu.json` 记录当前版本独立安装结果，历史 0.16 包证据保留在相应 Git 提交。

## 0.17 RankMix 历史验证（专项测试在本版重跑）

- 0.17 当时运行：447 passed、52 warnings、0 failed、0 skipped；完整测试耗时 237.31 秒。
- 该版 100 个配置均在其 runtime / sources lock / dependency identity 下完成合成训练和重载。
- `rankmix.json` 来自同一完整 pytest XML，包含 31 项专项结果；8 组完整作者训练步骤的值、梯度与 Adam 更新逐位一致。修正版的参考源码仅显式修正选中 ID 和 padding 候选，原始比较保留源码行为。
- 四配置分别在监督阶段和 RankMix 阶段中断，续训得到相同权重与历史；只有混合训练阶段可保存最终 best。
- 独立 wheel 的四配置完成训练与重复评价；CLI 的 RankMix 别名、setup、check、run、evaluate 另行实测通过。
- 原始作者 10 项数值检查和 3 项独立 TensorFlow 对照再次通过。IIB-MIL 仍待训练定义补齐，未计入已接入。

## 0.18 原型与 DP

0.18 当时的完整测试：482 passed、55 warnings、0 failed、0 skipped；耗时 250.9 秒。102 个配置均通过当前软件身份下的 CPU 合成训练、保存与重载。

新增 TPMIL 与 cDP-MIL 公开训练路径两个配置，来源和边界见 [PROTOTYPE_DP.md](PROTOTYPE_DP.md)。新增 35 项测试：8 个 TPMIL 作者完整训练步、2 个 cDP 作者预处理/训练步，均对照损失、有效梯度与 Adam 更新；4 个二/三类断点续训测试逐位相同。新增 SciPy 版本变更阻断旧 checkpoint 的测试。其余覆盖常量注意力、伪标签 detach、加权采样、概率推理随机状态、内容缓存和错误输入拒绝。

独立 wheel 在 `/tmp/milplus-wheel-0180-initial2` 安装并执行两个新增配置的三分类训练和评价；最终 wheel 已从 `/tmp/milplus-wheel-0180-final2` 重新加载同一结果并得到相同预测，记录见 `validation/wheel_cpu.json`。命令行 TPMIL 的 setup/check/run/evaluate 另在 `/root/milplus_prototype_validation_v18/cli_tpmil2` 验证。

cDP-MIL 公开主程序没有调用其变分 gamma 更新，故入口命名 `cdpmil_source`，软件通过不代表论文全部机制通过。INS 本轮仅固定源码并核实缺项，不计入通过方法数。

## 0.19 层级标签与 HMIL

0.19 当时的完整测试为 511 passed、0 failed、0 skipped，耗时 252.64 秒。103 个执行配置全部在当前软件身份下完成 CPU 合成训练、保存、重载和测试预测；10 项传统作者数值对照及 3 项独立 TensorFlow CAMIL 对照再次通过。

新增 29 项 HMIL/层级数据检查，机器记录见 `validation/hierarchy.json`：6 组作者完整 epoch 对照覆盖 18 次 Adam 更新，逐参数、梯度和平均损失一致（汇总损失容许 float32 求和舍入）；历史作者网络恢复标量分类头后逐张量一致。另含五个损失项的梯度、单实例与不同 FM 宽度、层级映射拒绝规则、切片/患者两种指标和 2 项精确续训。源码配方与论文公式的差异见 [层级分类说明](HIERARCHICAL_MIL.md)。

CLI setup/check/run/evaluate 通过，运行目录 `/root/milplus_hierarchy_validation_v19/cli_hmil`。独立 wheel 从 `/tmp/milplus-wheel-0190-initial` 导入，在 `/tmp/milplus-wheel-0190-validation` 完成 8 和 1024 维两组 HMIL 训练。全库 pytest 临时证据位于 `/root/milplus_hierarchy_validation_v19/final_pytest`。最终 wheel 再安装到 `/tmp/milplus-wheel-0190-final`，重新载入两组 checkpoint，预测与指标完全一致。合成演示数据不能作为真实 WSI 结果。

SCL-WC 只完成来源审查，仍属 pending，不计入 74 个已适配家族或 103 个执行配置。GPU、真实 WSI 和论文指标复现均未运行。

## 0.20 AdvMIL 对抗生存

0.20 当时的完整测试为 539 passed、0 failed、0 skipped，耗时 280.64 秒。107 个执行配置全部在当前软件身份下完成 CPU 合成训练、保存、重载和测试预测；10 项传统作者数值对照及 3 项独立 TensorFlow CAMIL 对照再次通过。

新增 28 项检查，机器记录见 `validation/adversarial_survival.json`。12 组完整作者 D/G 更新对照覆盖两骨干、源码/修正 BCE、三种标签可见性，共 72 次 Adam 更新，损失、梯度、参数一致。另检查真实 4×4 区域结构、多切片边界、完整区域采样、隐藏/留出结局隔离、L1/对抗/删失梯度、尾批、四种监督模式精确续训及不推进训练随机流的随机推理。

CLI setup/check/run/evaluate 通过，运行目录 `/root/milplus_adversarial_validation_v20/cli_advmil2`。独立 wheel 从 `/tmp/milplus-wheel-0200-initial2` 导入，在 `/tmp/milplus-wheel-0200-validation2` 完成四配置 8 维训练以及 ESAT 1024 维输入、作者默认 384/128 网络宽度训练，共五组。全库 pytest 临时证据位于 `/root/milplus_adversarial_validation_v20/final_pytest2`。

AdvMIL 源码目标与修正版、时间缩放/尾批修正及标签隐藏范围见 [方法说明](ADVERSARIAL_SURVIVAL.md)。自然缺失结局队列、Graph/Cluster 骨干、GPU、真实 WSI 和论文指标复现未纳入本轮完成声明。89 个其他方法家族仍待适配。

## 0.21 PIBD 多模态生存

0.21 当时的完整测试为 561 passed、0 failed、0 skipped，耗时 279.22 秒。108 个执行配置全部在当前软件身份下完成 CPU 合成训练、保存、重载和测试预测；10 项传统作者数值对照及 3 项独立 TensorFlow CAMIL 对照再次通过。

新增 22 项检查，机器记录见 `validation/pibd.json`。6 组完整作者训练循环对照覆盖两种批大小和三组目标权重，共 18 次连续 Adam 更新，梯度、参数逐位一致；另验证原型/通路/PoE/PID/CLUB 各分支梯度、字面 PoE、训练专用全局 RNA 归一化和分箱、标签不进入推理、采样填充、尾批，以及两种 sampler 的精确续训。

CLI setup/check/run/evaluate 通过，运行目录 `/root/milplus_pibd_validation_v21/cli_pibd`。独立 wheel 从 `/tmp/milplus-wheel-0210-initial` 导入，在 `/tmp/milplus-wheel-0210-validation` 完成 4 组训练与评价。输入维度为 16/32/64/768；前三组分别验证 none/train_standard/train_global_minmax_preserve_zero。768 维额外验证作者默认隐藏宽度 256、4096 patch 训练预算、512 patch 内部采样和 50 次高斯抽样，患者 batch size 显式设为 4。全库 pytest 临时证据位于 `/root/milplus_pibd_validation_v21/final_pytest`。

[PIBD 说明](PIBD.md) 区分公开源码与论文 PoE、填充和联合 CLUB 配方；没有以普通分类头替换该方法。GPU、真实 BRCA/COAD 多模态数据、论文数值复现尚未执行。88 个其他方法家族仍待适配。

## 0.22 MMP 多模态原型与完整生存训练

0.22 当时的完整测试为 606 passed、0 failed、0 skipped，耗时 313.24 秒。112 个执行配置全部在当前软件身份下完成 CPU 合成训练、保存、重载和测试预测；10 项传统作者数值对照及 3 项独立 TensorFlow CAMIL 对照再次通过。

新增 45 项检查，机器记录见 `validation/mmp.json`。四种图像原型/融合组合 × Cox/NLL 共 8 组完整作者训练循环对照，包含 48 次连续 AdamW 更新，梯度和参数逐位一致，覆盖 warmup 与 cosine。八种组合精确续训一致。另验证源 EM/OT token、可微 UOT 的旧 entropy 固定点与梯度、零距离、同时间 Cox 风险集、训练专用原型/RNA/分箱拟合、缓存失效、无标签推理、所有训练分支、患者尾批、可选身份嵌入、默认宽度和协方差数值保护。

CLI setup/check/run/evaluate 通过，运行目录 `/root/milplus_mmp_validation_v22/cli_mmp`。独立 wheel 从 `/tmp/milplus-wheel-0220-initial` 导入，在 `/tmp/milplus-wheel-0220-validation` 完成 4 组训练与评价，全部使用 1024 维输入、默认隐藏宽度 256 和 16 原型；患者 batch size 显式设为 4，交替验证 Cox/NLL。全库 pytest 临时证据位于 `/root/milplus_mmp_validation_v22/final_pytest`。

[MMP 说明](MMP.md) 明确 OT token 修复、冻结预处理、正距离图像 OT、非对称注意力及 UOT 梯度；不是将方法名称映射到普通 ABMIL。GPU、真实 BRCA/COAD 多模态数据、论文数值复现尚未执行。87 个其他方法家族仍待适配。

## 0.23 PathOmics 两阶段训练

0.23 当时的完整测试为 642 passed、0 failed、0 skipped，耗时 326.95 秒。115 个执行配置全部在当前软件身份下完成 CPU 合成训练、保存、重载和测试预测；10 项传统作者数值对照及 3 项独立 TensorFlow CAMIL 对照再次通过。

新增 36 项检查，机器记录见 `validation/pathomics.json`。三种预测分支 × 源码/完整白名单冻结共 6 组独立作者训练循环对照，包含 48 次连续 Adam 更新，梯度和参数逐位一致；覆盖预训练 MSE、阶段切换和生存 NLL。三个分支各在预训练中、阶段边界、微调中断点恢复，共 9 组精确续训。另验证原生 1024/256 网络的三种分子编码 × 两种 pooling 输出/梯度、预训练快照 clone、尾窗口、训练专用分箱与标准化、无标签对齐、隔离推理 RNG 和缺失输入拒绝。

CLI setup/check/run/evaluate 通过，运行目录 `/root/milplus_pathomics_validation_v23/cli_pathomics`。独立 wheel 从 `/tmp/milplus-wheel-0230-initial` 导入，在 `/tmp/milplus-wheel-0230-validation` 完成 4 组训练与评价，均使用 1024 维输入和默认隐藏宽度 256；三预测分支、两冻结策略、两阶段保存重载。全库 pytest 临时证据位于 `/root/milplus_pathomics_validation_v23/final_pytest`。

[PathOmics 说明](PATHOMICS.md) 记录源码不完整的冻结循环、最佳预训练权重别名、尾批、单模态输入限制和缺失的 bilinear 分支。GPU、真实 BRCA/COAD 多模态病例及论文数值复现尚未执行。86 个其他候选家族仍待适配。

## 0.24 PAMoE 组织先验与专家路由

0.24 当时的完整测试为 683 passed、0 failed、0 skipped，耗时 387.5 秒。119 个运行配置全部在当前软件身份下完成 CPU 合成训练、保存、重载与预测。10 项传统作者数值对照、3 项独立 TensorFlow CAMIL 对照再次通过。

新增 41 项专项检查，见 `validation/pamoe.json`。两任务 × 四配置，共 8 组独立固定作者完整训练循环运行六 epochs，逐步比较梯度、参数、AdamW 状态和余弦学习率；4 组原生 1024/512 网络输出与梯度对照；8 组精确断点续训。其余检查覆盖先验 CE、手工患者/专家路由、事件采样、Cox 并列时间、训练专用原型均值、哈希与身份变化拒绝、训练截断和 CLI 原型准备。

CLI setup/check/run/evaluate 的独立运行目录为 `/root/milplus_pamoe_validation_v24/cli_pamoe`。从 wheel 独立目录导入的 4 个原生宽度配置完成分类/生存训练与两次一致评价，详见 `validation/wheel_cpu.json`。

[PAMoE 说明](PAMOE.md) 保留专家索引修正版与源码行为的区别，也明确公开代码的填充、采样、并列时间、学习率以及原型准备边界。这是 CPU 合成软件验证；真实 BRCA/COAD、CUDA 和论文数值复现未执行。85 个候选家族仍待接入。

## 0.25 DGMIL 动态分布训练

0.25 当时的完整测试为 710 passed、0 failed、0 skipped，耗时 390.96 秒。121 个运行配置全部在当前软件身份下完成 CPU 合成训练、保存、重载与预测。10 项传统作者数值对照、3 项独立 TensorFlow CAMIL 对照再次通过。

新增 27 项专项检查，见 `validation/dgmil.json`。4 组独立作者分布公式对照（含原生 768 维）；4 组连续三个 epochs 的作者训练/实例选择对照，共 12 次 Adam 更新，逐步比较梯度、参数、BatchNorm、优化器和分布评分；两配置精确续训。其余检查覆盖并列实例、训练专用密度与概率拟合、线性投影边界、原始距离排名、患者聚合/配对 bootstrap、推理状态/RNG 不变性及输入变化拒绝。

CLI setup/check/run/evaluate 的独立运行目录为 `/root/milplus_dgmil_validation_v25/cli_dgmil`。从 wheel 独立目录导入的两配置使用原生 768 维特征，完成动态训练与两次一致评价，详见 `validation/wheel_cpu.json`。

[DGMIL 说明](DGMIL.md) 明确源码极端值索引与去重修正版、测试分布拟合修复、原始距离与附加概率报告层的边界。这是 CPU 合成软件验证；真实 BRCA/COAD、MAE 图像预训练、CUDA 和论文数值复现未执行。84 个候选家族仍待接入。

## 0.26 MambaMIL 序列分支与后端

0.26 当时的完整测试为 769 passed、0 failed、0 skipped，耗时 467.88 秒。124 个运行配置全部完成当前软件身份下的 CPU 合成训练、保存、重载与预测。10 项传统作者数值对照和 3 项独立 TensorFlow CAMIL 对照通过。初次回归中两处旧测试仍将 MambaMIL 当作待接入条目，现已改用真正待接入的 2DMambaMIL，并新增 MambaMIL 三配置 ready 检查；还补齐了 MambaMIL 家族默认 SR 映射，以及矩阵对未选择具体配置家族的阻断检查。CLI 别名完成 check/run/evaluate 端到端验证，完整回归已重新执行。

新增 57 项 MambaMIL 专项检查，详见 `validation/mambamil.json`：7 组独立作者扫描公式及梯度对照（含长度 257 的跨块梯度）；9 组完整模型输出/梯度对照和 3 组原生宽度梯度对照；6 组在同一数值内核下调用作者原始分类/生存 trainer，连续三轮、共 54 次 Adam 更新逐位一致；6 条分类/生存精确续训路径；另覆盖加权采样、训练专用分箱、尾累积窗口、事件/删失 NLL、多分类和后端身份检查。

独立 wheel 的三配置、分类/生存六条路径均使用 1024 维输入、512 隐藏宽度和默认训练选项，完成两个 epochs 及重复一致的重载评价。GPU 融合内核验证脚本已提供，当前没有 GPU 运行证据。

[MambaMIL 说明](MAMBAMIL.md) 保留序列顺序、作者初始化、完整训练配方及分箱/尾批修复的边界。真实 BRCA/COAD 实验和论文指标复现仍未执行，83 个候选家族尚待接入。

## 0.27 MSMMIL 完整配方

0.27 当时的完整测试为 812 passed、0 failed、0 skipped，耗时 542.44 秒。125 个配置全部通过当前软件身份的 CPU 合成训练、保存、重载和预测；10 项传统作者对照及 3 项独立 TensorFlow CAMIL 对照通过。

新增 43 项 MSMMIL 检查：14 组完整作者网络输出及全部参数梯度（含两种原生 1024 输入/256 隐藏宽度包长），5 组棋盘补零与反转恢复，三扫描分支和共享注意力激活，13 次完整作者 RAdam/Lookahead 参数更新，以及两种包数下三轮作者 training_step 与累积配方的 15 次更新逐位一致。三个断点均在新建对象后得到逐位一致的网络、动量、慢权重、历史及测试预测。实际 CLI 以 `MSMMIL` 完成 check/run/evaluate。

初次专项运行的一项恢复检查受到开发时修改源码影响，软件身份验证按预期拒绝评价；固定代码后该项重跑和上述全量回归通过。没有绕过身份保护。独立 wheel 分别使用 1024 和 1536 维输入、作者默认 256 中间宽度和两层结构，完整训练两个 epochs，重载预测与指标一致。

[MSMMIL 说明](MSMMIL.md) 记录了伪网格、FP32、尾窗口固定分母及稳定慢权重键的边界。CUDA 融合对照、GPU 神经训练、真实 BRCA/COAD 训练和原论文指标复现仍未执行。完整范围保留 164 条，其中 82 个家族有执行适配，82 条待接入。

## 0.28 MFE-MIL 联合重建配方

0.28 当时的完整测试为 879 passed、0 failed、0 skipped，耗时 670.13 秒。131 个配置全部通过当前软件身份的 CPU 合成训练、保存、重载和预测；10 项传统作者对照及 3 项独立 TensorFlow CAMIL 对照通过。

新增 67 项 MFE 检查：11 条任务路线逐位对照作者网络输出、联合损失和全部活跃参数梯度；原生 D1024/512 decoder 四层网络梯度对照；5 种窗口掩码、3 种包长下 timm SDPA 与显式注意力的值/梯度；全部 11 条路线对照两轮原始 trainer，合计64次 Adam 更新逐位一致；11 条路线的精确续训与重载预测。还验证了单 patch 修复、真实坐标缺失的阻断、跨切片坐标框架、大坐标排序、尾窗口实际分母、训练专用生存区间、非有限重建拒绝和家族名 CLI。初次缺坐标测试只清空了索引字段，但 H5 内仍有坐标，随后改成真正移除 H5 坐标，输入检查按预期拒绝。

独立 wheel 使用 MFE-ABMIL 分类 D1024、MFE-TransMIL 生存 D1536，decoder 均为作者默认512维/四层/16头、两个共享适配层，训练两个 epochs 后重复重载得到一致的预测和指标。

[MFE-MIL 说明](MFE_MIL.md) 记录 FP32 与原 CUDA AMP 的数值环境差异、源 CLAM bag-only 配方、不活跃参数、训练专用分箱及坐标/尾批修复。当前没有 GPU AMP、真实 WSI 或原论文指标复现证据。完整164条范围保留，其中83个家族有执行适配、81条待接入。

## 0.29 MamMIL 拓扑序列与 Mamba2

0.29 当时的完整测试为925 passed、0 failed、0 skipped。两个隔离CPU测试进程覆盖全部模块，墙钟耗时590.56秒；执行前后校验软件身份和测试源码哈希一致。134个配置全部通过当前软件身份的CPU合成训练、保存、重载和预测；10项传统作者对照、3项独立TensorFlow CAMIL对照通过。

新增46项MamMIL检查：Mamba2/SSD独立参考的6组状态扫描值/梯度、4组门控RMSNorm、2组完整split/conv/scan；字面/修正循环共6个完整图输出/梯度对照及两个原生D1024/512隐藏宽度/状态128对照；五条分类/生存训练路线各13次原始RAdam更新逐位一致，共65次；五条路线精确续训和重载。还验证小图/多连通分量MST/DFS/BFS、零向量余弦距离、缓存冻结及损坏拒绝、患者图边偏移、显式原生一患者一WSI契约，以及CPU构图和训练评价CLI。

独立wheel使用MamMIL分类D1024与患者union生存D1536，均采用原生512隐藏宽度/状态128/两扫描层，训练两个epochs并重复重载得到相同预测和指标。图资产与checkpoint绑定，备份/迁移必须保留冻结图。

[MamMIL说明](MAMMIL.md) 保留来源循环缺陷、显式修正、单线程HNSW、PyTorch SSD数值环境、训练专用生存分箱和多切片扩展边界。当前没有CUDA融合、真实WSI或原论文指标复现证据。完整164条范围保留，其中84个家族有执行适配，80条待接入。

## 0.30 ReCo-MIL 原型上下文配方

0.30 当时的完整测试为969 passed、0 failed、0 skipped，墙钟耗时515.82秒。138个配置全部通过当前软件身份下CPU合成训练、保存、重载与预测，10项传统作者对照及3项独立TensorFlow CAMIL对照通过。

新增44项ReCo-MIL检查：8组单/双尺度和频率加权的完整源码输出、联合目标、活跃梯度对照；4组原生D1024/1536、hidden512/K128的网络梯度；4条路线在epoch1/2/12/13/14/35/50上各14次Adam更新逐位一致，共56次，跨越原型解冻、tau到达与cosine结束边界；4条路线跨冻结边界精确续训；4组原始采样和MiniBatchKMeans中心/频率逐位对照。另检查不对称双尺度、无推理门控、惰性相似度参数、投影正则无梯度、训练专用原型和实际CLI。

独立wheel使用单尺度MLP D1024和双尺度图压缩lowD1024/highD1536，均为作者原生hidden512/K128/三增强，训练两个epochs并精确重载预测及指标。

首轮全库测试因7GB的/tmp内存盘占满而中断，首次wheel运行也记录了ENOSPC。已将本任务历史wheel验证目录转移到主磁盘并保存迁移清单；随后以主磁盘TMPDIR全量重跑，采用suite_r2及wheel_native中的成功证据。首次中断和失败日志保留，不计为通过。

[ReCo-MIL说明](RECOMIL.md) 记录固定公开源码、已核对的出版社/会议条目及未取得的论文全文、单尺度装配修复、训练专用聚类、初始化RNG路径和源训练/推理差异。当前没有GPU AMP、真实WSI或论文指标证据。完整164条范围保留，其中85个家族有执行适配，79条待接入。

## 0.31 MoMIL 三路SSD及完整训练策略

0.31 当时的完整测试为1018 passed、0 failed、0 skipped，墙钟耗时557.57秒。140配置完成当前软件身份下的CPU合成训练、保存、重载及预测；10项传统作者对照、3项独立TensorFlow CAMIL对照通过。

新增49项MoMIL检查：此固定2.2.4版本的6组SSD扫描和2组完整投影conv/scan/RMS值与梯度；8组分类/生存完整图和2组原生D1024/1536、hidden512/state64梯度；7组长度下原序、字面通道反转、平方转置与填充保留；独立初始化RNG；原分类trainer6次Adam更新；原通用生存trainer在完整窗口下8次更新（含可选L1）；对只修复尾分母/刷新条件的作者循环再对照2次更新；4路线精确续训。还验证采样、训练专用分箱、多片患者顺序、最后轮选模、协议文件写回、输入拒绝和家族CLI。

独立wheel使用分类D1024及显式生存扩展D1536，均为原生hidden512/state64、三路各两层，训练两个epochs后精确重载预测与指标。分类默认验证了最后轮checkpoint策略；生存协议保存在正确的survival_protocol.json中，包含最终epoch和优化步数。

开发阶段短窗口参考曾将损失合并一次反传，与原逐患者反传的浮点顺序不同；随后改成对原循环仅修复尾窗口分母/刷新，更新逐位通过。全库初次启动后发现生存协议写回路径需要修正，主动中断该轮并补上文件断言，再冻结新软件与测试身份，以suite_r2、profiles_r2和新wheel结果为准。旧中断记录保留，未计为通过。

[MoMIL说明](MOMIL.md) 记录强制构造器修复、字面通道反转、填充token保留、源CLI和启动脚本差异，以及原生生存工厂缺项与显式组合扩展。CUDA准入在本机实际返回not_run；无GPU训练、真实WSI或论文指标证据。完整164条范围保留，其中86个家族有执行适配，78条待接入。

## 0.31.1 SMMILe 空间与数值前置组件

0.31.1 当时的全库1114 passed、0 failed、0 skipped，墙钟526.60秒。140个已有执行配置重新完成当前软件身份下的CPU合成训练、保存、重载和预测；10项原作者数值检查及3项独立TensorFlow CAMIL对照再次通过。

新增63项空间检查和33项SMMILe数值检查：6组原始网格、8组原始SLIC/邻接、6组完整模型输出/梯度/Adam（共18次更新）、3组原生FM维度512/1024/1536梯度，以及精度、资产输入绑定、篡改/陈旧拒绝、空洞、内存预算、负包和空多标签等检查。完整模型状态参数的独立重载与无查询标签数值预测均通过。初次开发测试的CLI用例误将main返回的退出码当作JSON对象；修正测试读取标准输出后，全部96项及当前全库通过，原失败日志保留，未计为通过。第一次全库运行在复查发现复数坐标会丢失虚部时主动中断，补充显式拒绝和回归用例；第二次复查进一步将缺号ID检查的内存分配限制为实际观测标签数，补充2^40异常ID用例后冻结新身份，以suite_r3/profiles_r3为准。

0.31.1 当时SMMILe仍为pending_adapter：这里验证的是完整作者网络的数值组件及空间资产，尚未验收自动两阶段训练、multilabel任务接口和正式实例产物导出。[SMMILe说明](SMMILE.md)记录修复、来源、输入、已测内容及剩余工作。因此164条范围、86个已接入条目、78个待接入条目和140个执行配置均保持原有口径；没有把前置组件虚计成新的完整方法。GPU、真实WSI和论文指标复现仍未运行。

0.31.1 wheel在仓库之外解压并以独立包路径导入，与源码117个包文件逐字节核对。三个SMMILe数值路线分别使用D512/1024/1536、原生hidden128/attention64和三个refinement heads，各进行三次Adam更新后逐位重载；另完成一个ABMIL配置的两epoch训练与重复评估。数值路线没有计成完整执行配置；这也不是独立GPU或真实WSI验证。

## 0.32 SMMILe 完整两阶段与多标签任务

当前全库1189 passed、0 failed、0 skipped，墙钟675.31秒；143配置全部完成当前软件身份下的CPU合成训练、保存、重载及预测。10项传统作者对照、3项独立TensorFlow检查再次通过。

SMMILe新增48项完整训练检查与27项多标签协议检查；结合前版63项空间、33项网络检查，共171项。12组任务/损失/采样组合直接执行固定Git中的完整数值epoch与validation循环，216次Adam更新的参数、梯度、优化器状态逐位一致，24次验证目标和随机状态一致；覆盖MRF、负包一致性、BCE/bibce与阶段重置。6组切换前后精确续训通过；3条路线的查询标签扰动不改变native/实例分数或HDF5内容。6组构造器初始化与RNG、饱和bi-tempered端点、NaN梯度拒绝、原EarlyStopping重置反例、原始实例行序、类型/语义/缺失标签、患者聚合及164条矩阵路由也通过。

完整适配保留源slide/SP/refinement验证目标、源归约次序及标签组合平衡采样；用同一backbone的激活导出实例结果。原脚本日志不具备完整实例标注时的非训练报表被剥离，没有制造GT。将饱和概率裁到相邻可表示浮点数、保留第一轮refinement最优权重等明确修复，及持续随机流与默认40+20预算的边界，均见[SMMILe](SMMILE.md)。

SMMILe家族现已接入三个配置，整体164条、87条已适配、77条待适配、143运行配置。实际执行的CUDA准入在本机返回not_run；无GPU、真实WSI或论文指标复现证据。

0.32 wheel在仓库之外解压并从独立包路径导入，与源码120个包文件逐字节核对。三条完整路线分别使用D512/1024/1536、原生hidden128/attention64及3个refinement heads，各运行3个epochs（1基础+2refinement，显式缩短预算）。通用SMMILe名称正确解析执行路线；stage1最优初始化、resetAdam、源目标、最优checkpoint、重复评估的CSV和指标、实例数量/坐标/通道及实际文件hash均核验。该导入共享基础依赖环境，未宣称独立OS环境或GPU/真实WSI/论文分数复现。

## 0.33 PAMIL 完整分阶段训练

当前冻结软件全库1271 passed、0 failed、0 skipped，墙钟548.64秒；146配置全部完成CPU合成训练、保存、重载和测试预测。传统10项作者对照与3项独立TensorFlow比较通过。

新增76项PAMIL检查；10组任务/可选实例损失/采样组合直接执行固定Git训练主体，810次Adam更新逐步比较梯度、参数和优化器状态，30次原生验证目标及随机状态对照通过。三个独立两级KMeans主体、六组构造器初始化/RNG、原型类别筛选和push来源也被核对。9条跨warm/joint/push/LAST的精确续训、三条查询标签独立性、原始行序HDF5哈希、最后轮/验证选模、缺失或耗尽候选时的原子失败，以及冻结标注的FM/权重/坐标/行数/类型/切分约束通过。

PAMIL新增source分类、显式修正分类、多标签三配置。源CE-on-softmax、collate导致的单局部中心、固定prototype与LAST后fc冻结行为分别披露；修正路线有独立命名。另有6项全局真实数值输入检查，覆盖train/val/test的复数特征与坐标拒绝，不允许转成float时静默丢弃虚部。源push实际使用训练实例标注，绝不以复制真实slide标签制造GT。输入和源码差异见[PAMIL](PAMIL.md)。

164条范围全部保留：88条已接入、76条待接入。实际执行的CUDA准入返回not_run；GPU、真实WSI、论文数值复现均无证据。

0.33 独立wheel检查确认123个包文件与源码逐字节一致，三路线输入维度1024/1536/768、fc512、attention256的实际checkpoint形状正确；两级KMeans、实例标注推送、LAST训练、重载与重复CSV/指标及实例哈希均通过。源码包另外核查全部164份规格和最新源码、测试、配置、文档与证据。

## 0.34 Loss-Attention 完整配方与显式WSI迁移

当前冻结软件全库1325 passed、0 failed、0 skipped，墙钟581.43秒；149配置全部完成CPU合成训练、保存、重载和测试预测。10项传统作者对照与3项独立TensorFlow比较通过。

新增54项Loss-Attention检查。6组固定作者MIL完整epoch对照覆盖singleton、16实例、变长bag及两种dropout，逐步比较324次Adam更新；另外2组直接执行原image-temporal完整数值循环，包括162次旧Adam更新、每轮重置、源EMA及验证随机流。temporal步骤最大绝对差2.38418579102e-07，EMA最大差7.45058059692e-09，来自原4×4 AvgPool2d与变长bag均值的float32归约布局；没有声称源跨布局逐位相等。3组论文Eq.(5–6)输出/完整梯度、9个原调度端点、6路逐位续训、共享W/b、实例归约长度效应、query标签独立性、实例输出及错误输入/梯度/动量溢出拒绝通过。

传统特征与MNIST的omega=0由论文附录A4/A9核实，temporal配置是独立命名的image配方向WSI迁移。原始共享分类器、实例正则、mean/sum区别及原Adam epsilon位置均保留；原test选模改为validation-only，原型/实例真值未被额外引入。来源和范围见[Loss-Attention](LOSS_ATTENTION.md)。

全部164条仍保留：89条已接入、75条待接入。实际CUDA检查返回not_run；GPU、真实WSI、原论文分数复现均无证据。

0.34 独立wheel检查确认124个运行包文件与源码逐字节一致：1536维输入的256/128/64三层classical、1024维输入的500维source-MIL、768维输入的512维temporal均实际完成3轮完整训练。共享正则、训练EMA、Adam每轮重置、checkpoint真实层宽、重复CSV/指标和原行序实例文件哈希均核验。最终wheel的运行包字节与已执行的独立解压副本一致，元数据含最新第三方声明；源码包另核查164份规格及最新文档、配置、测试和证据。

## 0.35 MicroMIL 完整冻结FM迁移配方

当前冻结源码全库1370 passed、0 failed、0 skipped，墙钟563.10秒；150配置全部完成CPU合成训练、保存、重载和预测，10项传统作者对照及3项独立TensorFlow比较通过。

MicroMIL新增45项检查，包含4组作者完整网络及原DGL0.8.1 GAT类的三轮BCE/Adam训练比较，共180次优化更新；切片/患者、dropout0/.5均覆盖。独立图消息与DCE公式、全参数梯度、单实例、重复自环、两种训练单位精确恢复、标签独立的重复随机图推理、非法输入与非有限梯度拒绝均通过。明确的WSI特征迁移、原数据入口与train模式修正、源码与论文差异见[MicroMIL](MICROMIL.md)。

164条仍完整保留：90条接入、74条待接入。CUDA实际返回not_run，真实WSI、GPU和原论文指标均未验证。

0.35 独立wheel验证：125个运行文件逐字节匹配当前源码；在独立解包路径执行1536维切片bag与1024维患者bag两条原生hidden128/36簇/两层图路线，各三轮完整BCE/Adam训练，保存与重载后重复CSV和指标一致。MicroMIL及DGL参考源码按固定commit从外部加载。

## 0.36 ProbSA-MIL 作者库完整概率训练

全库1434 passed、0 failed、0 skipped，墙钟580.29秒；154配置全部完成CPU合成训练、保存、重载和预测。10项传统作者对照及3项独立TensorFlow比较通过。

新增64项ProbSA检查；直接执行固定作者Trainer._shared_loop/MILModelWrapper和完整网络的4组三轮训练，共108次Adam更新。独立稠密图参照与修正后的稀疏图路径最大参数绝对差1.49011611938e-08，没有声称跨归约布局逐位相同。另有高斯KL公式/梯度、padding排除、不稠密化断言、周期/LR边界、4路线精确续训、两种预测定义、标签独立性、实例输出和资源预算验证。

来源是原作者明确链接的torchmil固定版本；源码均值logit与论文概率混合分别命名。详见[ProbSA](PROBSA.md)。当前91家族已适配、73条待适配，GPU/真实WSI/论文指标仍未验证。

0.36 独立wheel验证：126个运行文件逐字节匹配源码；四条ProbSA配置以原生512投影/128注意力宽度、1000/5000次MC采样运行，Transformer为原生两层八头。全部完成三轮BCE/KL/周期退火及Adam预热、保存/最佳权重重载、重复CSV/指标和实例HDF5哈希校验。

## 0.37 GTP 完整官方图分类与GraphCAM

全库1487 passed、0 failed、0 skipped，墙钟556.58秒，156配置完成CPU合成训练、保存、重载和预测。10项传统作者对照、3项独立TensorFlow比较通过。

新增53项GTP专项。直接执行固定官方main.py完整训练内循环和helper.Trainer，源码/修正版分别在epoch0/20/100入口检验，共36次Adam更新；原生64宽度/100池化节点的模型、梯度、BN、Adam、调度与RNG状态逐位一致。独立mincut/正交公式与全梯度、官方多类GraphCAM/LRP/软分配映射、两路线精确续训、原坐标拓扑、标签独立性、无边/单节点/常量显示标记与资源预算均通过。

初次启动的suite在最终测试契约修订前已中断，保留作开发记录；当前证据只使用suite_final及其测试源码哈希。完整固定FM输入边界与修复见[GTP](GTP.md)。当前92家族已适配、72条待适配，GPU/真实WSI/论文指标和定位性能未验证。

0.37 独立wheel验证：127个运行文件逐字节匹配源码；两条GTP路线分别使用1536/1024维输入、原生64宽度/100池化节点/三层八头ViT，完整三轮CE+mincut+orthogonality训练。BN/Adam/调度/最佳重载、重复CSV/指标和原生GraphCAM HDF5及哈希验证通过。

## 0.38 SETMIL 完整空间网络与原始训练流程

全库1533 passed、0 failed、0 skipped，墙钟606.79秒；158配置通过CPU合成训练、保存、重载和预测。10项传统作者对照、3项独立TensorFlow比较通过。

新增46项SETMIL专项：原生1280维、48×48物理图、29,075,587参数的完整模型输出/CE/全部梯度独立对照；直接运行作者WSIFeatDataset和整个engine.train_one_epoch，两配置共42次AdamW更新，在epoch98/99/100边界检查原生StepLR100；模型、梯度、AdamW、调度和RNG逐位一致。真实格点缺口、双轴裁剪反例、80组无缓存/缓存与训练/验证的变换对照、两路线精确续训、最佳重载、无标签查询、预测视野记录和输入变更拒绝均通过。

保留源码及修正版裁剪、缓存视图范围，使用作者已经提供的纯PyTorch iRPE。两个配置是一个家族；未把特征金字塔误作多个光学倍率，也未以普通Transformer替代空间网络。完整边界见[SETMIL](SETMIL.md)。93家族已适配、71条待适配，GPU/真实WSI/论文指标未验证。

0.38 独立wheel验证：128个运行文件逐字节匹配源码；SETMIL两路线分别使用1536/1024维输入、原生48×48物理图、reduce7/token128、三路64维TPMF、六层16头iRPE。全部完成三轮增强/CE/clip/AdamW/StepLR训练，保存/最佳重载、重复CSV/指标和PPE覆盖哈希验证通过。

## 0.39 StableMIL 完整空间网络与分类/生存训练

全库1577 passed、0 failed、0 skipped，墙钟590.78秒；162配置通过CPU合成训练、保存、重载和预测。10项传统作者对照、3项独立TensorFlow比较通过。

新增44项StableMIL专项。原生1024维输入、hidden512/depth2/heads8的两任务完整输出/CE/NLL/全部参数与输入梯度独立对照；完整源分类与生存训练循环共四条路线、69次Adam更新，模型、梯度、Adam及RNG逐位一致。变量名/输入连接和修正尾批分支都在源对照中明确登记。四路线精确续训含原分类未更新梯度；物理数组、缓存并发/内容身份、原小包负维度反例、代表切片约束、训练专用分箱和标签独立性通过。

生存发布launcher本身不完整，本适配明确采用作者NLL类默认alpha.15/4bins和补齐的zero decay，不能声称找回了未提供的原实验参数。缓存按坐标、参数、源commit和依赖版本冻结，并保留完整物理融合/区域/近邻规则。详见[StableMIL](STABLEMIL.md)。94家族已适配，70条待适配；GPU、真实WSI和论文分数均未验证。

0.39 独立wheel验证：131个运行文件逐字节匹配源码。四条StableMIL分类/生存配置使用1536/1024维输入、原生hidden512/depth2/heads8、ratio2/aggregate256/k8，完整三轮Adam/CE或作者NLL训练及尾批行为。权重、Adam、未更新梯度、分箱、最佳重载、重复CSV/指标、物理几何数组与哈希验证通过。

## 0.40 MONAI 原图/序数训练

全库1626 passed、0 failed、0 skipped，墙钟687.15秒。全部167配置完成同一源码身份的CPU合成训练/保存/重载/预测；10项传统作者对照与3项独立TensorFlow比较通过。

新增49项MONAI专项：五个原生可训练ResNet50分支、完整四层依赖Transformer与四阶段特征图金字塔的输出/BCE/全部参数及输入梯度；源Compose多种子和白块填充；完整作者train_epoch两轮共30次AdamW更新，模型、梯度、AdamW、cosine、BN和RNG逐位一致。五配置精确续训、原生序数预测/患者簇bootstrap、官方ImageNet权重、真实OpenSlide解码合成金字塔TIFF、原图/权重身份、链式外部HDF5拒绝及真实CUDA/AMP精度门控检查通过。

保留发布教程完整流程，论文五教师伪标签扩展未实现，不能当作已经复现。详见[MONAI](MONAI_MIL.md)。95家族已适配，69条仍待适配；GPU、真实WSI及论文数值均未验证。

0.40 独立wheel验证：134个运行文件逐字节匹配源码。五条原生ResNet50配置使用官方ImageNet权重、六级序数标签、真实256px RGB及完整四层Transformer/四阶段金字塔，显式CPU预算batch2/tile2/eval_chunk3。两轮完整BCE/AdamW/cosine共90次更新；CNN权重与BN更新、两组AdamW、调度、最佳重载和重复CSV/指标验证通过。原生44块/batch4的CUDA验证及真实WSI仍未运行。

## 0.41 Campanella 原图两阶段训练

全库1676 passed、0 failed、0 skipped，墙钟928.90秒。168配置在当前冻结源码完成CPU合成训练、保存、重载与预测，10项传统作者检查和3项独立TensorFlow检查通过。

新增50项专项包括完整原生ResNet34输出/CE/全部梯度/BN，专用512→128循环每步状态和最终步BPTT。四项完整作者训练比较覆盖两轮CNN/Adam与两轮RNN/SGD、加权损失与短尾批，共12次更新，参数、全部梯度、优化器、BN和RNG逐位一致。真实OpenSlide ROI、多种mult、最佳CNN而非最后CNN、两个阶段边界精确恢复、测试标签独立性、半概率不同决策、坐标/权重/快照篡改与CLI全链路均通过。

当前全库同时重验MONAI49项专项与5条完整作者训练路径、30次AdamW更新。v0.40的五条完整256px独立wheel证明保持为历史归档，不冒充v0.41独立wheel结果。当前安装包证明单独记录于wheel.json。

本版本覆盖96条接入记录，68条仍待实现。CUDA、自然WSI和论文数值全部未运行；合成分类分数不是研究结论。详见[CAMPANELLA](CAMPANELLA.md)。

0.41 独立wheel验证：137个运行文件与源码逐字节一致。官方ImageNetV1初始化、完整ResNet34、真实224px OpenSlide区域、原生10步/128维循环，10轮CNN和2轮RNN完整训练共12次Adam/SGD更新。显式CPU批次4/4；最佳CNN哈希、冻结BN、优化器状态、验证时机、last/best和重复CSV/指标通过。CUDA、自然WSI及论文原始100/100轮的数值复现未运行。

## 0.42 MAMMOTH

当前源码全库1730项通过，无失败和跳过；176配置完成CPU合成训练与重载。54项专项覆盖原始专家软路由/低秩计算、六条作者头的18次完整CE/辅助目标/梯度/AdamW/RNG逐位对照、八配置精确续训、SmoothTop1对照、抽样与cosine、AMP溢出控制流和原生1024维/30专家/16头/10slots。ILRA命名修正及原非共享分支的错误均显式登记。

当前完整配置运行目录：`/data/milplus_mammoth_v42_r2/profiles`。本版独立wheel原生训练证明另记于wheel.json；此前各版本的wheel证明仍属历史。CUDA、天然WSI及论文分数复现未运行。详见[MAMMOTH](MAMMOTH.md)。

0.42 独立wheel原生验证：8/8分类路线通过，全部1024维输入、30专家、16头、10slots；每配置显式2轮CPU软件预算。核验查询映射真实更新、AdamW状态/步数、cosine到0、last/best重载和重复指标/CSV。此前Campanella/MONAI原生wheel验证保持历史身份，当前全库仍有对应完整作者回归。

## 0.43 PMIL Shapley

当前源码全库1821项通过，无失败、无跳过；182配置完成CPU合成训练与重载。91项专项覆盖原始Shapley估计与RNG、完整伪包划分、6组逐步目标/梯度/Adam对照，以及12组真实作者主程序对照。主程序保留原生1024输入/512隐藏宽度；一组验证普通多轮，另一组注入受控收敛观察值来检验降学习率/伪包增长/重复重划分/轮次提前结束。两轮选择的模型、完整分区和最终RNG逐位一致。六条路线在多个切换边界精确恢复。

当前运行目录：`/data/milplus_pmil_shapley_v43/profiles`。独立wheel证据单独记录；早期版本安装包证明仍属各自历史，不冒充当前安装包结果。CUDA、真实WSI及论文分数复现未运行。详见[PMIL_SHAPLEY](PMIL_SHAPLEY.md)。

0.43 独立wheel原生验证：6/6 PMIL源码/修正路线通过，1024输入、512隐藏、128/256注意力；显式2轮×2epoch和2伪包的软件预算，未缩小模型。核验投影参数更新、每轮Adam与条件实例头步数、Shapley分区哈希、最终轮最佳网络、重载及重复CSV。CPU合成验证，不是GPU/真实WSI/论文分数复现。

## 0.44 MultiAttentionMIL

当前源码1852项通过，无失败或跳过；184配置完成CPU合成训练和重载。31项专项包含原生512/1280维输入、128隐藏宽度、三层注意力与全部梯度独立对照，实际执行原 notebook 外层训练循环的2组108次Adam更新/RNG对照，源码推理Dropout与命名修复、患者/切片评价、精确恢复、重复CSV和坐标对齐注意力导出。

当前运行目录 `/data/milplus_msattention_v44/profiles`。原始公开材料为补充MNIST训练，WSI冻结特征适配范围见[MSATTENTION](MSATTENTION.md)。独立wheel证明单列，旧版本wheel仍属于历史记录。GPU、自然WSI和论文数值复现未运行。

0.44 独立wheel验证4/4原生组合通过：两条源码/修正路线，512/1280输入，原始三层128隐藏，完整10epoch补充训练预算。核验所有Adam步数、七个实际模块权重更新、验证最佳checkpoint、重载与重复CSV。CPU合成验证不代表GPU/WSI/论文分数复现。

## 0.45 TTIS

当前源码1932项通过，无失败或跳过；191配置完成CPU合成训练和重载。79项专项包括七配置的原生宽度前向/梯度、七组原始ModelInterface或命名CLAM联合损失训练共98次逐步Lookahead更新与调度/RNG对照，以及跨同步边界恢复、严格VAL CE选模、重复CSV、TEST目标独立性和资源错配拒绝。实际CPU聚类使用显式k-means++适配，不能冒称原生cuML验证。

当前运行目录 `/data/milplus_ttis_v45/profiles`。固定作者来源与语义资源边界见[TTIS](TTIS.md)。独立wheel验证另列；旧wheel记录保留历史版本。GPU、自然WSI与论文数值复现未运行。

0.45 独立wheel：七个TTIS配置均以原生1024输入/512隐藏/384注意力完成50epoch。核对完整优化器步数、Lookahead慢权重与cosine状态、CLAM源码/联合损失梯度边界、最佳模型及重复预测。CPU合成向量和显式CPU聚类，未声称自然MUSK/cuML/GPU/WSI/论文指标。

## 0.46 官方2DMambaMIL

当前冻结源码2030项通过，无失败/跳过；194配置完成独立训练和预测。98项专项包含原生1024/128/state16、一层/两层、分类/生存全部梯度与独立公式对照，6组原始外层训练入口84次严格优化器/调度/RNG对照，以及精确恢复、末次并列选模、原生pad注意力质量、真实坐标与数据边界。

当前运行目录 `/data/milplus_mamba2d_v46/profiles`。三条扫描定义、发布版缺陷、物理画布及CUDA接口见[MAMBA2D](MAMBA2D.md)。历史安装包证据属于对应版本归档；当前独立wheel证据另列。GPU、自然WSI、论文得分未运行。

0.46 独立wheel：6/6原生2DMamba组合完成20epoch、共2700次优化器更新，核验原始AdamW/cosine或NLL/32逐病例更新、训练专用分箱、任务选模、原生填充注意力和重复预测。CPU合成证据不代表原生CUDA/GPU/WSI或论文指标复现。

## 0.47 SI-MIL

当前全库2070项通过，无失败、错误或跳过；196配置全部完成独立训练和预测。SI-MIL两条原始/完整VJP配置保留真正双分支、PF-Mixer、耦合采样、BCE/KD/注意力损失与Adam/cosine。35项专项包括实际作者外层训练函数108次逐位对照、独立解析导数、TRAIN精确分箱、输入变更和精确续训。另5项检查保证仅精简已预测核验的新合成权重；原始权重哈希/大小及预测保留。

当前运行目录 `/data/milplus_simil_v47/profiles`，完整回归 `/root/MIL/validation/simil_v47_suite_r3`。首次漏配TF路径的中止记录和2069项运行后发现测试文件名冲突的恢复记录保留；完整2070项采用第三次全新运行，不聚合此前通过数。此前版本的专门报告保留其原始版本/哈希，不重新标成当前版本。完整源码、输入与梯度政策见[SI-MIL](SIMIL.md)。GPU、自然WSI和论文指标复现尚未运行。

0.47 独立wheel两条完整原生SI-MIL路线均通过39epoch，合计1404次Adam更新；核验本地冻结源码与wheel全部运行文件一致，所有milplus模块确实从解压wheel导入。梯度修正和预处理修正均显式登记。

## 0.48 HIPT

全库2101项、198配置在同一冻结运行源码下全新通过；无失败、错误或跳过。HIPT 31项专项含实际公开教师、作者真实平衡采样/优化器/训练循环108次逐位更新对照、三分类梯度、精确续训、真实16×16几何和选模边界。独立wheel两条原生384/192路线各20轮，共720次Adam更新，检查冻结/微调教师、最终轮选择、重复预测和坐标注意力乘积。

完整回归 `/root/MIL/validation/hipt_v48_suite`；独立配置 `/data/milplus_hipt_v48/profiles`；原生wheel训练 `/data/milplus_hipt_v48/wheel_native_r2`。仅在实际重载预测后精简新合成验证权重，保留输入、日志、预测与权重哈希；不删除历史或用户数据。既往专项报告保留原始版本身份，不伪装成当前运行。实际预训练权重和作者源码外部获取；原生DINO空间、完整物理区域、分类与生存来源边界见[HIPT](HIPT.md)。GPU、自然WSI和论文指标尚未验证。

## 0.49 H²-MIL

全库2150项、200配置在同一冻结运行源码下全新通过，无失败、错误或跳过。H²-MIL49项专项含作者notebook构图、真实外层求和训练40次逐位更新对照、42级独立池化归属/收缩边/输入梯度、独立分辨率与节点注意力、90组原选模边界、精确续训、输入绑定、三分类和测试推理隔离。独立wheel两条原生1024/256/128路线各60轮，共120次Adam更新，检查验证选模、重复预测及映射回真实高倍率行顺序的结构输出。

完整回归 `/root/MIL/validation/h2mil_v49_suite`；独立配置 `/data/milplus_h2mil_v49/profiles`；原生wheel `/data/milplus_h2mil_v49/wheel_native`。实际重载预测后只精简新合成验证权重，保留哈希收据；未删除历史或用户数据。开发阶段检查失败及修正保留在infrastructure_attempts.json；完整回归全新执行，不拼接旧通过结果。既往专项报告保留各自原始软件身份。GPU、自然WSI和论文指标尚未验证。

## 0.50 AGDR-MIL

全库2200项、202配置在同一冻结运行源码下全新通过，无失败、错误或跳过。AGDR50项专项覆盖两个来源配置、实际完整训练循环180次逐位参数/梯度/Adam/随机状态更新、4组实际作者验证 DataLoader 对照、原生512隐藏层下四种高分池化/二三分类/16及1024两种输入宽度的完整梯度，以及精确续训、测试推理隔离、源码选模与逐行导出。独立wheel两条原生1024/512路线使用完整200轮最大预算及原作者早停，实际在33/27轮结束，共1080次Adam更新；逐轮对照实际Earlystopping回调，重载验证集选中的权重，并重复预测与实例导出。

完整回归 `/root/MIL/validation/agdrmil_v50_suite`，独立配置 `/data/milplus_agdrmil_v50/profiles`，原生wheel `/data/milplus_agdrmil_v50/wheel_native`。新合成配置只在实际重载评价后精简权重并保留哈希；未删除历史或用户数据。开发时的常量输入浮点退化问题和明确拒绝规则见 infrastructure_attempts.json；当前证明来自修正后的全新完整回归。既往报告保留各自原始身份。GPU、自然WSI和论文指标尚未验证。


## 0.51 C2Aug

全库2239项、203配置在同一冻结运行源码下全新通过，无失败、错误或跳过；完整回归墙钟1561.53秒。新增39项专项包含六组原生TransMIL输出/CE/输入及全部参数梯度逐位对照、三种增强独立数学验证、完整双对比目标、全部原型组梯度、TRAIN索引库和两条独立完整训练流程90次参数/梯度/Adam/EMA/FIFO/RNG逐位比较。两处断点恢复、严格验证选模和测试推理隔离均通过。

原生512宽度的源码CLI与独立草案wheel分别使用1024/1536维特征、补充材料8views/256queue/8prototypes/256reduction，各完成3epoch、135次更新，合计270次。最终wheel核对全部154个运行文件逐字节相同，并从全新独立安装路径实际重载已选权重、生成相同预测。这里没有完成200epoch论文预算，CPU合成分数不是benchmark结果。完整论文参考重建、作者源码缺项和未披露参数的明确选择见[C2AUG](C2AUG.md)。

逐配置运行目录为`/data/milplus_c2aug_v51/profiles`。全新回归在开发机独立内存临时目录`/tmp/milplus_c2aug_v51_suite_r4`运行，双进程并启用内存/空间保护，数值收据及XML逐文件校验后归档到`/mnt/medeval-share/xiangyuli/mil_baseline_puls_work/validation/c2aug_v51_suite_r4`。此前系统盘写满、SSHFS默认缓存导致HDF5重写失败及同步挂载过慢的中止记录均保留在infrastructure_attempts.json；当前证明不拼接这些未完成运行的通过数。未删除或迁移历史数据，也未向他人目录写入结果。

既往专项报告保留各自版本身份。CUDA、自然WSI及论文指标仍未验证；164条候选尚未全部接入。

## 0.52 PathRWKV

完整2292项CPU回归、205个配置及四组独立CPUwheel原生768模型检查均在当前冻结源码下通过。53项PathRWKV检查包含原始训练流程、异质标签、真实坐标和多格式、采样/补零、Cox/缺失标签/双shift状态、全部梯度、精确续训和测试隔离。实际原始作者训练两种FM宽度共16次AdamW更新逐位对照通过。

H200原生CUDA单独通过两个未改写的作者算子、三种序列形状的五项输入梯度、矩阵状态、768整网FP32数学参考梯度、FP32/BF16独立作者原生CUDA逐位梯度与权重对照、四组CLI训练/验证选模/重载/重复预测，共32次更新。单实例衰减参数只影响未返回的后继状态，验证器显式识别该结构性零梯度；未放宽数值容差或其他梯度要求。数值证据见pathrwkv_cuda.json，CPU与CUDA包环境身份分别记录。全库回归收据归档在自己的GPFS目录，不以旧版本检查拼接为当前证明。另以原生四张切片×2000tile默认预算完成一次BF16训练更新，输出/损失/全部梯度/权重/AdamW动量与独立作者源码逐位一致。BF16纯张量参考梯度仍有73个MTL权重元素未通过原容差，9个量化最大值通道选择不同，失败证据完整保留；未将其改称通过或放宽容差。BF16原生训练按独立原始CUDA实现逐位一致验收。GPU验证仅覆盖此方法；自然BRCA/COAD训练与论文指标仍未执行。

## 0.53 ProtoSurv

当前源码完整2320项回归、207个实际配置运行均通过，无失败、错误或跳过。28项ProtoSurv专项包含实际原始网络/三项目标/全部梯度、作者原始训练/Adam/调度/验证/RNG、临床时间与标签边界、真实CLI、图与原型来源、精确续训和两种累积尾批策略。

独立wheel实际执行两种FM宽度×源码/修正四组原生128图隐藏层、768原型维度、40个原型、四融合层训练，保留100轮上限与原早停规则，分别完成21/26/21/26轮、合计94次Adam更新，验证集选模后实际重载并重复预测。合成图只验证工程，不代表自然BRCA/COAD结果或论文数值复现。

## 0.54 GAT-Mamba

当前完整2343项、209个实际配置运行均通过，无失败、错误或跳过。23项专项覆盖独立作者原始整网/全部梯度、实际原始图生成与训练器/Adam/随机验证/plateau/RNG、两条边特征路线、精确续训、分类器TRAIN/外部来源与患者隔离、真实CLI和nativeCUDA缺失时拒绝回退。

独立wheel四组原生64投影/16位置/80通道、状态16、卷积4、expand1、单块网络，保留batch16/Adam/Cox加L1、原始201轮上限与6次未改善早停，实际完成[7, 7, 39, 58]轮、合计111次更新，验证集选模后实际重载与重复预测一致。显式使用同版本作者CPU扫描参考；未据此声称nativeCUDA、自然BRCA/COAD或论文数值复现。

## 0.55 C²MIL

当前完整2361项、210个实际配置运行均通过，无失败、错误或跳过。18项专项覆盖独立作者整网/五项损失/全部梯度、实际作者训练循环连续三轮的Adam/VAL/RNG逐位对照、原始物理图和1500patch参考采样、批次合并与学习率/最后并列选模、动态聚类/偏差缓存恢复和真实CLI。

独立wheel两组1024/1536FM维度、128隐藏四层GCN/512隐藏1024聚类表征，各完成原始100轮、合计200次更新，验证集选模后实际重载与重复预测一致。CPU合成软件验证，不代表自然BRCA/COAD、CUDA或论文数值复现。
