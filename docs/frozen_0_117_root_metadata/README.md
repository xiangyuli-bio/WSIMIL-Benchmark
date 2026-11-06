# MIL_BASELINE_PULS

当前未发布候选 **0.117** 修复DSPA独立运行时预检缓存提前创建输出目录的问题：预检使用独立目录，结果目录继续拒绝覆盖。CPU及真实CUDA预检检查通过；0.116的首次完整启动失败记录保留，失败发生在模型构造和训练之前。
前版 **0.116** 写入 [DSPA2025 完整PAT／双路参考运行时](docs/DSPAMIL.md)：完整检查点、独立进程、真实进程锁恢复、Student推理，以及验证／导出／配对统计一致的等级判定和患者概率融合。
该候选仍需完整模型实际执行和原始资产验收，不增加已完成条目。软件阶段计数保持161／164，352套配置；默认发布目录保持151条。
前版 **0.115** 新增DSPA患者／概念输入绑定与50轮训练控制。
22项输入检查与7项训练控制检查通过，其中完整宽度双路参考头完成50轮／300次更新及第7次更新后的精确恢复。
这项控制检查没有包含PAT骨干或原始概念资产，不能解释为完整DSPA训练通过；新完整检查点运行时仍需单独执行验收。
上版 **0.114** 的DSPA已知机制参考组件和FEATHER数值后端绑定继续保留。
已核对三套原生输入维度、完整512维／门控384维网络、源分类头重置过程，以及公开检查点的文件指纹。
已写入原生特征／患者绑定、20轮验证选模微调、10轮无验证微调、冻结KNN20与统一入口候选。
仍未取得原始预训练检查点，完整预训练模型执行与发布验收尚未完成；FEATHER保持待接入。
软件阶段已验证条目仍为 **161／164，352套配置**，最后三项仍是DyKo、DSPA-MIL和FEATHER。

DSPA新增双路概念聚合、Teacher/PTG/Student及论文各项损失，两种显式掩码条件的完整原生宽度组件已在H200检查。
原始概念资产及对齐证据、完整PAT实际训练及发布验收仍未完成，DSPA保持待接入。
作者源码校验现在使用Git只读状态检查；历史目录测试保留原先精确范围，并另外覆盖当前352套执行合同。

### 0.110候选阶段记录

当前未发布候选工作树 **0.110** 继续开发 [DyKo完整原生适配候选](docs/dyko_native_candidate.md)。
已写入并通过各自软件阶段检查的条目保持 **161／164，352套执行配置**；其中146条标记已适配，15条继续验证。
DyKo、DSPA-MIL和FEATHER仍未完成接入。DyKo新增原始TITAN文件校验、原生特征／概念／类别语义绑定、完整作者模型构造与隔离训练入口，但原始权重与完整方法尚未运行验证，不计作第162条。
RMDL的4套源码／独立安装包软件流程已经通过，完整训练预算另行执行；真实BRCA／COAD实验、论文指标复现和整库最终验收仍未完成。
默认开发入口仍为 **0.98：151条／321配置**，稳定入口仍为 **0.62：117条／219配置**。

### 0.109冻结时的历史记录

当前未发布工作树 **0.109** 新增 [RMDL 完整两阶段参考实现](docs/RMDL.md)，注册表为 **161／164 个实现候选、352 个执行配置**。
此前160条已通过各自的软件阶段检查；RMDL已通过完整作者网络组件、原始Adam数值、聚合器跨进程恢复和一次实际全链路软件运行，源码／安装包与完整预算验证仍单独验收，暂不计为第161条软件阶段已验证接入。
RMDL保留原图与TRAIN像素标注、完整55.9M定位器、硬负例、空间NMS、真实区域重编码、300×1024聚合输入及80／256批量；未公开训练流程的重建选择与原文／源码差异均明确记录。
已发布开发版仍为 **0.98：151条／321配置**，稳定版为 **0.62：117条／219配置**。
整库最终验收、全部164条准入和真实BRCA／COAD实验仍未完成。软件验证不等于论文指标复现。

下面保留各阶段的历史实现与验证记录；其旧等待状态不自动代表当前状态。

Prompt-MIL 保留作者三套原始DINO权重、完整192维／12层／3头RGB224骨干、真实提示微调、双流DSMIL与三次前后向。
源码配方保留8个切片的梯度累积及尾批梯度跨轮携带；论文预算参考配方明确补充尾批更新和验证选模。
源码与论文的轮次、CRC学习率、GAP／CLS描述及评价分数差异见文档，独立安装和完整预算仍单独验收。
27项专项与接口检查通过，输入像素／内存布局也与原Kornia转换逐值核对。

父版本 IIB-MIL 三套完整参考配置、32项专项和6套源码／安装包软件流程已通过；其300轮完整预算验证保留独立身份。
完整作者IIB训练循环未公开，重建决定在 [IIBMIL.md](docs/IIBMIL.md) 明确说明。
当前状态与历史快照分别记录，见 [README_CURRENT.md](README_CURRENT.md)。

新增 [LNPL-MIL 完整原图流程](docs/LNPL_MIL.md)：真实RGB224／TRAIN有限patch标注、完整弱ResNet18、独立冻结ImageNet特征、SP／Top200、四层TOD和原始Lookahead/RAdam。源码条件与论文残差／辅助损失／不重叠superpatch条件分别命名；原作者没有发布的弱分类器配方明确作为参考重建。该公开入口正在单独验收，不计入已交付151条。此前冻结父版本的32项专项检查和完整软件链路通过，不代表本工作树已完成原生训练预算、独立安装包或整库验收。

新增 [KMP-MIL](docs/KMP_MIL.md)：完整原始CONCH、语义初始化、PFC／TC与路由损失，按原代码处理16-bag尾批，验证集最佳模型回载后建立无需重放旧切片的记忆库。14项专项检查、作者完整训练循环对照、两种配置的统一CLI及独立安装包三处精确中断恢复均已通过。源码／独立安装包×max／mean共四套完整20轮上限GPU条件通过，保留每任务33个TRAIN包、每包4097个patch与16＋16＋1真实更新；逐轮检查非终端Adam矩、梯度、全部状态及两次新进程重载。默认CPU命令行及整库、正式验收仍分别跟踪；不计作真实WSI或论文指标复现。

0.96新增 [QPMIL-VL](docs/QPMIL_VL.md)：真实完整 CONCH 构造及约 1.10 亿参数文本塔，原始 20／24／5 原型池、16-bag 更新、三项损失与四任务训练；max-query 和 mean-query 分别登记。32 项专项检查、两项默认 CPU 命令行，以及两个独立安装配置的公开 CLI、三处逐项精确中断恢复和双重重载均已通过。两个配置 × 源码版／独立安装包共四个完整原生 GPU 条件均已通过；整库与正式发布验收仍未完成，不计作真实 WSI 或论文指标复现。

下面保留各阶段的实现与验证记录；历史段落中的等待状态不自动代表当前状态，以本节计数及快照状态为准。

[NcIEMIL 完整参考重建](docs/NCIEMIL.md) 已接通原始RGB256、完整预训练Swin-Tiny最大实例弱监督训练、全实例评分与高低分筛选缓存、原始聚合器和统一CLI。K128／四分类与K512／二分类分别完成原始及显式修正KV两种配置的100轮原生GPU训练，并通过真实中断恢复与两次独立重载。作者未发布的过滤训练参数采用明确声明的参考设置，不能当作论文数值复现；安装包、整库与最终准入仍分别验收。

新增 [TaxoMIL](docs/TAXOMIL.md)：真实预训练 GPT-2 small、原始双分支与双生成头、四项源目标和三周期权重；完整原始数据、模型初始化、整轮训练的全部梯度／AdamW状态／随机数及验证生成已核对，真实完整状态写出恢复通过。原始全损失入口的 `coarse_ital`／`c_ital` 键名错误显式修正；源码学习率与论文学习率分别登记。0.88修正了百分比乘除次序，保留与原始验证的严格相等检查；源码／论文学习率 × BRACS／PANDA四套完整原生预算、CLI恢复与双重重载已通过（21／21／35／35轮，均保留100轮上限与耐心20）。当前开发版本的整库与最终安装包验收仍分别执行，不计作论文指标复现。

新增 [GMMamba](docs/GMMAMBA.md)：完整原生512维双向16状态 IMM、真实坐标10组KMeans、八头CSS与分类token；保留原始AdamW、首次epoch6选模、100轮上限和完整恢复。原始组件10项、完整训练与恢复7项、真实CLI4项检查通过；完整原生预算、整库与最终安装包检查仍待完成。缺失的BiMamba目录以明确固定版本依赖移植接上；原生512维和显式FM投影修正分别登记。IIB-MIL仍为原始组件核验阶段，没有登记可运行方法。

新增 [MUSE](docs/MUSE.md)：完整原始约8900万参数网络、12层文本塔、600条原始描述、8专家和每个训练包20次独立更新。17项专项检查全部通过，包括作者循环权重／Adam／随机状态逐项相同、真实中断恢复及方法名CLI。这里只登记明确命名的公开源码条件：已查出类别提示覆盖、文本重初始化、检索方向等源码与论文差异，完整原生预算、整库、最终安装包及论文机制对照仍需分别完成。

新增 [Libra-MIL](docs/LIBRAMIL.md)：完整原始512维 SOT、8头注意力、真实预训练CONCH编码的46条结构和3条RCC类别描述、可学习多模态原型；4／6原型配置的两套完整20轮CPU及CUDA源训练、每套540次更新、完整状态恢复和双重重载已通过。49项专项检查通过；整库和最终发布验收分别跟踪。

[DyKo](docs/DYKO.md) 已完成固定来源和30项原始组件核验，仍等待获授权的TITAN资产及完整适配，不计入已完成方法数量。


新增 [M4](docs/M4.md)：完整五专家、多代理卷积层级、每终点四组门控和独立 sigmoid 预测头，保留求和 BCE、Adam、32轮余弦调度和验证平均端点 AUROC 选模。原始完整两轮训练／Adam／调度／随机状态逐项相同，真实状态恢复及2048／1024维两个 FM 接口的 CLI 双重重载已通过预检查；完整原生预算、整库和安装包验收仍待完成。

新增 [ConVL](docs/CONVL.md)：完整原生 CONCH 文本塔、16 个动态提示、图像条件 MetaNet、四头概念注意力和三项辅助目标，保留原始 Lookahead／RAdam、患者完整多切片包、200轮上限／验证 NLL 耐心8／梯度累积2。原始 FP16 配置要求 CUDA；FP32 参考配置单独命名。完整原始两轮训练对照、超过1 GiB真实状态恢复、CLI训练选模及两次独立重载已通过预检查；四套完整 CUDA 原生预算、整库与最终安装包检查仍待完成。

新增 [MIL with Center Embeddings](docs/CENTER_EMBEDDINGS.md)：完整原始 ResNet34／FPN／32张256像素原图包网络、实例与包分类及中心损失、两套 Adam 和原始学习率重置。保留每轮全候选扫描、按原始正类分数选择32张图、原始增强、验证平衡准确率选模，以及包分类和中心分配两种推理。作者完整两轮训练对照和真实断点恢复已通过；完整原生20轮预算、整库及安装包验证仍分别验收。

新增 [ConMIL](docs/CONMIL.md)：完整原始文本语义注意力、真实预训练 CONCH 文本编码、立方 B 样条 KAN、50轮 Adam／余弦调度和原始验证准确率选模。正提示、正负提示、明确命名的多分类修正配置分别登记。36 项预检查通过，包含作者完整循环的权重／Adam／调度／随机状态逐项相同、实际4096训练上限与完整4097验证包、真实断点恢复和独立重载。原始 KAN 网格重载问题只以相同数值的可写缓冲区修复；四套完整原生预算已通过，整库与最终安装包检查仍须完成。

SparseConvMIL 当前冻结30项专项检查通过，四套完整512像素／200轮编译稀疏后端参考验证全部通过。C2C 的四套512像素／25轮 GPU 预算检查全部通过；DT-MIL、AMIML、SAM-MIL 的完整原生预算通过。整体验收仍单独检查。当前没有真实 WSI 或论文指标复现结果。

以下版本说明保留历史状态，当前数量以本段与注册表为准。

0.56 开发中：TRIAGE-MIL 完整 MASS 筛选、语义超图、原生网络和完整训练配方已进入候选验证；尚未纳入正式数量。

0.55 新增 C²MIL：完整自适应偏差移除、因果图三分支、五项联合目标、真实缩略图和TRAIN1500patch参考池，保留原始完整训练/选模配方并完整保存动态聚类状态。18项专项、全库2361项和210配置通过；独立安装包完成1024/1536维原生网络各100轮训练及选模重载。见 [C²MIL](docs/C2MIL.md)。

0.54 新增 GAT-Mamba 的真实空间/六类亚型图、完整原始 GAT/Mamba1.2 网络、Cox/L1/Adam/随机验证/plateau和最后并列选模配方；提供显式边条件化修正版。23项专项、全库2343项和209配置通过，四组原生网络/原始预算的独立安装包训练及选模重载通过。见 [GAT-Mamba](docs/GATMAMBA.md)。

0.53 新增 ProtoSurv 的原始图/组织多原型融合、三项联合目标及完整患者生存训练配方。28项专项、全库2320项和207个配置通过；四组原生网络/100轮上限与原早停规则的独立安装包训练、选模重载与重复预测通过。见 [ProtoSurv](docs/PROTOSURV.md)。

0.52 新增 PathRWKV 的完整原始 TimeMix/ChannelMix、多任务联合训练、原生 CUDA 及明确命名的损失/推理修正路线。全库2292项、205配置CPU检查通过；H200实际验证两个原生算子、梯度/状态及四组768隐藏层完整训练/重载/预测。见 [PathRWKV](docs/PATHRWKV.md)。

0.51 新增 C2Aug 的完整论文参考配置：原生 TransMIL、三种跨包增强、EMA 教师、bag FIFO InfoNCE 和全部原型组 KL，完整断点状态与 TRAIN 专用索引库。作者源码缺项与明确的重建选择见 [C2Aug](docs/C2AUG.md)。203 个配置的逐项运行及全库 2239 项检查已通过，无失败、错误或跳过。

0.50 新增 AGDR-MIL 的完整来源双分支、YAML/TCGA 两配置、原始训练/验证随机状态及逐行实例输出。保留无梯度硬分级器与 Sigmoid-CE，见 [AGDR-MIL](docs/AGDRMIL.md)。

0.49 新增 H²-MIL 的真实三层图接口、来源兼容和显式修正两条完整训练配置及结构导出。见 [H²-MIL](docs/H2MIL.md)。

面向 WSI 的 MIL 实验工程。目标是使用统一任务数据包，通过 `milplus run --data 路径 --method 方法` 启动实验，同时保留方法的实际训练机制。

**正式0.62已验收117条/219配置；后续候选仍须分别完成当前源码的准入检查。** 当前数量以文首与快照为准，以下历史版本段落保留当时状态。原先17个配置是首批交付，未覆盖完整计划。`milplus methods` 显示全部164条，`milplus methods --runnable` 显示执行配置；待接入不会静默替换为ABMIL。

0.48 新增 HIPT 的两条原生预训练层级分类配置，保留实际教师权重、完整物理网格、配套 DINO 特征空间与作者完整训练流程。见 [HIPT](docs/HIPT.md)。

0.47 新增 SI-MIL 的完整双分支、PathExpert 数据登记、训练集专用归一化和两条梯度路线；保留作者损失、优化器、选模和解释输出。它需要实际手工病理描述符，不能仅靠 FM 特征运行。见 [SI-MIL](docs/SIMIL.md)。

新增训练流程包括原型预计算、warmup、EMA 教师、空间与排序约束、平衡批次、伪包混合及 ReMix 协方差增强。详见 [本轮实现与差异](docs/EXPANSION.md)。**完整计划仍未全部实现，也未进行原论文数值复现。**

0.10 增加 MiCo 分类/患者生存与 EfficientMIL GRU/LSTM，保留训练专用聚类、上下文增强、APS、双分支损失和调度。范围与差异见 [上下文与循环网络](docs/CONTEXT_SEQUENCE.md)。

0.11 增加 SCMIL 连续生存分布与 DeepAttnMISL 表型簇生存配方，包含时间单位、患者多切片边界、删失似然、批内 Cox、L1 和验证损失调度。见 [连续生存与表型簇](docs/CONTINUOUS_SURVIVAL.md)。

0.12 增加 MIL-Dropout 的 ABMIL、门控 ABMIL、DSMIL、TransMIL 四配置，以及 PackMIL 的分类和离散生存两配置，保留重要实例掩码、Lookahead、残差 hyperslide 分支和专用混合标签。输入、配方与作者差异见 [MIL-Dropout / PackMIL](docs/DROPOUT_PACKING.md)。

0.13 增加 S-DFP 四骨干的两阶段提示训练、MFC-MIL 两骨干的频域与记忆干预，以及 DyHG 动态超图。关键训练状态、源码修正和范围见 [提示、频域与超图](docs/PROMPT_FREQUENCY_HYPERGRAPH.md)。

0.14 分别接入通道版 `camil_channel` 和上下文版 `camil_context`，包含作者优化配方、物理邻域构图和独立 TensorFlow 数值对照。两篇论文的命名、公式、兼容修正与范围见 [两个 CAMIL](docs/CAMIL.md)。

0.15 增加 HEALNet 图像/组学生存融合与 HE2RNA 基因表达回归，包含患者表达目标、空间 supertile 缓存、作者加权损失与调度。输入、完整配方和源码修正见 [分子预测与融合](docs/MOLECULAR_REGRESSION.md)。

0.16 增加 CATE 与 ViLa-MIL 的三个图文配置，保留完整概念损失、CLIP 提示学习、双尺度解码；记录源码 EOT 和文本冻结差异。文本资产准备、输入协议与验证见 [视觉语言方法](docs/VISION_LANGUAGE.md)。

0.17 增加 RankMix 的 DSMIL / FRMIL 两阶段训练及源码行为对照，共四配置。实例索引修正、预训练选模、混合损失和精确续训见 [RankMix](docs/RANKMIX.md)。IIB-MIL 的具体缺项见 [审计记录](docs/IIBMIL_AUDIT.md)。

0.18 增加 TPMIL（完整原型 KL 训练）和 `cdpmil_source`（公开代码的 DP 聚类/概率分类训练路径）。TPMIL 退化注意力修正、cDP 公开实现与论文机制的边界，以及 INS 的实际缺项见 [原型与 DP 方法](docs/PROTOTYPE_DP.md)。

0.19 增加 HMIL 的显式层级标签、完整公开训练配方和粗细两级输出。分类头修复依据、源码与论文差异、SCL-WC 待解决问题见 [层级分类与来源审计](docs/HIERARCHICAL_MIL.md)。

0.20 增加 AdvMIL 的 ESAT/ABMIL 与源码/修正 BCE 四配置，保留区域投影、噪声生成、双优化器交替训练、删失重建和显式标签隐藏协议。见 [对抗生存训练](docs/ADVERSARIAL_SURVIVAL.md)。

0.22 增加 MMP 的四种原型/融合配置，包含训练集原型构建、图像与分子通路融合、患者批 Cox/NLL、AdamW 与 warmup/cosine。输入要求与源码修复见 [MMP](docs/MMP.md)。

0.21 增加 PIBD 的完整公开多模态生存训练路径，包含原型瓶颈、PoE、解耦 Transformer、联合 CLUB 估计器和患者小批量训练。数据要求、源码差异与验证见 [PIBD](docs/PIBD.md)。

0.23 增加 PathOmics 的双模态、图像分支和组学分支，包含无监督对齐预训练、生存微调、独立优化器和预训练选模。源码冻结循环、快照和尾批修复、输入边界见 [PathOmics](docs/PATHOMICS.md)。

0.24 增加 PAMoE 两骨干、源码/修正版共四配置，支持分类与生存；包含组织原型准备、先验损失、患者批采样、完整优化流程和版本检查。索引修复及原型输入见 [PAMoE](docs/PAMOE.md)。

0.25 增加 DGMIL 动态实例选择、完整投影训练、负类分布缓存和两种并列项处理配置。使用原始距离计算排名指标；测试拟合修复、概率报告层及验证范围见 [DGMIL](docs/DGMIL.md)。

0.26 增加 MambaMIL 的 SR、Bi、单向三配置，支持分类和患者生存。序列重排序、显式扫描后端、训练配方及修复见 [MambaMIL](docs/MAMBAMIL.md)。

## 安装与第一次运行

Python 3.10 或更新版本。先按照实际硬件安装 PyTorch；当前 CPU 验证环境使用 Python 3.10、PyTorch 2.5.1+cpu。GPU 环境请使用 CUDA 版 PyTorch。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'

milplus demo --output /tmp/milplus_demo
milplus check --data /tmp/milplus_demo --method abmil --deep
milplus run --data /tmp/milplus_demo --method abmil --epochs 3 --output runs/demo_abmil
milplus evaluate --run runs/demo_abmil
pytest
```

本工作区已经安装好环境，可直接执行：

```bash
cd /root/mil_baseline_puls
source .venv/bin/activate
milplus methods
```

演示数据永久标记 `synthetic: true`。新运行拒绝覆盖已有目录；重复演示请选择新路径。

## 首批配置与新增方法

| 配置 ID | 保留的机制 |
|---|---|
| `mean_linear` | 原始 patch 特征均值，训练集拟合标准化和 logistic regression |
| `mean_mil` / `max_mil` | 学习实例分类头后对 logits 做均值/最大池化 |
| `abmil` / `gated_abmil` | 非门控/门控注意力，切片分类 |
| `clam_sb` / `clam_mb` | 官方 CLAM 模型，实例聚类损失与 bag 损失；CE 实例损失配置 |
| `dsmil` | 实例最大分支、关键实例查询、bag 分支、双分支 BCE |
| `transmil` | Nystrom 注意力、CLS token、序列方形填充、PPEG |
| `dtfd` | 两级优化器、伪包分类、AFS/MaxS/MaxMinS 蒸馏 |
| `ilra` | 低秩压缩/恢复、门控残差和非局部池化 |
| `wikg` | 特征构图、top-k 邻居、知识注意力和双交互；分块计算图分数 |
| `acmil_ga` | 多注意力分支、随机 top-k 掩码、分支监督和多样性损失 |
| `aem_abmil` | ABMIL 加注意力负熵正则 |
| `mhim_abmil` | v1 注意力路线；显式监督初始化、特征蒸馏、掩码和 EMA 教师 |
| `abmil_cox` / `gated_abmil_cox` | PULS 生存适配，患者 bag、完整风险集、Breslow 并列时间处理 |

新增配置：FRMIL、FocusMIL、DGR-MIL、RRT-MIL、CHOWDER、VarPool（分类/Cox）、H2T、PANTHER、IBMIL、Additive MIL、Lin-MIL、FourierMIL、RetMIL、PNEA-MIL、MR-Block、CAR-MIL、FC-MIL、ASMIL、AttriMIL、nnMIL、MAD-MIL、LiteMIL，以及 PseMix/ProtoDiv/ReMix 的 ABMIL 路线。

0.3 新增 SurvPath、MCAT、PORPOISE 和 Patch-GCN，保留作者多模态/图网络及离散生存目标，并增加患者组学、通路、分箱和图接口。安装、任务格式和适配差异见 [生存与多模态](docs/SURVIVAL.md)。

0.4 新增 SGPMIL、MMIL-Transformer（随机/真实坐标分组）和 PTCMIL，包含 GP 变分 KL、组间消息通信、提示聚类损失与移动上下文。安装、原始源码和适配差异见 [GP 与提示方法](docs/GP_PROMPT_MULTILEVEL.md)。

0.5 新增 CBMIL，保留聚类自适应选择、实例约束、双视图对比与批内混合，支持带哈希的外部聚类标签。配置、输入维度适配与原流程差异见 [CBMIL](docs/CBMIL.md)。

0.6 新增 CMTA、MOTCat 整包与 micro-batch 生存配置，保留跨模态 L1 对齐、最优传输和小包损失/风险平均。输入、依赖与配方差异见 [跨模态方法](docs/CROSSMODAL.md)。

0.7 新增 CS-MIL 和 HAG-MIL 的 CE 实例损失配置，增加物理中心对齐、真实四叉树、聚类采样与逐层训练接口。输入格式和适配差异见 [多尺度方法](docs/MULTISCALE.md)。

0.8 新增 SmMIL 的四个平滑配置、PSA-MIL、SlideGraph+，保留空间尺度、作者目标及优化流程。输入契约、公式约定和适配修正见 [空间方法](docs/SPATIAL.md)。

0.9 新增 Bayes-MIL 的三种变分/空间配置、CAPRMIL 三种上下文池化配置和 VarMIL。MC 推理、原始公式边界、数值修正与来源见 [Bayes、上下文与方差方法](docs/BAYES_CONTEXT_VARIANCE.md)。

这些配置含基础对照、变体和适配，不能按运行配置数量计作独立论文。详细差异见 [方法说明](docs/METHODS.md) 和 [扩展实现](docs/EXPANSION.md)，测试和逐配置运行证据见 [验证结果](docs/VALIDATION.md)。方法名如 `CLAM`、`DTFD-MIL`、`ACMIL` 会解析到明确的默认配置；同名歧义方法保留独立 ID。

```bash
milplus inspect --method mhim
milplus methods --json
milplus methods --runnable
milplus coverage --json
```

## 数据包：路径包含标签与协议

```text
my_task/
  task.yaml
  slides.csv
  split.csv
  features_uni_v1.csv
```

`task.yaml` 示例见 [数据协议](docs/DATA.md)。`slides.csv` 至少包含 `slide_id,patient_id,label`；存在跨来源身份映射时加入 `global_patient_id`。生存任务使用 `time,event`。`split.csv` 为 `patient_id,subset`，其中 subset 必须为 train、val、test。

特征索引为 `slide_id,path`，可附加 `coords_path,sha256,coords_sha256`。支持 `.h5/.hdf5` 的 `features`/`coords` 数据集、二维 `.npy` 和安全加载的张量 `.pt`。同一任务的每个 FM 必须完整覆盖同一套切片；不自动丢弃缺特征的病例。

```bash
milplus run --data /path/to/task --method clam_sb --fm uni_v1 --split fold0 \
  --seed 2026 --device cuda --config configs/clam.yaml --output runs/brca_clam_uni
milplus evaluate --run runs/brca_clam_uni --device cuda
```

训练仅使用 train 更新参数、val 选择 checkpoint。`evaluate` 明确执行锁定测试，训练循环不计算测试指标。

## BRCA、COAD 与 TRIDENT

原始 WSI 路径、已知标签范围及准备命令见 [H 集群使用流程](docs/H_CLUSTER.md)。目前 BRCA 的候选任务是 IDC/ILC；COAD 的 COAD/MACR 是形态任务，**不是 MSI 预测**。已有标签还需核验临床来源，原始切片还需完成解码、MPP 和组织 QC。

```bash
milplus fetch-source --source trident --destination external/TRIDENT
milplus trident --repo external/TRIDENT --python /path/to/trident/env/bin/python \
  --wsi-dir /path/to/raw --custom-list /path/to/task/wsis.csv \
  --output /path/to/trident_output --fm uni_v1 --mag 20 --patch-size 256 \
  --checkpoint /path/to/authorized/uni_weights
```

默认生成命令；加 `--execute` 才执行。TRIDENT 使用单独环境，模型权重、访问授权和完整依赖由其原生流程管理。此仓库不包含 FM 权重。

```bash
milplus index-features --data /path/to/task --feature-dir /path/to/trident_h5_features \
  --fm uni_v1 --dimension 1024 --weights '实际权重版本或哈希' \
  --preprocessing '实际 TRIDENT 提取协议、MPP、视野和预处理版本'
milplus check --data /path/to/task --method abmil --deep
```

## 批量实验与统计

```bash
milplus matrix --data /path/to/task --methods mean_linear abmil clam_sb dtfd \
  --fms uni_v1 uni_v2 --seeds 2026 2027 2028 --output matrix.json
milplus run-matrix --plan matrix.json --output runs/main --devices cuda:0 cuda:1
milplus report --runs runs/main --output runs.csv
milplus compare --left runs/A/test_predictions.csv --right runs/B/test_predictions.csv \
  --classes 2 --metric macro_auc --unit patient --repetitions 2000 --output comparison.json
```

`--methods all` 将全部 164 条纳入矩阵，保留尚未接入和输入不兼容的条目，不自动缩减为已实现子集。矩阵保留 blocked 组合；调度只执行 ready 组合，每个设备同时一个进程。不会自动用测试集调参，也不会静默覆盖或恢复已有目录。`compare` 要求两个运行具有完全相同的测试身份与标签；按患者整组 bootstrap，并保留重复抽到患者的采样权重。该接口比较一对运行，不允许把多个 seed 拼成新增独立病例。

## 恢复与实验产物

```bash
milplus run --data /path/to/task --method abmil --output runs/interrupted \
  --config configs/classification.yaml --device cuda --resume
```

恢复需要相同的配置、seed、设备、源码、依赖版本和输入指纹。从上一个完整 epoch 恢复；中断 epoch 重新执行。checkpoint 包含模型、优化器、Python/NumPy/Torch/CUDA RNG、最佳分数和 patience 状态。`mean_linear` 是单次拟合，不提供中途恢复。

每次运行输出 `request.json`、`inputs.json`、`environment.json`、`status.json`、`history.json`、`last.pt`、`best.pt`、验证预测及指标；显式测试后增加测试预测与指标。线性模型使用无 pickle 的 `linear.npz`。神经 checkpoint 是本程序产生的可信本地文件，不应加载未知来源的完整恢复文件。

## 工程结构

```text
src/milplus/
  cli.py                    统一入口
  data.py, preparation.py   任务包、特征和 TCGA 准备
  models.py, extended.py    首批神经模型
  expansion.py              扩展模型与专用损失
  prototype_models.py       原型/混合分布/混杂注意力
  instance_models.py        FRMIL 与 FocusMIL
  augmentation.py           PseMix / ProtoDiv / ReMix
  omics.py                  患者组学与通路对齐
  discrete_survival.py       训练集时间分箱、删失 NLL 与梯度累积
  multimodal.py             SurvPath / MCAT / PORPOISE
  graph_survival.py          Patch-GCN，患者多切片图
  cluster_contrastive.py     CBMIL 聚类选择、批内混合与对比训练
  gaussian_process.py       SGPMIL，GP 变分与 Monte Carlo
  multilevel.py              MMIL 分组及 message token 通信
  prompt_clustering.py       PTCMIL 与提示状态
  bayesian.py                Bayes-MIL 变分目标、空间栅格与 MC 推理
  context_variance.py        CAPRMIL 上下文聚合和 VarMIL 均值/方差
  slide_prompt.py            S-DFP 两阶段联合训练与提示状态
  frequency_causal.py        MFC-MIL 多分辨率、频域与记忆干预
  dynamic_hypergraph.py      DyHG incidence、超图传递与随机推理
  mil_dropout.py             四骨干重要实例掩码、Lookahead 与配方
  packing.py                 PackMIL 打包、ADS 与残差 hyperslide 目标
  stochastic.py              推理随机状态隔离
  upstream.py               固定作者源隔离加载
  teacher.py, recipes.py    教师、掩码、辅助损失和优化器
  engine.py, linear.py      训练、恢复、验证与测试
  metrics.py                患者聚合、指标、配对区间
  execution.py              矩阵、H 集群、TRIDENT、源码获取
  resources/                164 条候选、配置和固定源码版本
  vendor/                   保留来源和许可证的模型源码
method_specs/               全部 164 条独立规格与实际状态
tests/                      软件和机制验收
configs/                    可运行训练配置
docs/                       方法差异、数据、集群、进度与验证证据
```

接入工作以 [ADAPTATION_STATUS.md](docs/ADAPTATION_STATUS.md) 为准。多尺度、图文、多组学、持续学习、预训练切片模型等未适配方法不能通过改名或统一 CE 循环进入正式评测。

## 来源与许可

CLAM 官方模型和许可证随包保留；TransMIL 适配取自固定版本 MIL_BASELINE，保留 MIT 许可证。其他实现的来源与算法差异见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[方法说明](docs/METHODS.md) 和源码锁。RRT、FRMIL、DGR、SurvPath、MCAT、PORPOISE、Patch-GCN、SGPMIL、MMIL、PTCMIL、CBMIL 的作者仓库独立获取，不包含在发布包中；扩展源码和许可证记录见 notices。原始来源获取是只读下载，不等于训练适配完成。

0.27 新增 [MSMMIL](docs/MSMMIL.md)：保留作者三路 AFWMamba、共享注意力、RAdam/Lookahead 和梯度累积。当前全量 812 项测试通过、125 个配置 CPU 合成训练及重载通过；GPU 与真实 WSI 指标尚未验证。

0.28 新增 [MFE-MIL](docs/MFE_MIL.md) 五种分类／六种生存路线，保留共享适配器、窗口掩码、重建 decoder 和完整联合目标。全量879项测试通过、131个配置完成 CPU 合成训练与重载；真实 WSI、GPU AMP 和论文指标尚未验证。MamMIL 接入前的来源审计及 GMMamba 待解决的缺项见 [源码审查队列](docs/STATE_SPACE_REVIEW_QUEUE.md)。

0.29 新增 [MamMIL](docs/MAMMIL.md)：Mamba2/SSD、HNSW/MST/DFS/BFS图资产、源码与修正层循环及显式患者多切片扩展。全量925项检查通过、134个配置完成CPU合成训练与重载；原生512隐藏宽度/状态128的独立wheel运行通过。GPU、真实WSI与论文指标尚未验证。

0.30 新增 [ReCo-MIL](docs/RECOMIL.md) 单/双尺度、MLP/图原型压缩四配置，保留稀有实例、正交损失和完整warmup/tau/冻结训练。全量969项测试通过、138配置完成CPU合成训练与重载；原生512隐藏宽度/K128的独立wheel运行通过。公开源码版本适配与论文身份核查分别登记；GPU、真实WSI和论文指标尚未验证。

0.31 新增 [MoMIL](docs/MOMIL.md) 三路Mamba2分类及明确命名的生存组合扩展，保留通道反转、平方填充和源码最后轮/验证选模。全量1018项测试通过、140配置完成CPU合成训练与重载；原生512隐藏宽度/state64的独立wheel运行通过。构造器与训练协议修复、来源边界均已登记；GPU、真实WSI和论文指标尚未验证。

0.31.1 完成 [SMMILe 的空间资产与数值组件](docs/SMMILE.md)：真实网格、绑定特征/坐标的 SLIC 资产，以及二分类/多类/多标签三个作者网络的输出、梯度和 Adam 对照。SMMILe 的完整两阶段训练、多标签任务接口和实例产物导出尚待接入，因此保持 pending，已接入数量不增加。

0.31.1 当前全库 1114 项测试通过，140 个已有配置在新软件身份下重新完成 CPU 合成训练与重载。新增的 96 项 SMMILe 前置检查不计为完整方法准入。

0.32 接入 [SMMILe 完整三路线](docs/SMMILE.md)：自动两阶段、源验证目标和Adam/plateau、BCE/bibce、组合平衡采样、多标签任务及指标、精确续训和实例HDF5导出。完整计划仍保留164条；当前已接入87条，77条待接入。GPU、真实WSI和论文指标验证另行登记。

0.32 当前全库1189项测试通过，143配置重新完成CPU合成训练与重载。SMMILe的171项关联检查包含原作者完整数值循环对照、阶段衔接、续训及实例输出；全部164条范围仍保留，77条继续接入。

0.33 增加 PAMIL 原型注意力的源码分类、显式修正版和多标签三条完整分阶段训练路线；全库1271项检查及146配置复跑通过。其原型推送需要训练实例标注，输入条件与作者差异见 [PAMIL](docs/PAMIL.md)。剩余76条继续接入；GPU/真实WSI/论文指标仍未验证。

0.34 增加 Loss-Attention 的传统特征、MNIST源码头迁移和图像时序配方迁移三配置；保留共享W/b、加权实例正则、训练专用EMA及每轮重置的源Adam。1325项检查与149配置复跑通过。明确的WSI迁移范围与来源见 [Loss-Attention](docs/LOSS_ATTENTION.md)；剩余75条继续接入。

0.35 接入 [MicroMIL](docs/MICROMIL.md) 冻结FM迁移版，保留作者聚类、硬Gumbel选择、真实DGL图聚合和完整BCE/Adam训练。1370项全库检查与150配置复跑通过；专项含原DGL0.8.1 GAT完整训练对照、slide/patient精确续训和重复推理。74条继续接入；GPU、真实WSI和论文分数未验证。

0.36 接入 [ProbSA-MIL](docs/PROBSA.md) 的作者库ABMIL/Transformer概率注意力四配置；保留MC-BCE、稀疏图KL、周期退火和Adam预热，区分平均logits与预测概率混合。全库1434项和154配置复跑通过，73条继续接入。GPU、真实WSI及论文指标未验证。

0.37 接入 [GTP](docs/GTP.md) 的完整GCN/mincut/ViT、CE与双正则、Adam里程碑调度和原生GraphCAM；保留源码及显式修正两路线。全库1487项和156配置复跑通过。72条继续接入；GPU、真实WSI和论文指标未验证。

0.38 接入 [SETMIL](docs/SETMIL.md) 的真实物理PPE、3/5/7邻域TPMF、完整iRPE空间Transformer和增强/AdamW配方，分别保留源码行为与显式修正。全库1533项和158配置通过。71条继续接入；GPU、真实WSI和论文指标未验证。

0.39 接入 [StableMIL](docs/STABLEMIL.md) 完整物理融合、区域语义、局部/全局注意力、随机二维RoPE及分类/生存配方，显式区分原尾批行为与修正，补齐发布脚本的输入连接。全库1577项、162配置通过CPU合成验证；70条继续接入，GPU/真实WSI/论文指标未验证。

0.40 接入 [MONAI 原图 MIL](docs/MONAI_MIL.md)：完整可训练 ResNet50、五个原生分支、依赖 Transformer/特征图金字塔和序数 BCE/AdamW/cosine 配方。全库1626项、167配置通过CPU合成验证；69条继续接入。论文额外伪标签扩展、GPU、真实WSI和论文分数未验证。

0.41 接入 [Campanella MIL-RNN](docs/CAMPANELLA.md)：真实坐标流式读取、完整ResNet34硬实例CNN/Adam训练、验证最佳CNN和原生10步循环/SGD训练，支持跨阶段精确恢复。全库1676项和168配置通过CPU合成验证；68条仍待接入。GPU、真实WSI及论文数值未验证。

0.42 接入 [MAMMOTH](docs/MAMMOTH.md) 的原生专家路由与八条完整分类配方，保留 CLAM 实例目标和各作者聚合头。54 项专项、六个作者训练循环的18次更新逐位对照通过；全库1730项及176配置通过CPU合成验证。97条已接入，67条仍待实现；GPU、自然WSI和论文数值未验证。

0.43 接入 [PMIL Shapley](docs/PMIL_SHAPLEY.md) 的完整渐进伪包/多轮重启和六条源码/修正配方。91项专项、12组原生作者主程序及6组逐步Adam对照通过；全库1821项和182配置通过CPU合成验证。98条已接入，66条仍待实现；GPU、自然WSI及论文数值未验证。

0.44 接入 [Wibawa MultiAttentionMIL](docs/MSATTENTION.md) 原生三层残差注意力和完整补充 CE/Adam 训练，提供源码推理 Dropout 与命名修正版。31项专项、2组原 notebook 完整训练循环108次更新对照通过；全库1852项和184配置通过CPU合成验证。明确记录冻结WSI特征适配范围，未声称复现私有肿瘤筛选流程；GPU、自然WSI、论文分数尚未验证。

0.45 接入 [TTIS](docs/TTIS.md) 六个原生分类头与七配置、完整 Lookahead/AdamW/cosine 配方、TEST 专用两阶段聚类/FPS/特征融合。79 项专项、七组作者训练入口共98次逐步优化对照通过；全库1932项与191配置通过CPU合成验证。源码CLAM仅bag CE与联合损失变体、主特征语义空间与独立对齐流、CPU/GPU聚类边界分别记录。尚有64条待完成；GPU、自然WSI、论文分数未运行。

0.46 接入[官方2DMambaMIL](docs/MAMBA2D.md)三条原生/修正路线，完整物理网格、可学习填充、分类CE/AdamW/cosine及患者NLL配方。98项专项、6组原始外层训练入口84次逐步更新对照通过；全库2030项与194配置通过CPU合成验证。明确区分发布版自定义梯度、仅导数修正与论文公式修正；未将CPU证据标为CUDA或论文得分复现。尚有63条待适配。

0.47 冻结源码全库2070项、196配置通过CPU合成验证；SI-MIL 35项专项和2组原始训练函数108次逐步更新对照通过。新增5项验证文件保留检查；仅在真实重载/预测后，对新生成合成权重保存哈希并精简，不删除用户数据或历史归档。还有62条待实现。

0.48 冻结源码全库2101项、198执行配置通过CPU合成验证；HIPT 31项专项和两组实际作者平衡训练循环108次逐步更新对照通过。独立wheel两条原生20轮训练共720次Adam更新通过。尚有61条待接入；GPU、真实WSI与论文指标复现均未运行。

0.49 冻结源码全库2150项、200执行配置通过CPU合成验证；H²-MIL49项专项含40次实际作者逐步Adam对照、42级独立池化及90组源码选模边界。独立wheel两条原生60轮路线共120次更新通过。尚有60条待接入；GPU、真实WSI与论文指标复现均未运行。

0.50 冻结源码全库2200项、202执行配置通过CPU合成验证；AGDR-MIL50项专项含实际作者训练180次逐步Adam对照和4组原始验证/RNG对照。独立wheel两条原生1024/512配置按200轮最大预算及原生早停，分别完成33/27轮、共1080次更新。尚有59条待接入；GPU、真实WSI与论文指标复现均未运行。

## Development candidate 0.67.0: VLSA

The development candidate contains122 of164 method families and227 execution profiles. Formal main remains0.60.0 with115 admitted families. VLSA now includes native CONCH ordinal survival, full patient-window IFMLE+EMD training and strict continuation. Its first25 mechanism/training checks passed, but the published cached-notebook numeric comparison failed and remains unresolved; see [docs/VLSA.md](docs/VLSA.md). No VLSA reference reproduction or formal admission is claimed.

0.80 接入 [MIL-Adapter](docs/MIL_ADAPTER.md) 的全部26个实际配置：7类原生聚合器、ZS/random、TaskRes、CLIP和TIP；保留完整少样本/AdamW/余弦/最终权重配方，实际作者CONCH512及KEEP768原型，显式记录源代码缺陷、KEEP输入维度修正和患者隔离协议扩展。85项专项预检查全部通过，包括52套完整源训练循环逐项数值比对和六组真实断点恢复；冻结后的整库与52套完整原生预算、最终安装包检查仍须完成。正式版本仍为0.62/117条。

Development0.89 adds the complete NcIEMIL reference reconstruction: full raw
Swin-Tiny weak pretraining, all-patch scoring/identity-bound highK/lowK cache,
literal native source aggregator/SGD training and public preparation/run CLI.
Current development coverage is142of164 families/297profiles, with117stable
families retained as the prior release. All four NcIEMIL native100epoch source
conditions passed; complete current-wheel/global acceptance is pending.
Missing author filtering hyperparameters and source/paper differences remain
explicit in [NcIEMIL documentation](docs/NCIEMIL.md). No clinical result or
paper-score reproduction is claimed.
