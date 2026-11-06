# 生存与多模态方法

这组接口支持患者级随访和删失数据，包含 `survpath`、`mcat`、`porpoise`、`patchgcn`。后续增加的 CMTA 与 MOTCat 使用相同的患者/组学接口，方法特有损失与分组详见 [CROSSMODAL.md](CROSSMODAL.md)。它们使用固定作者源码和离散生存似然；此前的 `abmil_cox`、`gated_abmil_cox`、`varpool_cox` 继续使用完整风险集 Cox 目标。

这些是受控数据协议下的训练适配，尚未完成原论文数值复现。BRCA IDC/ILC 和 COAD 形态分类仍使用分类配置；不能仅把类别标签改名为 event/time 来启动生存实验。多模态方法在推理时也需要相应患者的分子数据。

## 安装与合成数据运行

在能够访问 GitHub 的开发机准备固定源码；GPU 离线任务需共享相同缓存，或设置 `MILPLUS_SOURCE_ROOT` 指向已准备目录。

```bash
python -m pip install -e '.[dev,graph]'
milplus setup --method survpath
milplus setup --method mcat
milplus setup --method porpoise
milplus setup --method patchgcn

milplus demo --output /tmp/milplus_survival_demo --survival --omics
milplus check --data /tmp/milplus_survival_demo --method survpath --deep
milplus run --data /tmp/milplus_survival_demo --method survpath \
  --config configs/survpath.yaml --epochs 2 --output runs/survpath_demo
milplus evaluate --run runs/survpath_demo
```

`--omics` 创建的是六个合成分子组，仅用于软件验证。真实 MCAT 数据应按作者定义提供六类分子签名，不能以任意六组基因声称复现了其原始输入。

## 任务数据约定

`slides.csv` 使用 `time,event`，其中 **event=1 表示发生终点，event=0 表示删失**。同一全局患者的切片必须具有一致终点和时间，并处于同一划分。时间单位、起点、终点来源及治疗条件写入任务说明。每位患者的全部切片合为一个训练和评价单位。

多模态任务在 `task.yaml` 增加：

```yaml
task_type: survival
training_unit: patient
evaluation_unit: patient
omics:
  table: omics.csv
  columns: [GENE_A, GENE_B, GENE_C, GENE_D]
  pathways: pathways.json
  normalization: train_standard
  preprocessing: '填写表达量单位、上游变换、检测来源、基因版本和病例对齐规则'
```

`omics.csv` 至少有 `global_patient_id` 和 columns 中显式声明的各列，每位患者一行。它必须恰好覆盖当前任务患者，列顺序由 `columns` 决定。没有 global_patient_id 的切片清单会使用 patient_id 作为全局身份。缺测值、重复患者、未知基因和非有限数值直接报错；不隐式补零、选基因、取对数或合并重复样本。

`pathways.json` 是有序列表：

```json
[
  {"name": "pathway_A", "features": ["GENE_A", "GENE_C"]},
  {"name": "pathway_B", "features": ["GENE_B", "GENE_C", "GENE_D"]}
]
```

组之间允许共享基因，单组内不允许重复。SurvPath 使用每个通路单独的 SNN；MCAT 要求恰好六组；PORPOISE 使用完整分子向量，不要求 pathways。表格与通路文件内容都进入数据指纹，修改后拒绝复用旧运行的评价或恢复。

`train_standard` 按唯一训练患者拟合均值和总体标准差，并保存到 checkpoint；常量列尺度设为 1。`none` 明确关闭这一步。所有 assay-level 处理须在 preprocessing 中说明。此统一标准化不自动等同于作者论文中的分子预处理。

## 生存配方与评价

- 默认四个时间区间，仅根据唯一训练患者的已观察事件时间拟合内部线性分位点。区间左闭，最前/最后区间覆盖训练范围以外的时间；边界相等时进入后一档。
- 事件时间不足或分位点重复时显式报错，需预先选择较少的 bins，不自动改变任务。
- 每档预测条件风险 `h`；生存概率为 `S(t)=∏(1-h)`。已发生事件使用此前存活概率乘当档风险；删失使用截至该档的存活概率。
- 训练采用 log-sigmoid 形式，避免 sigmoid 饱和后再取 log 的数值问题。与作者损失在非截断范围内做数值及梯度对照。
- `model.alpha` 是作者式删失损失权重：总损失为 event loss + `(1-alpha)` × censor loss，默认 0。不是类别混合权重。
- `model.accumulation_steps` 控制每次优化更新的患者数；最后不足整组时按实际患者数平均。`model.l1_weight` 是可选的所有可训练参数 L1；默认关闭。共同优化器、学习率、梯度裁剪、训练预算及验证早停沿用 PULS 配置。
- 风险分数使用 `-sum(S)`，风险越大生存越差。验证集 Harrell C-index 选模；显式 evaluate 后才生成测试指标。此指标没有替代竞争风险、时间依赖 AUC、IPCW C-index 或校准评价。

每个运行新增 `survival_protocol.json`，保存分箱、风险定义、分子列/组索引、训练集均值与尺度。预测同时保存逐档 `hazards_i`、`survival_i` 和患者 risk，便于核查及后续统计。

## 模型和兼容性修改

| 方法 | 保留内容 | 明确适配 |
|---|---|---|
| [SurvPath](https://github.com/mahmoodlab/SurvPath) | 通路 SNN，通路自注意力和双向通路/组织交互，分模态池化及生存头 | 原生 256 维内部结构；输入维度来自 FM；统一患者数据、标准化、分箱和训练预算 |
| [MCAT](https://github.com/mahmoodlab/MCAT) | 六类分子签名，原始 co-attention，两条 Transformer，门控聚合及融合 | small WSI 输入层维度参数化；旧 `_LinearWithBias` 使用当前 PyTorch 对应类；输出增加 logits；默认 concat，可选择 bilinear |
| [PORPOISE](https://github.com/mahmoodlab/PORPOISE) | 门控 WSI 注意力，分子 SNN，门控双线性张量融合与生存头 | 原始 CUDA 常量改为跟随输入设备和 dtype；small 分支；默认 bilinear，可选择 concat；不启用未适配的 LRB 分支 |
| [Patch-GCN](https://github.com/mahmoodlab/Patch-GCN) | GENConv、残差 DeepGCN、多深度特征拼接、门控池化及生存头 | 输入维度参数化；层数/宽度可配置，默认作者 4/128；输出增加 logits；精确物理 kNN 取代外部图预计算路径 |

作者源码不随 PULS 分发。载入时核对锁定 commit 和 tracked 修改，并从 Git blob 读取源码；兼容性修改位于 `upstream.load_survival`，关键替换必须唯一匹配，失配时报错。

## Patch-GCN 图接口

每张切片需要真实且对齐的 `[N,2]` 坐标。对每张切片独立构建有向物理 kNN，默认每节点 8 个非自身邻居，边方向为邻居到中心。一个节点时为空边；小 bag 使用可用邻居数。单张切片重复坐标会报错。

同一患者的多张切片按不相连的子图合并，再在患者范围做注意力聚合。不会跨切片连接坐标。实例预算在患者级选择节点，然后按保留坐标重新构图。使用 SciPy 精确搜索，其构图和 tie 行为与作者外部预计算/近似搜索不保证相同，应计入图构造时间。当前适配为 spatial 路线，不冒充 latent 或多倍率图路线。

## 验收边界

新增测试覆盖损失手算、作者值与梯度对照、极端 logits、分箱边界、训练集拟合、分子身份与列校验、数据变更阻断、跨切片图隔离、两种模态与 GENConv 梯度、末批累积、四方法训练重载及逐位断点恢复。

CPU 合成测试不能证明 H200 CUDA 图算子、真实病例分子质量或原论文性能。正式多模态结果还需要真实的患者分子清单与随访终点；公开图像与分子来源的使用条件分别核查。

## PIBD 公开训练路径

`pibd_source` 使用相同患者表与显式 RNA/通路接口，保留四时间区间原型、PoE、解耦 Transformer、CLUB 和联合小批量目标。可显式选择其专用的训练全局 min-max/原始零值恢复模式，见 [PIBD](PIBD.md)。该归一化模式仅支持 PIBD，其他方法不会隐式套用。
