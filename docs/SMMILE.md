# SMMILe：完整空间 MIL 与两阶段训练

固定来源：[ZeyuGaoAi/SMMILe](https://github.com/ZeyuGaoAi/SMMILe)，提交 `4c4c79e364e77c16207f211775ef04dda40ff20a`。作者 README 对应 [Nature Cancer 论文](https://www.nature.com/articles/s43018-025-01060-8)。根 LICENSE 为 GPL-3.0，README 另有 academic/noncommercial 表述；本仓库保留原始声明，从外部固定 Git 对象加载定义，没有打包或重新授权作者代码。

0.31.1 完成空间与数值前置组件；0.32 接入完整训练、任务、选模、续训和实例输出。具体通过的检查与版本见 [验证记录](VALIDATION.md)，GPU、真实 WSI 与论文指标复现单独登记。

## 三条执行路线

| 配置 | 任务与输出 | 原始目标与默认配置 |
|---|---|---|
| `smmile`（别名 `smmile_multiclass`） | 单标签多类；每类独立检测注意力与 patch softmax 分类 | 基本、score-drop、SP bag BCE 的平均；MRF 默认开启 |
| `smmile_binary` | 原作者单个阳性 head；负类为任务 class0、阳性为 class1 | 基本 BCE 加增强 bag BCE 的平均；默认关闭 MRF，适合作者稀疏病灶路线 |
| `smmile_multilabel` | 每个端点独立 sigmoid；不做类别间 softmax | 基本 BCE 加作者 bi-tempered 增强目标；按完整标签组合平衡采样；MRF 默认开启 |

`--method SMMILe` 根据任务数据包的 `smmile_variant` 选择上述路线。没有该字段时，multilabel 任务选择多标签路线，其余选择标准多类路线；不会仅凭两类就推断“肿瘤/正常”。明确指定执行配置会保留该配置的含义。实际配置 ID 记录到 `request.json`；`matrix --methods all` 保留全部164条记录，并为该家族选择对应任务路线。

例如 BRCA 的 IDC/ILC 或肺癌亚型同为两个类别，也可以采用标准 `smmile`。如果采用二分类专用 head，须明确指定 `smmile_variant: binary` 或 `--method smmile_binary`，不应把非阳性类别自动称为正常组织。

## 准备真实空间资产

先取得固定源码并安装准备依赖：

```bash
milplus setup --method SMMILe
python -m pip install '.[smmile]'
```

准备环境固定 `scikit-image==0.25.2`，这是工程版本，作者没有固定该依赖，不能称为作者历史环境。按实际 TRIDENT 提取协议准备普通数据包及有序的真实坐标，再执行：

```bash
milplus prepare-smmile \
  --data /path/to/task --fm uni_v1 \
  --output /path/to/new_smmile_assets \
  --coordinate-frame level0_pixels --patch-stride 512 \
  --patches-per-superpixel 16 --compactness 50 \
  --provenance '填写实际TRIDENT run、切块倍率、level0 stride与坐标来源'
```

`512` 是示例。应填写坐标单位下的真实步长，不能把缩放后的 tile 像素宽度直接当作 level0 步长。也支持 `micrometers`；两个轴须落在声明的物理格点上。保留坐标精度、空洞和 patch 身份，拒绝重复、非格点、复数与越界坐标，不生成伪网格。

命令写出新的 `manifest.json` 与 HDF5 资产，不修改任务文件，不覆盖已有目录。登记到 encoder：

```yaml
smmile_variant: multiclass
training_unit: slide
evaluation_unit: patient
encoders:
  uni_v1:
    # 保留原有 dimension、index、weights、preprocessing 等字段
    smmile:
      manifest: /path/to/new_smmile_assets/manifest.json
```

随后 `milplus check --data /path/to/task --method SMMILe --deep`。资产绑定特征内容、有序坐标、FM、权重身份和原始空间网格；变更或篡改会拒绝运行。NIC 轴顺序为作者的 `[feature,x,y]`；SLIC 在转置后构建邻接、再恢复网格，保留作者对背景0的邻接约定。预处理只使用各 slide 的特征，不拟合队列统计，也不读取真实实例标注。

默认上限为1,000,000格点和512MiB稠密分配预算，SLIC按三份float64画布估算，邻接另行检查。它们不是整个进程RSS上限；更大切片应显式配置 `--max-grid-cells`、`--max-dense-bytes`，不会自动裁剪、缩放或缩减 bag。训练 BatchNorm 需要至少两个真实画布格点。

## 多标签任务数据

必须有明确的端点顺序与完整检测状态：

```yaml
task_type: multilabel
classes: [endpoint_A, endpoint_B, endpoint_C]
smmile_variant: multilabel
multilabel:
  columns: [status_A, status_B, status_C]
  label_provenance: '填写实际标签来源、测定定义及纳入规则'
  empty_target_semantics: '三个端点均有明确阴性记录；未测和未知已在建队列时处理'
```

`slides.csv` 的对应列必须逐项明确为字符串 `0` 或 `1`。缺失、未检测、unknown 和其他值都会拒绝读取，不能以“没有阳性记录”替代阴性。一人多片在患者级训练或评估时必须有相同的端点向量；SMMILe 训练始终使用单 slide bag，患者评估可按固定规则平均概率。每个 train/val/test 子集须有各端点的阳性和阴性支持。

多标签输出独立概率，行和无需为1。报告 macro/micro AUROC、AP、F1、subset accuracy 和独立二元 log loss；阈值为作者的严格 `p > 0.5`。其他尚未支持多标签的方法会明确显示不兼容，不会把任务转换成普通多类。

## 两阶段、目标和选模

默认采用作者 README 的40个基础epochs加20个refinement epochs：

1. 每个 slide 的真实 feature grid 经作者卷积、BatchNorm、ReLU、dropout 和门控检测注意力。保留 score dropout（D1）、按superpixel采样（G10）及单标签路线的SP平滑。
2. 在阶段1按作者验证目标保存最优权重；阶段2严格载入该快照，清空Adam状态，并恢复初始lr和新的ReduceLROnPlateau。不存在快照时拒绝启动，不静默从头训练。
3. 阶段2使用三个实例refinement heads、弱标签候选和可选MRF。真实实例标注不进入训练目标。二分类/多标签的负包一致性须显式开启，并在task中登记 `smmile_negative_bag_semantics`；多标签一致性系数为原作者0.1，二分类为1。
4. 保留作者Adam（lr2e-5、wd1e-5）、不裁剪梯度；plateau以native验证损失调度，factor0.5、patience10。平衡采样使用训练集的类别/完整标签组合频率，而非独立正标签频率。
5. 验证目标保留SP采样及refinement候选CE；验证不使用score-drop、MRF或负包一致性。多类/二分类验证候选率为源码默认0.01，多标签沿用训练率。全局Python浮点累计次序与作者循环一致。

统一报告中的exclusive-class概率会归一化；作者原始bag scores、BCE和OVR排名另存。不同类别有不同检测注意力，原始多类bag scores不一定和为1，不能直接替换成CE或丢弃其原生排名。二类标准路线按源码使用阳性分数计算原生AUC。源验证目标按slide平均，患者级标准指标单独报告。

每次stage与epoch的参数、优化器、plateau、最佳快照和Python/NumPy/Torch/CUDA随机状态都保存，可在切换前后精确续训。训练时不计算测试指标。固定预算配方保留同一运行的随机流，省略原脚本训练前仅用于控制台汇报的validation summary；不宣称历史脚本相同seed的逐位复现。

原作者在stage2 epoch0后无条件重建EarlyStopping，可能在epoch1用更差权重覆盖第一轮最佳值。本适配显式修复为保留全部refinement轮次的最佳值；相等分数保存并重置计数。该差异有原控制器反例测试，不隐藏在方法名称后。

## 运行、续训和实例产物

真实数据包默认运行：

```bash
milplus run --data /path/to/task --method SMMILe --output runs/my_smmile --device cuda
milplus run --data /path/to/task --method SMMILe --output runs/my_smmile --device cuda --resume
milplus evaluate --run runs/my_smmile --device cuda
```

有明确来源的数据才可解除synthetic标记。软件演示使用：

```bash
milplus demo --smmile multiclass --classes 3 --patients 30 --output /tmp/smmile_demo
milplus run --data /tmp/smmile_demo --method SMMILe \
  --config configs/smmile_smoke.yaml --output runs/smmile_smoke
milplus evaluate --run runs/smmile_smoke
```

另可选择 `demo --smmile binary` 或 `demo --smmile multilabel`；多标签演示包含明确的全阴性向量。演示永久标记synthetic，短预算不能作为论文结果。

显式评估后有 `test_predictions.csv`、`test_metrics.json` 及 `test_instances/manifest.json`。每张slide的安全HDF5包含原始顺序的坐标、bag scores、实例分类scores、检测attention、二者乘积及最终refinement scores。二分类refinement沿任务两类顺序，其余为声明类别再加background。记录坐标frame与stride，不将attention乘积冒称为独立类别概率或组织面积。预测张量来自同一次backbone计算；测试标签只参与指标计算，不影响输出scores或实例文件。

## 修复、来源对照与验证边界

除选模修复外，保留0.31.1已记录的设备、常量span、空邻居、原始坐标精度修复。新增bi-tempered端点修复：float32中的源码上界`1-1e-10`会舍入为1并产生NaN梯度，将精确1裁到其相邻可表示浮点数，非饱和概率保持原公式；非有限梯度会在Adam更新前拒绝。原弱目标逐项除以增强数再相加，保留其运算次序；0.31.1的独立代数检查未覆盖完整trainer的这种浮点次序，本版以实际作者训练循环补足验证。

独立对照直接读取固定Git中的完整数值循环，去掉循环后的实例标注依赖报表与文件写入，不制造实例GT来满足作者日志。验证包括三条任务路线、BCE/bibce、平衡/随机采样、阶段切换、负包一致性、MRF、初始化RNG、优化器状态、完整续训、原始行序和无查询标签预测。具体数量与当前软件身份以 [VALIDATION](VALIDATION.md) 为准。

CUDA环境可运行 `python scripts/validate_smmile_cuda.py --output /path/to/new_cuda_check`。没有CUDA时实际记录 `not_run`，不会将CPU检查当成GPU、真实WSI或原论文数值复现。
