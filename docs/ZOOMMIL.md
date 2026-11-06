# ZoomMIL：三尺度完整训练

本接口对应 [histocartography/zoommil](https://github.com/histocartography/zoommil)
固定提交 `da7bb7fa71629d34e9b04859c1db791ac8330f5d`，MIT许可。
完整模型在运行时从固定Git源私有加载。方法包括三个512维投影、256维门控注意力、
两个辅助选样头、100次高斯扰动的可微Top-K及其原始自定义反向、Kronecker层级选样与求和分类。
没有以普通池化、硬Top-K训练、单一尺度或更窄网络代替原机制。

| 执行配置 | 原生默认配置 | 验证选模 |
|---|---|---|
| `zoommil_bright_source_repaired` | K12、sigma0.002、dropout0.25 | 逐切片weighted-F1首个严格最优 |
| `zoommil_crc_source_repaired` | K16，其余相同 | 逐切片CE首个严格最小值 |

两个配置都完成固定100轮、batch1、FP32、Adam lr1e-4/wd0、逆类别频率有放回抽N个TRAIN袋。
ReduceLROnPlateau依据逐切片验证CE均值，默认factor0.8/patience5。
原生默认8个数据读取进程；可以明确设置 `model.num_workers: 0` 做受限资源的软件检查。
完整DataLoader的base seed和随机采样行为均保留，验证集顺序读取。

`--method ZoomMIL` 默认选择BRIGHT配置。名称表示训练参数来源，实际癌种与类别由任务包登记；
输入维度可以随FM变化，内部512/256宽度不改变。不得据配置名称推断输入已是BRIGHT数据。

## 准备真实层级输入

先建立公共任务包：患者与切片身份、真实标签、患者划分、默认FM及低倍特征索引。
对每张切片提供一个H5，包含三个倍率的特征及实际层级地址。低倍特征必须与公共特征索引一致。
原作者H5格式可以直接登记；高倍数据需要覆盖所保留低倍组织节点的全部后代。

```csv
slide_id,path
SLIDE_A,pyramids/SLIDE_A.h5
SLIDE_B,pyramids/SLIDE_B.h5
```

`zoommil_schema.json` 的原作者地址格式示例：

```json
{
  "schema_version": 1,
  "kind": "zoommil_three_level_source_hierarchy",
  "synthetic": false,
  "patch_pixels": 256,
  "coordinate_frame": "source_recursive_row_col_indices",
  "levels": [
    {"magnification": 1.25, "feature_key": "1.25x_patches", "coords_key": "1.25x_coords"},
    {"magnification": 2.5, "feature_key": "2.5x_patches", "coords_key": "2.5x_coords"},
    {"magnification": 10.0, "feature_key": "10.0x_patches", "coords_key": "10.0x_coords"}
  ],
  "coordinate_provenance": "填写实际坐标、父子关系和提取版本来源",
  "weights_provenance": "填写相同冻结FM及具体权重版本",
  "preprocessing": "填写实际切块、采样和FM预处理",
  "multiscale_provenance": "填写真实倍率、物理视野和完整后代提取规则"
}
```

原作者地址按row/col及递归分支保存，子块顺序为00、10、01、11。
接口依真实地址规范化行顺序；不依据文件排序或特征行号猜测父子关系。
1.25x/2.5x/10x的完整节点比是1:4:64，5x/10x/20x是1:4:16。

也支持实际二维level-0左上角xy坐标。将 `coordinate_frame` 改为 `level0_top_left_pixels`，
并登记 `coarse_patch_size_level0`（低倍patch在level-0的真实边长）及 `patch_size_provenance`。
不同切片校准不同，可以用 `coarse_patch_size_attribute` 指定各H5中实际记录的边长属性名。
转换只依据实际物理位置、倍率及边长；不接受不在声明网格上的坐标、重复、缺失或跨根后代。
普通TRIDENT输出需按三个倍率对应关系整理到上述H5字段，并保留真实坐标与提取配置。
对各尺度独立筛组织后缺失的子块，应补做真实图像/FM提取；不能以零特征、复制或插值伪造。

```bash
milplus setup --method ZoomMIL
milplus prepare-zoommil --data /data/task \
  --index /data/task/zoommil_index.csv --schema /data/task/zoommil_schema.json \
  --output /data/task/zoommil/manifest.json
```

在所用encoder下登记 `zoommil: {manifest: zoommil/manifest.json}`，随后运行：

```bash
milplus run --data /data/task --method ZoomMIL --output /runs/zoommil
milplus evaluate --run /runs/zoommil
```

清单绑定全部层级与基本特征文件哈希、FM身份及实际空间排列。缺少尺度、坐标或完整后代时明确失败。
源Kronecker和MonteCarlo图超过显式 `model.max_dense_bytes` 时也会失败，不截断WSI或替换模型。

## 明确修复与检查点

原模型依据构造时CUDA可用性创建恒等矩阵；这里改为实际输入设备，保持原数值计算。
原loss保存器读取了没有传入的val_loss字段，此处提供实际计算的逐切片验证CE。
原F1保存器初始值0会漏掉全程F1为0的模型，此处定义首个有效检查点；后续仍取严格首个最优。
原论文与公开classifier描述不完全一致，主接口保留公开代码的实际单Linear分类头。

检查点包含模型、Adam、plateau调度器、实际加权采样顺序、全部更新/轮数、学习率轨迹及CPU/CUDA随机状态。
没有提前停止；短轮数仅作为明确声明的软件验证，不代表完成原生100轮预算。

## 证据范围

修订版已通过原生1024与完整1536维的直接原模型输出/损失/全部梯度/随机流对照。
真实0与8读取进程下，完整epoch与原始Dataset、Trainer、Adam、plateau的更新、指标和RNG一致；
另已实际执行两个配置的连续/中断恢复及重复选中模型推理。
当前35项专项检查的最终当前版本收据及独立安装包100轮原生预算、全库/全部235配置仍分别验收。
未运行自然BRCA/COAD或其他WSI，也未复现论文指标。

该接口是公开三尺度缓存特征路线。论文中的两尺度Eff/CAMELYON及按需在线FM提取不是公开模型的三输入接口，
没有用重复尺度代替，也不据缓存特征推理声称原论文的整体WSI/FLOPs提速已经复现。
来源差异与仍未覆盖的条件见 [ZOOMMIL_AUDIT.md](ZOOMMIL_AUDIT.md)。
