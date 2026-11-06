# 多尺度：CS-MIL 与 HAG-MIL

当前接入两个明确配置：`csmil`（二分类）与 `hagmil_ce`（分类/有序类别，CE 实例损失）。三尺度特征必须来自真实的尺度提取；单尺度向量复制、伪造坐标或改变张量形状不构成多尺度数据。数据接口不会从普通 TRIDENT 输出猜测尺度对应关系。

## 固定来源与输入

| 配置 | 作者仓库与固定提交 | 必要输入 |
|---|---|---|
| `csmil` | [hrlblab/CS-MIL](https://github.com/hrlblab/CS-MIL/tree/d51b97f411caeff092e397427003021d1b23f00f) | 同一组织中心的三个物理视野，逐行对应；可提供最细尺度的实例簇标签 |
| `hagmil_ce` | [BearCleverProud/HAG-MIL](https://github.com/BearCleverProud/HAG-MIL/tree/9fec1ca4390346b958212801d2d5a14c7ce36d6f) | 三层四叉树；每层特征、中心坐标和逐子区域的父索引 |

作者代码通过 `milplus setup` 单独取得，加载前验证 commit 与已跟踪文件，执行固定 Git blob 中的定义。未将作者源码打包到 wheel。CS-MIL 根目录有 MIT 许可；HAG-MIL 当前固定源码未找到根许可文件，许可核查仍待完成。

```bash
python -m pip install -e '.[multiscale]'
milplus setup --method csmil
milplus setup --method hagmil_ce
```

## 数据包格式

普通 `slides.csv`、患者划分及 encoder 维度约定保持适用。在 encoder 下增加 `multiscale`，尺度顺序固定为粗→中→细。下面的视野数值是格式示例，实际应填写切块时的物理视野。

```yaml
encoders:
  uni_v1_multiscale:
    dimension: 1024
    index: coarse.csv
    weights: your_exact_weight_revision
    preprocessing: your_recorded_extraction_protocol
    multiscale:
      kind: quadtree  # CS-MIL 改为 aligned_centers
      coordinate_frame: micrometers
      preprocessing: actual_patch_generation_and_alignment_version
      levels:
        - name: coarse
          field_of_view_um: 512
          index: coarse.csv
        - name: middle
          field_of_view_um: 256
          index: middle.csv
        - name: fine
          field_of_view_um: 128
          index: fine.csv
```

每个索引包含 `slide_id,path`，坐标可放 HDF5 的 `coords`，或通过 `coords_path` 指向独立数组。坐标必须是**同一张原始切片坐标系中的组织中心，单位 μm**。常见提取工具输出的是左上角像素坐标，须用实际 MPP、视野和提取级别正确转换，并记录转换过程。

- 三个索引必须覆盖完全相同的切片，不能共用同一特征文件。每个尺度为非空 `[N,D]`，同一 FM 的三个尺度具有相同 D。
- `aligned_centers` 要求三个尺度 N 相同、逐行中心相同（绝对容差 0.01 μm）。组织视野严格递减。CS-MIL 输入网络时转为细→中→粗。
- `quadtree` 要求相邻视野严格为 2:1。中/细尺度索引还需要 `parent_path`，内容为整数 `[N_child]`，每个值指向前一层的行号。NPY、PT tensor 或 HDF5 `parents` 均可使用。
- 每个父区域必须恰有四个子区域，中心偏移为 `(±父视野/4, ±父视野/4)`，四个象限不得重复。文件行顺序可以不是连续四子结构；模型通过显式父索引展开，选中父区域按注意力排名排列，其子区域保持原文件顺序。
- 文件内容、坐标和父索引都进入输入哈希；任何尺度内容变化会阻止旧 checkpoint 的评价或续训。
- 两个方法均按单张切片训练。患者评价可以融合切片概率，但不能把不同切片的空间结构直接合并。通用 `max_instances` 会被拒绝，须使用方法自己的采样参数。

最细层索引示例：

```csv
slide_id,path,parent_path
slide_a,fine/slide_a.h5,fine/slide_a_parents.npy
```

新提取的特征应先进行人工抽样可视检查。自动几何检查证明索引一致性，不能证明图像内容、MPP 元数据和区域读取本身无误。

## CS-MIL 保留的流程与差异

固定源码的入口为 `Train_Test_Code/MIL_global_Stage1_Training.py`，实际模型类在 `DeepAttnMISL_model_no21.py`，尽管类名沿用了 DeepAttnMISL，网络实际执行三尺度聚合。

1. 每张切片的最细尺度特征单独 KMeans（默认 8 簇，random_state=0，n_init=10）；只在训练准备阶段缓存训练切片的簇。验证/测试在各自切片内无监督聚类，不拟合跨队列的聚类器，也不修改训练缓存。
2. 每次每簇采样数取正态分布 `int(N(3,1))` 并截断为至少 1，取簇大小与采样数的最小值后**有放回**抽样。空簇放零占位并掩码。`clustering: provided` 使用粗尺度索引中的 `cluster_labels_path`，它必须与三个尺度共同的中心行序一致，标签范围为 `[0, clusters)`；允许空簇。
3. 共享 1×1 投影、三尺度评分和实例注意力均来自固定作者模型。原代码跨尺度评分直接加权，**没有对三个尺度做 softmax**，本实现保留此行为。
4. 二分类用稳定的 logits BCE；四个切片的梯度相加后更新，尾批也更新。原代码先 sigmoid、clamp 到 `[1e-5,1-1e-5]` 后算 Bernoulli NLL，因此在饱和区两者梯度不同，已明确采用稳定 logits 版本。

适配改变包括输入维度参数化、去掉硬编码 CUDA、稳定掩码 softmax、统一训练 RNG/梯度裁剪和验证选模。推理以固定种子独立采样，恢复原训练 RNG；原脚本验证采样随调用次数变化。网络输出和非饱和区损失梯度已对照作者代码；固定采样不等于多次采样集成。CS-MIL 原有 SimSiam 图像预训练没有在本配置中复现，此配置用于冻结 FM 特征。

## HAG-MIL 保留的流程与差异

三个独立 IAT 模型分别负责粗、中、细层，每个模型包含四个 IAM 层、Nystrom 注意力、Admin residual、实例注意力、跨层表征组合和实例分类器。

每张切片执行三次共享 Adam 优化器的独立更新：粗层训练→按该次前向的注意力选择父区域→中层训练→再选父区域→细层训练。选择索引停止梯度。粗/中层 bag CE 使用 0.1 label smoothing，细层不用；各层总损失默认为 0.5 bag CE + 0.5 实例 CE。推理只导出细层预测。默认训练/评价最多选择 7000/10000 个父区域。

`hagmil_ce` 显式采用作者代码允许的 CE 实例损失，**没有实现示例配置默认的 SmoothTop1SVM 路线**。`HAG-MIL` 别名解析到此配置，inspect 会显示实际配方。输入少于 `2*k_sample` 时拒绝有重叠的正负实例采样。

固定作者 `main.py` 把配置字典作为 IATModel 的第一个位置参数，未正确传入输入维度等配置。本适配按构造函数签名显式传入，并修复 singleton 时无条件 squeeze 破坏实例维的问题。实例 CE 在输入设备计算，保留梯度。依赖使用 `admin-torch==0.1.0`；当前统一环境的 `nystrom-attention==0.0.14` 与作者 environment.yml 的 0.0.11 不同，数值对照是在当前固定依赖下完成，不宣称旧依赖/原论文结果复现。学习率、梯度裁剪、患者协议、验证选模由可记录的 PULS 配置控制。

## 可复现的软件检查

```bash
milplus demo --output /tmp/csmil_demo --patients 20 --multiscale aligned_centers
milplus run --data /tmp/csmil_demo --method csmil --config configs/csmil.yaml --epochs 2 --output runs/csmil_demo
milplus evaluate --run runs/csmil_demo

milplus demo --output /tmp/hagmil_demo --patients 20 --multiscale quadtree
milplus run --data /tmp/hagmil_demo --method hagmil_ce --config configs/hagmil_smoke.yaml --epochs 2 --output runs/hagmil_demo
milplus evaluate --run runs/hagmil_demo
```

`hagmil_smoke.yaml` 缩小隐藏宽度和采样预算，只用于软件检查。正式结构配置见 `hagmil_ce.yaml`。两种方法的合成验证、逐位续训和作者函数对照不能用于论文性能表。

## H²-MIL 的剩余问题

已固定 [H²-MIL](https://github.com/lin-lcx/H2-MIL/tree/ea472fefd5eab553c58041a5acc203ac7d467d06) 的原始图代码。当前 `main.py` 在不同宽度的层后复用同一个 `LayerNorm(in_feats)`，分类头输出数写死为 4，并在 CE 前做 softmax；层级池化 fitness 参数的训练路径及图输入还需核查。此方法保留为待接入，没有将 HAG-MIL 或其他图模型替代它。
