# 两个 CAMIL：通道注意力与邻域上下文

两篇工作的缩写相同，接口和方法机制不同。PULS 使用明确 ID：`camil_channel` 和 `camil_context`；单独输入 `CAMIL` 会报名称不明确，避免选错论文。

| ID | 原始来源 | 固定提交 |
|---|---|---|
| `camil_channel` | [maojy0914/CAMIL](https://github.com/maojy0914/CAMIL)，[Bioinformatics 论文](https://doi.org/10.1093/bioinformatics/btaf024) | `5481d32a59b0146d9a96f1129bb12ad493acb81a` |
| `camil_context` | [olgarithmics/ICLR_CAMIL](https://github.com/olgarithmics/ICLR_CAMIL)，[ICLR 2024 论文](https://openreview.net/forum?id=rzBskAEmoc) | `5f8ef577e040293794d0f695ceb810f890c5e9f7` |

两个固定来源均未发现根目录许可证，许可状态保留待核。作者原始代码通过 `milplus setup` 单独取得；本包不重新分发作者源码。通道版运行时载入固定 Git 对象中的定义；上下文版为 PyTorch 方程移植，以独立环境运行固定 TensorFlow 定义核对数值。没有把这种软件对照写成论文指标复现。

## 通道版 `camil_channel`

完整网络包含输入投影、默认四个 Nystrom Transformer/MCAB 交替块、LayerNorm、门控注意力池化和分类头。可以显式使用非门控注意力，作为同一家族的配置变体。

MCAB 对当前序列补零成正方形，分别计算 3×3、5×5 深度卷积及逐点卷积，再用共享 MLP 处理全局最大/平均池化结果，生成原特征、3×3 分支和 5×5 分支的三个 sigmoid 门。这三个门不经过 softmax 竞争；源码将温度与 logits 相乘。输出裁去补零 token。

该方形布局来自序列顺序，不需要真实坐标，也不表示真实多倍率图像。FM 更换后仍应保持 patch 顺序一致。默认 hidden=512，四层；测试中另提供显式的小宽度配置，未将小宽度结果冒称为原始规格。

配方依据 `Camelyon/CAMIL.yaml`、`models/model_interface.py` 和优化器源码：

- CE；默认 temperature=1.2、dropout=0.15、lr=2e-4、weight decay=1e-5。
- 使用作者 RAdam，保留 `N_sma >= 5` 的分支和解耦权重衰减，不替换成新版 PyTorch RAdam。
- bias 和一维参数不施加 weight decay。
- Lookahead 的 k=6、alpha=0.5；作者在第一次同步时才创建 slow 权重，第一次六步同步实际不改变 fast 权重，保留此行为。
- 每两个 bag 更新一次；不足两个的最后一组仍按配置的累积数除损失，与作者 Lightning 配方的缩放一致。
- slow 权重按参数名保存，修复原封装使用进程对象 ID 保存状态的问题；RAdam 动量、步数和 slow 权重均可精确恢复。
- PULS 当前使用 FP32，并提供显式梯度裁剪。作者 YAML 的 AMP 配置未在此宣称验证；验证集选模遵守任务协议。

## 上下文版 `camil_context`

输入必须包含单张切片的冻结特征和对齐的真实坐标。训练不会把不同切片坐标系拼成一个邻域图；患者多片可在评价时聚合概率。

### 邻域构图

按作者提取器和数据加载器的两步处理实现：

1. 依据物理坐标的欧氏距离，为每个实例排序得到最多 16 个候选，包括自身。
2. 用原始 FM 特征计算这些候选的余弦距离；此步骤不使用学习后的投影特征，也不参与反向传播。
3. 排除自身距离后，对其余最多 15 个距离做 L2 归一化，再取 `exp(-distance)`；自身权重设为其他权重的最大值。
4. 最后只保留前 8 个候选及对应权重。不能先取 8 个再归一化，因为两种结果不同。

保留作者的排序方式，并对普通随机坐标及有距离并列的规则切块网格核对。重复坐标被明确拒绝。单实例时使用自身权重 1，这是作者预处理在空邻域上未定义的显式扩展。

构图按行分块，避免保存全量 N×N 距离矩阵；排序的时间复杂度仍与原方法有关。图默认缓存于 `~/.cache/milplus/camil_graphs`，可用 `MILPLUS_CAMIL_GRAPH_CACHE` 指定有空间的本地目录。缓存键包含特征、坐标和实现文件的内容哈希，数组校验和会被检查；更换 FM、坐标或代码会生成不同图。随机采样的训练 bag 不写磁盘缓存，避免每轮随机子包产生大量缓存。`model.graph_cache: false` 可禁用缓存；缓存不保存标签或预测。

### 原始网络的轴和目标

全局分支使用作者 TensorFlow Nystrom 的字面行为：在序列前补零到 landmark 数的整数倍、平均 landmark、六步近似逆、残差卷积与输出投影，再加输入残差。尤其需要注意：作者将 `[B, heads, N, dim_head]` 输入默认 NHWC 的 Conv2D，所以卷积沿 **heads** 轴，分组作用于 `dim_head` 通道。本移植保留这个固定提交的行为，没有偷偷换成常见 PyTorch Nystrom 的 token 轴深度卷积；这是源码观察，不是对历史论文实验的判断。

局部分支计算全局特征的 QK 分数，乘邻接边权并按行求和，再对所有节点做 softmax。该注意力加权原始输入投影得到 local 值，随后使用作者的平方 sigmoid 融合：

```text
g = sigmoid(-local)^2
combined = 2 * local * g + 2 * global * (1 - g)
```

最后通过门控 MIL 注意力汇聚。二分类采用单 sigmoid 的等价双 logit `[0,z]` 表示；多分类采用 softmax。运行时只计算选中邻边的 QK，数值和梯度与稠密 QK 乘稀疏邻接矩阵对照。

默认原生宽度 512；其他 FM 维度使用显式可训练线性投影，512→512 时为 identity。默认 heads=8、dim_head=64、landmarks=256、邻域注意力宽度 256、池化注意力宽度 128。初始化采用 Glorot 分布，并保留 TensorFlow 2.15 重复初始化器导致的 Q/K 和 V/U 初值相同；它们仍是独立可训练参数。没有要求两个框架的随机初始化逐位相同。

优化器保留 Keras Adam 的 epsilon 位置及默认值 1e-7，另以真实 Keras 更新核对。验证 loss 连续十轮未改善时学习率乘 0.2，绝对改善阈值 1e-4。原代码虽给层配置 L2，却没有在 GradientTape 的目标中加入 `model.losses`，因此当前配方不施加该项，`weight_decay` 必须为 0。默认 30 epochs，按验证 `log_loss` 选模；用户任务若用患者概率聚合，验证 loss 也按该任务单位计算，并非原始逐切片日志。

兼容修正和边界：

- 整数化 Conv2D 的 filters 参数；用一维 `tf.nn.softmax` 对照原 `K.softmax` 意图。原调用在当前参考 Keras 环境拒绝一维张量，已有实际失败检查。
- 保留单实例的节点轴；原 `tf.squeeze` 在该条件下无法满足 `[None,512]` 形状检查，已有实际失败检查。
- 作者原环境为 TensorFlow 2.6/Python 3.9，当前独立参考使用 TensorFlow 2.15.1/Python 3.10；兼容修正和数值容差公开，不宣称完整旧环境复现。
- 缺少坐标、坐标重复或跨切片拼接会明确报错；不生成虚构物理坐标。

## 使用

```bash
milplus setup --method camil_channel
milplus setup --method camil_context

milplus run --data /path/to/brca_bundle --method camil_channel \
  --config configs/camil_channel.yaml --device cuda --output runs/camil_channel
milplus run --data /path/to/brca_bundle --method camil_context \
  --config configs/camil_context.yaml --device cuda --output runs/camil_context
milplus evaluate --run runs/camil_context --device cuda
```

通道版依赖 `prompt` extra 中固定的 `nystrom-attention`。上下文版正常训练只使用 PyTorch；TensorFlow 仅用于独立对照：

```bash
python3 -m venv /tmp/milplus-camil-tf
/tmp/milplus-camil-tf/bin/python -m pip install -r validation/camil-tf-requirements.txt
MILPLUS_TF_PYTHON=/tmp/milplus-camil-tf/bin/python pytest tests/test_camil.py
```

专项测试覆盖原生四层通道模型的两种 FM 宽度、两种注意力池化、完整梯度，13 步作者 RAdam/Lookahead 对照、梯度累积尾批，真实作者构图及规则网格，缓存失效与损坏，稀疏/稠密邻域计算，单实例，以及两个方法的精确恢复。TensorFlow 对照覆盖二分类、多分类、N=19/256/257、损失、输入及所有活动参数梯度、优化器和调度器。CPU 软件验证通过不代表 GPU、真实 WSI 或论文性能复现完成。
