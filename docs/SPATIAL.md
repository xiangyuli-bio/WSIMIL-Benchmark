# SmMIL、PSA-MIL 与 SlideGraph+

本次接入 3 个家族、6 个执行配置。它们使用固定作者源码，并保留各自的空间机制和训练目标。软件验证是 CPU 合成特征验证；尚无真实 WSI、GPU 或论文数值复现结果。

## 安装和入口

```bash
python -m pip install -e '.[graph]'
milplus setup --method smmil
milplus setup --method psamil
milplus setup --method slidegraph

milplus check --data /path/to/task --method smmil_early --deep
milplus run --data /path/to/task --method smmil_early --config configs/smmil.yaml --output runs/smmil
milplus run --data /path/to/task --method psamil --config configs/psamil.yaml --output runs/psamil
milplus run --data /path/to/task --method slidegraph --config configs/slidegraph.yaml --output runs/slidegraph
```

使用 GPU 时加 `--device cuda`，环境需安装对应 CUDA PyTorch。`SmMIL` 默认解析为 `smmil_early`；其他配置必须显式选择。源码只在联网 CPU 环境获取，随后可将源码缓存和训练环境移入集群镜像/GPFS。

## 数据契约

每个 slide 有独立的 `[N,D]` 特征和严格同行对齐的 `[N,2]` 坐标。坐标必须有限且不重复，不能将多个 WSI 的坐标直接拼成一个平面。可在患者层聚合预测，但这些配置的训练单位为 slide。

在 `task.yaml` 的对应 encoder 下加入：

```yaml
encoders:
  uni_v1:
    dimension: 1024
    index: features_uni_v1.csv
    weights: 实际权重版本
    preprocessing: 实际切块与预处理记录
    spatial:
      unit: level0_pixels
      origin: top_left
      patch_size: 512
      provenance: 实际 TRIDENT 参数和坐标生成记录
```

`patch_size` 必须采用和坐标一致的单位。这里的 512 只是格式示例，不能从 FM 输入张量宽度猜测该值。TRIDENT level-0 坐标应使用真实 `patch_size_level0`；模型输入 resize 后的 224/256 等像素值可能不同。单位允许 `level0_pixels`、`micrometers`、`tile_grid`；tile_grid 的 patch_size 必须为 1。`origin` 是 `top_left` 或 `center`。

PULS 不自动推断物理尺度，不把随机网格标成真实组织坐标。演示数据单独声明为 synthetic。模型选项不能覆盖 encoder 的 patch_size，避免同一特征版本出现冲突定义。

## SmMIL / Smooth Operator

来源：[Franblueee/SmMIL](https://github.com/Franblueee/SmMIL)，锁定 `4e2d1e0e08ad6085afa1469e6180c61516b1991e`，Apache-2.0。

| profile | 实际机制 |
|---|---|
| `smmil_early` | 在注意力计算前平滑嵌入，并池化平滑后的嵌入 |
| `smmil_mid` | 在 tanh 前平滑 attention 的隐层投影 |
| `smmil_late` | 在 softmax 前平滑 attention logits |
| `smmil_transformer` | 作者带平滑的 Transformer，再接 early Sm attention pool |

图以 `sqrt(2)*patch_size` 为半径，无普通自环，边权默认为 `exp(-L2(raw_feature_i-raw_feature_j)/D)`，再做对称度归一化。单节点图有自环；多节点图里的孤立节点保持零邻接行。邻接图是固定的特征/坐标预计算量，不反传到原始 FM。

默认作者维度：嵌入 512、注意力 128；SmTransformer 为 2 层、8 heads、无 FFN。二分类单 logit BCE，输出报告用 `[0,z]` 的二类 softmax。训练默认 weight decay 为 0，epoch 线性 warmup 从 0.1 到 1，持续 `max(floor(epochs*0.1),5)` 个 epoch。`balance_loss` 是显式可选配置，只用训练标签计算正类权重。

已记录的兼容修正和边界：

- 原 pool 的 exact 模式引用不存在的 `self.alpha`，修正为 `self.sm_alpha`。exact 解线性系统，设置 `max_dense_bytes` 限制显式邻接矩阵大小。
- pool 的指数归一化改为稳定 masked softmax 等价形式；正常输入的输出/损失/梯度与原作者代码对照。
- 作者 SDPA 在 eval 中仍传入 dropout；PULS 在 eval 中将其设为 0。默认 dropout=0 时与原路径相同。
- 作者 builder 传入未被使用的 `sm_spectral_norm` 参数；本配置默认关闭。显式开启 `spectral_norm` 会调用模型类真实支持的参数，属于有记录的选项。
- **ApproxSm 与 ExactSm 的 alpha 参数约定并不相同**：作者 Approx 使用 `1/(1+coef)` 作为邻接递推系数，Exact 解 `(I+coef*(I-A))g=f`。保留该固定源码行为，不声称相同 alpha 下两种模式等价。
- 默认 trainable coef、近似迭代 10 步；epoch 边界恢复包含平滑参数与优化器状态。

## PSA-MIL

来源：[SharonPeled/PSA-MIL](https://github.com/SharonPeled/PSA-MIL)，锁定 `673a7a904b565fb413fbdd1b937d544e87c96562`，Apache-2.0。本接入是 **2026-09-28 重构后的分类实现**，不将历史 README 中的生存入口视作已经实现。

保留残差输入投影、空间多头注意力、距离先验、head diversity 和作者优化器分组。默认每个 head 宽 96、3 heads、1 个空间 block、2 层输入 residual adapter，attention pooling。

坐标除以实际 patch_size，转为 tile 单位。每个 head 的有效邻居数由 `round(pi*local_k²)` 决定，从距离最近的候选中选取；它不是“半径内所有点”的另一种图规则。Gaussian / exponential / inverse-quadratic 等衰减由作者代码实现。

训练目标为按训练类别频数倒数加权、批内重新归一化的 CE，**减去** `alpha × diversity entropy`。默认 Gaussian binning，alpha=0.001。距离衰减参数 `lambda_p` 采用普通参数 100 倍学习率；学习率和 weight decay 保留作者按剩余步数计算的分段线性调度。两个参数组及 RNG 都进入 checkpoint。

查询注意力可以分块，但作者实现仍构造 **B×N×N 的完整距离矩阵**。`max_distance_bytes` 限制这部分矩阵的估算字节数，它不是整个训练峰值显存上限。需要缩小 bag 时使用明确、记录在请求中的 `max_instances`；不能把这种采样结果当作完整 bag 协议。

## SlideGraph+

来源：[wenqi006/SlideGraph](https://github.com/wenqi006/SlideGraph)，锁定 `23fbf58371a86525ea2daf0eb0232b379b1b8831`。根许可证为 GPL-3.0，源文件头写 GPL-2.0-or-later；保留该差异，作者源码单独获取。

输入首先进行作者的空间/特征相似性 average-linkage 聚类，默认 lambda_d=0.003、lambda_f=0.001、聚类阈值 0.8。聚类中心取坐标均值并取整，聚类特征取均值；中心经 Delaunay 构图，默认删除距离超过 4000 的边。参数的长度单位就是声明的坐标单位。默认协议使用整数 level-0 像素；其他整数单位必须显式指定 lambda_d 与 edge_distance，不能直接沿用像素阈值。

网络使用作者 GNN：EdgeConv/max 邻居聚合，宽度 `[16,16,8]`，逐层 mean 池化的节点分数相加，dropout=0.5。训练使用同批异类 slide 对的 margin-1 hinge 排序损失，不替换为分类 CE。分层分批确保每批两类均存在且每轮每个样本恰好一次；少数类过少时批数降低，单批可大于请求的 batch_size。

标准化参数仅由训练图节点计算，使用样本标准差；每个 epoch 后在训练 scores 上拟合作者 Platt scaling，再将系数保存到 checkpoint。验证只用于选模，测试只由显式 evaluate 访问。

与作者示例的差异必须保留：

- 作者示例在划分前使用全部图计算标准化参数，PULS 改为 train-only。
- 原预处理按每张片删除近似常量特征列，可能导致输入宽度不同。PULS 距离计算使用相同的非恒定列筛选，输出特征保留共同 D 维；全训练恒定列的标准差设为 1。
- 原 Delaunay 对单节点、双节点或共线中心会失败。增加明确的低维扩展：单节点无边；一维仿射空间仅连接相邻顶点并按距离裁剪。非退化二维图与作者 Delaunay 对照；取整后中心碰撞直接拒绝。
- 使用作者 Platt 代码，唯一兼容修改为去掉已弃用的 `np.float` 别名；预测用稳定 sigmoid 计算同一表达式。
- 当前配置使用一个验证选出的 checkpoint。**没有声称复现作者快照 ensemble**；未使用测试集挑选或加权模型。
- hierarchical clustering 的 condensed 距离仍是 O(N²)，`max_pair_bytes` 只约束该数组，不是总 RAM 上限。不静默换成 kNN 聚类；此配置拒绝 max_instances，保证每轮图拓扑一致。

## 验收范围

专项测试覆盖作者输出、目标值与参数梯度；SmMIL 四种插入位置；padding 与独立 bag 推理一致性；图聚类/Delaunay 对照；空间契约、训练专用标准化/校准、参数分组、调度以及 6 个配置的逐位恢复。完整结果写入 `docs/validation`。

Bayes-MIL 已在 0.9 接入三个明确配置，保留作者正则公式并披露其边界，修正空间 stride 和零方差梯度；完整说明见 [BAYES_CONTEXT_VARIANCE.md](BAYES_CONTEXT_VARIANCE.md)。
