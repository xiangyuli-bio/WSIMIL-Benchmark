# MAMMOTH

官方来源：[mahmoodlab/MAMMOTH](https://github.com/mahmoodlab/MAMMOTH)，固定提交
`aeeaffcd12500ed8dddec67665a33eb129c9ed84`。
论文：[Mixture of Mini Experts](https://arxiv.org/html/2603.22198v1)。

这是一项特征变换插件，计为一个方法家族。八条分类配置为：

| 配置 | 聚合与目标 |
|---|---|
| `mammoth_mean` | 专家输出特征均值后分类，CE |
| `mammoth_max` | 选取全局最高 logit 所属的一个完整实例向量，CE |
| `mammoth_abmil` | 作者门控注意力与 CE |
| `mammoth_clam` | 作者实例约束、SmoothTop1 和 bag/instance 加权目标 |
| `mammoth_dsmil` | 作者双流与平均 logits 的 CE；与独立 DSMIL 的 BCE 配方分别登记 |
| `mammoth_transmil` | 作者 Nyström 与 PPEG、CE |
| `mammoth_transformer` | 作者全局 Transformer、CLS 和前馈模块、CE |
| `mammoth_ilra_repaired` | 作者双向低秩交互与非局部池化，显式修正路由宽度 |

核心代码从外部 Git 对象直接加载，未复制进发行包。作者声明 CC-BY-NC-ND-4.0；本项目提供接入代码和固定版本获取方式，原源码及其使用条件保持独立。没有把整个 MIL-Lab 预训练模型集合算作已经接入。

```bash
milplus setup --method mammoth
milplus run --data /path/to/task_bundle --method mammoth_abmil \
  --config configs/mammoth_abmil.yaml --output /path/to/new_run
```

数据包使用既有分类协议：冻结 FM 特征、明确标签和患者划分。每次训练读取一个完整 slide bag，不需要新的文件格式。FM 维度须整除专家头数；不支持的维度会报错，不会偷偷投影或补零。超出路由/注意力内存预算也会在分配前报错。

示例配置显式设置 16 GB 的保守激活估算预算，库内默认值为 2 GB；可按实际设备和完整 bag 长度调整 `model.max_dense_bytes`。这个参数控制提前检查，不会预分配对应大小的内存。

## 实现与训练

保留作者的查询映射、归一化、软路由、slot 汇聚、共享低秩专家、专家归一化和跨头拼接；同时支持显式 `keep_slots=false` 的原实现重组合路线。输出的 MIL 注意力对应专家 slots，不能直接冒充原始 patch 空间注意力；可单独取得原始 dispatch 权重。

训练实现依据论文 A2：AdamW、余弦调度、逆类别频率有放回抽样、单 bag 更新、输入特征 dropout。默认最多 20 轮、patience 5，至少训练 10 轮后才允许提前停止。调度器、AMP scaler、实际更新/溢出跳过计数及全局随机状态随 checkpoint 保存；AMP 溢出由 scaler 降低 scale 并跳过该次更新，FP32 非有限梯度直接报错。模型选择只读验证集；默认验证 log loss 是本项目明确选择，论文没有指定该指标。测试集仅由独立评价命令读取。

## 版本差异与范围

- 原 README/YAML 默认 30 专家、16 头、10 slots；论文主实验写 9、编码器消融写 6。默认遵循公开代码，slots 可显式配置。
- 原代码 auto-rank 四舍五入，论文公式向下取整；保留原代码计算，零秩/负秩会拒绝。
- ILRA YAML 的 `embed_dim=slot_dim=256` 被核心断言拒绝。命名修正版保持 ILRA 256 宽，将路由改为 128；不会声称这一修正是原论文默认值。
- 原非共享专家分支会广播出五维张量，随后归一化失败。明确拒绝 `share_weights=false`，保留作者推荐的共享分支；没有替换其数学实现。
- CLAM 的二分类 SmoothTop1 使用设备无关的等价损失，包括原阈值硬分支，与固定原始 `smooth-topk` 做数值/梯度比较。显式选 CE 时也保留实例约束。
- CPU 默认显式 FP32。论文的 CUDA 混合精度需要配置 `precision: amp`，无法使用时直接报错。
- 本次准入覆盖分类插件及八条完整分类训练路线；不把作者 ILRA 中尚未实现的 survival 分支算作可运行生存方法。

## 验证

`tests/test_mammoth.py` 检查原始模型头、损失/梯度/AdamW 更新、专家路由公式、SVM、完整抽样训练、调度和精确恢复，并检查原生维度与输入错误。

```bash
python scripts/validate_mammoth.py --device cpu --output /path/to/new_native_cpu
python scripts/validate_mammoth.py --device cuda --output /path/to/new_native_cuda
```

该脚本保留原生专家、头数、slot 数及模型宽度，对全部八条路线训练并重载预测；轮数是显式的软件检查预算。CPU 合成验证、GPU 验证、天然 WSI 和论文分数复现分别记录，不能互相替代。
