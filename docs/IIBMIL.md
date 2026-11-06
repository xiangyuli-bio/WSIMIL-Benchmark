# IIB-MIL：完整组件与明确命名的训练重建

固定来源为 [作者代码](https://github.com/TencentAILabHealthcare/IIB-MIL/tree/a354ec6ea2b9ea2159e6784d54a6afb355c04b50)。
源代码没有 LICENSE，不分发第三方源码；运行时从固定私有缓存加载。
公开代码提供完整模型、数据初始化、置信度损失、原型更新和推理；没有完整训练循环。
[论文出版社页面](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_54)
已核对元数据，未取得可读全文，不声称读取了全文或恢复了未知训练超参数。

三个运行配置保留原始 1280 维输入、500 行 bag 和所有原始层：

| 配置 | 隐藏维度 | Encoder 深度 | Batch | LR | 原始类别数 |
|---|---:|---:|---:|---:|---:|
| `iibmil_luad_gm_complete_reference` | 32 | 8 | 16 | 2e-4 | 2 |
| `iibmil_tcga_nsclc_complete_reference` | 256 | 3 | 16 | 1e-5 | 2 |
| `iibmil_tcga_rcc_complete_reference` | 256 | 4 | 4 | 1e-5 | 3 |

共同原生预算为 300 轮，AdamW、weight decay 1e-5、clip norm 0.01，
StepLR 每 100 轮乘 0.1。Encoder 两头、FFN 2048，Decoder 两层四头、五个查询。
配置名描述所采用的原始网络与优化配置，不改变实际输入队列。
不同 FM 的输入维度与其他类别数是明确记录的接口扩展，内部网络维度不缩减。

## 保留的原始定义

- TRAIN 银行包含每张 TRAIN 切片的全部实际实例；验证和测试实例不进入银行。
  `build()` 的初值为弱标签类 `1-1e-6`，其余类别共享 `1e-6`。
- TRAIN 有 50% 概率随机排列全部实例，再取前 500 行；预测只取前 500 行。
  不足 500 行时用零填充，并重复第一个实例 ID。重复的填充 ID 不更新银行。
- 原型采用类别匹配切片中有效实例的 top10%，逐个执行未经归一化的 EMA。
  动量为 `1-(1-0.99)*batch_size/class_matched_bags`，缺失类别跳过。
- 实例损失使用原始 PartialLoss。置信度 EMA 调度保留原代码的 200 分母，
  即使训练预算是 300 轮也不改成 300、不裁剪其值。
- 原始注意力未使用 padding mask，此处保持该行为并披露。
  原始定义但未进入 forward 的参数保留，允许这些参数没有梯度。

## 需要明确披露的训练重建决定

公开缺少 trainer，以下决定不能标记为已经恢复的作者配方：

- 原始 bag logits 的 CE 与有效实例 PartialLoss 各权重 1，两项均保持启用。
- 前 10 轮不更新置信度，原型每个 minibatch 更新。
- 顺序为现有置信度监督 → backward/clip/AdamW → detached 更新前特征的原型 EMA
  → 用本次 forward 的旧原型分数更新置信度；不在更新后重算分数。
- 实例损失与银行更新仅使用真实有效行；原始注意力的填充行为保持不变。
- 使用冻结 FM 的单一特征视图，DataLoader workers=0，保留最后不足 batch 的样本。
- 完成全部请求轮次，不提前停止；第 10 个零起始 epoch 起按验证 macro AUROC 选模型。

checkpoint 包含实际银行、初值、原型、优化器与调度器、置信度 EMA、每轮实例抽样、
完整 Python/NumPy/Torch/CUDA RNG。银行身份绑定原始实例、切片、弱标签和实际特征内容哈希。

## 使用

```bash
python -m milplus demo-iibmil --output /your/own/path/iib_demo --classes 2
python -m milplus run --data /your/own/path/iib_demo --method IIB-MIL \
  --device cuda:0 --output /your/own/path/iib_run
python -m milplus evaluate --run /your/own/path/iib_run --device cuda:0
```

真实冻结特征数据包使用相同入口，标签和患者划分由 `task.yaml` 绑定。
可选 H5 `patch_ids` 或特征索引中的 `patch_ids_path`；没有原始字符串 ID 时，
使用绑定实际内容的原始特征行身份，记录其来源，不构造伪坐标。
二分类别名选 NSCLC 配置，多分类别名选 RCC 配置；LUAD-GM 必须显式指定。
软件生命周期检查使用完整原生网络与 500 行输入，12 轮跨越预热；
完整预算检查仍需实际执行所有 300 轮。两者均不等于真实 WSI 或论文分数复现。
