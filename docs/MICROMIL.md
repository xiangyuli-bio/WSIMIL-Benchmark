# MicroMIL：完整冻结特征迁移配方

当前执行 ID 为 `micromil_frozen`，`MicroMIL` 是其别名。它属于计划中的 K 组：原始研究面向显微镜图像患者 bags；本实现明确迁移到 WSI 的冻结 FM 特征，不声称获得了原始 BreakHis / Seegene 数据或复现了论文分数。

## 固定来源和输入

[MICCAI 2025 官方页面](https://papers.miccai.org/miccai-2025/0566-Paper0707.html) 链接到[作者仓库](https://github.com/kimjongwoo-cell/MicroMIL)，固定提交 `66e637a891b7358198896c338b0e37828f3fc1f1`。实现读取其 `model.py` 的 DEC / ClusterAttentionModel Git 对象；`main.py`、`dataset.py` 和 requirements 已单独审查。作者仓库未提供根许可证，因此不在本项目分发第三方代码。

输入使用已有标准任务包：二分类标签、患者身份、固定 train/val/test 划分和每张切片 `[N,D]` 特征。允许 slide 或 patient 训练单位；patient 模式拼接该患者各切片的特征，不连接物理坐标。原方法不需要位置，因此不构造伪坐标。冻结特征必须记录真实 FM 权重和预处理来源。

源程序的 RGB 编码器被显式替换为特征边界上的 Identity。原编码器包在 `no_grad` 中；缓存特征还固定了推理预处理 / BN 模式，这与源码第一轮 CNN 训练模式不完全相同。此迁移差异保留在方法协议中，不将其当成原图流程复现。

## 保留的模型和完整训练

保留作者的 D×D 投影、Student-t 软聚类、可学习中心、每簇硬 Gumbel 代表选择、完整余弦相似度矩阵、逐行硬 Gumbel 有向边、附加自环、两层单头 GAT、dropout / LeakyReLU、图均值与分类头。原有重复自环不去重；单实例 bag 按源码复制为两个相同实例。GAT 的单头前缀维度按源码保留，无需替换成其他图网络。

训练逐 bag 执行 `BCE(softmax(logits)[1], label)` 和连续状态 Adam；默认 50 轮、lr=1e-4、weight_decay=1e-6、hidden=128、clusters=36、layers=2、dropout=0.5、patience=5。无额外聚类 KL 损失、伪标签、教师或预计算阶段。聚类中心通过代表选择的直通梯度更新；离散 `.nonzero` 边索引不提供梯度。单个重复实例无法提供有效的选择差异，不能据此宣称聚类学习成功。

保留 DataLoader / RandomSampler 的随机数行为，训练每轮重新进入 train 模式。使用验证 F1 选模、首次最优优先；patient 评价先融合患者概率。测试集不参与训练或选模。Adam、模型、计数、随机数与历史进入 checkpoint，支持精确中断恢复。

## 显式修复与差异

- 源 `Dataset_image` 返回三个值，训练循环仅解包两个；其数据路径与行号边界是本地硬编码。统一数据包替换这层，不沿用不明患者划分或吞掉文件读取异常。
- 原脚本 `model.eval()` 后未在下一轮恢复训练模式；此处每轮恢复。原脚本用 test F1 早停，没有保存最佳权重；此处验证选模并保存、重载 best / last。
- 显式声明图节点数，图设备跟随输入；略过未用的 `GCNConv` 顶层导入。可配置的隐藏宽度等参数确实传入模型。
- 推理依然使用硬 Gumbel 随机选择，但采用不依赖标签的固定逐 bag 随机流，并恢复训练 RNG。默认一次，允许登记多次概率平均，全部参数进入运行配置。
- 源码构造完整相似度矩阵；论文方法概述出现上三角描述。此路线遵循源码，不静默改为三角图。论文给出 lr=1e-3，源码默认 1e-4；此路线默认源码值，可显式调参。
- 使用 DGL 1.1.3 的真实 GAT / 消息传递；独立对照读取原依赖 DGL 0.8.1 的 GAT 类（提交 `1453bf2ceb23f413c23e3304dd94b40bf96b4cd7`，Apache-2.0），在同一后端执行完整作者训练循环。另有不调用 DGL 的邻居 softmax / 消息和 / 梯度公式对照。没有以 PyG 或简单池化替换模型。

## 使用

在有网络的 CPU 开发环境安装本项目和依赖、获取固定来源：

```bash
python -m pip install -e '.[dgl]'
milplus setup --method MicroMIL
milplus run --data /absolute/path/to/task_bundle --method MicroMIL \
  --output /absolute/path/to/new_run --config configs/micromil_frozen.yaml
milplus evaluate --run /absolute/path/to/new_run --subset test
```

`setup` 同时获取独立数值对照所需的原 DGL GAT 来源，采用稀疏检出，不下载其余工作树。GPU 环境需要与 Torch / CUDA 匹配并通过验证的 DGL CUDA 构建；PyPI 的当前 CPU 构建不能直接视为 GPU 可用。

## 验收边界

对应测试覆盖源完整前向 / 全部参数梯度、独立聚类公式、DGL 两版本的完整训练循环、真实图消息公式与重复边、slide / patient 两种逐位续训、预测标签独立性及重复性、非法输入和非有限梯度拒绝。全库及所有配置的最终结果以 `docs/validation/summary.json` 为准。

CPU 合成训练可证明软件路径及该固定版本的机制一致性。GPU、真实 WSI、多 FM 临床结果和原论文分数仍分别登记为 `not_run`。其余未接入方法继续保留，不由这个路线替代。
