# MSMMIL：三路扫描与完整训练配方

原始来源：[作者仓库](https://github.com/cool-breeze-and-rain/MSMMIL)，固定提交 `80382115f5e548bc39248fd726bae3db2ead8b7f`；[论文入口](https://www.sciencedirect.com/science/article/abs/pii/S0950705125009177)。根目录未提供许可证，本库不分发其模型源码，运行时读取用户获取的固定 Git 对象。其内置 Mamba 和优化器保留各自上游署名与许可。

`msmmil` 对应作者配置中的 `MKMIL(type='AFWMamba')`，支持切片级分类、不同 FM 的特征维度，以及患者级概率汇总评价。尚未登记其生存或患者拼包训练变体。

## 网络与输入顺序

默认网络是 `D→512→256`，两个 256 维残差 AFWMamba 模块，再经 `256→512`、LayerNorm、128 维注意力和分类头。ReLU、dropout=0.5、rate=10、状态宽度16、卷积宽度4、扩张2均来自作者配置。

每个 AFWMamba 包含三条具有独立卷积、状态和投影参数的扫描路径：棋盘子序列、交替行子序列及原始顺序。作者同一个 `BiAttn` 模块在三路结果上分别调用，然后求和并投影。严格保留补零、转置、反转与恢复顺序，不以普通 Mamba、BiMamba 或 SRMamba 替代。

输入保留特征文件中的 patch 顺序。rate 必须是正偶数；序列补到偶数高度的伪网格。这种布局不使用真实坐标，不能称为保留了 WSI 的物理二维结构。拒绝全局 `max_instances` 截断，也拒绝未经登记的多切片患者拼包。TRIDENT 提取结果应固定 patch 顺序和提取协议。

`model.hidden` 控制中间宽度，默认256；两侧投影为其两倍。缩小宽度用于软件测试时会记入配置，不能与作者原生宽度结果混报。保留作者递归初始化的实际行为，包括将标有 `_no_reinit` 的 dt bias 再次清零。

## 显式执行后端

默认 `backend: torch` 在作者完整三路计算图中使用因果卷积和精确选择性扫描递推。状态跨块保持可微，梯度检查点用于减少中间存储，没有截断序列或跨块梯度。完整网络已设计为与 MSMMIL 仓库独立的参考卷积/扫描公式对照。

`backend: cuda` 读取 **MSMMIL 自己锁定源码**中的融合算子，要求 GPU、`causal-conv1d==1.1.1` 和 `mamba-ssm==1.1.2`。包版本不能独自证明构建来源，因此还记录二进制哈希及 Torch/CUDA 信息。缺依赖时直接失败，不自动改用其他模型或内核。Torch 后端也可以在 CUDA 张量上执行，但长包吞吐需实际测量。

当前验证环境只有 CPU，未执行 CUDA 融合算子对照或真实 GPU 训练。提供的 `scripts/validate_msmmil_cuda.py` 可在配套环境中检查融合与参考实现的输出及全部参数梯度。

## 训练与恢复

- 每张切片一个 bag，交叉熵；两个 bag 累积一次更新。最后不足两个 bag 的窗口仍除以配置的累积数，与作者 Lightning 配方一致。
- 使用作者 `MyOptimizer/radam.py`，保留其整流边界和解耦 weight decay。bias 与一维参数不施加衰减，其余参数默认 `1e-5`；不能直接用新版 `torch.optim.RAdam` 视为同一数值实现。
- Lookahead `k=6, alpha=0.5`，第一次慢权重在第6次更新时才创建。作者实现使用 Python 对象 ID 存储慢权重，本库改用稳定参数名保存，保证新建模型后的恢复。
- 默认200 epochs、lr=`2e-4`、patience=40，以固定验证集的 log loss 选择权重，无默认梯度裁剪或学习率调度。
- FP32 对应发布配置的 `precision: 32`；历史 Lightning/APEX 运行环境未原样复刻，不能声称 AMP 的数值行为已被验证。
- checkpoint 保存完整网络、RAdam 动量、Lookahead 慢权重、步数、随机状态和后端身份。恢复时拒绝数据、配置、源码或后端身份变化。

## 使用

```bash
milplus setup --method MSMMIL
milplus check --data /path/to/task --method MSMMIL --deep
milplus run --data /path/to/task --method MSMMIL \
  --config configs/msmmil_classification.yaml --output runs/msmmil
milplus evaluate --run runs/msmmil
```

CPU 合成数据上的输出、梯度、连续参数更新和断点恢复检查见 [验证记录](validation/)。它们证明软件路径及所列机制的对应关系；真实 BRCA/COAD 结果和原论文指标复现仍需实际数据与计算实验。
