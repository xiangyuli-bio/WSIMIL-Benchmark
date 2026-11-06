# MambaMIL：序列分支、执行后端与训练协议

来源：[作者仓库](https://github.com/isyangshu/MambaMIL)、[MICCAI 2024 论文](https://arxiv.org/abs/2403.06800)。固定提交 `9aea39efc13fdcd3703b5b54c28a64887b4acbd7`。本库读取固定 Git 对象中的作者定义；不重新分发该仓库源码。作者 README 指定非商业学术用途；内含 Mamba 组件另有 Apache-2.0 许可。

## 三个执行配置

| ID | 序列处理 | 任务 |
|---|---|---|
| `mambamil_srmamba` | 原顺序 + 按 rate 转置的重排序分支，两路求和 | 分类、离散生存 |
| `mambamil_bimamba` | 原顺序 + 反向分支，恢复顺序后求和 | 分类、离散生存 |
| `mambamil_unidirectional` | 单向 Mamba，对应作者消融 | 分类、离散生存 |

三配置属于一个 MambaMIL 家族，不能计为三篇独立论文。默认使用 SR-Mamba。保留 `D→512 + GELU`、两层预归一化状态空间残差模块、最终 LayerNorm、128 维注意力和分类头。状态维度 16、卷积宽度 4、扩张系数 2。重排序 rate=5 依据作者发布的训练脚本。输入 D 来自任务数据包，支持不同 FM。

保留原特征文件的 patch 顺序。SR 分支先补零、转置序列，再执行自己的卷积和扫描，最后逆转置、裁去补零；Bi 分支沿实例维度反转。不会按伪坐标重排，也不把重排序换成反向扫描。患者生存输入按 slide_id 排序后拼接其切片，每张切片内部顺序不变；这与作者按患者表顺序拼接的输入顺序约定有差异，需要在正式特征协议中冻结。拒绝全局 `max_instances` 截断。

## 显式后端

`model.backend: torch` 是默认可用实现。网络主体和三种分支直接加载作者定义，融合 inner kernel 替换为 PyTorch 因果深度卷积、投影和同一选择性状态空间递推。扫描分块使用梯度检查点，每块传递可微的完整状态，未截断跨块梯度。`scan_chunk` 控制重计算与保存开销，不改变序列定义。

`model.backend: cuda` 使用固定源码中的融合 CUDA 调用，要求可用 GPU，以及 `causal-conv1d==1.1.1`、`mamba-ssm==1.1.2`。后者应从本仓库锁定的作者 `mamba/` 子目录构建。运行记录与 checkpoint 保存包版本、Torch/CUDA 版本及两个二进制扩展的 SHA256。模型恢复拒绝后端身份变化，不会静默降级。版本字符串只检查 API 兼容性；构建来源仍需由环境构建记录证明。

作者 README 的历史环境是 Torch 2.0.1/CUDA 11.8；PULS 当前基础环境是 Torch 2.5.1 CPU。不能直接将历史 Torch 环境作为已验证的 PULS CUDA 环境。PyTorch 后端可供软件验证，Python 递推的长包吞吐需要实测；不能用其 CPU 速度推断作者融合 CUDA 的效率。

作者 SR-Mamba 的非融合 `use_fast_path=False` 路径使用反向序列，还含卷积分支覆盖问题。本适配始终保留其融合路径定义的计算图，再明确选择扫描内核，不调用该非融合路径。

## 完整训练配方

- 分类：每切片一包，CE，一包一次 Adam 更新。默认 dropout=0，非加权采样。验证选模默认 log_loss，统计单位遵循数据包的 `evaluation_unit`。
- 生存：每患者一包，四区间离散风险、作者 eps 截断 NLL，alpha=0；dropout=0.25；按照 `(时间区间, 删失状态)` 联合类别逆频率有放回采样；每 32 位患者累积更新。
- 两者：lr=2e-4、weight_decay=1e-5、200 epochs、patience=20，无默认梯度裁剪或学习率调度。分类提前停止最早在作者零基 epoch>50 后触发；生存为 epoch>40。生存验证选模为患者 C-index。
- 保留作者模型初始化的实际行为：递归 initializer 会再次初始化 Linear，并将标有 `_no_reinit` 的 dt 投影 bias 清零。作者 Adam 工厂也未按 `_no_weight_decay` 标签分组，本库保留所有参数的同一 weight decay。没有以通用 Mamba 的初始化规则替换它。
- 完整库使用自己的确定性 shuffle、文件加载和随机状态记录，采样分布与作者一致；不能保证与作者 DataLoader 在同一 seed 下具有相同采样序列。

明确的协议修复：生存分箱仅拟合唯一训练患者的观察事件时间，避免作者在划分前拟合整个队列带来的信息重叠；最后不足 32 位患者的窗口按实际数量平均并更新，避免原循环把残余梯度留到下一 epoch。未改变完整窗口的原始目标。默认不使用作者可选的额外 L1。

## 使用

先在联网 CPU 开发机准备固定源码；任务环境可通过 `MILPLUS_SOURCE_ROOT` 指向共享的源码目录。

```bash
milplus setup --method mambamil
milplus check --data /path/to/task --method mambamil --deep
milplus run --data /path/to/task --method mambamil_srmamba \
  --config configs/mambamil_classification.yaml --output runs/mambamil_cls
milplus evaluate --run runs/mambamil_cls
```

生存任务使用 `configs/mambamil_survival.yaml`。无需额外组学。分类目前按切片训练，不接受未经登记的患者分类拼包适配。原始 SVS 文件夹仍需要先用 TRIDENT 提取特征并建立患者、标签、划分和特征索引。

未来在构建并核验 CUDA 环境后，运行专用检查：

```bash
python scripts/validate_mamba_cuda.py --output /path/to/new_cuda_check.json
```

该命令比较三分支、不同包长、作者默认 512 隐藏宽度下融合内核与 PyTorch 参考的输出和全部参数梯度。无 GPU 时写入 `not_run` 并返回非零；失败时保存原因，不当作通过。当前没有执行 CUDA 训练或此融合内核对照。

## 验证边界

测试分层进行：独立读取作者 `selective_scan_ref` / `mamba_inner_ref`，对照扫描、完整模型输出和梯度；再在相同数值内核下执行未经改写的作者分类/生存训练循环，对照连续更新、参数和 Adam 状态。后者专门验证 trainer，不宣称独立验证 CUDA 内核。

不同卷积/扫描求和顺序会产生浮点误差，softmax 公共偏置的理论零梯度在 float32 下尤其敏感；Adam 可放大这些近零差异。因此跨数值内核使用明确的输出/梯度误差界，完整 trainer 更新在同一内核下要求逐位一致。另验证三种分支的原生 1024 输入/512 隐藏宽度、跨块梯度、加权采样、训练专用分箱、尾批修复，以及分类/生存六条路径的精确续训。

逐项结果以 [验证目录](validation/) 中当前软件身份对应的实际通过记录为准。合成 CPU 检查不证明真实 WSI 指标，也不代表原论文数值复现。
