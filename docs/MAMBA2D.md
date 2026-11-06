# 官方 2DMambaMIL

来源为 [AtlasAnalyticsLab/2DMamba](https://github.com/AtlasAnalyticsLab/2DMamba)，固定提交
`9fe12aaa866ddbb83b66acc4ef400abf8613ce6a`。原始头、全部 Mamba/RMSNorm/卷积/门控模块、
二维扫描、AdamW 和生存损失均从经过校验的外部 Git 对象加载。
没有采用 MIL_BASELINE 的简化实现，也没有用一维 Mamba 或伪方格替代真实二维输入。
固定源码根目录没有 LICENSE；作者源码与编译产物不随本项目分发。

## 三个配置的含义

| 配置 | 扫描和梯度 |
| --- | --- |
| `mamba2dmil_source` | 保留发布版 PyTorch 前向和自定义反向传播，供源码行为对照 |
| `mamba2dmil_autograd` | 前向逐位保留发布版 PyTorch 算法，仅以自动微分修正反向传播 |
| `mamba2dmil_paper` | 两处垂直扫描修正及自动微分，对照原论文 Eq.5/6 的横向后纵向递推 |

方法名 `2DMambaMIL` / `mamba2dmil` 默认选择第三条显式修正路线。
每个配置都支持完整分类和离散生存训练。一条方法记录对应三个执行配置；
`paper` 标识公式对照的实现，并不表示已经复现论文指标。

发布版 PyTorch 扫描存在两个经数值核验的问题：

- 自定义反向传播在多个小网格上不符合自身前向的中心差分梯度。
- 垂直 down-sweep 两处读取已经被水平扫描修改的 `A`，而非该阶段的 `A_raw`。

修正路线保留其并行扫描控制流，逐项记录修改；按独立编码的
[论文公式](https://arxiv.org/html/2412.00678v3#S3.SS3)检查横向、纵向两级扫描及导数。
这些观察针对固定发布代码的 PyTorch 路径，不推断原论文 CUDA 实验使用了同样的反向传播。

## 真实坐标与声明的物理画布

输入为标准冻结特征任务包，H5/PT 中提供行对齐坐标；也支持单独的 `coords_path`。
对应编码器需要声明以下信息，数值必须来自真实切块协议和画布范围：

```yaml
encoders:
  uni:
    dimension: 1024
    index: features.csv
    # 保留实际权重版本及预处理信息
    spatial:
      unit: level0_pixels
      origin: top_left
      patch_size: 512
      provenance: "实际 level-0 坐标、网格间距和切块流程版本"
    mamba2d:
      max_coordinates: [181272, 88334]
      provenance: "实际统一画布的最大 x/y 坐标及其来源；示例数值须替换"
```

`max_coordinates` 是包含端点的物理坐标上界，顺序为 `[x,y]`。
`spatial.patch_size` 在本方法中是该坐标单位下的网格间距；不要混淆网络输入像素和 level-0 间距。
默认要求 top-left 坐标。除 level-0 pixels 外，框架的 micrometers/tile_grid 声明也可使用，
但同一画布、坐标和间距必须处于同一单位；tile_grid 的间距为 1。
坐标以原始精度读取，再按作者的 floor 除法映射至整数网格。
重复单元、负坐标、越界、非整数 level-0 像素、缺失或错位坐标直接拒绝。

画布包含非组织区域，使用作者的可学习 pad token 或显式零 token。
源模型把这些单元同时纳入扫描和最终注意力，不能在实现时悄悄屏蔽它们。
网格大小及扫描临时张量的规划估计写入协议。
`workspace_budget_bytes` 控制估计值的上限，默认 2 GiB；它不保证实际分配或自动微分峰值一定低于该值。
大画布需按实际硬件显式调整预算，或另行构建并验证原生 CUDA 内核；不通过截断 bag 改变算法。

分类以单张切片训练，可按既定任务协议进行患者级评价；选模仍保留逐切片 AUROC。
原生生存配置要求任务包为每位患者登记一张代表 WSI，并事先记录选择规则。
多张切片的独立坐标系直接拼接会覆盖单元，因此不支持这种隐式拼接。

## 完整训练配方与恢复

```bash
milplus setup --method 2DMambaMIL
milplus check --data /path/to/task_bundle --method 2DMambaMIL --deep
milplus run --data /path/to/task_bundle --method 2DMambaMIL \
  --config configs/mamba2d.yaml --output /path/to/new_run
```

默认 1024 维输入从 FM 登记读取，模型隐藏宽度 128、状态宽度 16、一层、Dropout .25。
作者 `apply(initialize_weights)` 的 Xavier 顺序原样保留，包括其对 `dt_proj` 初始化的覆盖。
原生多层 PyTorch 网络同样接入。未暴露已发布但不能正确处理模块对象的可选线性 PE 分支；
不支持的参数报错，不会默默忽略或关闭。

分类：整袋 CE，顺序 batch size 1，AdamW lr=1e-4、weight decay=1e-5，20 epoch，
逐 epoch cosine 至 0，不使用梯度裁剪或实例截取。按 VAL 切片 macro-AUROC 保存，
保留作者 `>=` 的并列时选择较晚模型规则。

生存：仅用训练患者的观察事件时间拟合分箱，原生 eps-clamped NLL、alpha=0，
随机无放回患者采样，AdamW 相同，无调度器。
发布版将每例损失除以 32，但每例仍执行 step/zero；这里明确保留该行为，
不将它描述为 32 例梯度累积。日志同时记录原始和实际优化损失。
按 VAL 患者 C-index 严格改善选模。首次合法零分仍保存，避免没有可加载的 checkpoint。
TEST 不参与选模。

模型、训练分箱、AdamW、调度器、更新/轮次计数、随机状态和网格/后端合同全部保存。
严格恢复拒绝输入、编码器、画布、扫描实现或配置变化。
作者后端诊断保留在 stderr，使 CLI 的评价 stdout 保持可解析 JSON。

H5 实例导出包含原始坐标、整数网格位置、完整画布注意力及组织 mask。
真实 patch 权重保持其在整个画布上的原始质量，不重新归一化为和为 1；
pad 区域的权重质量单列。这些是注意力权重，不是因果解释。

## 显式 CUDA 后端

默认 `backend: torch` 使用已声明的 PyTorch 扫描，张量设备由 `--device` 决定。
`backend: cuda` 仅属于 `mamba2dmil_source`，运行作者编译的 `pscan.so`，不施加 CPU 修正，
不自动退回 PyTorch。当前仅开放源码默认的一层 CUDA 网络，
其多层路径会重复重建已经展平的稀疏网格，未当作已验证功能开放。

按固定作者仓库的 `cuda_kernel/build.sh` 在匹配环境编译，向模型配置提供 `cuda_binary` 的绝对路径。
相邻的 `pscan.so.json`（随实际二进制文件名）须登记：

```json
{
  "source_commit": "9fe12aaa866ddbb83b66acc4ef400abf8613ce6a",
  "sha256": "<实际编译产物 SHA256>",
  "torch": "<实际 torch.__version__>",
  "cuda": "<实际 torch.version.cuda>"
}
```

二进制及登记文件的哈希写入运行合同，并核对 source commit 和 Torch/CUDA ABI。
GPU 不可用或产物不匹配时直接失败。
这提供明确的执行接口；GPU/CUDA 内核验证状态以当前验证报告为准，不能用 CPU 测试替代。

## 验证

```bash
python scripts/validate_mamba2d.py --device cpu --output /path/to/new_native_cpu
python scripts/validate_mamba2d.py --device cuda --output /path/to/new_gpu_torch
python scripts/validate_mamba2d.py --device cuda --backend cuda \
  --native-cuda-binary /path/to/pscan.so --output /path/to/new_native_cuda
```

测试分开核验原始算法、独立公式/数值导数、完整网络全部梯度，以及作者完整外层训练入口的更新。
后者对修正路线使用已通过独立数学核验的相同并行内核，逐位核对优化器与训练调度，
避免把串行与并行浮点舍入经 Adam 放大的差异误判为外层流程错误。
另检验原生宽度、多层、精确恢复、并列选模、重复预测、真实 pad 注意力质量、
训练专用分箱、输入错配拒绝和 CLI 入口。自然 WSI、原生 CUDA 和论文得分复现单独登记。
