# MFE-MIL：共享适配器、掩码重建与 MIL 联合训练

来源：[论文](https://arxiv.org/abs/2610.10225)、[作者仓库](https://github.com/AtlasAnalyticsLab/MFE-MIL)，固定提交 `813c8e8abf46c05afc2402c06a6c7ac77284fff2`。作者使用 CC BY-NC-SA 4.0，本库通过外部固定 Git 对象加载其定义，不在安装包内重新分发源码。其 decoder 使用锁定的 timm **0.9.8**，提交 `9afe0bb78e91ea4183cd1bf076f160ec50e9078b`，Apache-2.0。

## 执行配置

| ID | 作者头部 | 已接入任务 |
|---|---|---|
| `mfemil_mean` | MeanMIL | 分类、生存 |
| `mfemil_max` | MaxMIL | 分类、生存 |
| `mfemil_abmil` | DAttention | 分类、生存 |
| `mfemil_clam_sb` | MFE 发布版 CLAM-SB | 分类、生存 |
| `mfemil_clam_mb` | MFE 发布版 CLAM-MB | 分类、生存 |
| `mfemil_transmil` | MFE 发布版 TransMIL | 生存 |

`MFE-MIL` 家族名默认解析为 `mfemil_abmil`。六个配置属于一个训练策略家族。分类五配置和生存六配置对应发布的两个 sweep；不能计作六篇独立论文。

## 网络和目标

保留作者两层 `Linear(D,D)→LayerNorm→ReLU→Dropout(0.25)` 共享适配器，以及每个头的原始网络。MIL 分支对原始冻结特征运行适配器，按 sub_batch=8 分段并使用梯度检查点。分段会影响 dropout 随机流，不能无记录地改成单次全包计算。

重建分支先将原始 patch 特征 L2 归一化，按序列顺序填充到正方形，再放置不重叠的 3×3/5×5 窗口并随机补齐到 mask_ratio=0.75。保留作者循环上限和随机数消耗。共享适配器处理掩码后的特征，默认四层、512 维、16 头 Transformer decoder 将 mask token 与二维 sin/cos 位置编码结合，预测原始特征维度。MSE 只计算真实 patch 中被掩码的部分，填充位置不进入目标。

联合目标为 `0.7 × task_loss + 0.3 × reconstruction_MSE`，task_loss 是分类 CE 或作者 eps 截断离散生存 NLL。`alpha` 必须大于0；无重建的纯 MIL 对照应使用相应独立配置，避免把关闭核心训练机制的实验继续当作 MFE。

输入采用冻结 FM 特征。作者 `stop_grad=False` 保留；此时原始特征不是可训练编码器的输出。生存推理输出 `-sum(S)` 风险。推理只运行共享适配器和 MIL 头，不生成掩码，不调用 decoder。当前运行对象和 checkpoint 仍保留 decoder 参数；尚未提供删除这些参数的精简推理导出，显存成本报告需明确这一条件。

作者 MFE 训练循环不启用 CLAM 的实例聚类损失，因此这里保留其 **MFE-CLAM bag-only 配方**，不能作为完整独立 CLAM 的替代实现。作者声明但未使用的 `feat_scale`、`norm` 和 CLAM instance classifiers 同样保留并登记为不活跃参数。

## 顺序、物理坐标与修复

分类保留特征文件中的 patch 顺序，不要求额外坐标。生存必须提供真实坐标；原训练代码按 `y*1_000_000+x` 排序，本库按每张切片的 `(y,x)` 稳定字典序排列，再按 slide_id 拼接患者切片，避免大坐标键冲突和混合不同切片坐标系。单切片、原键无碰撞时与源排序一致。

掩码网格始终是填充后的序列伪网格。按坐标排序并不能保证稀疏组织的物理邻接关系恢复；不能将其报告为真实二维空间重建。作者论文和仓库也将该顺序描述为弱隐式先验。

另有以下明确修复：

- ABMIL/CLAM 的无参数 `squeeze()` 改为仅移除批维，支持单 patch，避免丢失实例维度。
- CLAM-MB 的参数为空的 CE 模块不强制 `.cuda()`，允许 CPU 软件验证。
- 非有限输入、目标或梯度直接失败，取消源码中的 NaN→0 损失替换。
- 生存区间只用唯一训练患者的观察事件时间拟合，验证与测试不参与分箱。
- 生存最后不足累积数的窗口按实际患者数平均并更新；完整窗口目标保持不变。源码固定除以 gc，并在末尾再次 flush。

## 配方与数值环境

| 任务 | 默认优化与模型选择 |
|---|---|
| 分类 | Adam，lr=1e-4，无 weight decay；200 epochs，patience=10；验证 macro AUROC；每切片一次更新 |
| 生存 | Adam，lr=2e-4，weight decay=1e-5；20 epochs，patience=5；验证患者 C-index；gc=32 |

默认不裁剪梯度、不加调度器、不做加权采样。生存可显式启用作者的区间/删失联合类别逆频率采样。分类训练单位为单切片，患者级评价可使用任务数据包预定义的概率汇总。评价单位沿用冻结的数据包协议；患者汇总结果与作者逐 WSI 评价不能混为同一实验条件。

PULS 默认 `precision: fp32`，用于当前 CPU 可验证路径；它是相对源码 CUDA AMP 的显式数值条件变化。`precision: amp` 要求实际 CUDA 设备，保存 GradScaler 状态，不自动降级。当前没有 GPU 训练验证。timm decoder 的 `attention_backend` 显式为 `sdpa` 或 `math`，后者用于数值对照；网络定义来自原始 timm，不通过 torchvision 导入其他视觉模型。

`decoder_dim/depth/heads` 可供软件验证缩小，默认仍为512/4/16。所有设置、特征身份、源码锁、随机状态、Adam 和 scaler 状态进入运行记录与 checkpoint。拒绝全局 `max_instances` 截断。作者测试集每 epoch 评价的行为未进入选模：本库仅在固定验证集选权重，测试由独立 evaluate 执行。

## 使用

```bash
milplus setup --method MFE-MIL
milplus check --data /path/to/task --method MFE-MIL --deep
milplus run --data /path/to/task --method mfemil_abmil \
  --config configs/mfemil_classification.yaml --output runs/mfe_cls
milplus evaluate --run runs/mfe_cls
```

生存改用 `configs/mfemil_survival.yaml`，数据包登记患者、事件、时间、固定划分和坐标。离线 Worker 需挂载提前获取的 MFE/timm 两个源码目录，并设置 `MILPLUS_SOURCE_ROOT`。

当前验证使用合成特征，逐项记录见 [验证目录](validation/)。来源输出、联合损失和梯度、完整作者 trainer、断点恢复及独立安装包分别检查。真实 WSI 指标、原论文分数以及 AMP/GPU 吞吐仍需实际实验。
