# CMTA 与 MOTCat

三个执行配置为 `cmta`、`motcat` 和 `motcat_micro`，对应两个方法家族。它们是患者级多模态生存方法，需要冻结 WSI 特征、随访时间/事件，以及同一患者的六组分子签名。仅有 BRCA/COAD 图像和形态分类标签时，不满足这些配置的输入条件。组学表、分组和患者 ID 格式见 [SURVIVAL.md](SURVIVAL.md)。

## 来源与安装

| 方法 | 原始来源 | 固定提交 |
|---|---|---|
| CMTA | [FT-ZHOU-ZZZ/CMTA](https://github.com/FT-ZHOU-ZZZ/CMTA) | `31340f6c74575668de35ee6c5d761467f11089e2` |
| MOTCat | [Innse/MOTCat](https://github.com/Innse/MOTCat) | `0da379f73d92c096122df139a9410d08b096e6c1` |

作者模型由 `milplus setup` 单独获取，不包含在 PULS 的 wheel/源码包中。两份 README 均说明非商业学术用途，外部代码不受本仓库许可证重新授权。

```bash
python -m pip install -e '.[transport]'
milplus setup --method cmta
milplus setup --method motcat

milplus demo --survival --omics --patients 20 --output /tmp/milplus_molecular_demo
milplus run --data /tmp/milplus_molecular_demo --method cmta \
  --config configs/cmta.yaml --epochs 2 --output runs/demo_cmta
milplus evaluate --run runs/demo_cmta

milplus run --data /path/to/survival_task --method motcat_micro \
  --config configs/motcat_micro.yaml --device cuda --output runs/motcat_micro
milplus evaluate --run runs/motcat_micro --device cuda
```

需要先为真实任务准备六组测量值；不会填造组学或自动把缺失值当成零。只使用唯一训练患者拟合分子标准化与生存时间分箱。

## CMTA 保留的机制

保留作者 small 模型的 WSI/分子嵌入、四个 Nystrom 编解码器、双向跨模态注意力、原始 PPEG 和融合头。PPEG 使用作者按序列排列并补齐的方格，不能解释为物理组织坐标。

训练目标为离散生存 NLL 加 `alignment_weight × (L1(P.detach(), P_hat) + L1(G.detach(), G_hat))`。默认 `alignment_weight=1.0` 对应作者 README 的启动示例（作者 argparse 默认值则为 0.5），并与删失权重 `alpha` 分开配置。仅对齐目标分支停止梯度；两个编码器仍通过生存目标和交叉模态路径学习。设为零会移除核心机制，因此正式 `cmta` 配置拒绝零值。

显式兼容修改：WSI 输入维度参数化；CLS token 和双线性融合常量跟随模型设备；返回原始 logits 以计算稳定的 log-sigmoid NLL。没有替换作者的 Nystrom 实现。

## MOTCat 保留的机制与依赖差异

保留作者六组分子 SNN、最优传输 co-attention、两条 Transformer、门控池化及 concat/bilinear 融合。默认使用非平衡 OT `pot-uot-l2`，可显式选择作者的平衡 `pot-sinkhorn-l2` 分支。代价和边缘权重在求解时 detach；流矩阵用于聚合 WSI value 特征，后者仍有梯度。不会宣称求解器本身可微。

安装固定 `POT==0.9.6.post1`，适配代码显式设置 `reg_type="entropy"`，保留作者 POT 0.9.3 的目标。新版默认值是 KL；直接沿用默认值会改变流矩阵。[POT 官方文档](https://pythonot.github.io/gen_modules/ot.unbalanced.html) 说明两种正则定义。测试用独立的旧式熵核固定点方程对照，并检查其与新版默认 KL 结果确有区别。

其他明确修改：输入宽度参数化；CUDA 常量跟随输入设备/dtype；singleton bag 的 squeeze 只去 batch 轴；全零代价归一化有数值保护，平衡分支全零边缘质量时采用均匀权重。这些退化输入处理不影响正常非零代价/质量分支。

## Micro-batch 配方

`motcat` 使用整包。`motcat_micro` 保留作者 `shuffle + array_split(N // size + 1)` 分组规则，包括 N 恰好为 size 整数倍时多分一组的边界行为。size 至少为 2，以避免作者公式在 singleton 输入下产生空组。

- 每个小包独立运行模型、计算 NLL；患者损失为小包损失的等权均值，不是先平均 logits 再计算损失。
- 逐小包反向传播，患者间再按 `accumulation_steps` 累积；最后不足整组时按实际患者数归一化。整个已加载特征 bag 仍在设备上，节省的是中间激活图，不保证完整输入常驻内存随小包上限缩小。
- 患者预测为小包生存曲线/风险的均值。导出的条件风险由平均生存曲线推导，保证 `cumprod(1-hazards)=survival`。
- 推理使用显式固定种子，并恢复调用前的 Python、NumPy 和 Torch 随机状态。与作者验证阶段每次随机划分不同，差异登记在 `survival_protocol.json`。

## 比较范围和证据

保留模型与核心目标，采用 PULS 的患者划分、训练集拟合、Adam、验证 C-index 选模、早停及梯度裁剪。CMTA 作者示例使用 SGD/不同预算，因此这些配置属于受控协议适配，不能称为原始完整超参复现。训练期间不评价测试集。

测试包括作者完整网络输出及目标梯度对照、对齐分支 detach 检查、两种融合/两种模态梯度、singleton/全零 OT、旧熵核对照、micro-batch 流式与整体目标梯度对照、末批累积、推理 RNG 隔离、患者预测、逐位精确续训和重载。当前验证是 CPU 合成数据；GPU、真实临床数据和论文指标复现状态分别见 [VALIDATION.md](VALIDATION.md)。
