# AGDR-MIL：完整来源配方及梯度边界

固定作者 [shuiguoc/AGDR-MIL](https://github.com/shuiguoc/AGDR-MIL) 提交 `8e830683c0a695e1e1983f01d35dc85471bf327c`。运行时从已校验的外部 Git 对象加载实际模型与激活函数，保留 512 维降维、硬分级、以高分实例引导的余弦重分配、低分子集随机屏蔽、高分/完整包双分支和双损失。作者代码和数据不随本库分发；固定树中未发现根许可证。

| 执行配置 | 来源参数 |
|---|---|
| `agdrmil_yaml_source`，`AGDR-MIL` 默认 | 实际 `config.yaml`：高/低阈值 0.95/0.5、低分屏蔽率 0.2、高分损失权重 0.3、高分 max 池化 |
| `agdrmil_tcga_source` | README TCGA 列：阈值 0.9/0.4、屏蔽率 0.3、高分损失权重 0.4、高分 max 池化 |

两者都使用原完整分支的注意力池化、ReLU、dropout 0.25 及原 Sigmoid 分类头，CE 直接接收这两个 Sigmoid 输出，再按相应权重组合。预测按作者验证代码对完整包 Sigmoid 输出再做 softmax。没有静默删除 Sigmoid、补造分级器损失，或将完整模型替换为 ABMIL。

README 的 C16 推荐高分 attention 池化，而实际 YAML 写 max。默认配置严格按 YAML；若明确研究 README 条件，可在配置中设 `model.high_pool: attention`。此外支持原模型的 mean/gate 高分池化，均须作为明确实验变体登记；不增加独立论文计数。

## 运行

只需已准备好的标准分类数据包与冻结 patch 特征，不要求额外坐标、组学或文本。输入维度来自所选 FM；原生隐藏维度始终为 512。坐标若已提供，会随实例结果保留。方法使用单张 WSI 为训练单位；同一患者的多张切片仍须在同一划分，患者评价按任务协议融合概率。

```bash
milplus setup --method AGDR-MIL
milplus check --data /path/to/task --method AGDR-MIL --deep
milplus run --data /path/to/task --method AGDR-MIL \
  --config configs/agdrmil.yaml --output runs/agdr_yaml
milplus evaluate --run runs/agdr_yaml

# BRCA/COAD 等 TCGA 分类任务可明确选择 README TCGA 条件。
milplus run --data /path/to/tcga_task --method agdrmil_tcga_source \
  --config configs/agdrmil.yaml --output runs/agdr_tcga
```

任务标签、患者映射、固定划分和 FM 登记须已完成；直接提供 SVS 文件夹不包含这些信息，也不能代替特征提取。暂不将原分类器当作生存模型或有序分级模型。

## 完整训练与选择

保留公开 YAML 的 200 轮预算、Adam `lr=2e-4`、`weight_decay=1e-5`、单包批次，无调度器和梯度裁剪，完整 bag 不截断。每轮通过作者相同的 PyTorch DataLoader 机制随机打乱；独立训练 generator 以源码常数 2000 初始化，状态随 checkpoint 保存。模型和全局随机状态使用实际运行登记的 seed；可在统一协议下明确更换 seed，不声称复用了作者原始实验的初始化。

验证与测试的原 DataLoader 虽不 shuffle，仍会消耗一次全局 CPU base seed，后续低分屏蔽继续消耗随机状态。本实现保留该行为。验证使用加权双分支逐 WSI CE，最后并列最优也保存并重置计数器；原生 patience 为 20，且零起算 epoch>20 后才可触发早停。测试推理在训练和选模期间封闭，仅在显式 evaluate 时进行。

重复 evaluate 从运行 seed 和固定遍历顺序重放原随机屏蔽，不将其包装成无随机性的原模型。中断恢复保留模型、双分支参数、Adam、全局随机状态和独立采样 generator。

源验证定义的逐 WSI accuracy、二分类阳性 AUC/多分类 weighted OVR AUC，分别保存为 `agdr_native_slide_accuracy`、`agdr_native_slide_auc`；统一报告仍按任务设置提供 slide/patient 指标，二者不可混称。AUC 未定义时记为 null，不沿用作者多分类异常路径中填 0 的做法；选模依据为有效的双分支 CE。预测还保留 `agdr_high_sigmoid_*`、`agdr_full_sigmoid_*` 两组原始分支输出。

## 必须披露的来源行为

分级注意力网络只通过布尔索引影响分组，原损失没有到该网络参数的梯度。当前接入保留其初始化后不更新的行为，并明确区分这个分级器与完整包分支的可学习聚合注意力。原模型还定义了一些未参与当前池化路线的模块，保留这些参数及其缺失梯度，不人为制造更新。

原始 min-max 分级在常数评分上没有定义，会造成空高分子集；至少两行有限特征且评分非退化才可运行。所有特征行完全相同、评分恒定或评分非有限时明确报错，不生成伪分组。相同特征经浮点矩阵核后可能产生微小的评分差异，因此也在输入端识别完全相同的行，避免将舍入噪声放大成伪分级。低分屏蔽在 eval 中仍执行，实例导出记录本次实际保留的行。

实例 H5 将分级值、类别为低/中/高的 `partition`、实际聚合行顺序 `retained_order` 和完整包注意力映射回原特征行；被丢弃行的注意力为 0。可用坐标同步保留。这不是因果归因，也不把随机分级值称为训练后学习的注意力。

## 验证状态

0.50 主库全库 2200 项及 202 个执行配置全新通过。该方法 50 项专项含实际作者训练循环 180 次逐步更新对照、4 组原始验证 DataLoader/屏蔽随机状态对照、四种高分池化与两种 FM 宽度的完整梯度、二/三分类、精确续训及源回调选模边界。独立 wheel 两条原生 1024/512 路线按完整 200 轮最大预算和作者早停分别完成 33/27 轮、1080 次更新，并真实重载、重复预测与导出。机器可读报告见 `docs/validation/agdrmil.json`、`summary.json` 和 `wheel.json`。CPU 合成检查不代表真实 WSI、GPU 或论文指标复现。
