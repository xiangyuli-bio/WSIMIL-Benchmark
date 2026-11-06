# DSPA-MIL / DSPA-U-MIL：机制候选与验收边界

官方来源：[AlexNmSED/DSPA-MIL](https://github.com/AlexNmSED/DSPA-MIL)，固定提交 `75d95d410374771a3eba4bd84fdfadc06acf0eed`，MIT。该固定版本 README 标注 incomplete；本仓库现已登记两套完整2025参考实现的公共代码profile，状态为 `adapted_under_verification`，不是原作者完整训练器或DSPA-U复现。统一入口见[使用说明](LAST_THREE_METHODS.md)。

已对原生 1024 维、12 层、8 头、3 个模式 token、6 个 Grade Group 的完整 SAT 网络执行前向、反向与 AdamW 更新，参数量为 **151,141,388**。这项组件可微性检查不是完整 DSPA 配方。

实际执行也确认了公开训练接口的缺项：模型返回字典，而训练循环解包为四个位置值，会在损失计算前失败；流程引用的 `WSI_Dataset_GS` 未在数据类文件中发布。注意力在 softmax 前把跨 token 的 logit 乘零，相关概率仍非零；没有静默修正这些源码行为。

当前候选在作者原生 PAT 上增加了[原论文](https://papers.miccai.org/miccai-2025/paper/0636_paper.pdf)可确定的概念相似性聚合、Teacher-Gate、伪文本生成器、Student-Gate、知识对齐和三个辅助模式分类目标。它仍需原始资产、患者数据、完整训练、独立包和发布验收。

证据：`tests/test_dspamil_source.py`；开发运行 `dspamil_v100/components_r1.xml`，3 项组件及 1 项完整目录保留检查通过。没有真实患者结果或论文指标复现。

新增代码为 `dspamil_equations.py` 和 `dspamil_reference.py`。原生 PAT 保留完整 12 层、1024 维、8 头和三个 token，共151,120,896个参数；作者 SAT 中的基线融合头由双路参考头替换。参考头采用显式1024宽度时，完整参考图为162,690,072个参数。H200上的257实例组件检查确认 PAT 与作者代码的模式输出、梯度、AdamW各状态及随机状态逐位一致；两种掩码条件的完整参考组件都完成了前向、反向、一次更新及不调用Teacher的推理。

这些检查采用合成特征和合成概念向量。它们没有完成50轮训练、实际概念生成、患者数据绑定、数值结果复现或完整方法准入。代码中的 `full_method_validation` 和 `formal_admission` 继续为假。

`EquationChoices` 要求声明概念权重沿实例或概念归一化、论文公式与文字中不同的gating-KL方向、teacher detach和文本L2的含义。辅助目标需要三列实际二元模式标签；GG4/GG5无法唯一确定它们。六分类logits到连续等级采用显式概率期望参考接口。

`HeadReference` 要求提供PTG、Teacher和Student的隐藏宽度、激活、dropout及来源说明。每个模式的PAT1024与概念1024特征拼接成2048，三个模式进入PTG的输入为6144；这一字面拼接接口解决的是论文记号歧义，未声称恢复作者未发布的头配置。

掩码条件分别保留作者的softmax前乘零行为，以及印刷公式对应的softmax后乘零、保持原分母的参考条件。未将两者混写为同一方法结果。全部实例与四个概念均进入各模式聚合。

原论文使用UNI图像特征并引用Quilt-1M文本模型。实际文本到UNI空间的映射、12条原始模式概念、患者级真实概念和头配置尚缺充分材料。当前组件要求已对齐的1024维向量；维度一致不能证明语义对齐。

0.115新增 `dspamil_data.py`：将原生UNI权重身份、特征文件、患者与切片划分、12条模式文本／3×4×1024向量、6条等级文本／6×1024向量、文本编码器检查点与对齐依据逐文件绑定。提供者的对齐说明仍是提供者声明，没有被标记为独立核实的原始论文资产。

`training_pattern_table` 必须恰好覆盖TRAIN切片，包含 `slide_id,global_patient_id,grade_group,pattern_3,pattern_4,pattern_5,annotation_provenance`。缺失、未知值、患者或等级错配，以及验证／测试行都被拒绝；不会从GG4／GG5猜测模式标签。验证和测试不能调用 `training_supervision`，将其行字段改成TRAIN也会被拒绝。所有非空实例保留，深检查可检出文件长度和修改时间均未变时的实际字节变化。

准备命令：

```bash
milplus prepare-dspamil --data /path/to/grade-task \
  --schema /path/to/dspa-input-schema.yaml --output /path/to/dspa-binding.json
```

完整schema由 `tests/test_dspamil_data.py` 的明确标记合成输入给出可执行例子；其中概念、权重文件和对齐说明仅供软件检查，不能用于原始方法结果。真实使用须替换为实际资产及来源依据。准备后在任务的UNI编码器中设置 `dspamil: {manifest: /path/to/dspa-binding.json}`。

`dspamil_training.py` 保留论文50轮上限、AdamW学习率2e-4／权重衰减1e-5及全部七个损失记录（含总和），以仅调用Student的验证QWK选模型。早停耐心与最小轮数、六logit等级判定、无放回采样、FP32、单切片更新、无调度／无裁剪都作为明确的参考重建选择记录。该层不允许通过配置悄悄改小轮数或换掉配方。

22项输入检查通过；7项训练控制检查通过，包含完整1024宽度Teacher／PTG／Student双路头的50轮／300次更新与第7次更新后恢复，权重、Adam状态、梯度、采样与随机状态严格相等。此隔离检查直接提供模式张量，**未执行PAT骨干、原始概念生成或完整DSPA训练**。

0.116写入完整 `dspamil_runtime.py` 和 `dspamil_driver.py`。该内部候选始终构建162,690,072参数、12层／1024维／8头PAT及完整1024宽度双路头，不允许通过参数改小图或50轮上限。完整检查点保存每个模型张量、Adam状态、梯度、采样、Python／NumPy／Torch／CUDA随机状态、实际数学后端、患者／概念合同和全部配方。文件对象序列化使ZIP前缀不含文件名或PID，完整载荷不被删减；每份检查点另有实际字节摘要。

恢复前检查实际父／子进程和启动时间，进程仍运行时拒绝重复拉起；确认终止的旧锁会单独保留。推理仅调用Student，不查询真实等级概念或辅助标签。真实概念加载、全模型源码／安装包／原生预算验证及发布验收仍待完成。

等级判定必须明确为 `half_up_expected_grade_reference` 或 `argmax_logits_reference`。前者用记录的Student概率，以float64固定六项加法顺序计算期望并作half-up取整；训练MSE的六logit概率期望接口仍保留。患者级评价先平均同一患者的六类概率，再作相同判定；验证选模与最终评价单位必须一致。通用结果评分和患者配对bootstrap保留该规则，不静默替换成分类argmax或累计概率头的round。bootstrap重复抽取的患者与其切片保留独立抽样身份。

以下是显式标记为合成数据的控制入口，不代表实际UNI提取、Quilt对齐或临床结果。完整训练保持50轮，此处仅展示命令，本轮接入不自动启动训练：

```bash
python -m milplus.dspamil_driver demo --output /path/to/new-synthetic-control
milplus run \
  --data /path/to/new-synthetic-control \
  --method dspamil_2025_released_mask_reference \
  --config configs/dspamil_2025_full_reference.yaml \
  --output /path/to/new-reference-run --device cuda:0
milplus evaluate \
  --run /path/to/new-reference-run --device cuda:0
```

另一掩码条件是 `dspamil_2025_paper_mask_reference`。示例config逐项声明论文未给出的配方选择，50轮是完整预算控制，不是原作者早停参数的恢复。`audit_checkpoint_updates` 只增加指定更新后的观察检查点，不改变训练步骤；`stop_after_train_batches` 可在真实更新边界退出，再用相同配置去掉该选项并加 `--resume` 恢复。原始概念／对齐、DSPA-U和完整验收缺项继续保留。

0.116首次完整启动在预检后遇到 `FileExistsError`：Torch相关缓存提前创建了待保留的输出目录，模型和训练尚未启动。0.117将预检缓存放到结果目录之外，保留结果目录的 `exist_ok=False` 保护。实际Torch2.5.1+cu124／CUDA12.4预检重现了旧问题，并确认新逻辑可严格创建全新结果目录；旧失败和缓存现场均保留。

检查点载荷还对字符串载体身份作规范化，避免恢复前后的历史切片字符串引用影响pickle编码；张量、NumPy数组、梯度与OrderedDict模型元数据均保留，实际序列化工具检查验证这些值及重编码字节。只有完成验证且编码字节摘要完全相同的检查点副本才允许共享存储，不以近似数值或缩小模型来节省空间。

DSPA-U属于不同的四模式及不确定性版本，完整说明尚未取得。当前候选保留2025年的三模式范围，不将其计作DSPA-U复现。官方仓库HEAD已再次核对，仍为上述 incomplete 提交。
