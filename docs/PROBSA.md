# ProbSA-MIL：作者库完整训练路线

执行配置：`probsa_abmil_source`、`probsa_abmil_probmean`、`probsa_transformer_source`、`probsa_transformer_probmean`。别名 `ProbSA-MIL` 默认使用 `probsa_abmil_probmean`。四个配置属于一个方法家族，不计作四篇独立工作。

## 来源及范围

[原方法仓库](https://github.com/Franblueee/ProbSA-MIL) 目前未提供原始实验脚本，明确指向作者维护的 [torchmil](https://github.com/Franblueee/torchmil) 中 ProbSmoothABMIL / TransformerProbSmoothABMIL。采用提交 `2472b8bee84b906eac9351f1ec49d430c577bbcd`，Apache-2.0；独立获取后读取固定 Git AST，不捆绑第三方库或凭空补写其原始实验数据。这里是明确命名的作者库版本适配，不是原始论文分数复现。

保留作者高斯注意力的均值与对数方差网络、重参数化采样、源码的采样值裁剪、softmax 聚合、MC 分类目标和图先验 KL。Transformer 路线使用作者完整 Encoder / Layer / 多头注意力与残差路径。确定性零方差不是这些路线的默认值。

## 输入与图

要求冻结 FM 特征、对应真实二维坐标、物理单位 / patch_size / 原点与预处理来源、切片标签、患者身份和固定划分。仅支持二分类和独立切片训练；患者评价通过既定概率融合，不把不同切片坐标拼成一张图。

图采用固定作者 `build_adj` / `normalize_adj`：半径为 sqrt(2)×patch_size，边权 exp(-L2/D)，对称度归一化，不补自环。图基于输入的原始冻结特征，不使用当前可训练投影、标签或测试结果。坐标读取保留 float64 精度，拒绝重复 / 非有限坐标。图缓存可有界淘汰，重建仍产生相同静态图；图边数在原代码分配之前检查。

## 完整配方

默认先使用 Linear(D,512)+ReLU 特征变换，注意力宽度128，训练采样1000次、评价5000次。Transformer 使用两层、八头、宽度512。默认 ABMIL batch=4、Transformer batch=1；允许显式登记其他 batch 和预算。

训练每个 batch 对所有 MC logits 计算 BCEWithLogits 均值，加上归一化的高斯图 KL；不能把这个损失换成对平均 logits 的一次 BCE。默认 Adam、lr=1e-4、weight_decay=0，LinearLR 从0.1倍起、10轮预热；完整100轮预算，验证切片 AUROC 选最佳权重，测试集锁定。可选正类权重只从 TRAIN 切片拟合并入 checkpoint。

默认五个 KL 周期，cycle_len=总更新步数/5，前 floor(0.8×cycle_len) 步线性增长，其余保持1。执行作者 CyclicalAnnealingScheduler 的实际调用顺序，包括首个 step 后系数仍为0的源码行为；也支持显式固定正系数。零 KL 被拒绝为 ProbSA 正式路线，确定性基线另行定义。

## 必要修正

- 原 KL 未接受 mask，会将 batch padding 的均值、方差和长度算进目标。现在逐 bag 使用真实有效长度，padding 无 KL 梯度。
- 原稀疏 KL 为取对角线将 N×N 图搬到 CPU 并稠密化；现在用稀疏 BMM 和对角 scatter 计算相同迹项，全程不创建稠密邻接矩阵。
- 作者 SDPA 无条件使用 dropout；评价时显式关闭。诊断注意力分支的 mask 极性也修正。正常模型训练不依赖该诊断分支。
- 原模型通过 MILModelWrapper 接受 TensorDict；本训练入口显式传入 Y/X/adj/mask，保留模型的完整 compute_loss。没有删除缺少的训练阶段，也没有把原库现成 Wrapper 误报为缺失 trainer。

## 两个预测定义

`*_source` 保留作者库的 sigmoid(mean(logits))；`*_probmean` 使用 mean(sigmoid(logits)) 的预测混合，即论文 Eq.19 的概率积分。两种结果不保证相等，所以不混名汇报。两者训练目标相同。

预测使用不依赖 query 标签的固定逐切片随机流，保留真实采样并恢复训练 RNG。输出 MC 混合的预测熵、条件熵和互信息；在 source 路线上，这些不确定性描述概率混合，而不是额外替代 source 的显示分数。实例 HDF5 提供原始行序对应的归一化注意力均值、未归一化高斯注意力均值 / 方差和真实坐标，不能将它们解释为 patch 癌症概率。

## 运行

```bash
milplus setup --method ProbSA-MIL
milplus run --data /absolute/task_bundle --method ProbSA-MIL \
  --output /absolute/new_run --config configs/probsa_abmil_probmean.yaml
milplus evaluate --run /absolute/new_run --subset test
```

原生尺寸和采样次数是默认值；缩小后的合成软件检查会显式记录参数。MC 工作集、Transformer 注意力工作集、图边数和缓存分别有预算上限，超出时报告输入条件，不静默裁剪 bag 或缩减采样数。

## 验证

专项检查包括完整作者网络 / 梯度、直接执行作者 Trainer._shared_loop 的四组三轮数值对照、独立 KL 公式与 padding 梯度、稀疏图不稠密化、周期 / LR 边界、四路线精确恢复、两种预测定义的数值反例、标签独立性、实例输出与内存边界。最终全库结果见 `docs/validation/summary.json`。

GPU、真实 WSI、原始队列和论文指标仍须独立验证；CPU 合成结果不作为研究结论。
