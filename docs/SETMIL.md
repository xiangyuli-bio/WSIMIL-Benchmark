# SETMIL：真实空间 PPE、TPMF 与空间 Transformer

两条执行配置：`setmil_source`、`setmil_repaired`。别名 `SETMIL` 默认选择修正版，它们属于同一方法家族。

## 来源与边界

固定[作者官方仓库](https://github.com/Louis-YuZhao/SETMIL)提交 `4243f633adb9e8a84cea11f92abf3d3b0ede126e`。使用实际训练入口 `WSIFeatDataset`、`T2T_ViT_iRPE` 和完整增强与训练逻辑。另一个 `util/feature_seq2_feature_map.py` 不是示例配置指定的数据路径，不能替代实际 PPE。

源代码保存在外部，经 Git 对象校验后在私有模块命名空间加载，不重新分发。上游 Transformer 文件有 Clear BSD 头，但固定仓库没有相应根 LICENSE，许可仍标记未核实。`timm` 的初始化/零概率 DropPath 和 EasyDict 数据结构采用明确的兼容导入；iRPE 使用作者已经提供的纯 PyTorch 路线，不用普通位置向量替代。

原示例使用 EfficientNet-B0 的1280维 patch 特征，也可以保存图像增强后的 `tr` 视图缓存。本适配器支持冻结 FM 特征迁移；不等同于重现原始 patch 编码器或图像增强提取过程。编码器版本、物理视野、切块和坐标必须由任务包登记。

## 输入和物理特征图

输入为单张切片 `[N,D]` 特征及行序一致的真实 `[N,2]` 坐标。标准坐标为 x,y，转换到作者 row,column 布局。要求 `encoder.spatial` 指定单位、origin、patch_size 和 provenance；可用 `patch_stride` 单独声明步长，未提供时采用 patch_size 的不重叠网格。坐标读取保留 float64。

按真实网格减去最小行、列，保留组织缺口的零单元，不生成序列伪网格。重复、非有限或未按声明网格对齐的坐标会被拒绝。原始 float64 稠密图和模型参数、Unfold、注意力缓冲均在分配前做预算检查，超过 `max_dense_bytes` 明确失败。

PPE 保留原生中心裁剪和对称零填充，默认输出48×48；不是将整张切片缩小到48×48。可能舍弃视野外 patch。预测 CSV 保存总 patch 数、增强前保留数量、保留行号哈希和坐标哈希，不能把局部输入结果称为看过全部组织。

## 完整网络和训练

原生模型：7×7、stride3 的 T2T 降维；3/5/7 三路独立邻域 T2T 融合；每路64维，拼接为192维；CLS token 和固定正弦绝对位置编码；6层16头 Transformer；按 head 独立、基于欧氏相对距离、作用于 query 的 iRPE bias；线性分类头。默认中间 token128维，dropout/attention dropout=.2。网络沿用作者行为忽略传入的 padding mask，没有暗中加入另一套注意力掩码。

完整保留训练增强：水平和垂直翻转、近邻插值缩放后随机裁剪/填充、全图均值填充的 cutout、90度整数旋转。先生成图，再按源代码 NumPy 顺序执行增强；验证和测试不做这些图级随机增强。

训练使用 CE、AdamW lr=2e-4、weight_decay=.05、batch4、保留尾批、梯度范数裁剪 .01；StepLR 在每个 epoch 训练后调用，step_size=100、gamma=.1。默认100轮，验证切片级 AUC 首次最优选模，patience20；患者级最终评价仍按任务声明的融合规则进行，不能用患者 AUC 偷换源切片选模指标。所有声明类别必须出现在验证集。

为可追溯续训，数据读取使用单进程，保留作者 prefetcher 提前一个 batch 取数的 RNG 消费顺序，设备传输同步执行。上游默认10个读取 worker 的并行调度未宣称逐位重现。模型、AdamW、StepLR、epoch 和全部 RNG 一致地写入 last checkpoint，最佳 checkpoint 独立保存。上游把最佳模型和当前 optimizer/epoch 混放的保存方式不用于续训。测试集仅在显式 evaluate 时使用，上游每轮测试的做法未沿用。

## 源码与修正版

`setmil_source` 保留两个经原代码确认的行为：

- 高、宽均超限时，第二次裁剪仍使用裁剪前的高度。例如60×70图裁到48时，实际得到6×48，再填充到48×48。
- 声明 `tr` 缓存后，各 subset 都随机选择每个 patch 的缓存视图，包括验证和测试。

`setmil_repaired` 在第二次裁剪前重新读取当前高度；缓存增强视图仅用于训练，验证和测试使用固定 `features`。网络结构、损失、优化器、学习率和图级增强均保持原路线。

可选缓存必须与特征保存在同一份已哈希的 HDF5 中，形状为 `[N,V,D]`，对齐原始 patch 行。任务示例：

```yaml
encoders:
  uni:
    # 另需填写标准特征维度、索引、权重和 spatial 元数据
    setmil_cached_views:
      dataset: setmil_views
      provenance: 固定编码器对已登记 patch 图像增强视图生成的冻结缓存
```

未声明缓存时，两条路线使用已登记的固定特征；不会伪造增强后的图像特征。缓存同属原 HDF5 的内容哈希，变更后评价和续训会拒绝旧实验。

## 运行与验证

```bash
milplus setup --method SETMIL
milplus run --data /absolute/task_bundle --method SETMIL \
  --output /absolute/new_run --config configs/setmil_repaired.yaml
milplus evaluate --run /absolute/new_run --subset test
```

专项检查包含原生1280维完整网络的输出、CE和全部梯度；直接执行作者 `WSIFeatDataset` 与多种随机增强；整个 `engine.train_one_epoch`、原数据集、AdamW/裁剪/调度/RNG的逐位比较；两条路线精确续训、最佳重载、预测覆盖记录、标签独立、缓存身份与预算拒绝。最终全库、全部配置和独立 wheel 结果见 `docs/validation/summary.json`。

CPU 合成验证与 GPU、真实 WSI、论文分数复现分别记录，后三项尚未执行。
