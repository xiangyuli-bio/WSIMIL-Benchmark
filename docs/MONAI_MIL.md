# MONAI：完整原图训练、依赖 Transformer 与特征图金字塔

这是计划 E 类的原图/编码器联合训练条目。五条执行配置属于一个来源家族：

| 配置 | 原生 `MILModel` 分支 |
|---|---|
| `monai_mean_tutorial` | 各实例线性 logits 均值 |
| `monai_max_tutorial` | 各阈值 logits 分别取最大 |
| `monai_attention_tutorial` | 原生注意力池化 |
| `monai_transformer_tutorial` | ResNet50 + 四层依赖 Transformer + 注意力 |
| `monai_pyramid_tutorial` | ResNet50 四组特征图 + 四阶段金字塔 Transformer + 注意力 |

`monai`、`monai_mil` 和 `Dependency-aware MIL / MONAI` 解析到依赖 Transformer 配置。三个池化分支是原框架中的对照，不另计三种独立论文方法。

## 固定来源和版本边界

- [MONAI 1.5.1](https://github.com/Project-MONAI/MONAI/tree/9c6d819f97e37f36c72f3bdfad676b455bd2fa0d)：`monai/networks/nets/milmodel.py`，Apache-2.0。
- [官方训练教程](https://github.com/Project-MONAI/tutorials/tree/d7c326e2b5b72423a082c9cee3e41035c4c6a42a/pathology/multiple_instance_learning)：完整 `train_epoch`、数据增强、优化器及验证流程，Apache-2.0。
- 原论文：[Accounting for Dependencies in Deep Learning Based Multiple Instance Learning for Whole Slide Imaging](https://arxiv.org/abs/2111.01556)。原始图像输入和两种 Transformer 是主模型；论文另报告了五模型集成、生成实例伪标签及再次训练的扩展。

本适配完整保留已发布 MONAI tutorial 的训练路线。**论文的五教师实例伪标签扩展没有在这些配置中实现，也不声称这些配置包含该扩展。** 它没有出现在本次固定的官方 trainer 中，不能通过给普通 BCE 起一个伪标签名称来补足。

论文和后续教程的超参数有差异。本配置使用发布教程的 AdamW、学习率、256px/44 tiles；没有把论文的 Adam、224px/56 tiles 或学习率混入同一版本。

## 输入：完整分辨率层级，而非冻结特征

数据包依旧提供患者主表、固定 train/val/test 划分与标签。`encoders` 中登记一个原图输入入口，`dimension: 3` 表示 RGB 通道。`index` 中每个 `path` 指向一张完整切片的已声明分辨率层级：

1. `kind: openslide`：单文件 WSI，例如 SVS、TIFF、NDPI、SCN。真实分辨率层级、尺寸和 downsample 由 OpenSlide 读取，不静默选择别的层级。多文件 MRXS 等需要额外伴随文件清单，当前明确拒绝。
2. `kind: decoded_rgb`：完整层级的 uint8 `[3,H,W]` NPY，或 HDF5 的 `pixels` 数据集。它保留在线随机网格和增强，不能用一组固定采样 patch 或 FM 向量冒充。转换过程、原始层级和 downsample 必须预先登记。

一个六级 ISUP 任务的相关字段如下；完整任务包仍需 `slides.csv`、`split.csv` 和 `images.csv`：

```yaml
schema_version: 1
task_id: panda_isup_locked
task_type: ordinal
classes: [ISUP0, ISUP1, ISUP2, ISUP3, ISUP4, ISUP5]
training_unit: slide
evaluation_unit: slide
slides: slides.csv
defaults: {fm: raw_resnet, split: fold0}
splits: {fold0: split.csv}
encoders:
  raw_resnet:
    dimension: 3
    index: images.csv
    raw_images:
      kind: openslide
      level: 1
      provenance: "填写真实扫描来源、层级与预处理版本"
      max_image_bytes: 536870912
      backbone:
        architecture: resnet50
        initialization: imagenet1k_v1
        path: /absolute/resnet50-0676ba61.pth
        sha256: 0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a
```

`images.csv` 至少包含 `slide_id,path`；可提供完整 `sha256`。原图内容、尺寸/层级、初始化权重、任务和患者划分均加入运行身份检查。`max_image_bytes` 在解码前检查；模型/特征图/注意力另有 `max_dense_bytes` 预算。大图需明确提高预算，或从确实存在的低分辨率层级建立新协议。不会自行缩小图像或换层级来通过预算。

默认不在线下载权重。ImageNet 初始化必须匹配上面的官方文件完整 SHA256；`initialization: explicit_random` 是有名、有记录的从零训练条件，不是缺权重时的自动回退。合成示例默认使用它。

## 保留的原生网络

ResNet50 的全部卷积和 BatchNorm 均参与训练，没有冻结 `.net`、预先计算特征或用小 CNN 替代。

普通 Transformer 原生宽度为2048，8头、4个 `TransformerEncoderLayer`，FFN2048、默认 dropout0；注意力为2048→2048→1。

金字塔从真实 `layer1..4` 特征图取空间均值，宽度分别256/512/1024/2048；中间拼接后为768、1280并投影回256，最后拼接成2304进入末级 Transformer。每级默认4层，最后注意力为2304→2048→1。没有将单尺度向量 reshape 成伪特征图。原论文文字的2308与源码2304有差异，本适配按发布源码执行。

训练始终执行完整 `[B,N,3,H,W]` 小批次，保留同一批次的原生 BatchNorm 统计。验证可分块计算 CNN，最后对全包执行聚合器；金字塔只保留各块特征图的空间均值，避免堆积完整特征图。多 bag 分块的 B/N 顺序明确处理并对照整个包的原生 forward。训练中请求分块会明确拒绝，不能用它暗中改变 BatchNorm。

## 原图预处理的实际行为

训练使用原生 MONAI `RandGridPatchd`、按像素亮度之和排序取最小的44块、逐块水平/垂直翻转与90度旋转（各概率.5）。源码网格随机偏移的默认上限是图像尺寸除以 patch size 的**余数**，不是任意一个完整 patch 的范围；这里保留实际实现。

`pad_mode=None` 只说明不做图像边缘填充。`GridPatch` 在训练块数不足44时仍会补白块，且这些白块参与模型；这一行为保留并记录数量。验证使用固定网格及 `.999*3*255*tile_size²` 前景阈值，读取全部保留块，不做训练增强。无前景或尺寸不足时明确失败。

所有输入按源码从 RGB uint8 缩放到 `[0,1]`。没有额外 ImageNet mean/std、染色归一化或 ICC 转换。每包的真实选块位置、层级/downsample 和白块数量产生哈希，验证预测 CSV 保存哈希与实际块数。

采用显式单 worker 随机流：每个 TRAIN bag 的原生 Compose 由已保存的 CPU Torch RNG 生成独立种子。它保留增强机制并允许精确恢复；不声称重现原多 worker/GPU 调度顺序。

## 序数目标与完整优化流程

K个有序等级使用K−1个累积二元标签：等级g的第j个目标为 `j < g`。源教程对其使用 `BCEWithLogitsLoss`。原生预测是 `round(sum(sigmoid(logits)))`，**不是 softmax 的 argmax**。无序的 BRCA 亚型等任务不能仅通过重排类别冒充这个序数任务。

预测保存各阈值、期望等级和取整等级；不人为构造类别概率。因此对应的普通 multiclass AUROC/log-loss 留空。报告原生 QWK、准确率、macro F1、等级 MAE 和验证 BCE。患者簇配对 bootstrap 已支持这些序数分数，也支持与 CE/softmax 分类头比较 QWK 等等级指标；分别保留取整和 argmax 的原始决策规则。不将多种子视为独立病例。发布教程按 WSI 评价，本配置要求 `evaluation_unit: slide`；患者概率融合不能直接照搬到这类阈值头。

默认50轮、batch4，保留最后不足一批的数据。使用 AdamW：CNN/attention/head 学习率3e-5、weight decay0；Transformer 单独学习率6e-6、weight decay.1。每轮训练后执行 `CosineAnnealingLR(T_max=epochs, eta_min=0)`。无裁剪，不用普通单 bag CE trainer 替代。

只用验证 QWK 选模，保持完整训练预算。原教程初始最佳QWK为0，若始终非正可能没有最佳文件；统一接口会保存第一个有限验证结果，然后按严格改善选择。这项执行修正明确登记。last/best、AdamW、cosine、AMP scaler和全部 RNG 都保存，测试仅在显式 evaluate 时运行。AMP 必须显式启用并有实际 CUDA，训练、验证前向与验证BCE都使用对应精度上下文，CPU不会静默模拟。

## 使用和验证边界

```bash
# CPU/GPU 环境须选择匹配的 Torch/Torchvision wheel；当前软件锁为Torch2.5.1。
python -m pip install -e '.[rawimages]'
milplus setup --method monai

# 网络允许的 CPU 主机下载一次，再带到离线环境；运行阶段只读本地。
curl -fL -o /absolute/resnet50-0676ba61.pth \
  https://download.pytorch.org/models/resnet50-0676ba61.pth

milplus run --data /absolute/ordinal_rgb_bundle --method monai \
  --output /absolute/new_run --config configs/monai_transformer_tutorial.yaml --device cuda
milplus evaluate --run /absolute/new_run --subset test --device cuda
```

合成原图示例可用 `milplus demo --raw-images --patients 30 --classes 6 --output /absolute/demo` 创建。示例 patch 为32px，试跑应在配置明确使用 `tile_size: 32`、`tile_count: 4`；正式原生默认256px/44块和完整网络不因此改变。

测试分别检查原作者五分支、原生四层网络、全部输入/参数梯度、源 Compose、多轮完整 `train_epoch`/AdamW/cosine/BN/RNG、精确续训、真实 OpenSlide 解码器对合成金字塔 TIFF 的读取、ImageNet 初始化和输入篡改。GPU脚本为 `scripts/validate_monai_cuda.py`，缺实际 CUDA 时记录 `not_run`。最终全库与安装包证据见 `docs/validation/summary.json`；CPU合成检查不等于真实 WSI 或论文数值复现。
