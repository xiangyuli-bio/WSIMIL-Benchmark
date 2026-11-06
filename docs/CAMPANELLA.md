# Campanella MIL-RNN：完整原图两阶段训练

`campanella_rnn` 对应计划 E 类的一条方法记录。执行的是原图 CNN 硬实例学习与专用循环聚合两个完整阶段，不是外部冻结 FM 特征加普通 RNN。输入仍使用统一患者、切片、标签和划分数据包。

## 固定来源

原作者代码：[MSKCC-Computational-Pathology/MIL-nature-medicine-2019](https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019/tree/daecb7293b5ea886ba222008f2592e1e29e9dac6)。固定树包含完整 `MIL_train.py` / `RNN_train.py`、测试入口和 README。该版本未发现 LICENSE，许可状态保留为未核实，作者源码与预训练权重均不打包分发。运行时从锁定 Git 对象加载作者 `rnn_single` 和分组筛选定义；ResNet34 使用固定 torchvision0.20.1。

## 原图与真实坐标

本方法需要原始单文件 WSI 和真实 `(x,y)` 切块坐标。TRIDENT 生成的真实 level0 坐标可用于索引，但仅有其 FM 特征向量不足以运行这个联合训练方法。不能把特征序号构造成方格来代替坐标。

`task.yaml` 的相关配置如下，其余患者主表、二分类标签和 train/val/test 划分沿用通用数据包：

```yaml
schema_version: 1
task_id: binary_wsi_locked
task_type: classification
classes: [negative, positive]  # 明确原方法的0/1顺序；不会从文件名推测标签
training_unit: slide
evaluation_unit: slide
slides: slides.csv
defaults: {fm: raw_resnet34, split: fold0}
splits: {fold0: split.csv}
encoders:
  raw_resnet34:
    dimension: 3
    index: images.csv
    raw_regions:
      kind: openslide
      coordinate_unit: level0_pixels
      coordinate_order: xy
      level: 0
      mult: 1.0
      provenance: "填写组织筛选、真实切块坐标及版本来源"
      max_grid_rows: 2000000
      max_grid_cache_bytes: 134217728
      max_tile_bytes: 268435456
      backbone:
        architecture: resnet34
        initialization: imagenet1k_v1
        path: /absolute/resnet34-b627a593.pth
        sha256: b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f
```

`images.csv` 至少有 `slide_id,path,coords_path`。`path` 指向 SVS、TIFF 等已支持的单文件 WSI，`coords_path` 指向整数 `[N,2]` NPY，或含 `coords` 数据集的 HDF5。外部链接、虚拟数据集、外部存储和软链接通向外部文件均拒绝。源 WSI、坐标文件、实际层级/downsample、坐标行顺序、权重和患者划分都有身份检查。多文件 WSI 需要完整伴随文件清单，当前不自动接受。

每次只读取一个真实区域，随后组成源配方的小批次；不会把整张 level0 WSI 解码到内存。坐标使用有上限的只读 LRU 缓存，全队列索引采用累积偏移，不创建海量 Python ROI 元组。源分组排名仍需要全体 TRAIN 的分数数组，该部分和 CNN/序列批次分别预估并受 `max_dense_bytes` 约束；超预算明确失败，不偷偷丢实例。

原源码在两个阶段有不同的读取尺寸：CNN 用 `round(224*mult)`，RNN 用 `int(224*mult)`；`mult!=1` 时再做 PIL 双线性 resize 到224。这个可能改变两阶段实际视野的差异保留并登记。`level` 必须真实存在；可提供 `downsample` 进一步核对。原点必须位于 WSI 内部，区域跨过边缘时保留 OpenSlide 的原生 RGBA→RGB 行为，不换用白色补边。

归一化严格使用原代码的 `ToTensor` 和各通道 `mean=.5,std=.1`，不额外做 ImageNet mean/std、随机翻转、颜色或染色增强。ImageNet 初始化只读本地完整 SHA256 匹配的官方 ResNet34V1；`explicit_random` 是明确命名的不同初始化条件，没有缺权重自动回退。

## 阶段1：训练硬实例筛选 CNN

默认完整 ResNet34 ImageNet 初始化，分类头512→2，100轮、batch512、Adam学习率1e-4、weight decay1e-4。每轮：

1. CNN置为eval，遍历全部 TRAIN 坐标区域，计算阳性概率。
2. 使用源 `np.lexsort` / `group_argtopk`，每张 WSI 选择top1；不使用实例真值。
3. 对选出的实例用 Python `random.sample` 重排。
4. CNN置为train，以切片弱标签做 CE，所有卷积和 BatchNorm 参与更新。

默认每10轮在 VAL 上计算每张 WSI 的最大区域阳性概率，阈值为 `>=.5`，以 `(FPR+FNR)/2` 的最小值选模，平局选择最新。阶段2始终使用验证最佳 CNN，不能使用最后一轮 CNN 或未训练的 ImageNet 编码器。显式缩短预算时，CNN最后一轮必须落在实际验证边界上。

## 阶段2：作者专用循环聚合器

使用最佳 CNN 对 TRAIN、VAL 和后续查询的真实坐标分别打分，选择10块并按阳性概率降序排列。筛选不读取查询标签；相同分数在源top-k集合内部按原坐标行顺序排列。这个衔接步骤是基于 README 定义实现的明确桥接，原仓库未提供完整生成脚本。

CNN冻结全部参数及 BatchNorm。每块输出512维特征，循环单元为作者的：

```text
h0 = zeros(batch, 128)
ht = ReLU(fc1(instance512) + fc2(h[t-1]))
logits = fc3(h10)  # 只对最后一步计算CE，完整反向传播至全部循环步骤
```

它不是 `nn.RNN`、GRU或LSTM。默认100轮、batch128、SGD学习率.1、momentum.9、weight decay1e-4、Nesterov启用；没有裁剪或调度器。默认保持降序序列；显式 `shuffle_sequence` 才做源代码中的序列重排。

原数据集用 `min(s, len(grid))`，不足10块、长度不同的混合 batch 会在默认collate处失败。本执行配置要求每张切片具有至少 `steps` 个真实区域，不暗中填充、缩短所有包或重复实例。`steps`、`ndims` 和两阶段批次等显式覆盖全部写入请求与方法协议。

阶段2每轮验证，保留源 `errors()` 的 logits `argmax` 选模规则。原 `RNN_test.py` 导出的最终预测则是阳性概率 `>=.5`。两者在恰好.5时不同：预测文件同时记录 `prediction` 与 `campanella_selection_prediction`；通用分类指标和患者簇bootstrap遵循已登记的导出决策规则，选模使用源训练规则。不会混用这两种判断。

## 保存、恢复和结果

统一入口执行完整两阶段，默认共200轮。两组优化器、当前阶段/轮次、CNN最佳快照及哈希、实例筛选缓存与坐标身份、循环初始化、更新次数和 Python/NumPy/Torch RNG 都进入 last/best。阶段1不会写成最终RNN最佳模型；中断后从各阶段边界恢复，不重新随机初始化已训练的RNN，也不改用最后CNN。测试只在显式 evaluate 时执行。

预测CSV保存原始坐标行、网格和选中位置哈希、真实层级、最佳筛选器哈希与两种决策。`method_protocol.json` 记录两阶段预算、优化器、输入、默认及显式覆盖。采用一个同步数据worker和统一可恢复随机流，不声称重现作者历史上分开启动两个CLI进程的种子状态。

```bash
python -m pip install -e '.[rawimages]'
milplus setup --method campanella_rnn
curl -fL -o /absolute/resnet34-b627a593.pth \
  https://download.pytorch.org/models/resnet34-b627a593.pth

milplus check --data /absolute/binary_raw_bundle --method campanella_rnn --deep
milplus run --data /absolute/binary_raw_bundle --method campanella_rnn \
  --config configs/campanella_rnn.yaml --output /absolute/new_run --device cuda
milplus evaluate --run /absolute/new_run --subset test --device cuda
```

合成ROI数据包可用 `milplus demo --raw-regions --patients 8 --backbone-weights /absolute/resnet34-b627a593.pth --output /absolute/synthetic` 创建。试跑预算需要另一个显式配置，例如 `cnn_epochs: 10,rnn_epochs: 2,test_every: 10,cnn_batch_size: 4,rnn_batch_size: 4`，并将总epochs设为12或让入口自动按两阶段求和。正式默认224像素、10步、128维循环状态和完整ResNet34不会缩小。

验证覆盖原作者完整两阶段训练、所有参数/输入梯度、Adam/SGD、BatchNorm与RNG，原生OpenSlide坐标读取、多种mult，以及最佳CNN/跨阶段续训/查询标签独立性。最终通过状态以 `docs/validation/summary.json` 为准。合成数据的软件一致性不能证明自然WSI、GPU或论文数值复现；这些状态独立登记。

## 显式实测区域缓存入口（2026-11-06）

新增 `raw_regions.kind: measured_rgb_region_cache`，仅改变读图 I/O。
它不是整张 WSI，不能调用 `slide()` 冒充 OpenSlide，也不接受冻结 FM 特征。
必须提供 `cache_manifest`：完整实际 RGB 区域 H5、原始坐标/行身份、逐袋源读取及交付收据、
真实 WSI stat、可选 quickhash 的真实值或显式缺省、level/downsample/size、全部像素和归档 SHA、源收据 SHA 均须绑定。
坐标 NPY 只是原始 H5 坐标的无损整数副本，不是生成的网格。
虚拟/外部 H5 像素、非原生尺寸、错位坐标、错误 patch ID、文件变化和合成数据冒充真实数据均拒绝。
单个缓存必须精确覆盖 CNN 与 RNN 各自的源读取尺寸；不能裁剪、补边或偷偷插值到另一读取尺寸。
因此这个缓存模式不接受 `mult=1.003` 的 CNN225/RNN224 尺寸分歧，原 OpenSlide 模式仍原样支持它。

BRCA 专项采用完整原始 CLAM level1/256RGB 与显式 `mult=8/7`：
作者 `round(224*mult)` 和 `int(224*mult)` 在此均为256，再执行原样 PIL 双线性224、
ToTensor 与 Normalize(mean=.5,std=.1)。该条件明确区别于历史论文倍率或源码示例 level0/mult1，
不冒称历史倍率、队列或分数复现；未改变 CNN/RNN、100+100 epochs、512/128批次或10步机制。
运行协议增加 `region_storage`，记录缓存 manifest 和实际读取条件，原 OpenSlide 协议不变。

真实原始 BRCA WSI 在原图节点执行未改的作者 `MILdataset` / `rnndata` 类，
3个 CNN 区域及完整10步 RNN 张量与 GPU 主机缓存入口逐位一致，完整1706行几何/坐标核对通过。
凭据 `campanella_real_original_region_equivalence.json`。这是输入实现对照，不是临床训练或GPU数值验证。

原有50项完整专项加9项缓存入口检查已在现有节点的CPU路线通过，未改原测试模型/宽度/预算。
另外3项保留完整 ResNet34、原样三轮两阶段软件预算及原恢复断言，核验缓存训练和直接读取的轨迹。
首次2项恢复通过，轨迹项因试图把不同 TIFF/H5 的文件身份摘要也当成相同而失败；
完整诊断确认差异仅为6个 `top_cache.identity`，没有参数、选择索引/概率、优化器、RNG或历史数值差异。
失败文件及日志保留。修正后的检查先分别按原公式验证各文件真实 stat 身份，并要求两种身份不同，
再逐位比较全部数值状态；未改模型或放宽数值容差。实际最终是否全通过以62项报告/attestation为准，
不能把这些短软件检查称为已完成真实200轮训练。

专项准备器等待全部856/2422295行原图及62项完整通过报告，才建立真实完整输入契约，
再交给同一资源感知队列。原生 batch512 的全队列CUDA峰值尚未测得；
队列采用单卡独占/128GiB显存预留，不把未测峰值写成已证实，也不为并发减小批次。
实际状态 `native_campanella_method_preparation.json`；源码候选仅新增缓存读图与协议字段，活动训练不受修改。

2026-11-05T20:02Z，R1的62项完整软件检查及逐位轨迹/恢复对照全部通过。
随后完整856输入构建发现318张源WSI的OpenSlide `quickhash-1` 属性为None，
其余原始stat前后、level/size、坐标和真实RGB交付证据均一致。
这是R1对可选属性的过强入口要求，不是原图损坏、坐标缺失或模型失败；原失败目录完整保留。
仓库与隔离R2候选改为保留真实None，增加`source_quickhash_available:false`，
同时强制完整真实stat字段、源stat前后一致和既有像素/归档/坐标/收据SHA，不能生成替代quickhash。
新增3项可选属性检查（合法None接受；空串/布尔拒绝），完整65项在独立节点CPU路径重新执行。
截至本次记录仍未宣称R2全绿或启动真实训练；当前V7队列只认已冻结R1，不会把尚未核验的新源码静默投入训练。
证据`campanella_optional_quickhash_full_cohort_diagnostic.json`、`campanella_region_actual_checks_state_r3.json`。

2026-11-05T20:15:23Z，隔离R2候选的全部65项检查实际通过，50项既有原始检查、
12项缓存/可选属性检查及3项完整缓存训练/恢复检查均未减少模型宽度、原案例预算或数值容差。
真实原始WSI的13个CNN/RNN输入张量在R2再次逐位通过。
完整XML和逐文件源码/案例摘要绑定于`campanella_complete_validation_attestation_r2.json`。
128项既有/新增队列控制检查亦在实际节点通过，当前V8只接受这个已冻结R2候选及真实None元数据，
拒绝伪造quickhash、剔除318张切片、换模型、改预算或输入不完整。
方法准备器使用新目录`inputs_native_campanella_measured_regions_r2`，旧失败目录不覆盖，
当前正在完整856 RGB归档的首用SHA核对；尚未宣称已进入训练或完成200轮预算。

2026-11-05T20:31Z，R2完整856患者/2422295原始行的实际深检与原生模型契约通过，
318个真实缺省quickhash如实保留，65项报告和原初始化/100+100epochs/512与128批次/10步均已绑定。
原生请求`campanella_native_regions2_20261106.json`获现有V8队列accepted，
数据目录和旧失败目录分别保留。当前为ready_for_training_attempt，不是已开训或完成。
