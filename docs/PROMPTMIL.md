# Prompt-MIL：完整原图提示微调

来源为 [作者仓库](https://github.com/cvlab-stonybrook/PromptMIL/tree/b7c088e983eb4e369b9fe346873e941da01ed2e6)，
固定提交 `b7c088e983eb4e369b9fe346873e941da01ed2e6`。
[作者论文全文](https://arxiv.org/abs/2303.12214) 已核查方法、三次前后向和训练配置。
原仓库没有统一 LICENSE；运行时私有加载固定原文，不分发作者实现或权重。

本接入需要每张切片的实际 RGB224 patch，不能用预先提取的 FM 向量代替。
完整编码器为 DINO ViT-Tiny，192 维、12 层、3 头、16 像素 token patch。
代码名字中的 `_192` 表示嵌入维度，实际图像输入为 224 像素。
逐项核验原始队列 teacher 的全部 149 个骨干状态键；不允许随机或部分加载编码器。
骨干冻结，仅新提示参数和完整 DSMIL 分类器训练。默认一个浅层提示 token，
也支持明确配置 `model.deep_prompt: true` 的完整 11 层深提示扩展。

每个 WSI 的训练保留三次计算：

1. 全部原图按原代码的均衡 `torch.chunk` 分块，无梯度提取临时特征。
2. 完整 DSMIL 同时计算 bag 和 max-instance 损失，各占 0.5，保留每个特征的梯度。
3. 再次前向每块原图，把对应特征梯度传回真实提示参数。

没有切片实例截断、缓存向量替代或梯度裁剪。原示例不启用 ImageNet 归一化，
保留 uint8 RGB 除以 255 的输入。代码实际采用非提示 token 的全局均值；
论文结构图中的 CLS 描述与实现细节的 GAP 不同，本接入保留可核验的作者实现。

## 分别命名的配方

| 队列条件 | 源码配置 | 论文预算参考 | 共同原始定义 |
|---|---|---|---|
| `brca` | 50 轮，LR 1e-4 | 40 轮，LR 1e-4 | AdamW，WD .01，训练块 512 |
| `bright` | 50 轮，LR 1e-4 | 40 轮，LR 1e-4 | Adam，betas .5/.9，WD .05，训练块 512 |
| `crc` | 40 轮，LR 1e-4 | 40 轮，LR 5e-4 | AdamW，WD .01，训练块 256 |

配置 ID 为 `promptmil_{brca,bright,crc}_dino_dsmil_{source,paper_reference}`。
BRIGHT 使用 CE；另外两套使用 one-hot BCE。学习率使用原始两组参数和余弦调度，
最小 LR 5e-6。原始提示组倍率为 1；论文中完整骨干微调的 0.1 倍率不强加到提示训练。

源码每 8 个切片累加未经除 8 的梯度并更新；不足 8 的尾部梯度跨 epoch 保留，
最终未更新的尾部梯度不参与模型输出。`source` 配置保留这一行为，
并按作者实际 `trainer.test(passed_model)` 行为选最后一轮。
其 TRAIN 少于 8 张时会明确失败，避免报告零优化器更新为训练成功。

`paper_reference` 保留论文轮次与学习率，并明确补充两项基准决定：
每个 epoch 更新最后一个不足 8 的尾批，按验证集的原始分数 macro OVR AUROC 选模。
这两项不是已经恢复的论文隐藏超参数。两种配置都不提前停止。
两种配置均保存尚未更新的实际提示/分类器梯度、优化器、调度器、抽样顺序和全部 RNG。

原代码在验证阶段同时监控测试数据；统一入口训练时只读取验证集，测试保持锁定。
该数据协议变化有明确记录，不声称复现完整原始 Lightning 数据加载顺序或论文分数。

## 分数与评价

作者即使在 CE 三分类中也返回独立 Sigmoid 列。
CSV 的 `promptmil_source_score_*` 保留这些原值，`promptmil_source_AUROC` 是逐列 macro OVR，
包括二分类中两个输出列的平均，不将其静默替换为正类单列。
`p_*` 是明确单独导出的 logits softmax，用于共享的概率评价；两类指标分别命名。
请求患者级评价时，分别对相应分数列做患者均值，属于明示的评价扩展。

## 准备和运行

联网 CPU 主机获取指定原始权重；GPU 任务直接使用已准备文件：

```bash
python -m milplus prepare-promptmil-assets --condition brca --output /own/assets/promptmil
```

原图索引表有 `slide_id,patch_id,path` 三列。按表顺序打包实际 RGB224，不裁剪、不缩放：

```bash
python -m milplus pack-promptmil --table /own/patches.csv --output /own/task/rgb
```

常规数据包保留患者、类别和固定划分，编码器 `dimension: 3`，index 指向打包的 `features.csv`。
准备 schema 示例：

```yaml
schema_version: 1
kind: promptmil_original_RGB224_and_cohort_DINO
condition: brca
pixel_key: rgb
patch_ids_key: patch_ids
normalization: source_uint8_div255_only
initial_weights: /own/assets/promptmil/tcga-brca-vitt.pth
pixel_provenance: 原始WSI切块协议及实际解码来源
patch_provenance: 实际唯一patch身份与行顺序
```

```bash
python -m milplus prepare-promptmil --data /own/task --schema /own/schema.yaml --output promptmil.json
# 在 task.yaml 对应 encoder 中添加 promptmil: {manifest: promptmil.json}
python -m milplus run --data /own/task --method Prompt-MIL --device cuda:0 --output /own/run
python -m milplus evaluate --run /own/run --device cuda:0
```

别名从已绑定的实际原始权重选择队列参考配置，不修改输入任务的标签含义。
支持声明其他类别数的输出头扩展；原始骨干维度、深度和 DSMIL 结构不缩减。
真实参考复现、独立安装包、完整原生预算与整库发布验收分别记录；
合成 RGB 软件测试不作为临床 WSI 或论文指标复现。

## 显式原作者合并JPEG输入（2026-11-06）

原HDF5 RGB224入口、无隐式缩放的`pack-promptmil`和全部模型/训练配置保持不变。
另支持`schema.storage: source_merged_jpeg_rgb224`，直接读取作者原生的合并JPEG格式。
它不是把RGB256文件伪装成224：作者固定提交的`script/merge_img_wsi.py`实际先以
`cv2.resize`缩放为224，再每200张纵向合并、RGB转BGR、按OpenCV默认选项写JPEG。
新索引保存完整原始patch IDs及顺序、每个200张/最后尾组的偏移/数量/路径/SHA；
准备器核验全部JPEG字节和真实RGB224几何，实际读取执行作者的`read_rgb_img`。

这里明确选用作者提供的可选依赖回退条件`use_jpeg4py=False`，即OpenCV解码，
不偷偷改用Pillow解码或声称恢复了作者未公开的历史jpeg4py/OpenCV版本。
schema必须声明`source_commit`、`merge_batch: 200`、
`jpeg_decoder: source_cv2_fallback`、实际`opencv_version`/`opencv_build_sha256`
及`jpeg_protocol_sha256`。普通RGB224 schema字段和原始队列DINO权重要求仍然有效。
新入口须安装额外依赖`pip install '.[promptmil-jpeg]'`；Kornia仅用于独立原作者Dataset对照，
模型仍使用已核验的原始全图块/三次计算、192维12层DINO和完整DSMIL。

索引文件示意（仅结构，SHA及真实路径必须由实际输入准备产生）：

```json
{
  "schema_version": 1,
  "kind": "promptmil_original_200_tile_merged_jpeg_bag",
  "protocol_sha256": "实际预处理协议SHA",
  "patch_ids": ["真实原始图块ID，按原顺序完整列出"],
  "chunks": [{"path": "000000.jpg", "offset": 0, "count": 1, "sha256": "实际JPEG SHA"}]
}
```

常规features索引的`path`指向各袋JSON，再用现有`prepare-promptmil`绑定新schema。
重复图块/文件、缺失或错序组、错误颜色/几何、改动JPEG字节、未声明存储/解码条件均拒绝。
不在加载时裁剪、补齐、重复或重采样；原模型自身的分块和梯度累积不变。

实际BRCA最大原图袋选取205行做固定输入接口对照，原作者`process_batch`产生的
完整200张组及尾组5张，与新准备流程的JPEG字节完全一致；作者完整Dataset
读出的float32张量和当前真实入口逐位一致。这是接口核验，不是只用205行训练。
另一个真实完整2899行袋已逐行准备、源RGB像素/SHA及全部JPEG回读核验通过，
全856袋/2422295行准备随后启动。它单列
`native_promptmil_source_cv2_jpeg224_original_coords`，不是UNI主组，也不声称论文5x队列复现。

全部17项原有原作者完整机制/两轮训练/恢复检查，以及10项新格式/拒绝/双轮
HDF5同像素对照（权重、Adam、待累积梯度、RNG）在实际节点CPU路径通过。
开发机3次137退出记录保留，不能把先前的部分stdout当完整通过；实际重测仅调整
已核验资产位置和无RNG副作用的资源日志，没有减少用例、宽度、轮数或原生分块参数。
证据在BRCA工作共享目录：`promptmil_original_complete_actual_runtime_r4.xml`、
`promptmil_jpeg_full_container_actual_runtime_r3.xml`、
`promptmil_real_original_preprocessing_equivalence.json`、
`native_promptmil_jpeg_pilot_state.json`、`native_promptmil_jpeg_full_launch.json`。
冻结候选仅改`promptmil_data.py`并新增`promptmil_jpeg.py`，未改运行中的源码或runtime2。
全856输入与单独任务契约未完成前，仍不能把该方法计作已放行或训练完成。

### 原队列准入已接通

2026-11-05T18:33Z，主接收控制器交接到V5（PID576431/start832762154），
64项新旧控制检查通过；4个当时活跃的repair2训练PID及启动时钟保留，
旧worker、已有修复/恢复通道、资源峰值保护和73任务并发上限不变。
仅新增此原生JPEG组的准入，不修改已运行模型的源码、环境、配方或进度。

独立方法准备器PID576845/start832774931已启动，等待完整856袋和2422295行。
就绪后核对全部完整JPEG收据，使用未削弱的原准备接口建立新任务和manifest，
验证原742/69/45划分、IDC0/ILC1、149张量原始BRCA教师和完整DINO/DSMIL输入。
它不直接训练；只提交一个新目录的请求，原调度器按真实资源余量准入。

配置沿用最早preflight已经登记的`promptmil_brca_dino_dsmil_paper_reference`：
40轮、AdamW1e-4/wd0.01、512分块、8切片梯度累积，保留原有命名的尾批flush/
VAL原始分数AUROC基准决定；没有切换为不同的50轮`source`条件，也没有临时缩短为软件测试预算。
512是全袋计算分块，不是截断实例。按最大完整袋的CPU张量堆叠、整袋GPU输入及
反向激活设置32GiB主机/32GiB显存预留；实际峰值仍待真实训练核验，
不通过降低patch数或网络宽度规避峰值。

V5独立核对完整队列、患者/类别、任务/manifest、实际像素协议、27项完整软件检查的
真实case IDs/XML/SHA、冻结两文件候选和原始权重；拒绝部分输入、换编码器/解码器/
配置、改轮次、缺乏峰值预留、重复运行和用户排除的方法。
来源：`scheduler_promptmil_controls_r1.xml`、`scheduler_v5_handoff_launch.json`、
`native_promptmil_method_preparer_launch.json`，状态：`native_promptmil_method_preparation.json`。
等待输入或成功排队仍不是实际训练开始/完成。
