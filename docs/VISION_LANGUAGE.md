# CATE 与 ViLa-MIL

本轮接入两个方法家族、三个执行配置。它们需要任务级文本资产；只有 WSI 特征目录不满足输入要求。正式运行沿用 `milplus run --data TASK --method METHOD --config CONFIG`，患者划分、验证选模、逐样本预测与断点续训均使用公共引擎。

## 原始来源与范围

| 配置 | 固定原始来源 | 保留的机制 |
|---|---|---|
| `cate` | [HKU-MedAI/CATE](https://github.com/HKU-MedAI/CATE/tree/a10fd45146849239ac0c052c45d76c02c81bba6b) | CIB 随机重参数化、top-10 概念引导 InfoNCE、KL、梯度截断的 CFI、DAttention、三项加权目标 |
| `vilamil_source` | [Jiangbo-Shi/ViLa-MIL](https://github.com/Jiangbo-Shi/ViLa-MIL/tree/68a11cf0d5cf092dd980f0da1cb38ccac8747a82) | 原公开代码的提示位置与可训练文本编码器、双尺度原型/上下文解码器、两尺度 logits 求和与 CE |
| `vilamil` | 同上，显式修正版 | 将 EOT 读取位置向后移 16；按论文冻结文本编码器，梯度仍通过编码器传到可学习提示词；其余解码器保持作者实现 |

CLIP 文本 Transformer 与 BPE tokenizer 使用 [OpenAI/CLIP 固定版本](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6)。不执行作者入口脚本，不下载模型权重；读取本地 `weights_only=True` tensor state dict。CATE 和 ViLa 根目录未发现许可证，源码在安装时单独取回；发行包不复制这些第三方实现。CLIP 为 MIT。

安装及准备源码：

```bash
python -m pip install -e '.[visionlanguage]'
milplus setup --method cate
milplus setup --method vilamil  # 同时取回固定的 OpenAI CLIP 源码
```

## CATE 数据与配方

原仓库的 `catemil.py` 默认聚合输入宽度为 512，CATE 拼接后是 1024。本适配根据实际概念宽度 D 使用 2D 输入，保留聚合器 512/128 内部层宽。原 `cate.py` 只有标签 0/1 分支，因此当前配置明确限二分类；不将其宣称为已经实现 RCC 三分类。

原始图像特征和文本概念必须属于同一 VLM 对齐空间。原生参考是 CONCH；换成未对齐的 UNI 特征，仅维度相同也不满足条件。`task.yaml` 的当前 encoder 增加 `embedding_space`，文本元数据声明相同标识及实际权重、预处理。可在各 encoder 内分别配置 `vision_language`，或使用任务级默认。

```yaml
vision_language:
  kind: concept_anchors
  metadata: text/text.json
encoders:
  conch:
    dimension: 512
    index: features/conch.csv
    embedding_space: conch_v1_image_text_WEIGHT_HASH
```

`text/text.json` 示例（文件路径可用任务目录相对路径或绝对路径）：

```json
{
  "classes": ["IDC", "ILC"],
  "synthetic": false,
  "model": "CONCH-v1",
  "weights": "ACTUAL_WEIGHT_VERSION_AND_HASH",
  "preprocessing": "ACTUAL_CONCH_TEXT_ENCODING_SETTINGS",
  "text_provenance": "fixed class descriptions, recorded before test evaluation",
  "embedding_space": "conch_v1_image_text_WEIGHT_HASH",
  "groups": [
    {"name": "IDC", "path": "text/idc.npy", "prompts": ["An image of invasive ductal carcinoma"]},
    {"name": "ILC", "path": "text/ilc.npy", "prompts": ["An image of invasive lobular carcinoma"]},
    {"name": "normal", "path": "text/normal.npy", "prompts": ["An image of normal breast tissue"]}
  ]
}
```

每个 `.npy` 为 `[概念数,D]`，每行都需登记对应原始 prompt。最前两组按任务类别排序；后续组作为 class-agnostic negatives。CFI 按作者代码使用前两组所有概念，InfoNCE 使用每组均值。不会无条件把一组概念压成一个向量。作者 `generate_concept_anchor.py` 生成的 tensor list 可转换：

```bash
milplus prepare-text --kind concept_anchors \
  --features /path/concept_tensor_list.pt \
  --metadata /path/descriptors.json --output /path/task/text
```

导出命令保留顺序、写入输入权重和产物哈希，不运行 CONCH。必须先用相应 CONCH 文本编码器生成真实概念。

[CATE 论文](https://arxiv.org/html/2411.09894v1) 给出 CE + λP·PIM + λS·SIM，Adam、学习率 1e-5、batch size 1。仓库没有完整 trainer；本适配补齐三项目标，**要求显式传入两个正权重**，不猜测作者默认值。示例 `configs/cate.yaml` 的 1.0/0.1 是工程起步值，正式实验需预注册或在验证侧选择。原 top-k=10 要求至少 10 个实例；过短 bag 会报错。梯度截断、原始点积、56.3477 系数与整体样本标准差按代码保留；标准差增加 1e-8 下限。保留训练重参数化，推理使用隔离 RNG 的一次固定种子抽样，不消费后续训练随机数。

## ViLa 双尺度及文本输入

图像两个尺度都必须真实存在，可以具有不同实例数。当前实现不要求逐 patch 对齐；作者 forward 没有使用坐标。原生 RN50 图像与文本输出维度均为 1024。两个尺度不能指向同一个文件，不能把同一张量复制后称为双尺度。

```yaml
vision_language:
  kind: learned_dual_scale_prompts
  metadata: text/text.json
encoders:
  clip_rn50:
    model: OpenAI-CLIP-RN50
    weights: ACTUAL_RN50_WEIGHT_HASH
    dimension: 1024
    index: low_features.csv
    dual_scale:
      high_index: high_features.csv
      low_field_of_view_um: 512
      high_field_of_view_um: 128
      provenance: ACTUAL_EXTRACTION_PROTOCOL
```

两张索引表均为 `slide_id,path`，必须覆盖完全相同的 slide。上面的物理视野仅说明配置形式，应填写自己的真实切块记录。任务 `classes` 顺序决定文本提示顺序。

先准备 `descriptors.json`：

```json
{
  "classes": ["LUAD", "LUSC"],
  "model": "OpenAI-CLIP-RN50",
  "weights": "ACTUAL_RN50_WEIGHT_HASH",
  "preprocessing": "OpenAI CLIP BPE; exact descriptors, no silent truncation",
  "text_provenance": "record source or LLM/model/prompt/version used for descriptions",
  "prompt_order": ["low", "high"],
  "prompts": ["LOW_LUAD_DESCRIPTION", "LOW_LUSC_DESCRIPTION", "HIGH_LUAD_DESCRIPTION", "HIGH_LUSC_DESCRIPTION"]
}
```

导出已合法取得的 RN50 tensor state dict（`.pt` 不是 TorchScript archive）：

```bash
milplus prepare-text --kind learned_dual_scale_prompts \
  --features /path/RN50_state_dict.pt \
  --metadata /path/descriptors.json --output /path/task/text
milplus check --data /path/task --method vilamil --deep
milplus run --data /path/task --method vilamil --config configs/vilamil.yaml
```

导出保留 RN50 文本权重、真实 BPE token ids、原始描述、权重与产物哈希。长描述若无法在 77 token 中同时容纳 16 个学习 token，会明确拒绝；需主动缩短并记录描述，不静默截掉 EOT。数据包加载时重新校验 token 与文本匹配。不会用缓存的静态文本向量替代需要梯度传播的 prompt learning。

[ViLa 论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ViLa-MIL_Dual-scale_Vision-Language_Multiple_Instance_Learning_for_Whole_Slide_Image_CVPR_2024_paper.pdf) 描述冻结文本编码器；公开 trainer 未冻结它。因此提供两个有明确命名和运行协议记录的配置。源码还在插入 16 个上下文后读取未移动的 EOT 下标；短文本会读到共享上下文，测试可复现类别特征相同。默认修正这个位置；`vilamil_source` 保留可比的旧行为，不能用于声称该问题已修复。原源码创建 ReduceLROnPlateau 却没有调用 step，适配维持恒定 Adam。选模使用 PULS 验证 CE，patience 20；原 stop_epoch=80 和 few-shot 抽样不自动套用，数据子集必须由任务清单明确冻结。

## 软件验证范围

`tests/test_vision_language.py` 覆盖 CATE 原生 512 维及其他对齐维度、两个类别的完整输出/目标/梯度对照；ViLa 二/三分类的作者输出和梯度对照；EOT 位置、文本冻结、上下文梯度和单原型；三配置精确续训、模型/优化器状态与预测重载；文本身份/维度/权重/哈希和独立尺度的拒绝路径；原始 CLIP BPE tokenization。

```bash
milplus demo --vision-language concept_anchors --output /tmp/cate_demo
milplus run --data /tmp/cate_demo --method cate --config configs/cate.yaml
milplus demo --vision-language learned_dual_scale_prompts --output /tmp/vila_demo
milplus run --data /tmp/vila_demo --method vilamil --config configs/vilamil_demo.yaml
```

这些演示使用明确标记的随机合成特征和小型随机 CLIP 文本网络，仅验证软件。真实 RN50/CONCH 预训练权重、GPU、真实 BRCA/COAD 和论文指标复现仍未运行。

## BRCA 专项的原作者 CATE 文本资产（2026-11-06）

20:57:12Z，专项目录从固定作者 `src/prompts.py::brca_prompts` 找回了确切的 IDC/ILC 提示，
无需另造临床描述。按原 `generate_concept_anchor.py` 字面循环保留六组
IDC、ILC、脂肪细胞、结缔组织、坏死组织、正常乳腺组织，分别 110/110/110/88/44/66 行，共 528 条。
全部使用实际 CONCH1.0 原始权重 SHA256 `40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c`，
完整 150 个文本张量严格加载、原生 causal/padding/CLS 投影/L2，CPU FP32，没有使用患者标签。
原样作者导出循环在独立完整 factory 和仅省去无关组件的文本塔上逐条输出逐位一致；
现有 `prepare-text` 和 `VisionLanguageData` 实际导出/回读也完全相同。

资产在 GPU 工作区 `cate_original_brca_text_assets_20261106_r1/text/text.json`，
完整输入 tensor list SHA256 `b8451f1397fb4994a7e31ab1b5ed4399bba1d7e671f2fa859c930fa91b5575ee`。
共享凭据为 `cate_original_brca_text_verified_r1.json`。
导出器的冻结相对路径 helper 将运行元数据落在其 GPU 本地源码目录；
完成后仅把两份收据原字节镜像到共享目录，来源/目标/SHA另记 `cate_author_text_receipt_transport_r1.json`。
没有为修正元数据位置重做计算或改变已完成资产。
全部文本资产另以原字节备份到专项共享目录 `cate_original_brca_text_assets_backup_r1`，
11个文件约2.25MB，逐文件SHA核对；其清单的原绝对路径未被暗改成备份路径。

该项只是文本输入完成，不是 CATE 临床训练或全方法验收。
全部 856 袋 CONCH 图像流尚在生成，不能用 UNI 特征替代。
重新核对[论文附录 C.1](https://arxiv.org/html/2411.09894v1)：其公开了 PIM/SIM 权重消融，
正文讨论 SIM 约 30 的效果，但这不明确等同于主实验唯一权重对。
示例配置 1.0/0.1 仍只是工程示例，不冒充作者默认值，实际训练需明确实验选择。
