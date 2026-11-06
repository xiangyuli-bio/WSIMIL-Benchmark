# FOCUS：完整预训练文本与切片训练路线

这是独立的 0.61 开发候选，尚未正式计数。正式主库目前仍以其已验收版本为准。不能把本目录的候选配置计入正式覆盖率。

来源为 `dddavid4real/FOCUS@66c4015d5ba09657f4c8183bc06947faecd5b01f`。依赖使用本次实际审阅的 `mahmoodlab/CONCH@141cc09c7d4ff33d8eda562bd75169b457f71a62`；FOCUS 原仓库没有给出 CONCH 的历史提交锁，本项目不声称还原了未公开的历史依赖版本。

## 使用

安装匹配设备的 PyTorch 和本项目的 `focus` extra 后，运行：

```bash
milplus setup --method FOCUS
milplus prepare-focus --data /absolute/task_bundle \
  --text-metadata /absolute/focus_text.json \
  --output /absolute/task_bundle/focus_manifest.json
```

把生成路径写入该数据包 `task.yaml` 的 `encoders.<fm>.focus.manifest` 后：

```bash
milplus check --data /absolute/task_bundle --method FOCUS --deep
milplus run --data /absolute/task_bundle --method FOCUS \
  --output /absolute/new_run --device cuda
milplus evaluate --run /absolute/new_run
```

权重由用户按原始条款自行提供，代码包不包含权重，也不自动请求任何受限模型的访问权限。当前已验证的原生文本权重是 FOCUS 作者 README 链接的公开 `david4real/FOCUS` 数据集中的 `conch.pth`，版本 `bc2c38dfbdf542b5c0616fd48ffaa68d0baf983b`，大小 802235437 字节，SHA256 为 `40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c`。FOCUS 的 Apache 许可证不替代 CONCH 及其权重的独立非商业条款。私有缓存不随安装包发布。

`focus_text.json` 示例：

```json
{
  "schema_version": 1,
  "kind": "focus_native_prompt_text",
  "synthetic": false,
  "classes": ["LUAD", "LUSC"],
  "checkpoint": "/absolute/private/conch.pth",
  "weights_sha256": "40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c",
  "weights_provenance": "Actual authorized checkpoint and its original source/version",
  "prompt_provenance": "Versioned fixed class descriptions, not patient-specific held-out labels",
  "feature_order_provenance": "Actual extraction order and version, preserved for all methods",
  "feature_preprocessing": "Actual before/after projection, normalization and extraction transform",
  "alignment_policy": "native_conch",
  "prompts": [
    {"role": "low", "class": "LUAD", "text": "Actual fixed low magnification LUAD description"},
    {"role": "low", "class": "LUSC", "text": "Actual fixed low magnification LUSC description"},
    {"role": "high", "class": "LUAD", "text": "Actual fixed high magnification LUAD description"},
    {"role": "high", "class": "LUSC", "text": "Actual fixed high magnification LUSC description"}
  ]
}
```

示例描述为占位内容，真实实验需固定实际类描述及来源。原生模式要求编码器登记 `model: conch_ViT-B-16`、完整 512 维、相同 `weights_sha256` 和明确一致的 `feature_preprocessing`。原始仓库没有充分说明图像特征是否取对比投影前后、是否归一化；登记应采用实际提取流程，不能替作者猜定。

换用其他 FM 时，声明 `alignment_policy: cross_fm_projection_extension` 和非空 `alignment_provenance`。保留完整输入 D→512 的可学习投影，文本仍为实际预训练 CONCH。该条件不声称新的 FM 与原文本已经语义对齐，也不能标为原生 CONCH 复现。

## 保留的实际方法

模型直接加载固定 Git 源码的 `FOCUS`、`PromptLearner`、`TextEncoder` 定义；私有加载 CONCH 的实际完整文本类与 BPE tokenizer。12 层、768 宽度、12 头、128 长度、512 输出和 85,547,520 个文本参数均保留且参与梯度更新。没有缓存最终文本向量或冻结其 Transformer。

原始提示处理用共享 16×768 上下文替换原始 token 的 1–16 位置，保存位置 0 前缀与 17 起后缀；保留两组共 2C 描述，但只用后 C 个高倍描述作为查询。FOCUS 的文本调用使用无 causal mask 的 Transformer、位置 0/BOS 池化及投影；不是 CONCH 标准的 causal/CLS `encode_text`。这些看起来不寻常的源码行为都予以保留。

视觉分支为完整 D→512/LN/ReLU/Dropout0.25。先在连续 8 行内按相似矩阵均值加有偏标准差判断冗余，再按各类别文本点积的平均相关性选取至多 8192 行并还原行顺序；随后在 8 行块内保留首行和相邻余弦小于 0.8 的行。8 头交叉注意力及分类 CE 完整保留。原源码没有使用物理坐标，也没有使用第二尺度输入；不能把这里的连续序列运算声称为真实坐标邻接或双尺度融合。

仅省去原工厂创建后从未被 FOCUS 保存或使用的 CONCH 视觉塔。所有实际文本权重严格加载。省去视觉塔的随机初始化会改变无用随机数消耗，因此不声称与原完整工厂具有逐种子的冷启动一致性；对照采用相同初始完整状态和 RNG。

## 训练、选模与恢复

每次一张完整切片，使用原始 RandomSampler 或显式选择的 `N/class_count` 加权有放回采样。保留作者三个启动脚本的 Adam 1e-4、weight decay 1e-5、200 轮上限与 early-stopping。main.py 的裸参数默认 lr 实际为 1e-3，虽帮助文字写 1e-4；本路线明确采用启动脚本值。采用用户已登记的 TRAIN/VAL/TEST 清单；若要复现作者 16-shot/10 次划分，必须提供其实际对应划分，不把合成或全队列实验声称为原 few-shot 结果。源码虽创建 ReduceLROnPlateau，但没有调用其 step，本路线保留常数学习率。SGD(momentum0.9) 为源码显式选项。

原 `validate` 实际传给 EarlyStopping 的是验证集切片错误率，而非注释所称的验证损失。选择第一个严格最优 VAL error；并列计入 patience，patience20 且 epoch_index>40 才停止，最早完成 42 轮。没有用测试集或 AUC 替代这个标准。外部患者融合如被任务显式要求，仅用于评价，不改变单切片训练或源验证选模。

检查点包含完整文本、提示、视觉参数、固定 token 前后缀、Adam 状态、训练采样次序、计数和所有 RNG。数据准备绑定权重、类描述、每张切片完整有序特征的内容哈希；类、提示、权重或行内容变化时拒绝静默续训。推理不使用未知病例标签参与特征选择。

## 验证边界

已通过 17 项不同专项检查（16 项核心/契约与 1 项完整续训），并额外复查最终 CLI 的混合输入拒绝。覆盖实际完整预训练文本张量、提示位置、原源码前向和全部梯度、原训练循环/优化器/RNG 对照、超过 8192 行的真实筛选分支、独立压缩计算、固定 VAL-error 选模及逐位续训。两轮统一入口训练、选中检查点重载与 TEST 预测也已通过。短轮数软件验证与原始 200 轮上限/最早 42 轮停止预算分别登记。

当前不宣称自然 BRCA/COAD、GPU 或论文指标复现；独立安装包原始预算、全库与全部配置验收状态由后续实际报告决定。
