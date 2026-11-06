# CoD-MIL：真实双尺度、诊断文本和完整训练配方

0.60 隔离候选，尚未正式计入方法数。执行配置 `codmil_source_repaired`，
别名 `CoD-MIL`。这是一条完整的作者源码修正路线，不是只加载一个注意力模型类。

固定来源：[Jiangbo-Shi/CoD-MIL](https://github.com/Jiangbo-Shi/CoD-MIL)，
`0fd748dc91ffd889bf13dbffd4db633b158809d6`。
[论文](https://doi.org/10.1109/TMI.2024.3485120)。源码未发现根目录许可证，
实现由用户私下 `setup` 获取，不将作者模型或训练代码复制进发行包。

## 方法实际执行什么

保留作者 `models/model_CoT.py` 的完整机制：

1. 低倍率实例经过门控注意力，产生逐实例加权特征；并非先平均成一个向量。
2. 原始文本特征经过两层可学习 MLP；低倍率每类诊断文本与加权实例相乘，求和得到低倍率 logits。
3. 根据当前模型预测类别的文本相关性，屏蔽最高10%的低倍率实例；剩余实例与类别及正常组织文本
   竞争，计算作者的辅助交叉熵。不是用真实标签挑选预测区域。
4. 同一相关性选前16个低倍率区域，通过真实映射定位高倍率 feature 行；保留源码可能产生的重复行。
5. 高倍率实例经过同一个门控注意力，与高倍率每类诊断文本比较；高低倍率 logits 相加。
6. 训练损失为主分类 CE + 1×掩码正常文本竞争 CE。验证选模也使用完整总损失。

作者声明的普通分类器和自注意力头在该路由中没有被调用。它们的状态保留，
但不虚构额外损失、梯度或实际使用的 Transformer 聚合。
整个 patch FM 冻结；文本的 MLP 与实际聚合器参与训练。提供的文本 embedding 本身固定。

## 原始版本问题和明确修正

对固定提交的实际执行表明：

- 肾癌三分类启动脚本与模型固定二分类的文本切片不一致；类别2会使原始 CE 失败。
  本配置将低/高倍率切片明确扩展为 C 和2C，并保存有序类别定义。
- 原始模型固定输入1024，而提供的 PLIP/QuiltNet 文本 tensor 是512维。
  本配置分别登记完整视觉 D 和原始文本 T，用 T→D→D 的可学习 MLP。
  D=1024、T=1024、C=2 时，实际作者输出与所有有效输入/参数梯度逐位一致。
- 外部硬编码映射路径改为已校验的物理资产；不再通过 `rstrip('.tif')` 猜测资源名。
- 单个低倍率实例时将布尔 mask 保持为一维，避免无意义的维度消失。
- 未使用的旧 `_LinearWithBias` 采用现代 `nn.Linear` 等价类；不借此替换实际聚合机制。
- 训练只使用训练集，选模只使用验证集；测试需显式 `evaluate`，不用于选超参数或 checkpoint。

以上是固定代码的观察和命名修正，不能据此断言历史论文实验有同样问题。
源码的最后一个文本行被排除在正常文本竞争之外，本配置保留该行为并要求显式
`excluded_last` 角色，避免偷偷改变背景损失定义。

## 数据包准备

标准任务数据包仍然提供患者、slide、标签、固定划分和主特征索引。
主特征是低倍率；必须保留与行序一致的真实坐标。另准备以下 CSV：

```csv
slide_id,fine_path,fine_coords_path,map_path
slide_a,features_20x/slide_a.h5,features_20x/slide_a.h5,maps/slide_a.npy
```

必须覆盖全部注册 slide，每个高倍率 feature 和 map 使用独立文件。高倍率 H5 有
`features`、`coords`；独立 NPY/PT 也可用。map 为 `[低倍率实例数,K]` 整数矩阵：
每个值是对应的实际高倍率 feature 行号，缺失位置写 `-1`。不得复制低倍率特征冒充另一倍率。

物理 schema 示例，尺寸单位与坐标单位相同：

```json
{
  "schema_version": 1,
  "kind": "codmil_physical_chain_prompts",
  "synthetic": false,
  "coordinate_unit": "level0_pixels",
  "origin": "top_left",
  "coarse_patch_size": 512,
  "fine_patch_size": 256,
  "mapping_provenance": "实际10x/20x共同level0坐标与行映射版本",
  "fine_extraction_provenance": "真实高倍率提取器、权重和预处理版本"
}
```

`encoder.spatial` 的 unit/origin/patch_size 必须与低倍率 schema 一致，含非空 provenance。
系统检查坐标行数、唯一性、高倍率 patch 的物理包含关系、映射范围和同一父区域内的重复 child。
这里是组织 patch 区域，不要求细胞级注释。重叠低倍率区域可指向同一真实高倍率 patch，
原始索引产生的重复权重不会被暗中去除。

文本 metadata 的 `kind` 是 `codmil_ordered_text_embeddings`，`schema_version:1`，
包含与任务一致的 `classes`、`synthetic`、原始 `dimension`、`features` 路径、模型、权重、
预处理、语义空间与文本来源。`prompts` 必须按以下顺序逐行记录真实文字：

1. C 个 `role:low`，每条 `class` 与任务类别顺序一致。
2. C 个 `role:high`，同样的类别顺序。
3. 至少一个 `role:normal` 的正常组织概念。
4. 一个 `role:excluded_last`，保存原始被排除的最后一行语义。

`alignment_policy:matched_vlm` 要求视觉/文本模型、权重、维度和语义空间一致。
替换为其他 FM 时，必须显式选 `learned_projection_extension` 并提供
`alignment_provenance`；不能把任意 FM 和文本称为天然对齐。这种输入扩展单独汇报。
系统不调用外部服务生成诊断文本，也不从测试标签生成案例专属提示。

```bash
milplus setup --method CoD-MIL
milplus prepare-codmil --data /path/to/task \
  --index codmil/index.csv --schema codmil/schema.json \
  --text-metadata codmil/text.json --output /path/to/task/codmil/manifest.json
```

将输出路径登记为该 FM 的 `encoder.codmil.manifest`。准备器绑定真实文件字节、完整行数、
坐标、schema、文本及 source index；不会复制大数据，也不会自动生成第二倍率。
之后只需：

```bash
milplus run --data /path/to/task --method CoD-MIL --config configs/codmil.yaml --output /path/to/new_run
milplus evaluate --run /path/to/new_run
```

## 完整训练和恢复

默认 source 配方：最多200epoch，Adam lr1e-4、weight_decay1e-5，每批一张slide；
默认实际 `RandomSampler` 完整遍历训练集，也可显式使用作者的 N/class_count 加权重采样。
保留固定验证 DataLoader 对 RNG 的实际消耗。workers=0 是确定性读取选择，没有额外 feature 增强。
源码虽声明 plateau scheduler，却没有调用其 step；本配置保留常数学习率，不冒充实际变更的调度。

以完整 VAL 总损失首次严格改善保存最佳模型；patience20，只有 epoch_index>50 才能提前结束。
这与源码 CoD 分支一致，transFG 的80条件不属于这里。关闭 early_stopping 时按作者选择最后一轮。
Checkpoint 包括所有模型参数、固定文本、Adam状态、每轮训练顺序、计数和全部 RNG；
对不一致的输入、文本、非训练集实例身份或计数明确拒绝。

默认不截断 bag、不额外 clipping。原生线性图和加载规模超过明确的 `max_dense_bytes` 时提前拒绝，
用户可根据 GPU 内存显式调整，而不会暗中换成小维度模型。

## 验证状态

本配置尚待正式验收。已完成端到端合成试跑；实际原始1024二分类源码输出和所有梯度，
1024/1536完整输入、多类别与不同 T 的所有分支，以及完整作者 train_loop 的
Adam参数/状态/训练顺序/RNG 对照正在进入专项与全库验收。
合成验证不代表自然 BRCA/COAD、CUDA或论文数值复现。
