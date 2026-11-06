# PAMoE：组织先验、专家路由与患者批训练

原始来源为 [PAMoE 作者仓库](https://github.com/wjx-error/PAMoE)，固定提交 `7a26f4cc842d2875cfdee33e77de1495cae9dabe`，对应 CVPR 2025 的 *Learning Heterogeneous Tissues with Mixture of Experts for Gigapixel Whole Slide Images*。代码在独立缓存中从已验证的 Git blob 读取，不打包重新分发；未发现根目录许可证。

## 四种配置与两种任务

| ID | 骨干 | 专家索引 |
|---|---|---|
| `pamoe_transmil`，别名 `pamoe` | 作者 Nyström TransMIL | 修正患者/专家索引 |
| `pamoe_transmil_source` | 同上 | 保留公开源码索引 |
| `pamoe_transformer` | 作者四层 Transformer | 修正患者/专家索引 |
| `pamoe_transformer_source` | 同上 | 保留公开源码索引 |

四个配置属于同一方法家族，均接入分类和患者生存。虽然历史候选台账将 PAMoE 归入 F 组，这里的公开方法只需要图像特征与组织原型，不需要患者组学输入。生存任务按患者合并多张切片，提供 time/event；分类按任务数据包的已声明单位运行。

输入维度从 FM 数据包取得，原作者为 UNI 1024 维；隐藏宽度默认 512，其他宽度是单独的实验设置。默认 4 个组织原型专家 + 2 个额外专家，capacity factor=2，保留前 20000 个训练 patch、评价使用全部 patch。真实组织坐标用于检查原型的标签顺序；TransMIL 自身采用作者的补齐方形序列与 PPEG，不应称为真实组织空间重建。

## 组织原型必须与 FM 一致

作者流程先用 CONCH 对 patch 做组织分类，再在对应的 UNI 特征空间内按组织类别求均值。CONCH 的标签与 UNI 的特征必须指向同一 patch。不能把 CONCH 嵌入直接当成 UNI 原型，也不能将 UNI 原型直接用于另一个 FM。

当前实现提供均值生成器，不自动下载或执行 CONCH 分类器。需要准备每张训练切片的整数组织类别 NPY，以及 CSV：

```text
slide_id,path,sha256,coords_sha256
slide_a,labels/slide_a.npy,<标签文件 SHA256>,<规范化有序坐标 SHA256>
```

每个标签数组是 `[N]` 整数，类别从 0 开始；N 与特征行数一致。坐标摘要按 `milplus.io.digest_object(Bundle(...).load(row, require_coords=True)[1].tolist())` 计算。标签和坐标摘要应在生成标签时登记，不能事后用不明顺序的特征重新填入。CSV 必须恰好覆盖当前划分的训练切片。真实组织类别名称、分类器权重、提示词和处理协议写入 provenance。

```bash
milplus prepare-tissues --data /path/to/task \
  --assignments /path/to/tissue_index.csv \
  --output /path/to/task/assets/pamoe_uni_fold0 \
  --names tumor stroma inflammatory necrosis \
  --provenance '实际分类器/权重/提示词/处理协议的版本标识'
```

以上类别名仅为数据格式示例，须与实际分类器对应。生成器遍历全部训练切片，使用 float64 累加再保存 float32 均值；拒绝缺失类别、缺失标签和错位坐标。作者原脚本则随机打乱切片，在追加完整一袋后以 `cnt > 500000` 停止，因此保留末袋并可能超过 500000；它还会静默跳过缺失文件。这些差异保存在 manifest 中。

生成后在 `task.yaml` 对应 encoder 中加入：

```yaml
encoders:
  uni:
    # 保留已有 dim、index、权重及预处理登记
    pamoe:
      manifest: assets/pamoe_uni_fold0/manifest.json
```

Manifest 包含 tensor hash、encoder/preprocessing identity、组织名称、训练患者 ID、逐切片特征和标签摘要以及 synthetic 标记。训练特征更新后必须重建原型；更换 FM、切分或原型时恢复训练会被拒绝。测试辅助函数只生成带 synthetic 标记的模拟标签，不能用于真实数据。

外部原型也可手工登记 `kind: external`，提供同样的 encoder identity、tensor/hash、names、provenance。必须明确 `fit_scope: external_disjoint` 并列出与本数据包不重叠的全局患者 ID，或显式使用 `external_unverified` 承认拟合范围未知。身份声明仍依赖数据提供者的真实性；后者不能宣称已排除测试暴露。作者附带的 `.pt` 原型未自动纳入，其训练范围和 FM 版本需另行核实。

## BRCA 专项原生组织资产流程（2026-11-06）

专项运行目录新增独立资产准备流程，未修改上述通用生成器、方法源码或正在运行的冻结环境。
执行作者 `classification_CONCH.py` 的原始 `run_zeroshot_wogt` / `compute_w_loader`：
实际 CONCH1.0 权重、`force_image_size=224`、batch64、FP32，保留原始 JSON 的九类顺序
`TUM,STR,LYM,DEB,ADI,BACK,MUC,MUS,BLD` 和 40 个别名 × 22 个模板；ID≥4 按源代码归为 Other4。
公开提示词含结直肠相关表述，未擅自改写成乳腺提示词，不声称专家标注或组织分类临床准确率。
这不是正在生成的 448 像素 CONCH 特征流，不能混用两者。

实际 GPU 核验包括：完整原始 factory 与独立视觉加载器的状态/真实图块输出逐位一致，
全部 880 条提示词生成的分类权重逐位一致，以及原作者分类函数的真实 65 行 batch64+tail1 输出一致。
GPU 容器未挂载原始 WSI，首次原图比较失败记录保留；在现有原图节点执行未改的原作者 Dataset 类，
无损传输首/中/末三个实际 RGB patch 后，与 GPU 缓存像素及官方 224 处理张量逐位比较通过。
证据为专项共享目录 `pamoe_native_tissue_classifier_verified_r2.json`，不覆盖 R1 失败证据。

原型拟合只使用既有 742 个 TRAIN 患者的原始 UNI H5 软链接视图，捕获实际目录顺序并显式 seed2026，
按作者整袋前缀选中 168 张、500974 行；没有历史随机顺序复现声明。
逐袋产生真实组织分类后，将执行原样 `make_prototypes_classifier.py` 的 float32 四类均值，
不是通用生成器的全 TRAIN / float64 条件。Other4 仅在原型均值拟合时排除，不能删掉训练袋中的这些 patch。
要求四类非空、完整坐标/patch ID/SHA/训练患者账本；缺失、错位或非有限值直接保留失败，不制造替代原型。
最终 manifest 使用现有 `kind:tissue_mean` / `fit_scope:training_patients` 的真实 TRAIN 子集契约。
截至 2026-11-05T19:30Z，组织标签仍在生成；这不是原型全部就绪或 PAMoE 已完成训练。

已有公开源代码包含 subtype / CE 分支，本次沿用既有注册 `pamoe_transmil`，不是将那 25 个任务不匹配方法
改造成分类器。只在完整 856 UNI 输入、真实原型和训练范围核验通过后，才由同一队列接收默认原生训练；
无额外 epoch/model/prompt 变更。UNI 历史权重版本未知、公开实验的倍率/患者队列不同等限制仍保留。
实时资产状态为 `native_pamoe_tissue_preparation.json`，提交状态为 `native_pamoe_scheduler_submission.json`。

### 真实 TRAIN 原型完成并准入

2026-11-05T20:51:45Z，168 张选定 TRAIN 切片的全部 500974 行完成原样组织分类，
原作者 float32 四类均值已实际计算并通过完整 856 人 UNI 输入契约。
四个原型 `TUM/STR/LYM/DEB` 分别来自 33031/189166/24252/20353 行；
234172 行 Other4 仅不参与原型均值，后续完整 MIL bag 不删这些行。
没有使用 VAL/TEST 拟合原型，742/69/45 划分和全部 2422295 行 UNI 输入保持。
原型 NPY SHA256 为 `40c3e10dee2bdc1fd4b4ae91680407943b68ae3b4b6b2dc9bd2c1475bf5214e2`。

20:51:51Z，现有 V8 调度器接受 `pamoe_native_trainpriors1_20261106.json`。
此时状态仅 `ready_for_training_attempt`，并非已开始或完成分类训练。
保留既有 `pamoe_transmil`、100 轮上限及原 VAL 停止规则，不修改科学配方；
主机/GPU 预留 32/64 GiB 是保守准入值，不宣称已实测临床训练峰值。
证据为 `verified_input_intake_receipts/pamoe_native_trainpriors1_20261106.json`，
以及 GPU 工作区 `inputs_native_pamoe_uni_tissues/native_contract_receipt.json`
（SHA256 `417b46d7fbf20b19c35c33ece504361aa9aa9f5e8743caefe1a95c8772bac7d5`）。
原型、协议、计划和全部168份组织分类原字节已备份到专项共享目录
`pamoe_native_prior_assets_backup_r1`，173个文件约8.86MB，逐文件SHA核对。
没有复制742份特征；备份中的原清单仍绑定GPU原路径，恢复使用须按原路径恢复或显式重建版本。

## 训练机制与明确差异

- 保留作者的 expert-choice top-k 路由、FFN 专家、组织先验 CE 和骨干结构。先验目标为每个组织原型的余弦相似度在 **整个批次 B×T** 上的 softmax；过滤后目标不再归一化。路由器接收其梯度。
- 作者将 `[B,E,K,D]` 按 B-major 展平，但之后以 expert-major 切片，B>1 时混用患者/专家输入。`_source` 保留该行为；其他配置显式按 `[:, expert]` 取值。B=1 时两者一致。修正版与手工按患者/专家循环计算的结果及输入梯度对照。
- 输入按批次零填充，没有 mask；保留作者的零值剔除与再填充，预测可能受填充影响。默认评价一次只预测一个 bag；不宣称跨批次组合不变性。
- 分类使用交叉熵；生存使用作者累计风险 Cox 损失，加 `0.01 × prior_loss`。此 Cox 不是 Breslow/Efron ties 实现：并列时间的顺序可能影响损失，显式记录并测试，不悄悄换目标。
- 生存采样器沿用作者按事件条件抽样、无放回、完整批次路线，剩余删失患者或不完整批次不使用。分类采用随机顺序完整批次。默认 batch size=8；训练患者不足一个完整批次直接报错。
- AdamW 默认 lr=0.0002、weight decay=0.01，每批一次更新。源码的 gc=2 和 lambda_reg=0.0001 参数没有进入实际更新计算，因此不虚构累积或 L1 项。
- 保留 CosineAnnealingLR `T_max=5, eta_min=0.00001`，超过第 5 epoch 后学习率会回升。默认训练 100 epochs、patience=5，最早在第 26 epoch 停止；模型只按 validation 选择。CLI `evaluate` 才评价 test。
- 原型注册为不可训练 buffer，进入 checkpoint；优化器、scheduler、随机数、已采样患者和更新计数支持精确恢复。作者使用普通属性加载原型，适配器不在每次运行时不受控地 pickle 加载它。
- 当前验证为 FP32 单设备 CPU。没有证明 AMP/DataParallel 或 GPU 分片下先验行为一致。非有限数值直接报错，避免沿用源码跳过病例或 epoch 后掩盖失败。

## 运行

```bash
python -m pip install -e '.[prompt]'
milplus setup --method pamoe
milplus check --data /path/to/task --method pamoe --deep
milplus run --data /path/to/task --method pamoe \
  --config configs/pamoe.yaml --output runs/pamoe
milplus evaluate --run runs/pamoe
```

切换到公开源码索引用 `--method pamoe_transmil_source`；换 Transformer 用相应 ID，所有差异进入运行协议。Transformer 可显式提供 `layer_types: [pamoe, ffn, pamoe, ffn]`，TransMIL 固定两层，不接受此参数。

验证包括独立读取固定作者完整训练循环的 8 组六 epoch 对照（两任务 × 四配置），逐步比较梯度、参数、AdamW 状态与 scheduler；原生 1024/512 网络对照；八组精确续训；组织先验和索引手工公式、训练原型范围、输入哈希及 CLI 检查。最终机器记录见 `validation/pamoe.json`。全部是 CPU 合成软件验证；真实 BRCA/COAD、CUDA 和论文数值复现仍未执行。
