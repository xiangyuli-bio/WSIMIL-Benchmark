# TTIS：测试时实例选择

作者来源为 [QuIIL/TTIS](https://github.com/QuIIL/TTIS)，固定提交
`ceb08d3bd8bc795845358850e314132b6231a921`。一条方法记录包含七个配置：

| 配置 | 原生分类头与训练目标 |
| --- | --- |
| `ttis_abmil_source` | ABMIL，作者 TTIS 训练入口的 bag CE |
| `ttis_clam_source` | CLAM-SB，作者 TTIS 入口重新计算的 bag CE |
| `ttis_rrt_source` | RRTMIL，bag CE |
| `ttis_s4_source` | S4MIL，bag CE |
| `ttis_transmil_source` | TransMIL，bag CE |
| `ttis_wikg_source` | WIKG，bag CE |
| `ttis_clam_full_loss` | 显式变体：优化 CLAM 返回的 bag/instance 联合损失 |

`TTIS` 默认指向 `ttis_abmil_source`。所有头、两阶段聚类函数、FPS、视图融合、
Lookahead 和优化器工厂均从单独获取的固定 Git 对象加载，不用同名简化模型替代。
WIKG 配置还需要图依赖，在仓库根目录执行 `python -m pip install -e '.[graph]'`。
源仓库没有根许可证，内嵌 MUSK 另有 CC-BY-NC-ND-4.0 条款；作者源码与权重不随本库分发。

## 输入与准备

任务数据包须提供冻结 patch 特征、分类标签、固定患者划分和真实坐标。
坐标必须与特征行一一对应，单位为 level-0 pixels 或 micrometers；不接受生成的伪网格。
同一 FM 的聚类清单仅覆盖当前 TEST 切片。TRAIN、VAL 始终使用完整 bag。

还需登记冻结语义资源。以下 JSON 展示格式，数值和路径必须来自实际导出：

```json
{
  "schema_version": 1,
  "synthetic": false,
  "classes": ["negative", "positive"],
  "model": "MUSK with the actual weight revision",
  "weights_sha256": "<64 lowercase hex characters of the actual weight file>",
  "embedding_space": "<actual text embedding space>",
  "visual_embedding_space": "<actual visual embedding space>",
  "provenance": "<export script, model revision, preprocessing and prompt source>",
  "dimension": 1024,
  "text_path": "/absolute/path/to/normalized_text_vectors.npy",
  "prompts": ["<exact negative prompt>", "<exact positive prompt>"],
  "prompt_class_indices": [0, 1],
  "logit_scale": 2.65926
}
```

`text_path` 为 `[prompts, dimension]` 的归一化浮点向量。`logit_scale` 是作者
`Musk.zeroshot` 实际使用的乘数，须记录实际导出值；不擅自添加指数运算。
每个语义列产生一个视图；多条同类提示保留独立列及映射，不提前平均。
合成演示资源有明确的 `synthetic: true`，不能登记为真实 MUSK 输出。

```bash
milplus setup --method TTIS
milplus prepare-ttis --data /path/to/task_bundle \
  --semantic-metadata /path/to/semantic.json \
  --cluster-backend cuml --output /path/to/new_ttis_assets
```

默认预算为 256 个簇：先按真实坐标做 16 个空间簇，再在各空间簇内对特征做至多
16 个子簇。原生后端需要 CUDA、cuML 和 CuPy，使用 k-means||；不可用时直接报错。
显式 `--cluster-backend sklearn_cpu` 使用同一两阶段函数及 k-means++，属于已命名的
CPU 适配，不能把其结果声称为 cuML 逐位复现。

准备命令输出 manifest 路径，将其写入任务的对应编码器配置：

```yaml
encoders:
  uni_v1:
    # 保留原特征索引、维度和权重信息
    ttis:
      manifest: /path/to/new_ttis_assets/manifest.json
```

## 两种语义空间政策

默认 `source_primary` 保留作者源码的操作：直接归一化主 FM 特征并与 MUSK 文本向量点乘。
作者 UNI 配置没有额外的 MUSK 图像流或对齐投影；1024 维相同不证明空间对齐。
该行为会记录在运行协议，不能以此宣称任意 FM 与 MUSK 语义空间兼容。

跨 FM 使用已对齐的语义引导时，另外准备真正的 MUSK 图像特征，CSV 包含
`slide_id,path`，每个 H5 保留与主特征完全一致的原始坐标和行顺序：

```bash
milplus prepare-ttis --data /path/to/task_bundle \
  --semantic-metadata /path/to/semantic.json \
  --guidance-index /path/to/musk_test_features.csv \
  --cluster-backend cuml --output /path/to/new_aligned_assets
```

此时政策为 `aligned_secondary`，必须登记相同的图像/文本 embedding space。
聚类仍使用主 FM 特征；额外语义视觉流仅决定语义视图起点。其额外提取成本应单列。
manifest、文本和语义登记文件纳入运行签名；特征、坐标、次级视觉流分别绑定内容哈希。
更改资源后不能复用原 checkpoint 的输入签名。

## 训练、推理和恢复

```bash
milplus run --data /path/to/task_bundle --method ttis_abmil_source \
  --config configs/ttis.yaml --output /path/to/new_run
```

原始配方：batch size 1、随机打乱整袋、50 epoch，Lookahead/AdamW，学习率 1e-4、
eps 1e-10、weight decay .001；bias 和一维参数不衰减。Lookahead alpha=.5、k=6，
慢权重在第六步首次建立，非第零步。每轮 cosine 调度至 1e-7，周期 50。
按 VAL 的逐切片 bag CE 严格改善选模，默认 patience=10；测试标签不参与选模或视图生成。
原配置主特征为 1024、隐藏 512，ABMIL/CLAM 注意力宽度为 384；其他头保留各自原生结构。
主 FM 宽度从数据包读取，
RRT 原生投影仍要求 512 隐藏宽度。

只有 TEST 启用 TTIS。每个簇保留原来的 .3 FPS 比例，分别从质心、注意力
（S4 为特征范数）及每列语义分数起点采样；退化相同点时保留作者重复索引行为。
多个视图按特征 L2 范数的 softmax 加权，然后通过原分类头，不替换成概率平均。
CLAM 推理仅向未使用的辅助损失提供常数占位标签，实际 TEST 标签不进入网络。
导出的 H5 保存原始坐标和各视图的原始行索引，不把选择索引称为注意力或因果解释。

checkpoint 包含完整模型、Adam 状态、Lookahead 慢权重、调度器、更新计数、随机状态
及输入合同。CPU 路线显式采用 FP32；作者 BF16/GPU 设置需另行验证。

## 验证范围

```bash
python scripts/validate_ttis.py --device cpu --output /path/to/new_native_cpu
python scripts/validate_ttis.py --device cuda --output /path/to/new_native_cuda
```

专项检查覆盖七配置原生输入/宽度的前向、完整梯度、选择索引、独立源码训练入口、
逐次 Adam/Lookahead 更新和 cosine 状态，以及跨同步边界恢复、重复 CSV、TEST 标签独立性、
真实两阶段 CPU 聚类、资源哈希和坐标错配拒绝。安装包验证另外保留原生宽度训练证据。
具体结果以 `docs/validation` 当前版本记录为准。CPU 合成检查不代表自然 WSI、
GPU 或论文数值复现；天然 MUSK 权重与实际患者数据均需用户的数据资源。
