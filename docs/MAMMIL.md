# MamMIL：拓扑序列、Mamba2/SSD 和完整训练

固定来源：[作者仓库](https://github.com/Vison307/MamMIL)，提交 `64736e1a06a23e6814dfee664ff02b27c72dfedb`；[论文](https://arxiv.org/abs/2403.05160v3)。根许可证 GPL-3.0；内含 Mamba 和其他组件保留其原始署名。源码单独获取，通过固定 Git 对象的指定定义加载，不在本包中重新分发或重新许可。

## 三个配置及来源差异

| 配置 | 层循环 | 任务和输入 |
|---|---|---|
| `mammil_source` | 字面发布的 zip(layers,ppeg) | 切片分类、每患者恰好一张 WSI 的离散生存 |
| `mammil` | 显式修正为执行全部扫描层 | 同上；家族名 MamMIL 默认指向此配置 |
| `mammil_patient_union` | 修正循环 | 多切片患者生存扩展，按 slide_id 拼接互不连通的图 |

发布源码创建两层扫描和一层 GENConv，forward 对两列表 zip，默认只执行第一层扫描。`source` 保留该行为，包括最后扫描层没有梯度；修正版执行两层，在层间保留原 GENConv/DeepGCN 图交互。它们属于同一个方法家族。原作者生存数据层要求每患者一张 WSI；多切片扩展单独命名，不冒充原生流程。扩展中各切片保持独立坐标框架、边和遍历，边索引加偏移，禁止跨片连边。

输入维度从 FM 特征登记读取，原生隐藏维度512、两层、状态维度128、卷积宽度4、expand=2、head_dim=64。每层包含四种排列（原序、DFS前序、DFS后序、BFS）各自的正反扫描，恢复实例顺序后平均，再经过原输出投影。保留原初始化、共享投影、分方向卷积/时间参数、RMSNorm和GENConv。分类为原 CE；生存为原截断 NLL、四区间、alpha=0。

## 图输入和冻结资产

必须提供与每行特征严格对应的真实坐标。读取作者 `h5toPyG.ipynb` 的 HNSW、余弦距离和图构建定义：坐标8NN图、特征latent 8NN、余弦边权、最小生成森林、DFS前序/后序以及每连通分量随机根的BFS。默认 M=16、post=0、efConstruction=400、查询ef=90。显式改成单线程建索引、固定 graph_seed=2024，记录与作者并行建图环境的差异。实例少于9时邻居数上限截到N，单实例图允许空边；重复坐标拒绝处理。

每张切片的图保存为内容寻址 NPZ。键包含特征/坐标内容、图生成代码、来源提交、参数、nmslib/networkx/PyG版本；六个数组另有内容校验和。首次准备允许建图；训练读取、断点恢复和评价要求已有图与保存的manifest完全一致，缺失或损坏会失败，不静默重建。HNSW 不承诺跨任意构建环境逐位一致，因此**图资产必须与 checkpoint 一起保留、备份或转移**。迁移后通过原配置的 `model.graph_cache` 指向完整资产目录。

可以先在 CPU 开发机准备，避免占 GPU 做预处理：

```bash
milplus setup --method mammil
pip install -e '.[topology]'
milplus check --data /path/to/task --method mammil --deep
milplus prepare-mammil-graphs --data /path/to/task --graph-cache /persistent/mammil-graphs --output /path/to/new_graph_manifest.json
```

配置 `model.graph_cache: /persistent/mammil-graphs` 后，训练使用相同参数。默认缓存位于数据包的 `.milplus-cache/mammil`。latent图和边权保留在资产中，但作者网络实际只消费空间边和三种遍历。每片构图只使用自身特征与坐标，不使用标签或队列拟合。拒绝 `max_instances` 随机截断，避免破坏图与排列的一致性。

## 数值后端

默认 `model.backend: torch` 明确实现作者 **Mamba2/SSD** 方程：head共享状态、dt softplus/bias、深度因果卷积、分组 B/C、skip D、门控RMSNorm和输出投影；使用分块梯度检查点，跨块完整状态可微传递，无梯度截断。没有用 Mamba1 或普通 Mamba 代替。PULS 的 FP32 PyTorch 路径便于软件验证，长包性能必须实测。

`model.backend: cuda` 明确要求 GPU、从固定作者 `mamba/` 构建的 mamba-ssm==2.2.2、causal-conv1d==1.4.0 和 Triton；缺失时失败，没有自动回退。记录 Torch/CUDA、依赖版本、Mamba Python源码和causal二进制的内容身份；checkpoint恢复拒绝后端变化。版本检查不能证明包的构建来源，仍需要保留环境构建记录。作者历史 Torch2.1.2/PyG2.5.2 与本次 CPU Torch2.5.1/PyG2.6.1 不同。当前尚无融合 CUDA 验证，不能把本次结果视为生产 GPU 环境已通过。

在实际准备好的GPU环境中，可执行融合内核准入检查：

```bash
python scripts/validate_mammil_cuda.py --output /path/to/new_cuda_check.json
```

它对照源码/修正层循环、多个包长、原生隐藏宽度的完整网络输出与参数梯度。没有GPU时记录 `not_run` 并返回非零；不把缺少设备视为通过。本次CPU环境的实际结果为 `not_run`。

## 完整配方与运行

作者自定义 RAdam（不替换成 torch.optim.RAdam），lr=1e-4，weight_decay=.05；bias及一维参数不衰减，每包一次更新，250 epochs、patience20；分类验证log_loss，生存验证C-index。默认没有裁剪或调度。原参数分组与优化器状态参与逐位更新对照。统一库使用可追溯的确定性数据加载和shuffle，不保证与原DataLoader在同seed下顺序一致。

生存时间分箱仅拟合训练患者的观察事件时间，修复作者划分前全队列分箱；验证和测试标签不会影响分箱或图。推理只向作者forward传入用于占位损失的虚拟label/c，不读取查询真值来生成logits。

```bash
milplus run --data /path/to/task --method mammil --config configs/mammil_classification.yaml --output runs/mammil_cls
milplus run --data /path/to/single_slide_survival --method mammil_source --config configs/mammil_survival.yaml --output runs/mammil_surv_source
milplus run --data /path/to/patient_survival --method mammil_patient_union --config configs/mammil_survival.yaml --output runs/mammil_surv_union
milplus evaluate --run runs/mammil_cls
```

先用 TRIDENT 提取特征，并建立患者、标签、划分与真实坐标登记。单独一个SVS目录不能替代任务数据包。

## 验证范围

独立读取作者 selective_scan_ref、ssd_selective_scan、RMSNorm参考和完整split/conv/scan参考，对照不同包长、分组、门控顺序与跨块梯度。完整模型对照字面/修正循环、1/7/20实例及原生D1024/512隐藏宽度/状态128的输出与全部活跃梯度；对源码不活跃参数显式验证无梯度。原始分类/生存training_step在相同数值内核下对照五条路线各13步RAdam更新；数值内核的独立验证另行进行，不能将同内核trainer对照称作CUDA内核验证。另检查五路线精确续训、图冻结/损坏拒绝、森林遍历、患者图扩展和实际CLI。

当前通过数量、完整库回归和独立wheel结果以 [验证目录](validation/) 的当前软件身份记录为准。CPU合成验证不代表真实WSI训练、GPU吞吐或论文指标复现。
