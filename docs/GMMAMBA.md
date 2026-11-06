# GMMamba

来源：[titizheng/GMMamba](https://github.com/titizheng/GMMamba)，固定提交 `70576f1aaeef0d7041d6860799ade1ac905b85aa`；论文为 [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_GMMamba_Group_Masking_Mamba_for_Whole_Slide_Image_Classification_ICCV_2025_paper.pdf)。原始库未提供 LICENSE，代码从单独获取的 Git 对象加载，不随本包分发。

完整保留坐标 KMeans 分组、两层双向 Mamba 的 IMM、硬实例掩码、组注意力、CSS 最大池化／跨注意力／八头注意力，以及含分类 token 的全512维分类模块。不是普通 Mamba、均值池化或小模型替代。

作者库导入 `mamba.mamba_ssm.BiMamba`，但没有发布这个目录。本实现明确使用 MambaMIL 提交 `9aea39efc13fdcd3703b5b54c28a64887b4acbd7` 的真实 BiMamba 定义作为依赖移植：两个方向分别有卷积、输入投影、状态矩阵、时间投影和跳连，再求和投影。协议记录该依赖来源与文件哈希，不能把它称为作者未发布依赖的逐字复原。`backend=torch` 是明确的完整递推实现；显式 `backend=cuda` 必须满足原1.1.2／1.1.1编译依赖，不会自动回退。

两条配置分别登记：

- `gmmamba_mambamil_bimamba_source`：原生512维条件，保留作者未使用的 `_fc1`。
- `gmmamba_fm_projection_repaired`：明确启用原始 `_fc1` 的线性投影、GELU与0.25 dropout，再进入相同的512维 IMM，支持其他 FM 维度。它是命名修正配置。

源代码硬索引掩码不向 `mask_attention` 反传梯度；这部分完整保留并测试，没有把未训练的打分器宣称为已学习。CUDA 移植只把布尔掩码分配到输入设备。保留每组按原始 feature 行序的实例，删除最低分的 `max(1,floor(0.3*N))` 个实例。空组／单例组无法满足原始 IMM，明确拒绝，不重复、补造实例或自动降低组数。

输入必须有真实坐标及单位、原点、patch_size和来源声明，特征和坐标行严格对应。保留原始 HDF5 坐标 dtype；提前将整数坐标转成 float32 可能改变 sklearn KMeans 分组。完整患者划分、标签、原始文件和坐标字节纳入 checkpoint 合同，不使用未绑定版本的 `.pkl` 特征缓存。

训练保留原始两读进程、随机 TRAIN／VAL和顺序 TEST、逐 slide 交叉熵、AdamW `lr=1e-4,weight_decay=1e-5`、100轮上限。按原始 VAL AUC+F1 严格提升选模，epoch_index必须大于5，耐心20且epoch_index大于50才终止。优化器参数顺序保留 IMM→classifier→CSS。

原入口在每次验证改善后读取 TEST。统一实验延后实际 TEST 访问，同时只重放该顺序 DataLoader 在父进程消耗的一次 int64 base-seed随机数；实际原始 HDF5 Dataset／原始 TEST推理是随机状态对照。两个子读进程的随机状态不影响父进程，eval网络没有随机运算。这个移植不会利用测试标签选择模型。

使用方式：

```bash
milplus setup --method GMMamba
milplus run --data /path/to/prepared_task --method GMMamba --output /path/to/run
milplus run --data /path/to/other_fm_task --method gmmamba_fm_projection_repaired --output /path/to/other_run
milplus evaluate --run /path/to/run
```

当前为开发接入。完整原生预算、整库、安装包与论文机制／缺失依赖对照分别验收；合成数据运行不能替代真实 WSI 或原论文指标复现。

## BRCA 完整真实队列的原坐标分组核验

2026-11-06T01:23:21Z，在现有节点 CPU 对全部856患者、2422295行真实原始
UNI特征及整数坐标执行固定作者的`coords_grouping`，保持10组、random_state0、
n_init=auto和每组原行序；同时调用本项目未改动的`validated_groups`并逐位比较输出。
全部856袋的原生组容量通过，没有空组或单例组，全队列最小实际组含8个patch。
逐袋原坐标SHA匹配已核验来源，原文件大小/mtime前后不变；没有补patch、减少组数、
丢弃患者或创建全队列之外的临床子任务。

这仅是实际完整输入的分组前置核验：使用的是1024维UNI原数组，不曾把它传入原生512维
GMMamba网络，没有创建投影、模型、优化器或训练。不能称为512维接口、GPU、原生训练预算
或论文复现已经通过。原生512宽度仍是UNI实验的真实阻塞。

已询问是否允许单列现有CONCH512的补充实验、不计入UNI比较；截至本次更新尚无批准，
因此没有切换FM或提交该训练。原生条件和命名投影修正版继续分开，不自行启用后者。
原作者缺失BiMamba依赖的明确移植边界仍然保留。

凭据：`gmmamba_complete856_original_group_capacity_verified_r1.json`（含完整856逐袋记录），
`gmmamba_complete856_original_group_capacity_launch_r1.json`。实际耗时46.95秒，
自身峰值RSS977629184字节；仅当前CPU诊断的观测，不是模型训练峰值或整体耗时估算。
