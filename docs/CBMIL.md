# CBMIL：聚类选择与双视图训练

`cbmil` 使用 [wwu98934/CBMIL](https://github.com/wwu98934/CBMIL) 的固定提交 `980252ac796e4f68ad97d7b0b0de4aef4bb00157`。当前完成分类/有序分类的软件适配，尚未运行原论文数据协议或 GPU 实验。

```bash
milplus setup --method cbmil
milplus run --data /path/to/task --method cbmil --config configs/cbmil.yaml --output runs/cbmil
milplus evaluate --run runs/cbmil
```

## 实际训练机制

每张切片划分为固定数量的非空簇。作者网络通过 CLAM 产生各簇的 key，以 NonLocalRanking 对实例排序，按簇大小比例选取实例，同时形成簇级融合表征。PPMIL 同时聚合所选实例和簇表征，再融合分类。保留作者的 CLAM 正/负实例监督及亚型分支。

每个训练 batch 包含至少两张切片，每张提供完整簇视图和第二个簇视图。两次前向的融合特征进入对比投影头，计算 NT-Xent。最终损失：

```text
(1 - cl_weight) × (0.7 × bag CE + 0.3 × instance CE) + cl_weight × NT-Xent
```

作者 `mixup` 在两条视图分支内分别混合已选特征，系数从 `[mixup_alpha, 1)` 均匀抽取，保留主 bag 标签，簇融合特征不混合。它不是 Beta 系数或软标签 MixUp，适配器保留其实际行为。默认 alpha=0.8、cl_weight=0.1、temperature=1。

`view_instances: null` 保留作者默认数据入口行为：第二视图使用全体实例，差异来自选取路径、dropout 和批内混合。显式指定 view_instances 时按簇比例抽取第二视图，每簇至少一个实例。这个可选设置会改变训练条件，需记录配置。

## 聚类输入与数据约束

`clustering: kmeans` 是 PULS 的明确预处理设置：在每张切片的原始冻结 FM 特征上，以固定 clustering_seed、10 次初始化和指定迭代上限执行欧氏 KMeans，不使用标签。不依赖可训练输入投影的当前权重，也不跨切片或跨患者拟合中心。训练划分的分配在 prepare 中一次生成并进入 checkpoint；验证/测试仅对当前切片重新计算确定性分配，没有训练/测试联合聚类。

作者仓库使用预计算簇文件，但未在当前固定提交中提供完整聚类生成程序。该 KMeans 预处理不被声称为作者原始聚类的复现。

已取得聚类结果时使用 `clustering: provided`，并在当前 FM 的特征索引添加：

```csv
slide_id,path,cluster_labels_path,cluster_labels_sha256
slide_A,features/slide_A.h5,clusters/slide_A.npy,可选实际SHA256
```

聚类数组可为 `.npy`、HDF5 的 `cluster_labels` 数据集，或 tensor / tensor dictionary `.pt`；形状为整数 `[N]` 或 `[N,1]`，逐行对应特征。标签非负，允许非连续编号，但非空簇数量必须等于配置的 clusters。内容哈希进入输入清单，修改后旧 checkpoint 的继续训练或评价会被拒绝。现有作者 list-of-indices JSON 可先转换为逐实例数组，并核对每个实例恰好出现一次。

任何聚类退化、实例少于簇数、标签数量不对齐都明确报错，不自动减少簇数。当前训练单位为单张切片；`evaluation_unit: patient` 可以按统一协议平均切片概率。不同切片的簇编号不被视为可直接合并的共同语义，不能使用多切片拼接的 patient 训练模式。

## 适配与原实现差异

- 作者 model 文件引用不存在的 `models.mil`，实际 CLAM 类位于 `models/clam.py`；隔离加载时注入后者，算法定义不替换为通用模型。
- 作者簇 key、CLAM 内部表征与对比投影的维度假定为 512。输入 FM 恰为 512 时使用 Identity；其他维度添加一个可训练线性输入投影到 512。该投影计入参数和成本，必须在多 FM 比较中披露；不把形状不匹配静默广播或截断。
- 保留作者 selected_instances 的比例舍入、零填充、截断和 top_fuse 路径，包括某个小簇分配到零个所选实例时的原行为。
- 作者尾批允许一张切片，此时 NT-Xent 没有负样本。PULS 将最后一个单例并入前批，完整保留样本并保证每批至少两张；最多比 batch_size 多一张，显存预算需包含此情况。
- 对比掩码跟随 similarity 所在设备；CUDA 分支尚未运行验证。
- 使用统一的验证侧选模、完整 checkpoint、梯度裁剪和训练预算。没有在每个 epoch 查看测试集，也尚未加入作者可选的余弦学习率调度。
- 任务级 max_instances 会破坏固定聚类行顺序，因此拒绝该选项；使用 selected_instances 和 view_instances 管理方法内部预算。

## 验证与状态

专项测试覆盖原作者 train_forward 与适配器损失/输出对照（512 和另一输入宽度）、NT-Xent 手算值与梯度、各机制参数梯度与更新、混合系数、第二视图采样、尾批、聚类数组校验、输入变更阻断、两种聚类模式的训练/重载/精确续训及验证随机状态隔离。

这些是 CPU 合成特征软件检查。模型实例损失、对比损失实际更新不等于原论文精度已经复现。固定仓库根目录没有发现许可证文件，许可核查仍待完成，作者代码不随 PULS 发行包分发。
