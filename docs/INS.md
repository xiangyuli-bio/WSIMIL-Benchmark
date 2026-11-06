# INS：完整冻结特征训练路线与明确的源码修正

这是隔离的0.59候选，尚未正式计数。入口 `ins_frozen_source_repaired`，`INS` 别名指向它。
本接入对应用户方法清单中 INS 的条件范围：选择并固定冻结特征路线。

固定作者源码：[miccaiif/INS](https://github.com/miccaiif/INS)，
`cd97ba2fa7df2a951f218e6de405f664129931c8`。
[论文v2](https://arxiv.org/html/2307.02249v2) 定义实例对比、原型伪标签和袋级约束；
作者 README 明确允许原始RGB或预提取SSL特征两种输入。本配置实现后者，不冒充RGB端到端实验。
来源没有发现根目录许可证，作者代码由 `milplus setup --method ins` 获取；发行包不复制这些实现。

## 输入不是一个普通注意力池化器

`train_onlybag_feat_by_bagcls.py` 真正执行的是 `model.py::INS` 与 `SupConResNet`，
将完整 feature reshape 为 `32×(D/32)` 并复制成3通道，然后训练 ResNet18。
它没有使用名字看似更合适的 `model_feat.py`。因此本路线明确保留这一输入条件：

- FM 自身保持冻结，但新增的特征空间 ResNet18 参与训练；这不是直接更新原始FM。
- 必须 D>=32 且 D 能被32整除，例如768、1024、1536、2560。所有通道原样进入变换；不裁切或填充。
- 重排位置代表FM通道，不是组织坐标，也不是RGB图像。它可能影响跨FM比较，应作为方法条件披露。
- 原生作者 ResNet18 为 `[2,2,2,2]`，输出512；实例头512→2、投影器512→512→128。
  按论文“不预训练”的设置将 source `pretrained=True` 明确改为False，使用作者实际随机初始化结构，
  没有隐式下载ImageNet权重，也没有换成一个小MLP来通过测试。

任务是作者论文的二分类MIL条件。`classes[0]` 是负/参考袋，`classes[1]` 是阳性袋。
参考袋不一定意味着正常组织：论文也把LUAD/LUSC分别编码为负/正。
负袋内实例使用参考类别；阳性袋内实例的候选集合为 `[1,1]`，不能从真实patch标注获知其类别。
原始事件/删失生存任务需要单独定义，不能偷偷变成作者的二分类预后终点。

## 完整执行的机制与训练顺序

实际加载作者 `model.py`、`resnet.py`、`alexnet.py` 和 `utils/utils_loss.py` 的固定Git对象。

1. Query与momentum key两个完整编码器、两层投影器、实例分类器全部保留。
   key参数按m=.999更新；BN缓冲按各自实际forward更新，不凭空改成参数EMA。
2. IWSCL：warmup用作者MoCo目标，之后使用由当前实例分类和历史key类别建立的SupCon目标。
   原始函数的所有-pool分母与论文Eq.5印刷的仅非family分母不同；本配置明确使用实际源码公式。
3. 原型：128维双原型，按作者顺序用detached query和当前伪类做m=.9的逐实例更新后归一化。
   当前伪标签先使用上一份原型快照。源码在warmup期间也更新原型；warmup结束才更新confidence与监督mask。
4. Confidence：每个真实TRAIN patch有稳定ID；负袋初始 `[1,0]`，阳性袋初始 `[.5,.5]`。
   使用作者partial CE、prototype argmax与EMA，epoch结束按 `.95→.8` 的实际源函数公式更新系数。
5. 每个epoch先完成实例训练，再对每张TRAIN slide做一次完整mean-feature袋约束CE更新。
   不把这两个阶段压成一个普通CE训练循环。

默认完整配方：150epoch，20epoch warmup，实例batch64，queue8192，SGD lr=.01、
momentum=.9、weight_decay=.001；实例CE权重1、对比权重.5、袋CE权重1。
前20epoch只抽阳性袋patch，之后抽全部TRAIN patch。使用实际world1
`DistributedSampler(seed=0+epoch)` 与 `DataLoader(drop_last=True)`，不复制实例凑尾批。
每个epoch的袋训练按固定manifest次序进行，不丢袋。

作者冻结特征数据集将弱/强dropout都设为 `training=False`；这两个输入view在此路线中确实相同。
本入口没有默默启用额外增强。query/key仍是不同的参数更新流，且保留作者batch shuffle。
单rank的shuffle使用与原始CUDA代码一致的CPU `randperm` 再转到目标设备。
独立对照实际初始化单rankGloo，执行原始all_gather/broadcast；不把未验证的多rank控制器说成已支持。

原始LR默认milestone为700/800/900、decay=.1，150epoch范围内不会触发。可显式选择作者cosine，
实际参数均记录。默认不提前停止；只在warmup之后以VAL macro AUROC保存首次严格最佳checkpoint。
原始代码每epoch访问test且没有完整选模保存，PULS将其改为VAL选择、TEST显式evaluate。

## 必须公开的修正

- 原始 `model.py` 的袋路径对512维latent做softmax，却从不调用已经声明的512→2袋分类头；
  后续CE把它当作类别分数。本入口明确启用这个原始分类头，并在logits上做CE。
  输入为mean backbone512特征；不把它与论文Eq.10中可能指128投影向量的q记号混成同一个设置。
- 原始 feature dataset 根据真实patch label生成候选集合与point，且会按正patch比例筛选slide。
  这些信息不能用于仅bag标签的benchmark。本入口只用bag label和原始feature行，不读取patch标注或做该筛选。
- 原始queue的随机初始槽位有随机标签。本入口增加valid mask与真实TRAIN patch IDs，
  在被真实key写入之前不作为训练实例参与对比。保留原始完整容量、循环写入、query/key/queue顺序。
- 两个teacher head和额外mlp确实存在于该路由的source state，但没有进入forward；
  保留它们的状态和参数清单，不虚构教师注意力训练或强行添加额外loss。
- 原始script导入缺失的Cervical dataset；PULS使用标准任务清单和TRAIN索引，
  不调用外部硬编码路径，也不需要作者未公开的临床数据才能使用该冻结特征路线。

## 数据读取、内存与恢复

全局实例ID按TRAIN slide顺序、再按每张slide的原始feature行确定。逐batch读取真实行；
H5使用有上限的只读句柄缓存，NPY按行读，PT用有上限的解码bag缓存。
超过PT缓存容量的单文件仍需临时完整解码，这是格式限制，不声称PT支持随机行读取。
不将整个多癌种FM数据集拼接进内存。confidence和queue是训练状态，不含VAL/TEST ID。

袋约束训练仍对完整bag保留原生CNN与BN计算图，不以隐蔽分块或冻结BN改变训练。
`max_dense_bytes` 在输入复制和激活分配前给出保守估计；大bag需要按实际GPU内存显式调整预算。
推理在固定eval BN下分块编码全部实例，以float64累加512特征均值再调用原始分类头；
这与完整均值在数学上相同，浮点约简与原始一次性mean可有微小差异，专项检查设明确容差。

完整checkpoint包括编码器、key、双原型、有效queue/ID、confidence、SGD动量、epoch、
EMA系数及Python/NumPy/Torch/CUDA RNG。恢复时核对固定代码、训练输入哈希、完整行映射、
弱标签、队列身份与confidence结构。多种状态不匹配会拒绝，而不会重置后假装继续训练。

## 使用与当前验证边界

```bash
python -m pip install '.[ins]'
milplus setup --method ins
milplus demo --ins --patients 20 --output /path/to/new_synthetic_ins_bundle
milplus run --data /path/to/prepared_binary_bundle --method INS \
  --config configs/ins.yaml --output /path/to/new_run
milplus evaluate --run /path/to/new_run
```

任务路径绑定患者/slide/label、固定划分和FM特征索引；无需手工构造patch真值。
若任务明确设置患者级评价，使用统一患者融合并记录为外层评价扩展；原作者比较单位是slide。

目前仅为候选：原生1024/1536完整网络机制原型及实际原作者单rankGloo网络/梯度/SGD对照已执行；
统一训练、精确恢复、输入和全部软件配置仍在验收。未完成正式接入、GPU、自然BRCA/COAD或论文数值复现。
