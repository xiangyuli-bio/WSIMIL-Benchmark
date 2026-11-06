# Gaussian-process、分组通信与提示聚类方法

0.4 增加三个方法家族、四个执行配置：`sgpmil`、`mmil`、`mmil_coords`、`ptcmil`。均从锁定作者提交读取模型定义，作者源码单独获取，不包含在发行包中。本轮实现分类/有序分类接口；PTCMIL 的作者生存训练路径尚未接入。

## 安装与运行

```bash
python -m pip install -e '.[gp,prompt]'
milplus setup --method sgpmil
milplus setup --method mmil
milplus setup --method ptcmil

milplus run --data /path/to/task --method sgpmil --config configs/sgpmil.yaml --output runs/sgpmil
milplus run --data /path/to/task --method MMIL-Transformer --config configs/mmil.yaml --output runs/mmil
milplus run --data /path/to/task --method mmil_coords --config configs/mmil.yaml --output runs/mmil_coords
milplus run --data /path/to/task --method ptcmil --config configs/ptcmil.yaml --output runs/ptcmil
milplus evaluate --run runs/ptcmil
```

有序任务仍使用类别 logits，按任务定义选择 QWK；没有将这些分类网络声明为作者专门的 ordinal 模型。例子使用共同训练预算和验证侧选模，尚未复现作者论文实验。

## SGPMIL

来源：[mandlos/SGPMIL](https://github.com/mandlos/SGPMIL)，commit `c9dee272e3fec897dba8136f386ddd2d685da4ed`。额外依赖固定 `gpytorch==1.14.2`、`linear-operator==0.6.1`。

保留作者特征投影、GP 注意力、可学习诱导点、Cholesky 变分分布、LinearMean 和 Scale-RBF + ConstantKernel。训练目标为 **Monte Carlo 平均 logits 的交叉熵 + kl_factor × KL(q(u)||p(u))/训练 bag 数**。不是先对每次抽样计算交叉熵再取平均。预测使用多次抽样概率的均值，与平均 logits 的 softmax 有区别。

训练 bag 数来自实际训练单位：slide 模式按切片，patient 模式按患者。验证和测试不会参与该分母。诱导点、变分均值和 Cholesky 参数参与 AdamW 更新。

`sampling: var` 保留固定作者代码的广播行为：一个 Monte Carlo 样本中的实例共享同一个标准正态随机数，再乘各自标准差。因此它不是独立边缘抽样，也不是完整联合协方差抽样。`sampling: cov` 使用作者 `rsample` 协方差路径，长 bag 的内存成本需实测。默认 sigmoid 注意力不强制归一化，不能解释为实例概率之和等于 1。

验证/测试仍进行 Monte Carlo 推断，但使用固定 `evaluation_seed` 并恢复调用前的 Python、NumPy、CPU 和当前 CUDA 设备随机状态，避免验证次数改变后续训练抽样。固定种子使用相同随机数比较不同 checkpoint；它没有消除 Monte Carlo 近似误差。CUDA 数值与重复性尚未验证。

PULS 使用共同预算、梯度裁剪和验证早停，尚未实现作者 Lightning 中的 warmup/cosine 调度。配置文件保留作者常用学习率并明确记录其他控制设置，不宣称它等于完整作者实验配方。

## MMIL-Transformer

来源：[hustvl/MMIL-Transformer](https://github.com/hustvl/MMIL-Transformer)，commit `d363f43bf0949e3fc172e8d94f99deb335bbe171`。

保留作者分组、组内注意力、message token 的跨组通信、首层组掩码和后续消息传播。`mmil` 支持 random、seq、embed 分组；`mmil_coords` 使用真实坐标 KMeans。坐标与特征必须逐行对齐，不允许把不同切片的坐标系拼成同一空间。没有坐标时不会制造网格。

作者 `camelyon16` 分支包含输入线性投影；适配器对所有 FM 使用该投影分支，输入宽度来自特征包。此内部开关不改变数据集、标签或划分。默认每组最多 4300 个实例，超过时按作者规则抽样；这是额外于任务级 max_instances 的预算，训练和推理成本应一并登记。

兼容性修改：保留 singleton 的索引轴；只移除 batch 轴；遍历实际聚类标签，避免非连续标签丢失实例；若聚类退化使掩码后没有任何组，则明确报错。有效的非退化分组已与未修改作者网络做随机、顺序、嵌入分组三类数值对照。

验证关闭组掩码，并隔离固定种子的组排序/抽样。作者公开提交以推理和权重为主；当前 CE trainer 是 PULS 的受控训练适配，不冒充已经取得并复现原作者完整训练程序。

## PTCMIL

来源：[ubc-tea/PTCMIL](https://github.com/ubc-tea/PTCMIL)，commit `083734f4c511513d8e6a7cba9c7d65e9afd58f69`。额外依赖 `nystrom-attention==0.0.14`。

保留作者正交初始化的 prompt、全局 prompt 聚类、逐组局部 Nystrom 注意力、CTM 学习加权 token 合并和分类头。损失为分类 CE + `ptc_weight × ||PPᵀ-I||`；默认权重 0.1。没有把 prompt 聚类简化成普通 CLS 注意力。

训练在相邻 bag 间传递 detached prompt 状态，按作者 momentum 规则更新。每个 epoch 开始重置，验证/测试始终从学习到的 prompt 参数出发，不读取训练 bag 的移动上下文。上下文是 checkpoint buffer；优化步完成后才更新。中断恢复以完整 epoch 为界，与不中断训练逐位对照。

兼容性修改仅涉及设备相关张量与模块加载：`.cuda()` 改为跟随输入设备，identity 注册为 buffer，从固定 Git blob 提取实际使用的定义，避开未使用的 TCFormer/MMCV 图像模块。保留作者空簇的合并行为和 token 顺序。可选的 token-labeler、额外 checkpoint 和生存流程没有在本分类配置中启用。

## 验证与许可记录

专项测试覆盖三个家族的真实前后向、损失、参数更新、保存/重载、四配置逐位续训、验证状态隔离、GP 两条抽样路线、KL 数值、MMIL 组间通信、singleton/聚类退化、PTC prompt/CTM 梯度和手算加权合并。测试数据为合成特征；不证明真实 WSI 性能。

MMIL 仓库附 MIT 许可；SGPMIL 附非商业修改条款；在固定 PTCMIL 提交根目录未发现许可证文件，许可状态保留待核。各仓库的许可不由 PULS 根许可证替代。下载不等于取得原论文数值复现证据。
