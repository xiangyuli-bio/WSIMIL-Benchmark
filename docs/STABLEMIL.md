# StableMIL：物理融合、区域语义和局部/全局注意力

一个方法家族，四条显式配置：

| 配置 | 任务 | 主要行为 |
|---|---|---|
| `stablemil_classification_source` | 分类 | 完整作者网络、CE/Adam、原尾批梯度跨 epoch 保留 |
| `stablemil_classification_repaired` | 分类 | 原网络，加退化几何/小包和尾批累积修正 |
| `stablemil_survival_source` | 生存 | 完整 hazard 网络、作者 NLL/Adam/gc；补齐发布脚本中缺少的输入与任务连接 |
| `stablemil_survival_repaired` | 生存 | 同上，加退化几何/小包和尾批权重修正 |

`StableMIL` 别名按任务包选择对应修正版。源码版保留有科学影响的原行为；两种配置都修复阻止执行的设备、变量和接口错误，不能称为未改动的原作者启动脚本。

## 固定来源及完整性

[作者仓库](https://github.com/theeeqi/stableMIL)，提交 `67b279157680c8c96104823a9e6f2ef20b679d11`；论文 DOI `10.1109/TMI.2026.3682009`。固定仓库没有根 LICENSE，作者源码继续外部获取，不重新分发。原环境要求的 `timm==1.0.21` 固定到 `625386b96151ec49dede06190b0deaa3909cd4de`，原 Mlp、LayerScale、DropPath 和 tuple helper 也从校验过的 Git 对象读取，Apache-2.0。

使用完整物理分组、融合、语义 token、双向局部/全局注意力、随机二维 RoPE 和原始分类/生存训练循环，未用普通 Transformer 或一个通用 CE trainer 替代。代码中的实际注意力拓扑和尺度保持原样，没有根据论文标题臆造额外 entropy penalty。

## 输入、物理预处理和可复用缓存

需要冻结 FM 特征 `[N,D]`、同行序真实坐标 `[N,2]`，以及 `encoder.spatial` 中的单位、origin、patch_size、provenance。物理坐标以 float64 读取，使用源代码 floor 除法转换到参考尺度。

原预处理固定 `reference_size=512` 个 level0 pixel。本适配器默认读取已登记的 `spatial.patch_size`，也可在 model 配置明确指定 `reference_size: 512`。这是可追溯的 FM/切块协议迁移；不同物理视野不能暗中混为原论文设置。

保留源代码的 bbox 分组和 ratio（默认2）：每组按原排序最多保留 ratio² 行，少于该数时以组均值填充；之后用均值或原 learnable mapping 投影。不会把一个组的全部 patch 平均后假称原实现。实际保留行号、原始/融合实例数和完整数组哈希均进入几何记录。

保留原预处理的最小外接旋转矩形、原 `find_region_ag` 分区、区域排序、KD-tree 最近邻、距离过滤及自身回填。默认 k=8，距离阈值为预处理文件实际使用的 `6*sqrt(2)`（融合后的网格单位）；训练 parser 中的 `3*sqrt(2)` 没有替换已生成的邻居掩码。

训练单位保持单张 WSI 的独立坐标系。分类可在最终评价时按任务定义融合患者的多张切片概率；生存任务要求每个患者一张已经冻结选择规则的代表切片。遇到多张切片会明确拒绝，不能像原数据加载器一样不说明地只取第一张，也不能拼接不同坐标系。

预处理只由坐标和声明参数决定，不读取标签，不拟合全队列。几何在首次使用时写入运行目录 `stable_geometry/`，也可以通过 `model.geometry_cache_dir` 指定跨 FM/种子复用的目录。缓存键包含原坐标行序、参数、作者 commit 及数值依赖版本，文件与各数组都有哈希。原子完成目录避免并发读取到半份文件；每次载入检查内容和身份。内存 LRU 有独立预算，不需要每轮重做凸包与近邻计算。预测 CSV 写入几何哈希和实际保留数量。

## 完整网络

默认按发布 CLI 使用 hidden512、depth2、heads8、aggregate_num256、ratio2，而不是混用网络类的 depth3/aggregate128 默认值。

网络包含原特征融合和 Linear/GELU、学习 query 的区域语义 token、每个局部 token 对物理邻居和全部语义 token 的注意力、语义 query 的全局注意力、FFN、随机投影二维 RoPE、局部与语义 token 的联合均值及规范化分类头。训练保留原 p=.8 随机坐标投影及90度旋转，验证/测试使用原轴向 RoPE。默认 attention dropout 为0，头部 dropout 和 DropPath 也为0；不静默引入新的增强或位置索引。

## 训练、选择和准确的源码边界

分类：默认200轮、Adam lr=1e-4、无 weight decay、无梯度裁剪、无 scheduler；单 bag 随机采样，累积4个 bag。源码版保留未 flush 的尾批梯度及下一轮的继续累积，checkpoint 必须保存这些梯度。修正版在每个尾窗口更新，并按实际窗口长度归一化。

生存：保留作者 `NLLSurvLoss` 的 clamp、事件/删失定义与 hazard 累积生存概率。源 loss 类默认 alpha=.15，数据类默认4 bins；训练200轮、Adam lr=1e-4、gc4。源码最后一个窗口虽然会更新，但仍除以名义 gc；修正版改用实际窗口数。验证使用源代码规定的 alpha=0 NLL。

发布生存 launcher 缺少多个使用中的 parser 字段，并默认错误的任务名称；模型需要完整预处理输入，训练调用却只传特征和坐标。因此这里明确补齐为 NLL/4 bins/alpha.15/zero decay 的可执行设置，zero decay 是显式补齐值，不能声称已恢复作者未提供的实验参数。未把有问题的 Cox/其他未登记 loss 自动替代进来。

时间分箱只拟合 TRAIN 中唯一患者的观察事件；不沿用原数据层从全队列生成边界的做法。时间、事件、删失和代表切片规则必须由任务包提供。测试标签既不参与分箱，也不进入模型前向。

两种任务都以验证 bag CE / patient NLL 选模，保留 EarlyStopping 的同分覆盖规则和 epoch>50 才可提前停止的限制，默认 patience20。测试只在显式 evaluate 时运行。通用患者 AUROC 或 C-index 不替换原验证 loss 的选模作用。

共同执行修复还包括：绕过访问不存在 `self.fuse` 的 relocate；修正分类 trainer 的 `region_sorted_index` 拼写；在正确设备创建 mask；补齐生存预处理、任务、loss 和参数连接；用一致的 last/best/optimizer/RNG 保存方式替代错误的 EarlyStopping 参数绑定与可变最佳模型引用。原生数学和优化更新单独与源函数对照。

## 明确命名的几何修正

源码版对零轴跨度、少于5个/共线的融合点或融合点不足 k+1 的输入给出错误，避免沿用原崩溃或 NaN。

修正版在零跨度轴使用一个物理 bin，在退化几何中使用一个语义区域，并按实际可用邻居数分配全局 mask。非退化的大包仍执行完整原路径；没有添加伪坐标或修改真实 patch 行序。

## 运行与验证状态

```bash
milplus setup --method StableMIL
milplus run --data /absolute/task_bundle --method StableMIL \
  --output /absolute/new_run --config configs/stablemil_classification_repaired.yaml
milplus evaluate --run /absolute/new_run --subset test
```

生存任务使用对应 survival 配置。原生1024维/hidden512的分类和生存网络、完整梯度、原始预处理数组、两个完整训练循环、四条配置精确续训（含未更新的梯度）、缓存、源小包反例、标签独立性和训练专用分箱分别验证。最终全库、全部配置及独立安装包结果记录在 `docs/validation/summary.json`。

当前没有 GPU、真实 WSI 或原论文指标复现证据；CPU 合成结果只用于软件验证。
