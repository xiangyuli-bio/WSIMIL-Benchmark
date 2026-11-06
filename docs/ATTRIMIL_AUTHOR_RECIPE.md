# AttriMIL 完整原作者配方与明确修正

来源为 `MedCAI/AttriMIL`，固定提交
`e5736b9b4b5d3d4a64340b6ff5fa7fd470a338a9`，Apache-2.0。
运行时从经检查的固定 Git 对象加载原始模型、约束和验证定义；
第三方源码不复制进本包。

展示名 `AttriMIL` 选择 `attrimil_author_200_main_repaired`。
原有技术 ID `attrimil` 保留原受控移植，不将其结果改称作者配方结果。

| 配置 | 最大轮数 | 早停 |
| --- | ---: | --- |
| `attrimil_author_200_main_repaired` | 200 | 原始 main 显式关闭 |
| `attrimil_author_200_function_earlystop_repaired` | 200 | 原始函数默认打开 |

## 保留的机制和训练流程

- 类别专属的门控注意力与原生 adaptor，维度为 D→D/2→D，
  注意力中间维度 D/2；不缩小模型、不截断 bag。
- SGD lr2e-4、momentum0.9、weight_decay1e-5，batch1；
  不额外增加 scheduler、梯度裁剪或 AMP。
- 完整损失为 CE + 1×spatial + 5×rank，保留原始 class0 排除规则。
- TRAIN 使用按类别平衡的 WeightedRandomSampler，有放回抽取 N 次，
  每条权重为 N/该类别训练数量。
- positive/negative queues 每类容量4，存放实际原始特征，
  每轮重新创建；队列更新调用作者的 rank_constraint。
- 验证集完整 CE 严格改善选模，初始最优值10000。
- 函数默认早停条件为 `retain > 20 and epoch > 50`，
  其中 epoch 从0开始；main 配置保留完整200轮。

## 坐标与邻居

必须提供真实 patch 坐标和物理尺度元数据。缺少邻居时使用该 patch 自身，
与原始 `create_3coords.py::find_nearest` 一致。九列顺序为：

```text
self,left,right,up,down,left_up,left_down,right_up,right_down
```

沿用作者代码的列名和对应坐标偏移，不按名字重新解释或交换对角线。
空间步长由任务包登记的 patch 尺寸/步长和读取尺度决定；
可显式提供 `encoders.<fm>.attrimil.neighbor_step` 两个有限正坐标距离。
不使用序列位置构造伪网格；重复坐标直接拒绝。

## 原始缺陷与修正

1. 原始 spatial_constraint 对 `[1,C,N]` attributes 使用 `A[:,c]`，
   在真实多 patch bag 中索引失败。明确修正为 `A[0,c]`。
2. 原始 train_loop 的日志引用未定义的 `loss_ins`，实际 utils
   也不导出它。独立原始参考仅将这个未使用日志值设为0；
   包内训练直接省去该日志，不更改目标函数。
3. 原始模型的 A_raw 只在 CPU 分配。明确改为与输入特征同设备，
   CPU 原生输出、梯度和更新仍与未修改原模型对照。
4. 原始 summary 按 DSMIL 的四输出解包，但 AttriMIL 返回五输出。
   统一评价使用原模型的五输出概率，从选定权重重新执行。

原始报错证据与这些修正分别保留；上述配置以 repaired 命名，
不声称未经修改的原始程序能够完成训练。

## 统一数据与恢复

患者不重叠的固定任务清单替代原程序的内置划分；统一 seed 显式登记，
原始 main 未明确设置模型 seed。数据按任务清单的 slide_id 排序。
CPU 同进程加载与作者 CPU 设置一致；CUDA 运行也使用 num_workers0
以记录完整顺序，区别于原始 CUDA num_workers4。

检查点保存模型、SGD、全部 RNG、实际 weighted draw 顺序和游标、
两侧有序队列、选模状态及历史。支持实际 minibatch、末批选模前及
完整 epoch 边界恢复；`--stop-after` 是可恢复中断，不是降低200轮预算。
恢复时补建 DataLoader 不额外消耗原训练轨迹的 RNG。

训练不创建 TEST 数据集、不执行 TEST 选模。完整特征和坐标字节、
患者划分、源码、软件和数值设置在请求中绑定；恢复与评价均检查。
完整训练结束后，独立评价入口重新载入最佳模型及其状态。

```bash
milplus setup --method AttriMIL
milplus check --data /path/to/prepared_task --method AttriMIL --deep
milplus run --data /path/to/prepared_task --method AttriMIL \
  --output /path/to/new_run --device cpu
milplus evaluate --run /path/to/new_run --subset test --device cpu
```

## 验证边界

机制测试使用原始512D/1024D模型及1025个以上 patch 的长 bag，
对照输出、梯度、SGD、RNG、队列与真实作者邻居预处理。
完整轨迹测试使用原始 Generic_Split、HDF5 loader、
get_split_loader、train_loop 和 validate 作为独立参考。
参考 trainer 仅修正上述空间索引和未定义日志。

这些软件输入均为合成特征；测试通过不代表真实 WSI 结果、
GPU 验证或原论文指标复现。当前版本的执行结果以版本证书和
保留的实际日志为准，测试收集数量不等于已执行数量。
