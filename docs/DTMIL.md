# DT-MIL：完整空间网络与明确命名的参考训练

0.63 独立开发候选，尚未正式计入方法数。主库正式状态以已验收版本为准。
本方法是 Li 等人的 MICCAI2021 *DT-MIL: Deformable Transformer for Multi-instance
Learning on Histopathological Image*，不是同缩写的航空时序 MIL 或其他双层 MIL。

固定作者来源：`yfzon/DT-MIL@f52530811d0b1b591e10eab2fad9fd2ac1573b98`，Apache-2.0。
[作者代码](https://github.com/yfzon/DT-MIL)、[会议入口](https://miccai2021.org/openaccess/paperlinks/2021/09/01/160-Paper0828.html)、[论文 DOI](https://doi.org/10.1007/978-3-030-87237-3_20)。
运行时私有加载固定 Git 对象；不需要把其他框架中的同名模型当作原方法。

## 使用与真实输入

```bash
milplus setup --method DT-MIL
milplus check --data /absolute/prepared_task --method DT-MIL --deep
milplus run --data /absolute/prepared_task --method DT-MIL --output /absolute/new_run --device cuda
milplus evaluate --run /absolute/new_run
```

数据包包含固定患者划分、分类标签和每张切片完整 FM 特征及真实 `[x,y]` 坐标。
坐标可以位于相同 HDF5 的 `coords`，或登记 `coords_path`；必须与特征逐行对应。
例如 `task.yaml` 的 encoder 元数据：

```yaml
spatial:
  unit: level0_pixels
  origin: top_left
  patch_size: 256
  patch_stride: 256
  provenance: Actual scanner level0 coordinate extraction and tile configuration/version
```

单位和步长必须填写实际提取条件，不能复制示例当作所有数据的配置。程序按真实
位置减去共同原点并生成行列格，不把变长序列随意排列成方阵。重复坐标、偏离
登记步长、非有限或超出精确范围的坐标、过大的稠密画布都会报错，不自动下采样。

作者数据允许预先提取的 TRAIN 增强视图。需要时在对应 encoder 中登记：

```yaml
dtmil_cached_views:
  dataset: train_views
  provenance: Actual same-FM weight/preprocessing and view-generation recipe/version
```

该 HDF5 数据集必须是与原始行对齐的有限 `[N,V,D]` 浮点数组。它和常规特征一起
做内容哈希；仅 TRAIN 随机选择视图，VAL/TEST 始终使用登记的常规向量。不制造
额外光学尺度，也不在该聚合阶段更新切块编码器。

## 两个完整配置

| ID | 数据处理与任务条件 |
|---|---|
| `dtmil_source_crop_reference_train` | 保留发布版第二次裁剪沿用旧高度的行为，以及二分类标签范围；单独声明有限参考坐标修正和重建训练 |
| `dtmil_repaired_reference_train` | 修正第二次裁剪高度、明确支持 C 类标签；使用相同完整网络、有限参考坐标修正及完整参考训练 |

`DT-MIL` 默认解析为第二条。两者都不是仅 forward 的模型包装；都包含完整训练、
验证选模和状态恢复。它们不等于无法取得的作者历史完整训练器。

## 原始网络与采样算子

完整保留：输入 D 维的 BN/1×1Conv/BN/LeakyReLU/1×1Conv、256 维、八头、六层
可变形编码器、六层普通自注意力/交叉注意力解码器、正弦空间编码、每头四个
采样点、可学习查询拼接分类头和发布版交叉熵。默认100查询来自原参数解析器；
README 的推理示例另外使用2查询，二者不能不加说明地视为同一个模型设置。
所有实际超参数及覆盖值保存到请求与方法协议。

算子使用固定作者代码中的 `ms_deform_attn_core_pytorch`，即完整双线性采样与
零边界数学参考。没有替换为普通注意力。该 Torch 路径支持相应设备的算子，但
当前尚不声称已验证 CUDA，亦不声称达到作者编译 CUDA 扩展的速度。原始完整
数学算子的输出与 value/location/weight 全部梯度已与独立四邻点公式对照。

作者公开路径实际使用普通解码器，因此 `transformer.reference_points` 线性层
没有参与该路径，保留其状态但不虚构辅助损失。解码器内部还保留源代码未使用
padding mask 的行为。未将其他 DETR 的 two-stage、多尺度工厂开关冒充为本次
已验证的模型。

## 明确修正与信息预算

发布版先把高度裁到60，再在宽度裁剪时沿用原高度。实际100×100输入会保留
20×60而不是60×60；该行为在第一条配置保留，在第二条修正，并记录实际保留
行数和行 ID 哈希。默认60×60中心视野是原始输入策略，不代表全 WSI 所有组织
均进入每次预测。全矩阵比较应披露这一输入条件和被裁切比例。

源数据居中填充，而有效区域计算读取第一行/列；它们全为 padding 时，原始
参考坐标发生除零。单层网格中原式 `grid/valid_ratio*valid_ratio` 本来会抵消。
两条配置均只把零分母换成1，保持每个正比例和实际网格不变，并明确登记这个
有限边界修正。内部空洞仍按源码作为未掩码的零向量处理。

原始 `label>0` 将类别2变成1，二分类原范围0/1不变；多分类只在明确修正配置中
按登记的 C 类计算。不会先合并标签后冒充三分类结果。

## 完整参考配方与复现边界

公开 `main.py`、`engine.py` 只提供推理，未发布 optimizer.step/backward 或完整
训练循环。参数解析器给出50epochs、batch20、主学习率2e-4、backbone2e-5、
offsets/reference点学习率乘数0.1、weight decay1e-4、clip0.1、lr_drop40。
本库据此提供明确重建的完整参考配方：AdamW三个不重叠参数组、每轮 StepLR
（gamma0.1）、完整50轮、不猜测历史早停；TRAIN随机排列并保留最后不足20张的
批次，依次执行作者原始翻转、缩放/裁剪/填充、cutout与90度旋转。

优化器类型、调度调用时序、同步数据读取和第一个严格最优 VAL log-loss 选模为
明确的 benchmark 参考选择，不把解析器中的数字当作作者曾实际执行的证据。
也可显式登记 VAL macro-AUC 选模，保存完整配置。测试集只在完成后加载选中
检查点评价，真实标签不参与网络推理。

检查点保存全部参数、BN统计、AdamW状态、LR分组、StepLR、完整训练时域、
实际 TRAIN 次序、物理裁剪/增强哈希、更新计数和所有 RNG。输入内容、来源、
坐标、配方或调度改变时拒绝静默续训。

当前源码数值与数据边界审查属于软件证据。完整独立安装包预算、全库、全部
配置和最终选模重载仍待验收；没有自然 BRCA/COAD、GPU 或论文指标复现结论。

当前最终代码的23项专项全部通过：完整三种原生维度/60map/6+6层/100查询的输出与全部梯度、独立双线性全梯度、实际作者数据/AdamW参数组/裁剪/StepLR/RNG、两条三轮逐位续训、缓存视图与CLI拒绝条件。当前全库2513项、221配置与四组完整50轮独立包预算仍待验收。
