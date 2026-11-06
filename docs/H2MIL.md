# H²-MIL：真实三层图与完整训练

固定作者 [lin-lcx/H2-MIL](https://github.com/lin-lcx/H2-MIL) 提交 `ea472fefd5eab553c58041a5acc203ac7d467d06`。运行时从外部已验证 Git 对象载入完整 RAConv、IHPool 和 GCN 定义，保留两级图卷积与层级池化、读出求和及分类器。该提交未找到根许可证，作者代码与数据不随本库分发。

| 配置 | 行为 |
|---|---|
| `h2mil_repaired`（`H2-MIL` 默认） | 明确修正 LayerNorm 维度、C 类分类头及 logits 交叉熵；支持多分类 |
| `h2mil_source_compat` | 仅修正无法运行的 LayerNorm 维度；保留四输出 softmax 后再做 CE，仅接受二分类任务 |

两条路线都保留完整原始机制和专用训练流程。配置不是“读入特征后改用 ABMIL”。作者原始 LayerNorm 在不同宽度间复用而报错，因此没有声称完全未经修改的源码能够运行。

## 实际输入

每张 WSI 必须提供三个真实图像层级的特征：整张缩略图、低倍率区域、高倍率 patch。三者来自同一登记的编码器空间、具有相同维度；缩略图和低倍率特征不能通过高倍率向量求均值伪造。

主任务的标准特征 H5 保存高倍率 `features` `[N,D]` 和实际 `coords` `[N,2]`。另有每张 WSI 一个层级 H5，仅包含：

| H5 数据集 | 形状与含义 |
|---|---|
| `thumbnail_features` | `[1,D]`，实际缩略图图像特征 |
| `coarse_features` | `[C,D]`，实际低倍率图像区域特征 |
| `coarse_grid` | `[C,2]`，低倍率实际网格索引，唯一非负整数 |
| `fine_grid` | `[N,2]`，与主特征行对齐的实际高倍率网格索引 |
| `fine_parent` | `[N]`，每个高倍率节点所属的低倍率行号，零起算 |

主坐标必须恰好等于 `fine_grid * fine_stride_level0`；父节点的低倍率格点必须等于 `fine_grid // 2`。两个倍率的物理边长比例为 2；不接受将任意特征序列排成方形。每个低倍率节点至少有一个实际子节点，所有网格唯一。原始逐轴归一化要求每个倍率在 x、y 方向均有非零跨度；退化单行/单列输入明确拒绝。

构图保留同倍率八邻域及真实双向父子边。节点顺序为缩略图、低倍率、高倍率原始行顺序。坐标先按作者方式逐倍率、逐轴归一化，网络再映射到 `[-1,1]`。这保留作者的归一化行为，不声称归一化距离等于物理长度。

建立 `hierarchies.csv`，严格只有 `slide_id,path`，覆盖任务中的每张切片，不能包含标签列。下面在已有标准任务数据包上生成准备声明；来源说明必须替换为实际提取记录：

```python
from milplus.data import Bundle
from milplus.h2mil_data import COMMIT, encoder_identity
from milplus.io import write_json

bundle = Bundle('/path/to/task', fm='your_fm')
schema = {
    'schema_version': 1,
    'kind': 'h2mil_thumbnail_coarse_fine',
    'source_commit': COMMIT,
    'synthetic': False,
    'coordinate_frame': 'level0_pixels',
    'fine_stride_level0': 1024,  # 替换为真实非重叠高倍率 patch 的 level0 边长
    'encoder_identity_sha256': encoder_identity(bundle),
    'patching_provenance': '实际倍率、MPP、图像边长与坐标生成记录',
    'thumbnail_provenance': '实际缩略图读取、缩放与特征提取记录',
    'coarse_feature_provenance': '实际低倍率图像编码记录',
    'fine_feature_provenance': '实际高倍率图像编码记录',
    'encoder_provenance': '同一编码器、权重版本及输入变换记录',
}
write_json('/path/to/h2_schema.json', schema)
```

`encoders.<FM>.spatial` 使用 `unit: level0_pixels`、`origin: top_left`，`patch_size` 与实际 `fine_stride_level0` 相等。本接口要求非重叠图像层级；示例数字不能代替真实提取尺度。编码器及声明的身份绑定不等于已经证明真实图像提取来源，仍需保留提取日志与权重记录。

```bash
milplus setup --method H2-MIL
milplus prepare-h2mil --data /path/to/task --fm your_fm \
  --index /path/to/hierarchies.csv --schema /path/to/h2_schema.json \
  --output /path/to/task/hierarchy.json
```

在 `task.yaml` 对应 encoder 下加入 `h2mil: {manifest: hierarchy.json}`。准备清单绑定原特征、坐标、层级文件、编码器、源版本与几何 SHA；读入时继续检查实际父子关系，重新填写文件哈希不能让错误几何通过。

```bash
milplus check --data /path/to/task --method H2-MIL --deep
milplus run --data /path/to/task --method h2mil_repaired \
  --config configs/h2mil.yaml --output runs/h2mil
milplus evaluate --run runs/h2mil
```

## 保留的训练与源码修正

公开代码默认 60 轮、Adam `lr=5e-5`/`weight_decay=5e-4`、固定顺序连续批次（默认 32）。每图 CE 求和后每批更新一次，尾批不丢弃；不使用调度器、随机打乱、梯度裁剪或 patch 截断。选模采用最后并列最优的验证集逐 WSI AUC，且仅在该轮训练过程的预测 AUC>0.8 或零起算轮数>总轮数−10 后具备资格。完整预算没有额外早停；测试集只在选模后使用。

原实现的 `LayerNorm(in_feats)` 会在输入、隐藏层和半宽分类层间重复使用而发生形状错误。这里按实际宽度提供 PyG 图级 LayerNorm；相同宽度仍共享对应模块。`h2mil_repaired` 另将头改为实际 C 类，并对 logits 做 CE；`source_compat` 保留原四概率 CE 和二分类指标条件。

来源兼容路线输出四个原始概率及原 argmax 到 `h2_original_p_*`、`h2_native_prediction`。标准二分类报告把原第 1 列作为阳性概率，其补数作为阴性概率，并明确使用 0.5 阈值；排名指标使用未改写的原第 1 列。标准二分类准确率可能不同于四类 argmax 的原生准确率，后者单独保存在验证指标 `h2_native_accuracy`，不混称原文结果。

IHPool 的投影 fitness 在当前公开代码中仅影响硬排序/归属，投影参数没有直接损失梯度；若干已定义池化层也未参与 forward。本接入保留并记录这些行为，没有凭空加入直通梯度。分组取整及第二级 `ratio>=1` 的特殊分支也保留。稀疏边收缩替换原来的 N×N 临时二值邻接矩阵，已与独立归属和边收缩计算对照；未更改池化语义。设备分配跟随输入，旧 PyG 调用通过兼容接口执行。

[论文](https://cdn.aaai.org/ojs/19976/19976-13-23989-1-2-20220628.pdf) 的设置为 512 像素 patch、1024 维 KimiaNet 特征及 50 轮；这与公开代码默认的 60 轮、固定四输出和第二级比例 4 等存在差异。当前配置名称指向已披露的源码与修正路线，不声称论文数值复现。更换 FM 需要重新生成三个真实层级的特征，不能只替换高倍率分支。

## 结构输出与验证

实例 H5 保留原始坐标、原节点类型/父节点、两次池化的归属、fitness、后续拓扑，以及映射回原高倍率行顺序的最终簇。fitness 投影没有直接损失梯度这一边界写入文件语义，不将其包装为因果归因或可学习注意力权重。

软件验证使用明确标记的独立高斯层级特征：

```bash
milplus demo --h2mil --output /tmp/h2mil_demo
```

完整范围须分别核验作者构图、两级池化、原始外层训练、精确续训、结构导出、三分类扩展和安装包。0.49 正式全库 2150 项、200 配置通过；本方法 49 项专项含实际作者 40 次逐步更新、42 级独立池化和 90 组原始选模边界。独立 wheel 两条原生 1024/256/128 维路线各 60 轮、共 120 次更新与真实重载通过。报告见 `docs/validation/h2mil.json` 和 `summary.json`。GPU、自然 WSI 和原论文数值实验尚未运行。
