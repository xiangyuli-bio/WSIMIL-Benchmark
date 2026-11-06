# PAMIL：原型注意力与完整分阶段训练

本适配对应 [Jiashuai-Liu/PAMIL](https://github.com/Jiashuai-Liu/PAMIL) 的 MICCAI 2024 原型注意力工作，固定提交 `7fe90096f64697bd6b68d4d7209e31f437a6b08f`。它与同名动态策略 PAMIL 是不同条目。作者仓库没有根目录许可文件；运行时读取外部固定 Git 源码，本库不打包复制其模型文件。

## 三个明确的执行配置

| ID | 标签 | 保留或修正的行为 |
|---|---|---|
| `pamil_prototype_source` | 单标签分类 | 保留作者 CE(softmax)、两阶段聚类中 `len(cors)==1` 的行为、冻结原型、LAST 后 fc 不恢复的行为 |
| `pamil_prototype_repaired` | 单标签分类 | 显式改为 CE(logits)、每片 min(N,local_clusters) 局部中心、按类索引实例挖掘、LAST 后恢复 fc |
| `pamil_prototype_multilabel` | 多标签分类 | 独立 sigmoid/BCE、每片局部聚类、按类索引实例挖掘与设备兼容修正 |

`--method pamil_prototype` 会按 task_type 选择分类修正版或多标签版；`PAMIL` 本身存在同名歧义，必须指定版本。输入宽度可随已登记 FM 改变，默认 fc/attention 宽度仍为 512/256。更改宽度属于显式实验配置。

全部配置保留两条预测路径：实例注意力聚合与原型分支。原型一致性项仍是源代码对 sigmoid 输出计算的非归一化 KL 项，未擅自改成 softmax KL；实例聚类和原型聚类项保留原始符号、归约和系数。可选 CLAM 式实例分类使用伪目标；它不同于 push 使用的真实/外部实例标注。多标签源码的 `elif` 会在启用实例分类后关闭原型聚类项，此行为明确保留。

## 额外监督与数据准备

固定版本的 active `push` 按实例类别筛选可投影的 patch。该路线需要**训练集的实例标签**，不能把 slide 标签复制到所有真实 patch 并宣称是原方法复现。普通 BRCA/COAD 特征包如果没有这类标注，仍可运行本库其他合适方法；PAMIL 本路线会明确报告缺少输入。

在独立 CSV 中提供且仅提供训练 slide 的标注路径：

```csv
slide_id,path
TRAIN_SLIDE_A,annotations/TRAIN_SLIDE_A.npy
TRAIN_SLIDE_B,annotations/TRAIN_SLIDE_B.npy
```

每个文件是安全可读取的整数 `[N]` 数组，与该 slide 的特征、真实坐标行顺序完全一致。`0..C-1` 对应 task.yaml 中 classes 的顺序；`-1` 表示不参与投影的背景/未标注实例。其他原始编码需先按明确规则转换并记录来源。禁止 pickle/object 字典、浮点类别、缺行、重复 slide、验证/测试 slide 混入。

```bash
milplus prepare-pamil --data /data/task \
  --annotations /data/task/train_annotations.csv \
  --output /data/task/pamil/manifest.json \
  --provenance '标注来源、版本、类别映射和背景处理规则'
```

随后在选定 encoder 下登记：

```yaml
encoders:
  uni:
    # 原有 dimension/index/权重/预处理登记保持完整
    pamil:
      manifest: pamil/manifest.json
```

manifest 将标注绑定到 FM、split、类别顺序、特征及坐标文件的内容哈希。训练输入检查和每次 push 都会核查；不为验证/测试准备或读取实例 GT。特征重新排序、坐标修改、切分变化后，必须重新建立有证据的对齐标注。

## 完整训练流程

1. 在独立种子随机流中，使用 TRAIN slide 进行局部 KMeans，再对局部中心做全局 KMeans。原聚类脚本与训练脚本是两个进程；合并入口会恢复训练 RNG，避免预计算改变训练随机流。显式使用 `n_init=10`，不受新版 sklearn 默认值变化影响。
2. warm 与 joint 按源代码冻结 prototype Parameter；`fc`、注意力、分类头参与 Adam。原配置中的其他分组学习率和 scheduler 未被 active trainer 使用，本适配也不伪造其效果。
3. 在 push 时，依据原型—类别权重确定类别，仅在包含该类的训练 bag 中，从对应实例标注中逐个选择不重复 patch。保存 slide、原始行号、原始精度坐标、类别、相似度和源文件哈希。
4. 将选定原始特征原地复制到原型中，保持 Parameter 对象和优化器的关系。若候选耗尽或某原型没有合格标注，则原子失败，不进入源代码的无限 while 或静默零向量替换。
5. LAST 冻结 fc，训练注意力与分类头；阶段结束后的源码行为和修复行为分别登记。允许多个有明确间隔的 push，所有声明的 push 必须落在预算内。

源码还需要解开 collate 包裹的实例标签并转换标量类别，才能在当前 NumPy/PyTorch 中执行 active push。原型得分全零时，用分母 1 保留零值，避免 0/0。修正版使用稳定 log-sigmoid；source 配置保留原 `log(sigmoid)`，非有限值会报错。默认不启用可选实例分类；source 分类配置拒绝该损坏分支，修正版则保留正确的每类索引及 one-hot 训练目标。

默认 350 epochs、80 warm、分类 330/multilabel 320 时 push、30 LAST 配置与作者设置相同。350 总预算会截断分类的最后 30 轮窗口，实际只有 20 轮 LAST；本适配保留这个边界。短预算必须同时包含 warm、joint、push、LAST，不能只跑普通 CE 就标为完整路线。

默认 `validation_selection: false` 对应作者 `early_stopping=false` 的最后一轮权重。启用 `validation_selection: true` 时按原 bag+ER+实例聚类验证目标选模，保存并列最优；最佳权重可能早于 push。原 trainer 没有依据 stop 标志提前跳出，适配同样完成固定预算。

```bash
milplus setup --method pamil_prototype
milplus run --data /data/task --method pamil_prototype \
  --config configs/pamil_prototype_repaired.yaml --output runs/pamil
milplus evaluate --run runs/pamil
```

不允许通用 max_instances 截断或额外梯度裁剪。缺少额外标注时不会退回 ABMIL。输出包含完整阶段记录、聚类及 push 来源；实例 HDF5 保存原行序注意力、未归一化原型相似度及可用的真实坐标，三者语义独立。推理不传入查询标签。

仅做软件演示时，可生成永久标记为 synthetic 的小数据包；其中的实例标签只用于测试代码，不能迁移为临床标注生成方法：

```bash
python -c "from milplus.pamil_demo import create_pamil_demo; create_pamil_demo('/tmp/pamil_demo')"
milplus run --data /tmp/pamil_demo --method pamil_prototype \
  --config configs/pamil_smoke.yaml --output runs/pamil_demo
milplus evaluate --run runs/pamil_demo
```

## 验证边界

独立检查直接执行固定 Git 的网络、训练循环数值主体、聚类主体和 push，列出兼容修正后比较完整 epoch 的损失、参数、Adam 状态和随机流；没有把适配器自己当作参考实现。另检查跨阶段续训、实例导出、标签独立性、失配输入与投影失败。

CPU 合成检查和已提取特征只验证软件与机制，不是 GPU 验证、真实 WSI 实验或论文分数复现。所有 164 条候选仍保留，PAMIL 的三配置只计一个家族。最终运行证据统一登记在 [VALIDATION.md](VALIDATION.md) 和 `docs/validation/`。
